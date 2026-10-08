"""
Build a larger evaluation set and run the pipeline over it.

The stored evaluation covers 26 dates that were picked by hand over time.
That is too few to say much (8 out of 10 has a 95% interval of 49-94%) and
hand-picking is its own bias. This script draws a reproducible sample of
detected anomalies and runs the full pipeline on the ones that have no
up-to-date output yet.

Sampling
--------
From the single-day ("shock" and "shock+trend") anomalies in
results/anomaly_detections.csv, stratified by year and direction so that no
period or direction dominates, with a fixed seed. Every labelled date
(data/labeling/labels_by_written_rule.csv) is always included, so the
labelled set stays a subset of what is evaluated. The chosen dates are
written to data/eval/eval_dates.csv, and that file - not the random draw -
is what later runs use, so the set does not shift under you.

Days that are not real moves
----------------------------
On some days the price series switches from an expired futures contract to
the next and shows a move no contract made. The checked ones are listed in
data/labeling/contract_switch_dates.csv. The detector no longer flags them,
and they are dropped here as well, wherever a date list is read, so an older
list that still names them does no harm. The answer key marks its own two
such dates EXCLUDE, and they do not count as labelled.

Running
-------
Each date is run as its own `python -m src.pipeline --date ...` process, one
after another, with a pause between them (GDELT and the free model tiers are
both rate-limited). A date is skipped if it already has an output from the
current pipeline version, so the script can be stopped and restarted, and
re-running it after a pipeline change redoes only what is stale.

Expect several minutes per date: a date that has never been retrieved makes
six GDELT queries at 10-25 seconds each, often with retries. The script
prints a line a minute while a date is running and an estimate of the time
left after each one; the pipeline's own output goes to
data/eval_cache/logs/<commodity>_<date>.log.

Failed news queries
-------------------
GDELT regularly rejects or times out a query even after retries. The run
carries on with whatever the other queries and sources returned, so a date
can finish "current" having searched with only part of its queries, and a
refusal on such a date may say more about the API than about the news. The
status line counts these dates.

--retry-incomplete deals with them in two steps. First it asks GDELT again
for just the queries that are not in the cache, slowly, one attempt at a
time, with no model call (--fill-cache does this step on its own). Then it
re-runs the pipeline only for the dates that now have more of their queries
answered than their stored run had; a date whose queries failed again is
left alone, since running it again would change nothing.

Usage:
    python -m scripts.run_eval_set --sample 100            # choose the dates (no runs)
    python -m scripts.run_eval_set --run                   # run what is missing or stale
    python -m scripts.run_eval_set --run --max 20          # ... at most 20 this session
    python -m scripts.run_eval_set --run --labelled-first  # hand-labelled dates first
    python -m scripts.run_eval_set --fill-cache            # re-ask failed news queries only
    python -m scripts.run_eval_set --run --retry-incomplete  # ... and re-run the dates that gained
    python -m scripts.run_eval_set --status                # what is done / stale / missing

Then:
    python -m src.evaluation.batch_runner --existing-only
    python -m src.evaluation.metrics
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "eval_dates.csv"
LABELS_PATH = REPO_ROOT / "data" / "labeling" / "labels_by_written_rule.csv"
LOG_DIR = REPO_ROOT / "data" / "eval_cache" / "logs"
MANIFEST_COLUMNS = ["date", "direction", "z_score", "pct_move", "anomaly_type",
                    "labelled", "stratum"]


def load_shock_anomalies(commodity: str = "coffee") -> pd.DataFrame:
    from src.config.commodities import get_commodity
    df = pd.read_csv(RESULTS_DIR / get_commodity(commodity).anomalies_file)
    df["pct_move"] = 100 * df["price"].pct_change()
    shocks = df[df["anomaly_type"].isin(["shock", "shock+trend"])].copy()
    shocks = shocks[~shocks["date"].isin(not_real_moves(commodity))]
    shocks["direction"] = shocks["z_score"].apply(lambda z: "up" if z > 0 else "down")
    shocks["stratum"] = shocks["date"].str[:4] + "-" + shocks["direction"]
    return shocks


def labelled_dates() -> set:
    """Dates the answer key says should be explained or refused. Dates it
    marks EXCLUDE (not a real move) are not labelled dates."""
    if not LABELS_PATH.exists():
        return set()
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        return {row["date"] for row in csv.DictReader(f)
                if (row.get("expected_outcome") or "").strip().upper() in ("EXPLAIN", "REFUSE")}


def not_real_moves(commodity: str = "coffee") -> set:
    """The checked contract-switch days: flagged moves that no contract made."""
    from src.config.commodities import load_contract_switches
    return set(load_contract_switches(commodity))


def stratified_sample(shocks: pd.DataFrame, n: int, seed: int, always: set) -> pd.DataFrame:
    """n dates across year-direction strata. Every date in `always` is taken
    first; the rest are drawn at random, each time from whichever stratum
    currently has the fewest dates, so the strata end up as even as their
    sizes allow."""
    rng = random.Random(seed)
    chosen, filled, pools = [], {}, {}
    for date, stratum in zip(shocks["date"], shocks["stratum"]):
        filled.setdefault(stratum, 0)
        if date in always:
            chosen.append(date)
            filled[stratum] += 1
        else:
            pools.setdefault(stratum, []).append(date)
    for stratum in sorted(pools):
        rng.shuffle(pools[stratum])
    while len(chosen) < n:
        open_strata = [s for s in sorted(pools) if pools[s]]
        if not open_strata:
            break
        stratum = min(open_strata, key=lambda s: filled[s])
        chosen.append(pools[stratum].pop())
        filled[stratum] += 1
    picked = shocks[shocks["date"].isin(chosen)].sort_values("date").copy()
    picked["labelled"] = picked["date"].isin(always)
    return picked


def write_manifest(sample: pd.DataFrame) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = sample[MANIFEST_COLUMNS].copy()
    out["z_score"] = out["z_score"].round(2)
    out["pct_move"] = out["pct_move"].round(2)
    out.to_csv(MANIFEST_PATH, index=False)


def read_manifest() -> list:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found - run with --sample N first.")
    unreal = not_real_moves()
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return [row["date"] for row in csv.DictReader(f) if row["date"] not in unreal]


def output_state(date: str, commodity: str = "coffee") -> str:
    """'current', 'stale' (older pipeline version, ended in a fault, or decided
    under a guard rule that has since changed) or 'missing'."""
    from src.pipeline import PIPELINE_VERSION
    path = RESULTS_DIR / f"pipeline_output_{commodity}_{date}.json"
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return "missing"
    if str(data.get("pipeline_version", "1")) != PIPELINE_VERSION:
        return "stale"
    if (data.get("outcome") or {}).get("is_fault"):
        return "stale"   # a fault is worth retrying
    if not guard_decision_is_current(data):
        return "stale"
    return "current"


def guard_decision_is_current(data: dict) -> bool:
    """Is the stored explain/refuse decision the one the current direction
    guard gives for this run's stored model answer and evidence tally?

    The guard is a fixed rule applied after the model answers, so changing
    the rule only changes the runs the old and new rule disagree on (the
    first version also withheld on a tie). Those are redone; everything else
    stays current without calling the model again."""
    from src.rag.outcome import guard_fires
    if data.get("decision_mode") == "blind":
        return True          # decided by the blind-evidence rule; no guard involved
    result = data.get("explanation_result") or {}
    withheld = "withheld_explanation" in result
    if not withheld and result.get("decision") != "EXPLAINED":
        return True          # the model itself refused; no guard rule changes that
    return withheld == guard_fires(data.get("direction_summary") or {})


def failed_queries(date: str, commodity: str = "coffee") -> tuple:
    """(queries that failed after retries, queries sent) in the stored run for
    this date; (0, 0) when there is no output or it recorded neither."""
    path = RESULTS_DIR / f"pipeline_output_{commodity}_{date}.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8")).get("retrieval_meta") or {}
    except (ValueError, OSError):
        return 0, 0
    return int(meta.get("network_failures") or 0), int(meta.get("network_attempts") or 0)


def incomplete_dates(dates: list, states: dict, commodity: str = "coffee") -> list:
    """Current dates whose run lost at least one news query."""
    return [d for d in dates if states.get(d) == "current" and failed_queries(d, commodity)[0] > 0]


def gdelt_requests(date: str, commodity: str = "coffee") -> list:
    """(query, params, startdatetime, enddatetime, maxrecords) for every GDELT
    request the pipeline would SEND for this date - built by the pipeline's
    own function, so the cache keys are the ones the pipeline looks up.
    Queries the pipeline only reads from the cache are left out: nothing is
    gained by asking for them."""
    from src import pipeline
    from src.config.commodities import get_commodity
    cfg = get_commodity(commodity)
    plan = pipeline.gdelt_query_plan(date, cfg.gdelt_queries,
                                     **pipeline.search_arguments(cfg))
    return [(r.query, r.params, r.startdatetime, r.enddatetime, r.maxrecords)
            for r in plan if r.live]


def uncached_requests(date: str, commodity: str = "coffee") -> list:
    from src import pipeline
    from src.rag.gdelt_client import cache_get
    return [r for r in gdelt_requests(date, commodity)
            if cache_get(pipeline.GDELT_CACHE_DIR, r[1], verbose=False) is None]


def gained_queries(date: str, commodity: str = "coffee") -> int:
    """How many more of this date's queries are answered in the cache now
    than its stored run had answered. Above zero, a re-run would see evidence
    the stored run did not."""
    path = RESULTS_DIR / f"pipeline_output_{commodity}_{date}.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8")).get("retrieval_meta") or {}
    except (ValueError, OSError):
        return 0
    answered_then = (int(meta.get("cache_hits") or 0) + int(meta.get("network_attempts") or 0)
                     - int(meta.get("network_failures") or 0))
    requests_now = gdelt_requests(date, commodity)
    answered_now = len(requests_now) - len(uncached_requests(date, commodity))
    return max(0, answered_now - answered_then)


# The wait after a rejected request. In 764 logged requests the answer rate
# was one in five whether the wait was 3, 5 or 10 seconds or several minutes
# (results/EVALUATION.md, section 9b), so a longer wait buys nothing that the
# logs can show and only makes the pass slower. It stays above GDELT's one
# request every 5 seconds. Each fill prints its own answer rate by wait.
WAIT_AFTER_REJECTION = 10.0
# Stop asking after this many rejections in a row. At one answer in five a
# run of 40 has a chance of about one in ten thousand of starting at any given
# request; past that the API is not answering and it is time to come back
# later. (Twenty would be reached by chance in about a third of 180-request
# passes.)
MAX_REJECTIONS_IN_A_ROW = 40


def why_failed(exc: Exception) -> tuple:
    """('rate_limited' | 'refused' | 'error', text) for a failed GDELT fetch.
    'refused' is GDELT answering with a message instead of results - what it
    does when it will not run a query as written - and asking again cannot
    help."""
    cause = exc.__cause__ or exc
    status = getattr(getattr(cause, "response", None), "status_code", None)
    if status == 429:
        return "rate_limited", "HTTP 429"
    if "not valid JSON" in str(cause):
        return "refused", str(cause)[:300]
    return "error", f"{type(cause).__name__}: {str(cause)[:200]}"


def fill_cache(dates: list, commodity: str = "coffee", rounds: int = 12,
               gap_seconds: float = 6.0) -> dict:
    """Ask GDELT for the requests that are not in the cache. No model is
    called and no pipeline output changes; this only fills the cache.

    One attempt per request per round, and never sooner than gap_seconds
    after the previous one (GDELT asks for at most one request every 5
    seconds). Each answer is cached as it arrives, so this can be stopped and
    restarted. It ends when everything is answered, after `rounds` passes, or
    after MAX_REJECTIONS_IN_A_ROW rejections in a row."""
    from src import pipeline
    from src.rag.gdelt_client import cache_put, fetch_gdelt

    asked = answered = in_a_row = 0
    refused = {}                    # query -> GDELT's message; not asked again
    errors = {}                     # other failures -> count
    by_wait = {}                    # wait before the request -> [asked, answered]
    wait = None
    gave_up = False
    for round_no in range(1, rounds + 1):
        missing = [(date, r) for date in dates for r in uncached_requests(date, commodity)
                   if r[0] not in refused]
        if not missing or gave_up:
            break
        n_dates = len({date for date, _ in missing})
        rate = f"{answered} of {asked} answered so far. " if asked else ""
        print(f"Round {round_no} of {rounds}: {len(missing)} news request(s) across {n_dates} "
              f"date(s) are not in the cache. {rate}", flush=True)
        for i, (date, (query, params, start, end, maxrecords)) in enumerate(missing, 1):
            if query in refused:
                continue
            if wait is not None:
                time.sleep(wait)
            tally = by_wait.setdefault("first" if wait is None else f"{wait:g}s", [0, 0])
            tally[0] += 1
            asked += 1
            try:
                response = fetch_gdelt(query, maxrecords=maxrecords,
                                       startdatetime=start, enddatetime=end,
                                       max_attempts=1, verbose=False)
            except RuntimeError as exc:
                kind, text = why_failed(exc)
                if kind == "refused":
                    refused[query] = text
                    print(f"    GDELT would not run this search, so it will not be asked again:\n"
                          f"      {query}\n      {text}", flush=True)
                    wait = gap_seconds
                else:
                    if kind == "error":
                        errors[text] = errors.get(text, 0) + 1
                    in_a_row += 1
                    wait = max(WAIT_AFTER_REJECTION, gap_seconds)
            else:
                cache_put(pipeline.GDELT_CACHE_DIR, params, response)
                answered += 1
                tally[1] += 1
                in_a_row = 0
                wait = gap_seconds
            if i % 10 == 0 or i == len(missing):
                print(f"    {i}/{len(missing)} asked this round; {answered} of {asked} answered "
                      f"in total", flush=True)
            if in_a_row >= MAX_REJECTIONS_IN_A_ROW:
                print(f"    {in_a_row} requests in a row were rejected. Stopping: the news API "
                      f"is not answering now. Run the same command again later; what was "
                      f"answered is kept.", flush=True)
                gave_up = True
                break
    still = sum(len(uncached_requests(date, commodity)) for date in dates)
    print(f"Asked {asked} time(s), {answered} answered. {still} news request(s) are still "
          f"not in the cache.", flush=True)
    if by_wait:
        print("Answered, by the wait before the request: "
              + "; ".join(f"{k}: {v[1]} of {v[0]}" for k, v in by_wait.items()), flush=True)
    for text, n in errors.items():
        print(f"  {n} request(s) failed for another reason: {text}", flush=True)
    return {"asked": asked, "answered": answered, "still_missing": still,
            "refused": refused, "by_wait": by_wait}


def status(dates: list) -> dict:
    states = {date: output_state(date) for date in dates}
    counts = {s: sum(1 for v in states.values() if v == s) for s in ("current", "stale", "missing")}
    print(f"{len(dates)} dates in the evaluation set: {counts['current']} current, "
          f"{counts['stale']} stale (older pipeline version, a fault, or decided under an "
          f"older guard rule), {counts['missing']} not run")
    incomplete = incomplete_dates(dates, states)
    if incomplete:
        print(f"  {len(incomplete)} of the current dates ran with one or more news queries "
              f"failed; --run --retry-incomplete asks again for just those queries and "
              f"re-runs the dates that gain from it")
    return states


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {rest % 60:02d}s"


def archive_output(date: str, commodity: str = "coffee"):
    """Before a date is run again under a new pipeline version, keep its
    existing output under results/archive/pipeline_v<N>/. Re-running writes
    to the same file, and the earlier runs are the "before" that any claimed
    improvement has to be measured against. Returns the archived path, or
    None if there was nothing to keep (no output, same version, or already
    archived)."""
    from src.pipeline import PIPELINE_VERSION
    path = RESULTS_DIR / f"pipeline_output_{commodity}_{date}.json"
    try:
        version = str(json.loads(path.read_text(encoding="utf-8")).get("pipeline_version", "1"))
    except (ValueError, OSError):
        return None
    if version == PIPELINE_VERSION:
        return None
    target = RESULTS_DIR / "archive" / f"pipeline_v{version}" / path.name
    if target.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    return target


def run_one(date: str, commodity: str, heartbeat_seconds: float = 60.0) -> int:
    """Run the pipeline for one date as a subprocess, writing its output to a
    log file and printing a line every heartbeat_seconds so a slow date is
    visibly still running rather than looking hung. Returns the exit code."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{commodity}_{date}.log"
    # Unbuffered, or the log stays empty until the date finishes and the
    # "live output" the heartbeat points at is not live.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    started = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.pipeline", "--date", date, "--commodity", commodity],
            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                return proc.wait(timeout=heartbeat_seconds)
            except subprocess.TimeoutExpired:
                print(f"    ... still running ({_fmt_duration(time.time() - started)}; "
                      f"live output in {log_path.relative_to(REPO_ROOT)})", flush=True)


def preview_reading(dates: list, commodity: str = "coffee") -> None:
    """For each date, print the documents the blind reading would be shown,
    without calling a model or saving anything (the pipeline's
    --preview-reading). The full output of each goes to a log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    for i, date in enumerate(dates, 1):
        log_path = LOG_DIR / f"preview_{commodity}_{date}.log"
        print(f"[{i}/{len(dates)}] {date} ...", flush=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            code = subprocess.run(
                [sys.executable, "-m", "src.pipeline", "--date", date, "--commodity", commodity,
                 "--preview-reading"],
                cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        text = log_path.read_text(encoding="utf-8", errors="replace")
        start = text.find("--- Reading preview")
        if code != 0 or start < 0:
            print(f"    did not finish (exit {code}); see {log_path.relative_to(REPO_ROOT)}")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if lines:
                print(f"    last line: {lines[-1][:200]}")
            continue
        print(text[start:].rstrip(), flush=True)


def last_log_line(date: str, commodity: str = "coffee") -> str:
    """The last non-empty line of a date's log, which for a run that exited
    with an error is the error."""
    try:
        lines = (LOG_DIR / f"{commodity}_{date}.log").read_text(
            encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "see the log"
    return next((line.strip()[:160] for line in reversed(lines) if line.strip()), "see the log")


def run_pending(dates: list, max_runs: int = None, sleep_seconds: float = 20.0,
                commodity: str = "coffee", retry_incomplete: bool = False) -> None:
    states = status(dates)
    pending = [d for d in dates if states[d] != "current"]
    if retry_incomplete:
        incomplete = incomplete_dates(dates, states, commodity)
        if incomplete:
            fill_cache(incomplete, commodity)
            gained = [d for d in incomplete if gained_queries(d, commodity) > 0]
            print(f"{len(gained)} of the {len(incomplete)} date(s) that lost news queries now "
                  f"have more of them answered and will be re-run. The other "
                  f"{len(incomplete) - len(gained)} are left as they are: nothing new to "
                  f"show the model.\n", flush=True)
            pending += gained
    if max_runs:
        pending = pending[:max_runs]
    if not pending:
        print("Nothing to run.")
        return
    print(f"Running {len(pending)} date(s), {sleep_seconds:.0f}s apart. Each date takes "
          f"several minutes, most of it waiting on the news API. Safe to stop (Ctrl+C) and "
          f"restart: finished dates are skipped.\n", flush=True)

    durations = []
    failures = 0
    for i, date in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {date} ...", flush=True)
        archive_output(date, commodity)
        started = time.time()
        code = run_one(date, commodity)
        took = time.time() - started
        durations.append(took)

        tier = "?"
        path = RESULTS_DIR / f"pipeline_output_{commodity}_{date}.json"
        if code != 0:
            tier = "DID NOT FINISH"     # any tier on disk is from an earlier run
        elif path.exists():
            try:
                tier = (json.loads(path.read_text(encoding="utf-8")).get("outcome") or {}).get("tier", "?")
            except ValueError:
                pass
        remaining = len(pending) - i
        eta = (sum(durations) / len(durations) + sleep_seconds) * remaining
        failed, sent = failed_queries(date, commodity) if code == 0 else (0, 0)
        print(f"    {tier}  [{output_state(date, commodity)}]  {_fmt_duration(took)}"
              + (f"  ({failed} of {sent} news queries failed)" if failed else "")
              + ("" if code == 0 else f"  (exit {code}: {last_log_line(date, commodity)})")
              + (f"   about {_fmt_duration(eta)} left" if remaining else ""), flush=True)
        failures = failures + 1 if code != 0 else 0
        if failures >= 2 and remaining:
            print(f"\nStopping: {failures} dates in a row did not finish. The same error "
                  f"would repeat for the other {remaining}. Fix it, then run this again.")
            break
        if remaining:
            time.sleep(sleep_seconds)
    print()
    status(dates)
    print("\nNext:\n    python -m src.evaluation.batch_runner --existing-only\n"
          "    python -m src.evaluation.metrics")


def main():
    parser = argparse.ArgumentParser(description="Build and run a stratified evaluation set")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Choose N anomaly dates and write the manifest")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run", action="store_true",
                        help="Run the pipeline for manifest dates with no current output")
    parser.add_argument("--max", type=int, default=None,
                        help="With --run or --fill-cache: at most this many dates")
    parser.add_argument("--sleep", type=float, default=20.0,
                        help="With --run: seconds between dates (default 20)")
    parser.add_argument("--labelled-first", action="store_true",
                        help="With --run: do the hand-labelled dates before the rest, so "
                             "decision accuracy is available early")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="With --run: ask again for the news queries that failed, then "
                             "re-run the dates that now have more of them answered")
    parser.add_argument("--fill-cache", action="store_true",
                        help="Ask GDELT again for every query not in the cache (no model "
                             "call, no pipeline run), for every date in the set")
    parser.add_argument("--rounds", type=int, default=12,
                        help="With --fill-cache: passes over the missing requests (default 12)")
    parser.add_argument("--preview-reading", action="store_true",
                        help="For the first --max dates (or the --dates given), show which "
                             "documents the blind reading would be shown. No model call, "
                             "nothing saved.")
    parser.add_argument("--dates", default=None, metavar="D1,D2",
                        help="With --preview-reading: these dates instead of the first --max")
    parser.add_argument("--providers", default=None,
                        help="With --run: model provider(s) to use, e.g. groq - so the whole "
                             "set is answered by one model family instead of whichever has "
                             "quota left (default: the configured order)")
    parser.add_argument("--status", action="store_true", help="Show what is done and what is not")
    args = parser.parse_args()

    if args.sample:
        shocks = load_shock_anomalies()
        sample = stratified_sample(shocks, args.sample, args.seed, labelled_dates())
        write_manifest(sample)
        by_year = sample["date"].str[:4].value_counts().sort_index()
        print(f"Wrote {len(sample)} dates to {MANIFEST_PATH} "
              f"({int(sample['labelled'].sum())} hand-labelled, "
              f"{int((sample['direction'] == 'up').sum())} up / "
              f"{int((sample['direction'] == 'down').sum())} down)")
        print("  by year: " + ", ".join(f"{y}: {c}" for y, c in by_year.items()))
    def in_order(dates: list) -> list:
        if not args.labelled_first:
            return dates
        labelled = labelled_dates()
        return [d for d in dates if d in labelled] + [d for d in dates if d not in labelled]

    if args.fill_cache:
        # With --max, fill only the dates a --run with the same options would
        # take first, so a short check does not wait on the whole set.
        to_fill = in_order(read_manifest())
        fill_cache(to_fill[:args.max] if args.max else to_fill, rounds=args.rounds)
        if not args.run:
            print("\nNext:\n    python -m scripts.run_eval_set --run --retry-incomplete")
    if args.preview_reading:
        if args.dates:
            to_preview = [d.strip() for d in args.dates.split(",") if d.strip()]
        else:
            to_preview = in_order(read_manifest())
            to_preview = to_preview[:args.max] if args.max else to_preview
        preview_reading(to_preview)
        return
    if args.providers:
        os.environ["EXPLAINER_PROVIDERS"] = args.providers     # inherited by each date's run
    if args.run:
        # Before anything slow: a mistyped provider name or a missing key would
        # otherwise fail every date, minutes into each one.
        from src.rag.explainer import check_providers
        print(f"Model provider(s): {', '.join(check_providers())}")
    if args.run:
        run_pending(in_order(read_manifest()), args.max, args.sleep,
                    retry_incomplete=args.retry_incomplete)
    elif args.status or not (args.sample or args.fill_cache):
        status(read_manifest())


if __name__ == "__main__":
    main()
