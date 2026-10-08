"""
Robustness evaluation: does the system refuse when it should?

An explanation rate says how often the system produces an explanation. It
says nothing about whether those explanations are warranted - a system that
explained everything would score 100%. The claim this project makes is the
opposite one: it declines rather than invent a cause. That claim can only be
tested on inputs where declining is the right answer. This module builds
those inputs and measures how often the system explains anyway.

Each experiment changes exactly one thing about a normal run and holds the
rest fixed. All of them go through src.pipeline.explain_from_documents, the
same code path the CLI uses.

  true        Real anomaly, its own evidence, the true move. The control:
              the explain rate here is the system's coverage.

  flip        Same date, same evidence, but the model is told the move went
              the OTHER way (+6.7% becomes -6.7%). Evidence that explains a
              rise cannot explain a fall. If the system explains both, its
              explanations are being fitted to whatever move it is told
              about.

  placebo     A quiet trading day (no anomaly on or near it), with the news
              actually retrieved for that day, and a fabricated large move
              whose size is drawn from real anomalies. Nothing happened, so
              there is nothing to explain: every explanation here is a false
              one. This is the false-explanation rate.

  transplant  Real anomaly and true move, but the evidence is another
              anomaly's (same direction, at least 60 days away), with its
              publication dates shifted to fit. See the report for how to
              read this one - a high rate is expected and marks a limit of
              the design rather than a model failure.

  closed_book The same model, asked about the same move with NO documents.
              On real dates this is the memorization control: an event the
              model saw in training can be "explained" without retrieval, so
              retrieval only demonstrably matters on dates after the model's
              knowledge cutoff (pass --cutoff). On placebo days any answer
              is an invention, which gives a no-retrieval hallucination rate
              to set beside the pipeline's.

Two verdicts are recorded per trial: what the MODEL decided, and the final
decision after the deterministic direction guard. Reporting both shows how
much of the refusal behaviour is the model's and how much is the guard's.

Trials are cached (data/eval_cache/), so an interrupted run resumes and a
finished run can be re-reported without calling anything. Faults are never
cached.

Needs working retrieval, the embedding model and an LLM key, so it has to run
on a machine with network access.

News for the placebo days is fetched first, with the same patient pass that
fills the cache for the evaluation set (the news API answers about one
request in five). A placebo day is only run once every one of its news
requests has been answered: a day run on part of its news has less to build
a false explanation from, and a refusal there would flatter the result. Days
still waiting are listed in the report and picked up by the next run.

For the blind rule every trial keeps its readings, so the report also shows
what stricter versions of the rule would have decided on the same readings -
on the invented moves, where every explanation is false, and on the real
ones, where a stricter rule may cost explanations.

Usage:
    python -m src.evaluation.robustness                      # everything
    python -m src.evaluation.robustness --experiments placebo,flip
    python -m src.evaluation.robustness --providers groq --cutoff 2024-06-30
                                        # one model family; split by its knowledge cutoff
    python -m src.evaluation.robustness --prompt-versions blind,v2
                                        # the blind rule next to the earlier model-decides path
    python -m src.evaluation.robustness --report-only        # rebuild the report from cache
    python -m src.evaluation.robustness --providers groq --experiments placebo --n-placebo 12
                                        # the refusal test alone, on 12 quiet days
"""

import argparse
import contextlib
import copy
import io
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (   # noqa: E402
    EXPLAINED, REFUSED, FAULT, fmt_rate, load_labels, rate,
)

RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "data" / "eval_cache"
EVIDENCE_CACHE = CACHE_DIR / "evidence"
TRIAL_CACHE = CACHE_DIR / "trials"
# Blind readings of a date's documents. A reading does not depend on the
# move, so the true-move and flipped-move trials of a date share one.
READING_CACHE = CACHE_DIR / "readings"

# Not a prompt: the name the harness uses for the blind-evidence decision
# (src/rag/blind_evidence.py) so it can be run next to the prompt versions of
# the earlier decision path on the same evidence.
BLIND = "blind"

EXPERIMENTS = ["true", "flip", "placebo", "transplant", "closed_book"]

# A placebo day must be quiet on its own and have no flagged day close by.
PLACEBO_MAX_ABS_Z = 0.5
PLACEBO_MAX_ABS_CUMULATIVE_Z = 1.0
PLACEBO_CLEAR_TRADING_DAYS = 5
TRANSPLANT_MIN_DAYS_APART = 60

CLOSED_BOOK_PROMPT = """You are asked about a coffee futures price move. No documents are provided.

DATE: {anomaly_date}
MOVE: {price_description}

Using only what you already know, state what caused this specific move on this specific date.

If you do not know what happened on this date, say so. Do not infer a cause from general knowledge of coffee markets, from the season, or from the size and direction of the move - a plausible guess is not an answer.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "unknown": true | false,
  "cause": "...",
  "confidence": "high" | "medium" | "low"
}}
"""


# ---------------------------------------------------------------------------
# Date selection
# ---------------------------------------------------------------------------

def load_detections(commodity: str = "coffee") -> pd.DataFrame:
    from src.config.commodities import get_commodity
    return pd.read_csv(RESULTS_DIR / get_commodity(commodity).anomalies_file)


def real_dates(labels: dict, limit: int = None, df: pd.DataFrame = None) -> list:
    """Real anomaly dates to test: every hand-labelled date plus every date
    that already has a stored coffee run, oldest first - restricted to dates
    the detections file actually flags. A stored run on a day that is not a
    detected anomaly (the pipeline will run on any date if asked) is not a
    "real anomaly" and would blur the control condition."""
    dates = set(labels)
    for path in RESULTS_DIR.glob("pipeline_output_coffee_*.json"):
        dates.add(path.stem.replace("pipeline_output_coffee_", ""))
    if df is not None:
        flagged = set(df.loc[df["anomaly_flag"].fillna(False).astype(bool), "date"])
        dates &= flagged
    out = sorted(dates)
    return out[-limit:] if limit else out


def select_placebo_dates(df: pd.DataFrame, n: int, seed: int) -> list:
    """Quiet trading days, spread across years.

    Quiet means: not flagged, a small daily z-score, a small cumulative
    z-score, and no flagged day within PLACEBO_CLEAR_TRADING_DAYS trading
    days either side - so news about a nearby real move cannot be what the
    system latches onto.
    """
    flagged = df["anomaly_flag"].fillna(False).astype(bool).to_numpy()
    near_flag = [False] * len(df)
    for i, is_flagged in enumerate(flagged):
        if is_flagged:
            lo = max(0, i - PLACEBO_CLEAR_TRADING_DAYS)
            hi = min(len(df), i + PLACEBO_CLEAR_TRADING_DAYS + 1)
            for j in range(lo, hi):
                near_flag[j] = True

    quiet = df[
        (~pd.Series(near_flag, index=df.index))
        & (df["z_score"].abs() < PLACEBO_MAX_ABS_Z)
        & (df["cumulative_z_score"].abs() < PLACEBO_MAX_ABS_CUMULATIVE_Z)
    ]
    if quiet.empty:
        return []

    rng = random.Random(seed)
    by_year = {}
    for date in quiet["date"]:
        by_year.setdefault(date[:4], []).append(date)
    for dates in by_year.values():
        rng.shuffle(dates)

    # round-robin across years so no single year dominates the sample
    picked, years = [], sorted(by_year)
    while len(picked) < n and any(by_year[y] for y in years):
        for year in years:
            if by_year[year] and len(picked) < n:
                picked.append(by_year[year].pop())
    return sorted(picked)


def fabricated_move(df: pd.DataFrame, date: str, seed: int) -> dict:
    """An anomaly record for a quiet day, claiming a move the size of a real
    shock anomaly. Size and sign are copied from a randomly chosen real
    single-day anomaly, so the fabricated moves have the same distribution as
    the real ones."""
    rng = random.Random(f"{seed}:{date}")
    shocks = df[df["anomaly_type"].isin(["shock", "shock+trend"])].reset_index()
    donor = shocks.iloc[rng.randrange(len(shocks))]
    prior = df["price"].shift(1)[donor["index"]]
    pct = 100 * (donor["price"] - prior) / prior
    row = df[df["date"] == date].iloc[0]
    return {
        "date": date,
        "price": float(row["price"]),
        "z_score": float(donor["z_score"]),
        "anomaly_flag": True,
        "anomaly_type": "shock",
        "pct_move": float(pct),
        "direction": "up" if donor["z_score"] > 0 else "down",
        "placebo": True,
        "true_z_score": float(row["z_score"]),
        "move_copied_from": donor["date"],
    }


def flipped(anomaly: dict) -> dict:
    """The same anomaly with the direction of the move reversed."""
    out = dict(anomaly)
    for key in ("pct_move", "z_score", "cumulative_pct_move", "cumulative_z_score"):
        if out.get(key) is not None:
            out[key] = -out[key]
    out["direction"] = "down" if anomaly.get("direction") == "up" else "up"
    out["flipped"] = True
    return out


def transplant_pairs(anomalies: dict, seed: int) -> dict:
    """date -> donor date. The donor is another real anomaly in the SAME
    direction (so this is not a second flip test) at least
    TRANSPLANT_MIN_DAYS_APART days away."""
    rng = random.Random(seed)
    pairs = {}
    for date, anomaly in sorted(anomalies.items()):
        d = datetime.strptime(date, "%Y-%m-%d")
        candidates = [
            other for other, a in sorted(anomalies.items())
            if other != date and a.get("direction") == anomaly.get("direction")
            and abs((datetime.strptime(other, "%Y-%m-%d") - d).days) >= TRANSPLANT_MIN_DAYS_APART
        ]
        if candidates:
            pairs[date] = rng.choice(candidates)
    return pairs


def redate_documents(documents: list, donor_date: str, target_date: str) -> list:
    """Copy another date's evidence and shift every publication date by the
    gap between the two anomalies, so each document is as many days before
    the target move as it was before its own. Without this the transplant
    would be given away by the dates and would test date-checking rather
    than content."""
    from src.rag.evidence import annotate_recency
    gap = datetime.strptime(target_date, "%Y-%m-%d") - datetime.strptime(donor_date, "%Y-%m-%d")
    out = copy.deepcopy(documents)
    for doc in out:
        for field in ("publication_date", "session_date"):
            value = doc.get(field) or ""
            try:
                doc[field] = (datetime.strptime(value[:10], "%Y-%m-%d") + gap).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return annotate_recency(out, target_date)


# ---------------------------------------------------------------------------
# Evidence and trials (cached)
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def news_complete(date: str) -> bool:
    """True when every news request the pipeline sends for this date has an
    answer in the cache."""
    from scripts.run_eval_set import uncached_requests
    return not uncached_requests(date)


def fill_news(dates: list) -> dict:
    """Ask the news API for every request these dates still lack, keeping at
    it the way the evaluation set's cache-filling pass does."""
    from scripts.run_eval_set import fill_cache
    return fill_cache(dates)


def _evidence_path(date: str, retrieval: str, text_max_age: int = None) -> Path:
    """Where a date's retrieved evidence is cached. The name carries the
    pipeline version (a new version may send different queries) and, for
    cache-only retrieval, how many of the date's queries the GDELT cache
    holds - so evidence gathered before the cache was filled in is not
    served afterwards as if it were complete."""
    from src.pipeline import PIPELINE_VERSION
    tag = f"p{PIPELINE_VERSION}"
    if retrieval == "cache":
        from scripts.run_eval_set import gdelt_requests, uncached_requests
        total = len(gdelt_requests(date))
        tag += f"__q{total - len(uncached_requests(date))}of{total}"
    if text_max_age is not None:
        tag += f"__text{text_max_age}"
    return EVIDENCE_CACHE / f"{retrieval}__{date}__{tag}.json"


def evidence_for(date: str, retrieval: str = "live", use_cache: bool = True,
                 text_max_age: int = None) -> tuple:
    """(documents, retrieval_meta_dict) for a date's window.

    retrieval="live"  the full production retrieval (GDELT, news aggregator,
                      article fetch).
    retrieval="cache" on-disk GDELT and text caches only: fast, repeatable,
                      but only possible for dates retrieved before.
    text_max_age      as in the pipeline's blind path: article text is fetched
                      only for documents at most this many trading days old,
                      the ones the reading can be shown. None fetches all.
    """
    path = _evidence_path(date, retrieval, text_max_age)
    if use_cache and path.exists():
        cached = _read_json(path)
        if cached:
            return cached["documents"], cached["retrieval_meta"]

    from src import pipeline
    from src.config.commodities import get_commodity
    cfg = get_commodity("coffee")
    kwargs = dict(cache_dir=pipeline.GDELT_CACHE_DIR, fallback_dir=None, return_meta=True)
    if retrieval == "cache":
        kwargs.update(commodity_key=None, gdelt_live=False, fetch_text="cache")
    else:
        kwargs.update(commodity_key=cfg.key, fetch_text_max_age=text_max_age)
    with contextlib.redirect_stdout(io.StringIO()):
        documents, meta = pipeline.retrieve_evidence(
            date, cfg.gdelt_queries, cfg.semantic_reference_query,
            **pipeline.search_arguments(cfg), **kwargs)

    if documents and not any(d.get("semantic_score") is not None for d in documents):
        raise SystemExit(
            "Semantic scores are missing (the embedding model did not load). The gate "
            "would run on keyword scores alone, which is not the system being evaluated. "
            "Fix the model load before running this.")
    meta_dict = meta.to_dict()
    # An outage is not evidence. If nothing could be retrieved at all, do not
    # remember that as this date's evidence - the next run should try again.
    # A partial failure (some queries rate-limited, others answered) is kept:
    # that is the normal state of this retrieval, and the trial records how
    # many queries failed so thin evidence is visible in the report.
    if meta.retrieval_ran() and documents:
        _write_json(path, {"date": date, "retrieval": retrieval,
                           "documents": documents, "retrieval_meta": meta_dict})
    return documents, meta_dict


def run_signature(providers: list, retrieval: str, seed: int) -> dict:
    """Everything that makes one harness run comparable with another. Stored
    in every trial and part of its cache key, so trials made under different
    settings never answer for each other and are never pooled in a report."""
    from src.pipeline import PIPELINE_VERSION
    from src.rag.explainer import normalize_providers
    return {
        "providers": "+".join(normalize_providers(providers)) if providers else "default",
        "retrieval": retrieval,
        "seed": seed,
        "pipeline_version": PIPELINE_VERSION,
    }


def _signature_slug(signature: dict) -> str:
    return (f"{signature['providers']}__{signature['retrieval']}"
            f"__s{signature['seed']}__p{signature['pipeline_version']}")


def _trial_path(experiment: str, date: str, prompt_version: str, signature: dict) -> Path:
    return TRIAL_CACHE / (f"{experiment}__{date}__{prompt_version}__"
                          f"{_signature_slug(signature)}.json")


def pipeline_trial(experiment: str, anomaly: dict, documents: list, retrieval_meta: dict,
                   prompt_version: str, providers: list = None, extra: dict = None) -> dict:
    """Run the post-retrieval stage once and reduce it to a trial record."""
    from src import pipeline
    from src.rag import outcome as oc

    blind = prompt_version == BLIND
    try:
        output = pipeline.explain_from_documents(
            anomaly, copy.deepcopy(documents), retrieval_meta,
            prompt_version=None if blind else prompt_version, providers=providers,
            decision_mode="blind" if blind else "legacy",
            reading_cache_dir=READING_CACHE if blind else None, verbose=False)
    except Exception as exc:  # noqa: BLE001 - one bad response must not stop a long run
        return {"experiment": experiment, "date": anomaly["date"],
                "direction": anomaly.get("direction"), "prompt_version": prompt_version,
                "verdict": FAULT, "model_verdict": FAULT, "guard_fired": False,
                "n_documents": len(documents),
                "reason": f"{type(exc).__name__}: {exc}"[:200], **(extra or {})}
    er = output["explanation_result"]
    tier = output["outcome"]["tier"]

    if tier in oc.FAULT_TIERS:
        verdict = model_verdict = FAULT
    else:
        verdict = EXPLAINED if tier in oc.EXPLAINED_TIERS else REFUSED
        # What the model itself decided, before the direction guard.
        model_said_explain = er.get("decision") == "EXPLAINED" or "withheld_explanation" in er
        model_verdict = EXPLAINED if model_said_explain else REFUSED

    text = er.get("explanation") or (er.get("withheld_explanation") or {}).get("explanation")
    blind_block = er.get("blind_evidence") or {}
    kept = ({"readings": blind_block.get("readings") or [],
             "rule_variants": blind_block.get("variants") or [],
             "pct_move": anomaly.get("pct_move"),
             "reading_calls": er.get("reading_calls")}
            if blind and "blind_evidence" in er else {})
    return {
        "experiment": experiment,
        "date": anomaly["date"],
        "claimed_move": output["price_description"],
        "direction": anomaly.get("direction"),
        "prompt_version": BLIND if blind else (er.get("prompt_version") or prompt_version),
        "decision_mode": "blind" if blind else "legacy",
        "verdict": verdict,
        "model_verdict": model_verdict,
        "guard_fired": "withheld_explanation" in er,
        "tier": tier,
        "model_used": er.get("model_used"),
        "provider": er.get("provider"),
        "n_documents": len(documents),
        "retrieval_failures": (retrieval_meta or {}).get("network_failures", 0),
        "n_accepted": output["gate_result"]["accepted_document_count"],
        "gate_decision": output["gate_result"]["decision"],
        "direction_summary": output["direction_summary"],
        "evidence_recency": output["evidence_recency"],
        "explanation": text,
        "reason": er.get("reason"),
        **kept,
        **(extra or {}),
    }


# What version 11 added to the rule: a report supports an explanation only if
# it is about this market, gives an event as the reason and describes a move
# of this size (src/rag/blind_evidence.py).
FIT_THE_MOVE = ("this_market", "event_reasons", "sized_support")

# Other versions of the rule the report scores from stored readings
# (src/rag/blind_evidence.py describes each): (key, label, variants added,
# variants taken away). The first row is the pipeline as each trial ran.
RULE_ROWS = (
    ("as_run", "As the trials ran", (), ()),
    ("fit", "A report must fit the move: this market, an event as the reason, "
            "a move of this size (the rule from version 11)", FIT_THE_MOVE, ()),
    ("before_fit", "Any report with a reason counts (the rule before version 11)",
     (), FIT_THE_MOVE),
    ("size", "Ignore reports of a move under a quarter the size of the claimed one",
     ("size_consistent",), FIT_THE_MOVE),
    ("reports", "Only reports of a price move, not background events",
     ("reports_only",), FIT_THE_MOVE),
    ("gate", "Only articles that cleared the relevance filter itself", ("gate_only",), ()),
)


def _row_variants(trial: dict, extra: tuple = (), without: tuple = ()) -> tuple:
    return tuple(dict.fromkeys(x for x in tuple(trial.get("rule_variants") or ()) + tuple(extra)
                               if x not in without))


def rule_verdict(trial: dict, extra: tuple = (), without: tuple = ()) -> str:
    """What a blind trial would have ended with under another version of the
    rule, recomputed from its stored readings. A trial that kept no readings
    (it stopped before the reading, or predates this field) keeps its verdict."""
    from src.rag import blind_evidence
    if trial.get("readings") is None or trial.get("verdict") not in (EXPLAINED, REFUSED):
        return trial.get("verdict")
    decided = blind_evidence.decide(trial["readings"], trial.get("direction"),
                                    variants=_row_variants(trial, extra, without),
                                    actual_pct=trial.get("pct_move"))
    return EXPLAINED if decided["decision"] == "EXPLAIN" else REFUSED


def sizes_in_words_recorded(trials: list) -> bool:
    """Did every stored reading record how big the document's words made the
    move? Readings made before version 11 did not, and for those the "fit"
    row can only count a size the document states as a number."""
    readings = [r for t in trials if t.get("prompt_version") == BLIND
                for r in t.get("readings") or []]
    return bool(readings) and all("move_size" in r for r in readings)


def rules_on_stored_readings(trials: list) -> list:
    """For the blind trials: the explained rate of each condition under each
    row of RULE_ROWS. On the placebo row every explanation is a false one."""
    rows = []
    blind = [t for t in trials if t.get("prompt_version") == BLIND
             and t.get("readings") is not None]
    for key, label, extra, without in RULE_ROWS:
        if key != "as_run" and all(
                set(_row_variants(t, extra, without)) == set(t.get("rule_variants") or ())
                for t in blind):
            continue        # the same rule the trials ran under: nothing to show
        row = {"key": key, "label": label}
        for experiment in ("placebo", "flip", "true"):
            scored = [rule_verdict(t, extra, without) for t in blind
                      if t["experiment"] == experiment]
            scored = [v for v in scored if v in (EXPLAINED, REFUSED)]
            row[experiment] = rate(sum(1 for v in scored if v == EXPLAINED), len(scored))
        rows.append(row)
    return rows


def closed_book_trial(anomaly: dict, providers: list = None, source: str = "real") -> dict:
    """Ask the model about the move with no documents at all."""
    from src import pipeline
    from src.rag import explainer

    description = pipeline.describe_move(anomaly)
    prompt = CLOSED_BOOK_PROMPT.format(anomaly_date=anomaly["date"],
                                       price_description=description)
    record = {"experiment": "closed_book", "date": anomaly["date"], "source": source,
              "claimed_move": description, "direction": anomaly.get("direction"),
              "prompt_version": "closed_book"}
    try:
        info = explainer._call_model_ex(
            prompt, temperature=explainer.EXPLANATION_TEMPERATURE,
            max_output_tokens=explainer.EXPLANATION_MAX_OUTPUT_TOKENS, providers=providers)
        parsed = json.loads(info["response"].text)
        if not isinstance(parsed, dict):
            raise ValueError("response is not a JSON object")
    except Exception as exc:  # noqa: BLE001 - model or parse failure
        return {**record, "verdict": FAULT, "model_verdict": FAULT,
                "reason": f"{type(exc).__name__}: {exc}"[:200]}

    explainer.log_call(anomaly["date"], getattr(info["response"], "usage_metadata", None),
                       "CLOSED_BOOK", model=info["model"], provider=info["provider"],
                       latency_ms=info["latency_ms"], documents_in_prompt=0,
                       prompt_version="closed_book")
    cause = (parsed.get("cause") or "").strip()
    answered = (not parsed.get("unknown")) and bool(cause)
    verdict = EXPLAINED if answered else REFUSED
    return {**record, "verdict": verdict, "model_verdict": verdict,
            "model_used": info["model"], "provider": info["provider"],
            "explanation": cause or None, "confidence": parsed.get("confidence")}


def with_current_guard(trial: dict) -> dict:
    """The trial with its final verdict set by the current direction-guard
    rule. A trial stores the model's own verdict and the evidence tally, and
    the guard is a fixed rule over those two, so trials cached under an
    earlier rule are re-scored here instead of calling the model again."""
    from src.rag.outcome import guard_fires
    if (trial.get("model_verdict") != EXPLAINED or "direction_summary" not in trial
            or trial.get("decision_mode") == "blind"):     # no guard in that path
        return trial
    fires = guard_fires(trial.get("direction_summary") or {})
    return {**trial, "guard_fired": fires, "verdict": REFUSED if fires else EXPLAINED}


def cached_trial(path: Path, run, use_cache: bool = True) -> dict:
    """Return the cached trial at `path`, or run it and cache the result.
    A fault is returned but not cached, so the next run retries it."""
    if use_cache and path.exists():
        cached = _read_json(path)
        if cached:
            return cached
    trial = run()
    if trial.get("verdict") != FAULT:
        _write_json(path, trial)
    return trial


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_trials(experiments: list, real: list, placebo: list, df: pd.DataFrame,
               prompt_versions: list, retrieval: str, seed: int,
               providers: list = None, sleep_seconds: float = 0.0,
               use_cache: bool = True, verbose: bool = True, waiting: list = None) -> list:
    """Run (or read from the cache) every trial asked for. Placebo days whose
    news requests have not all been answered are not run; their dates are
    appended to `waiting`."""
    from src import pipeline
    from src.config.commodities import get_commodity
    from src.rag import blind_evidence

    cfg = get_commodity("coffee")
    # The blind reading is only ever shown recent documents, and the pipeline
    # fetches article text for those alone. The earlier model-decides path
    # reads the whole window, so it needs the text of all of it.
    text_max_age = (blind_evidence.MAX_AGE_TRADING_DAYS
                    if list(prompt_versions) == [BLIND] else None)
    tag = run_signature(providers, retrieval, seed)
    trials = []

    def log(message):
        if verbose:
            print(message, flush=True)

    def do(path, run):
        was_cached = use_cache and path.exists()
        trial = cached_trial(path, lambda: {**run(), "run": tag}, use_cache)
        trials.append(trial)
        log(f"    {trial['experiment']:<11} {trial['date']}  "
            f"{trial.get('prompt_version', ''):<11} -> {trial['verdict']}"
            + (" (guard)" if trial.get("guard_fired") else "")
            + (" [cached]" if was_cached else ""))
        if not was_cached and sleep_seconds:
            time.sleep(sleep_seconds)

    anomalies = {}
    uses_real_dates = any(e in experiments for e in ("true", "flip", "transplant", "closed_book"))
    for date in (real if uses_real_dates else []):
        try:
            anomalies[date] = pipeline.get_anomaly(date, cfg.anomalies_file)
        except SystemExit:
            log(f"  {date}: not in the detections file, skipped")
    pairs = transplant_pairs(anomalies, seed) if "transplant" in experiments else {}

    needs_real_evidence = any(e in experiments for e in ("true", "flip", "transplant"))
    for date, anomaly in anomalies.items():
        log(f"  {date} ({anomaly['direction']})")
        if needs_real_evidence:
            documents, meta = evidence_for(date, retrieval, use_cache, text_max_age=text_max_age)
        for version in prompt_versions:
            if "true" in experiments:
                do(_trial_path("true", date, version, tag),
                   lambda: pipeline_trial("true", anomaly, documents, meta, version, providers))
            if "flip" in experiments:
                do(_trial_path("flip", date, version, tag),
                   lambda: pipeline_trial("flip", flipped(anomaly), documents, meta,
                                          version, providers))
            if "transplant" in experiments and date in pairs:
                donor = pairs[date]
                donor_docs, donor_meta = evidence_for(donor, retrieval, use_cache,
                                                      text_max_age=text_max_age)
                do(_trial_path("transplant", date, version, tag),
                   lambda: pipeline_trial(
                       "transplant", anomaly, redate_documents(donor_docs, donor, date),
                       donor_meta, version, providers, extra={"evidence_from": donor}))
        if "closed_book" in experiments:
            do(_trial_path("closed_book", date, "closed_book", tag),
               lambda: closed_book_trial(anomaly, providers, source="real"))

    if "placebo" in experiments or "closed_book" in experiments:
        for date in placebo:
            fake = fabricated_move(df, date, seed)
            log(f"  {date} (placebo, claimed {fake['pct_move']:+.1f}%)")
            if "placebo" in experiments:
                if not news_complete(date):
                    log("    waiting: not all of this day's news requests have been answered")
                    if waiting is not None:
                        waiting.append(date)
                    continue
                # A quiet day has never been retrieved before, so this is
                # always live retrieval regardless of --retrieval.
                documents, meta = evidence_for(date, "live", use_cache,
                                               text_max_age=text_max_age)
                for version in prompt_versions:
                    do(_trial_path("placebo", date, version, tag),
                       lambda: pipeline_trial("placebo", fake, documents, meta, version,
                                              providers, extra={"placebo": True}))
            if "closed_book" in experiments:
                do(_trial_path("closed_book_placebo", date, "closed_book", tag),
                   lambda: closed_book_trial(fake, providers, source="placebo"))
    return trials


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _explained_rate(trials: list, key: str = "verdict") -> dict:
    graded = [t for t in trials if t.get(key) in (EXPLAINED, REFUSED)]
    return rate(sum(1 for t in graded if t[key] == EXPLAINED), len(graded))


def summarize(trials: list, labels: dict, cutoff: str = None) -> dict:
    versions = sorted({t["prompt_version"] for t in trials
                       if t["experiment"] != "closed_book"})
    summary = {"by_prompt_version": {}, "n_trials": len(trials),
               "n_faults": sum(1 for t in trials if t.get("verdict") == FAULT),
               "models": {}}
    for t in trials:
        name = t.get("model_used") or "not recorded"
        summary["models"][name] = summary["models"].get(name, 0) + 1

    for version in versions:
        block = {}
        for experiment in ("true", "flip", "placebo", "transplant"):
            rows = [t for t in trials
                    if t["experiment"] == experiment and t["prompt_version"] == version]
            if rows:
                block[experiment] = {
                    "model_only": _explained_rate(rows, "model_verdict"),
                    "with_guard": _explained_rate(rows, "verdict"),
                    "gate_passed": rate(sum(1 for t in rows if t.get("gate_decision") == "EXPLAIN"),
                                        len(rows)),
                }

        # Paired: of the dates explained under the true move, how many were
        # ALSO explained under the opposite move?
        true_by_date = {t["date"]: t for t in trials
                        if t["experiment"] == "true" and t["prompt_version"] == version}
        flip_by_date = {t["date"]: t for t in trials
                        if t["experiment"] == "flip" and t["prompt_version"] == version}
        for key, label in (("model_verdict", "model_only"), ("verdict", "with_guard")):
            base = [d for d, t in true_by_date.items()
                    if t[key] == EXPLAINED and flip_by_date.get(d, {}).get(key) in (EXPLAINED, REFUSED)]
            both = [d for d in base if flip_by_date[d][key] == EXPLAINED]
            block.setdefault("explained_both_directions", {})[label] = {
                **rate(len(both), len(base)), "dates": both}

        # Against the hand labels, for the control runs.
        graded = [(labels[d]["expected_outcome"].strip().upper(), t)
                  for d, t in true_by_date.items()
                  if d in labels and t["verdict"] in (EXPLAINED, REFUSED)]
        for key, label in (("model_verdict", "model_only"), ("verdict", "with_guard")):
            exp = [t for want, t in graded if want == "EXPLAIN"]
            ref = [t for want, t in graded if want == "REFUSE"]
            block.setdefault("labelled", {})[label] = {
                "explain_recall": rate(sum(1 for t in exp if t[key] == EXPLAINED), len(exp)),
                "refusal_specificity": rate(sum(1 for t in ref if t[key] == REFUSED), len(ref)),
            }
        summary["by_prompt_version"][version] = block

    closed_real = [t for t in trials if t["experiment"] == "closed_book" and t.get("source") == "real"]
    closed_placebo = [t for t in trials
                      if t["experiment"] == "closed_book" and t.get("source") == "placebo"]
    closed = {"real": _explained_rate(closed_real), "placebo": _explained_rate(closed_placebo)}
    if cutoff:
        closed["cutoff"] = cutoff
        closed["real_before_cutoff"] = _explained_rate([t for t in closed_real if t["date"] <= cutoff])
        closed["real_after_cutoff"] = _explained_rate([t for t in closed_real if t["date"] > cutoff])
        for version in versions:
            rows = [t for t in trials if t["experiment"] == "true" and t["prompt_version"] == version]
            summary["by_prompt_version"][version]["true_before_cutoff"] = \
                _explained_rate([t for t in rows if t["date"] <= cutoff])
            summary["by_prompt_version"][version]["true_after_cutoff"] = \
                _explained_rate([t for t in rows if t["date"] > cutoff])
    summary["closed_book"] = closed
    if any(t.get("readings") is not None for t in trials):
        summary["rules_on_stored_readings"] = rules_on_stored_readings(trials)
        summary["sizes_in_words_recorded"] = sizes_in_words_recorded(trials)
    return summary


# ---------------------------------------------------------------------------
# Report and chart
# ---------------------------------------------------------------------------

def make_chart(summary: dict, version: str, path: Path) -> bool:
    """One figure: share of trials explained, per condition. Returns False if
    matplotlib is unavailable (the report is still written)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    block = summary["by_prompt_version"].get(version, {})
    closed = summary.get("closed_book", {})
    bars = []   # (label, rate dict, colour)
    good, bad, grey = "#3b7a57", "#b5483a", "#7d8a97"

    def add(label, r, colour):
        if r and r.get("n"):
            bars.append((label, r, colour))

    add("Real anomaly\n(should explain\nsome)", block.get("true", {}).get("with_guard"), good)
    add("Direction\nflipped", block.get("flip", {}).get("model_only"), bad)
    add("Direction flipped\n+ guard", block.get("flip", {}).get("with_guard"), bad)
    add("Quiet day,\ninvented move", block.get("placebo", {}).get("model_only"), bad)
    add("Quiet day\n+ guard", block.get("placebo", {}).get("with_guard"), bad)
    add("No documents,\nquiet day", closed.get("placebo"), grey)
    add("No documents,\nreal anomaly", closed.get("real"), grey)
    if not bars:
        return False

    fig, ax = plt.subplots(figsize=(max(7.5, 1.35 * len(bars)), 4.6))
    xs = range(len(bars))
    values = [b[1]["pct"] for b in bars]
    lower = [b[1]["pct"] - b[1]["ci95"][0] for b in bars]
    upper = [b[1]["ci95"][1] - b[1]["pct"] for b in bars]
    ax.bar(xs, values, color=[b[2] for b in bars], width=0.62)
    ax.errorbar(xs, values, yerr=[lower, upper], fmt="none", ecolor="#222222",
                capsize=4, linewidth=1)
    for x, b in zip(xs, bars):
        ax.text(x, min(b[1]["ci95"][1] + 3, 104), f"{b[1]['k']}/{b[1]['n']}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Explained (%)")
    ax.set_title("How often the system explains, by what it was given\n"
                 "green: an explanation can be right; red: every explanation is wrong; "
                 "grey: no retrieval", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.005, "Bars show 95% Wilson intervals. Prompt " + version + ".",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def generate_report(summary: dict, trials: list, labels: dict, config: dict,
                    chart_name: str = None) -> str:
    L = ["# Robustness Evaluation", ""]
    L.append("How often the system explains a move, under conditions where explaining is "
             "and is not the right answer. Each condition changes one thing about a normal "
             "run; everything goes through the same code path as the CLI.")
    L.append("")
    L.append(f"{summary['n_trials']} trials ({summary['n_faults']} faults, excluded from "
             f"rates). Real anomaly dates: {config['n_real']}. Quiet placebo days: "
             f"{config['n_placebo']}. Retrieval for real dates: {config['retrieval']}. "
             f"Seed {config['seed']}.")
    models = ", ".join(f"{name} ({count})" for name, count in summary["models"].items())
    L.append(f"Models that answered: {models}.")
    L.append("")
    if chart_name:
        L += [f"![Explained rate by condition]({chart_name})", ""]

    for version, block in summary["by_prompt_version"].items():
        if version == BLIND:
            L += ["## Blind-evidence rule", "",
                  "The model reads the recent documents without being told the move, and a "
                  "fixed rule decides (src/rag/blind_evidence.py). No direction guard is "
                  "applied on this path, so the two columns are the same.", ""]
        else:
            L += [f"## Prompt {version} (model decides, then the direction guard)", ""]
        L += ["| Condition | Right answer | Explained: model alone | Explained: with direction guard |",
              "|---|---|---|---|"]
        rows = (("true", "Real anomaly, own evidence, true move", "explain where supported"),
                ("flip", "Same evidence, direction of move reversed", "refuse"),
                ("placebo", "Quiet day, real news, invented move", "refuse"),
                ("transplant", "Another anomaly's evidence, dates shifted", "see note"))
        for key, label, right in rows:
            if key in block:
                L.append(f"| {label} | {right} | {fmt_rate(block[key]['model_only'])} | "
                         f"{fmt_rate(block[key]['with_guard'])} |")
        L.append("")

        both = block.get("explained_both_directions", {})
        if both.get("model_only", {}).get("n"):
            L.append(f"**Direction sensitivity.** Of the dates the model explained under the "
                     f"true move, it also explained the opposite move for "
                     f"{fmt_rate(both['model_only'])}; with the guard, "
                     f"{fmt_rate(both['with_guard'])}. Evidence for a rise cannot explain a "
                     f"fall, so the right figure is 0%.")
            if both["with_guard"].get("dates"):
                L.append(f"Still explained in both directions with the guard on: "
                         f"{', '.join(both['with_guard']['dates'])}.")
            L.append("")

        if "placebo" in block and version == BLIND:
            p = block["placebo"]
            L.append(f"**False-explanation rate.** On quiet days with an invented move the "
                     f"system explained {fmt_rate(p['with_guard'])}. Nothing happened on "
                     f"those days, so every one of these is an explanation of a move that "
                     f"did not take place. The relevance filter passed "
                     f"{fmt_rate(p['gate_passed'])} of them.")
            L.append("")
            false_ones = [t for t in trials if t["experiment"] == "placebo"
                          and t.get("prompt_version") == BLIND and t.get("verdict") == EXPLAINED]
            if false_ones:
                L += ["What it built them from, for a person to read:", "",
                      "| Quiet day | Invented move | What the system said |", "|---|---|---|"]
                for t in sorted(false_ones, key=lambda x: x["date"]):
                    said = (t.get("explanation") or "-").replace("|", "/").replace("\n", " ")[:600]
                    L.append(f"| {t['date']} | {t.get('claimed_move') or '-'} | {said} |")
                L.append("")
        elif "placebo" in block:
            p = block["placebo"]
            L.append(f"**False-explanation rate.** On quiet days with an invented move, the "
                     f"gate let {fmt_rate(p['gate_passed'])} through to the model. The model "
                     f"explained {fmt_rate(p['model_only'])}; after the guard, "
                     f"{fmt_rate(p['with_guard'])}.")
            L.append("")

        rule_rows = summary.get("rules_on_stored_readings") if version == BLIND else None
        if rule_rows and any(r["placebo"]["n"] or r["true"]["n"] or r["flip"]["n"]
                             for r in rule_rows):
            shown = [(k, title) for k, title in
                     (("placebo", "Invented moves explained (right answer: none)"),
                      ("flip", "Reversed moves explained (right answer: none)"),
                      ("true", "Real moves explained"))
                     if any(r[k]["n"] for r in rule_rows)]
            L += ["**The same readings under other versions of the rule.** No model is called "
                  "again: each row re-applies the rule to the readings these trials stored. A "
                  "stricter rule is worth adopting only if it explains fewer invented moves "
                  "without explaining fewer real ones, and it has to be confirmed on days it "
                  "was not chosen on (a different `--seed` draws other quiet days).", ""]
            if any(r["key"] == "fit" for r in rule_rows) and not summary.get(
                    "sizes_in_words_recorded", True):
                L += ["These readings were made before version 11 and did not record how big "
                      "each document's words made the move. The version 11 row can therefore "
                      "count only a size stated as a number, which makes it stricter here "
                      "than the rule as it now runs.", ""]
            L += [
                  "| Rule | " + " | ".join(title for _, title in shown) + " |",
                  "|---|" + "---|" * len(shown)]
            for r in rule_rows:
                L.append(f"| {r['label']} | " + " | ".join(fmt_rate(r[k]) for k, _ in shown) + " |")
            L.append("")

        lab = block.get("labelled", {})
        if lab.get("with_guard", {}).get("explain_recall", {}).get("n"):
            L += ["Against the hand labels (control runs):", "",
                  "| | Explained when it should | Refused when it should |", "|---|---|---|",
                  f"| Model alone | {fmt_rate(lab['model_only']['explain_recall'])} | "
                  f"{fmt_rate(lab['model_only']['refusal_specificity'])} |",
                  f"| With guard | {fmt_rate(lab['with_guard']['explain_recall'])} | "
                  f"{fmt_rate(lab['with_guard']['refusal_specificity'])} |", ""]

        if "transplant" in block:
            L.append("**Reading the transplant row.** The evidence is real coffee news about a "
                     "real move in the same direction; only the period is wrong, and the dates "
                     "were shifted to hide that. Nothing in the documents tells the model they "
                     "describe a different week, so a high rate here is expected. It is not a "
                     "false-explanation rate. It marks what this design cannot catch: evidence "
                     "that is on-topic and direction-consistent but about another event. The "
                     "system trusts retrieval to return the right period.")
            L.append("")

        if "true_after_cutoff" in block:
            L.append(f"Real anomalies explained, by model knowledge cutoff "
                     f"({summary['closed_book']['cutoff']}): on or before "
                     f"{fmt_rate(block['true_before_cutoff'])}; after "
                     f"{fmt_rate(block['true_after_cutoff'])}.")
            L.append("")

    closed = summary.get("closed_book", {})
    if closed.get("real", {}).get("n") or closed.get("placebo", {}).get("n"):
        L += ["## No-documents control", "",
              "The same model, asked about the same move with nothing retrieved. It was told "
              "to say it does not know rather than guess.", ""]
        if closed["real"].get("n"):
            L.append(f"- Real anomalies: gave a cause for {fmt_rate(closed['real'])}")
        if closed.get("placebo", {}).get("n"):
            L.append(f"- Quiet days with an invented move: gave a cause for "
                     f"{fmt_rate(closed['placebo'])} - each one an invention")
        if "real_after_cutoff" in closed:
            L += [f"- Real anomalies on or before {closed['cutoff']}: "
                  f"{fmt_rate(closed['real_before_cutoff'])}",
                  f"- Real anomalies after {closed['cutoff']}: "
                  f"{fmt_rate(closed['real_after_cutoff'])}"]
        L.append("")
        if "real_after_cutoff" in closed:
            L.append("A cause given without documents for a date before the cutoff can come "
                     "from training data, so those dates cannot show that retrieval is doing "
                     "the work. Dates after the cutoff can.")
        else:
            L.append("No --cutoff was given, so these are not split by the model's knowledge "
                     "cutoff. Look up the cutoff for each model listed above and re-run with "
                     "`--report-only --cutoff YYYY-MM-DD`; without that split, an answer on a "
                     "real date cannot be told apart from recall of training data.")
        L.append("")

        rows = [t for t in trials if t["experiment"] == "closed_book"
                and t.get("source") == "real" and t["date"] in labels]
        if rows:
            version = next(iter(summary["by_prompt_version"]), None)
            rag = {t["date"]: t for t in trials
                   if t["experiment"] == "true" and t["prompt_version"] == version}
            L += ["### Labelled dates, side by side", "",
                  "For a person to read: does either answer name the documented cause? "
                  "Nothing here is scored automatically.", "",
                  "| Date | Documented cause (label) | No documents | With retrieval |",
                  "|---|---|---|---|"]

            def cell(text):
                return (text or "-").replace("|", "/").replace("\n", " ")[:260]
            for t in sorted(rows, key=lambda x: x["date"]):
                label = labels[t["date"]]
                with_docs = rag.get(t["date"], {})
                L.append(f"| {t['date']} | {label['expected_outcome']}: "
                         f"{cell(label.get('known_cause'))} | "
                         f"{t['verdict']}: {cell(t.get('explanation'))} | "
                         f"{with_docs.get('verdict', '-')}: {cell(with_docs.get('explanation'))} |")
            L.append("")

    def _median_docs(experiment):
        counts = sorted(t["n_documents"] for t in trials
                        if t.get("experiment") == experiment and t.get("n_documents") is not None)
        return counts[len(counts) // 2] if counts else None

    real_docs, placebo_docs = _median_docs("true"), _median_docs("placebo")
    if real_docs is not None and placebo_docs is not None:
        degraded = sum(1 for t in trials if t.get("experiment") == "placebo"
                       and (t.get("retrieval_failures") or 0) > 0)
        L += ["## Evidence volume", "",
              f"Median documents retrieved: {real_docs} for real anomaly dates, {placebo_docs} "
              f"for placebo days. {degraded} placebo trial(s) ran on retrieval where at least "
              f"one query failed.", "",
              "This matters for reading the false-explanation rate. If placebo days retrieved "
              "much less than real dates, the system had less to build a false explanation "
              "from, and the rate above understates what it would do with a full set of "
              "articles.", ""]

    if config.get("placebo_waiting"):
        L += ["## Quiet days not run yet", "",
              f"{len(config['placebo_waiting'])} quiet day(s) were left out because the news "
              f"API had not answered all of their requests: "
              f"{', '.join(config['placebo_waiting'])}. A day run on part of its news has "
              f"less to build a false explanation from, so it is not run until the news is "
              f"complete. Run the same command again to pick them up.", ""]

    L += ["## Limits", "",
          "- Sample sizes are small; every rate carries its 95% Wilson interval.",
          "- A placebo day's news is whatever was retrievable for that window when this was "
          "run. Live retrieval is rate-limited and not perfectly repeatable; the retrieved "
          "evidence is cached so a re-report uses the same documents.",
          "- An explanation on a real anomaly is counted as explained, not as correct. "
          "Whether it names the documented cause is for the side-by-side table and a human.",
          "- The direction guard and prompt v2 were written after two should-refuse dates "
          "were found explained in earlier runs. The placebo days are dates those changes "
          "were not designed on. The flip and control conditions reuse the real anomaly "
          "dates, including those two, with inputs the changes were not designed on.",
          "- The guard first withheld on a tie as well. That was dropped after it withheld "
          "three explanations on dates labelled as having a documented cause (1-1, 1-1 and "
          "3-3 tallies), so the guard column is scored under the current rule: more accepted "
          "documents against the move than with it. Cached trials are re-scored, not re-run.",
          "- The blind-evidence rule was written after the earlier runs had been read, and "
          "its two settings (recent means the same day or the two trading days before; the "
          "newest and most direct group of evidence decides) were fixed before any blind "
          "reading was run. The flip condition cannot be passed by tuning those settings: "
          "the reading is made without the direction, so the same documents cannot support "
          "both a rise and a fall.",
          "- Gemini 3 models are run at their default temperature (Google advises against "
          "lowering it), so trials answered by one are not exactly repeatable.",
          ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_cached_trials(signature: dict = None, experiments: list = None,
                       prompt_versions: list = None) -> tuple:
    """Cached trials for one run configuration.

    Returns (trials, other_signatures). Trials made under a different
    provider set, retrieval mode, seed or pipeline version are NOT returned -
    pooling them would report one rate over two different systems.
    other_signatures lists what else is in the cache, for the error message.
    """
    trials, others = [], set()
    wanted = set(experiments) if experiments else None
    for path in sorted(TRIAL_CACHE.glob("*.json")):
        trial = _read_json(path)
        if not trial:
            continue
        if signature is not None and trial.get("run") != signature:
            others.add(_signature_slug(trial["run"]) if isinstance(trial.get("run"), dict)
                       else "unknown")
            continue
        if wanted is not None and trial.get("experiment") not in wanted:
            continue
        if (prompt_versions and trial.get("experiment") != "closed_book"
                and trial.get("prompt_version") not in prompt_versions):
            continue
        trials.append(trial)
    return trials, sorted(others)


def run(experiments: list = None, n_real: int = None, n_placebo: int = 30,
        prompt_versions: list = None, retrieval: str = "live", seed: int = 7,
        cutoff: str = None, providers: list = None, sleep_seconds: float = 4.0,
        report_only: bool = False, use_cache: bool = True, verbose: bool = True,
        fill: bool = True) -> dict:
    experiments = experiments or EXPERIMENTS
    prompt_versions = prompt_versions or [BLIND]
    labels = load_labels()
    df = load_detections()
    real = real_dates(labels, n_real, df)
    placebo = select_placebo_dates(df, n_placebo, seed)
    waiting = []

    if report_only:
        signature = run_signature(providers, retrieval, seed)
        trials, others = load_cached_trials(signature, experiments, prompt_versions)
        if not trials:
            hint = (f" The cache holds trials for other settings: {', '.join(others)} "
                    f"(providers__retrieval__seed__pipeline-version) - pass the matching "
                    f"--providers / --retrieval / --seed." if others else "")
            raise SystemExit(f"No cached trials in {TRIAL_CACHE} for "
                             f"{_signature_slug(signature)}. Run without --report-only "
                             f"first.{hint}")
    else:
        if verbose:
            print(f"=== Robustness evaluation: {', '.join(experiments)} ===")
            print(f"{len(real)} real dates, {len(placebo)} placebo days, "
                  f"prompt {', '.join(prompt_versions)}, retrieval {retrieval}\n")
        if fill and "placebo" in experiments:
            lacking = [d for d in placebo if not news_complete(d)]
            if lacking:
                if verbose:
                    print(f"Fetching news for {len(lacking)} quiet day(s) first. The news API "
                          f"answers about one request in five, so this takes a while; what "
                          f"is answered is kept.\n")
                fill_news(lacking)
        trials = run_trials(experiments, real, placebo, df, prompt_versions, retrieval, seed,
                            providers=providers, sleep_seconds=sleep_seconds,
                            use_cache=use_cache, verbose=verbose, waiting=waiting)

    trials = [with_current_guard(t) for t in trials]
    summary = summarize(trials, labels, cutoff)
    config = {"experiments": experiments, "n_real": len({t["date"] for t in trials
                                                        if t["experiment"] == "true"}) or len(real),
              "n_placebo": len({t["date"] for t in trials
                                if t["experiment"] == "placebo"}) or len(placebo),
              "prompt_versions": prompt_versions, "retrieval": retrieval,
              "seed": seed, "cutoff": cutoff, "placebo_waiting": sorted(waiting)}

    chart_name = None
    versions = list(summary["by_prompt_version"])
    if versions:
        chart_version = (BLIND if BLIND in versions
                         else "v2" if "v2" in versions else versions[-1])
        if make_chart(summary, chart_version, RESULTS_DIR / "robustness_chart.png"):
            chart_name = "robustness_chart.png"

    report = generate_report(summary, trials, labels, config, chart_name)
    kept = archive_previous_report(run_signature(providers, retrieval, seed))
    if kept and verbose:
        print(f"\nThe report of the earlier run was kept in {kept}")
    _write_json(RESULTS_DIR / "robustness.json",
                {"config": config, "summary": summary, "trials": trials})
    (RESULTS_DIR / "robustness_report.md").write_text(report, encoding="utf-8")
    if verbose:
        print("\n" + report)
        print(f"\nSaved {RESULTS_DIR / 'robustness_report.md'}")
    return {"config": config, "summary": summary, "trials": trials}


def archive_previous_report(signature: dict):
    """Keep the report of an earlier run before this one overwrites it.

    results/robustness_report.md is always the latest run. A run made under
    other settings (another pipeline version, seed, provider) is a different
    measurement, and the code that made it can no longer rebuild it once the
    pipeline version has moved on, so its report, figures and chart are copied
    to results/archive/robustness/<its settings>/ first. Returns that folder,
    or None when there was nothing to keep (no earlier report, the same
    settings, or already kept)."""
    import shutil
    previous = RESULTS_DIR / "robustness.json"
    if not previous.exists():
        return None
    try:
        trials = json.loads(previous.read_text(encoding="utf-8")).get("trials") or []
    except (ValueError, OSError):
        return None
    earlier = next((t["run"] for t in trials if isinstance(t.get("run"), dict)
                    and {"providers", "retrieval", "seed", "pipeline_version"} <= set(t["run"])),
                   None)
    if earlier is None or _signature_slug(earlier) == _signature_slug(signature):
        return None
    target = RESULTS_DIR / "archive" / "robustness" / _signature_slug(earlier)
    if (target / "robustness.json").exists():
        return None
    target.mkdir(parents=True, exist_ok=True)
    for name in ("robustness.json", "robustness_report.md", "robustness_chart.png"):
        if (RESULTS_DIR / name).exists():
            shutil.copy2(RESULTS_DIR / name, target / name)
    return target


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def main():
    for stream in (sys.stdout, sys.stderr):   # model text can hold any character
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    parser = argparse.ArgumentParser(description="Robustness evaluation of the explain/refuse decision")
    parser.add_argument("--experiments", default=None,
                        help=f"Comma-separated subset of: {', '.join(EXPERIMENTS)} (default: all)")
    parser.add_argument("--n-real", type=int, default=None,
                        help="Use only the N most recent real anomaly dates (default: all)")
    parser.add_argument("--n-placebo", type=int, default=30,
                        help="Number of quiet placebo days (default: 30)")
    parser.add_argument("--prompt-versions", default=BLIND,
                        help="Comma-separated decision paths to run on the same evidence: "
                             "'blind' (the pipeline's blind-evidence rule, the default) "
                             "and/or the earlier model-decides prompts 'v1', 'v2'. "
                             "Example: blind,v2")
    parser.add_argument("--retrieval", default="live", choices=["live", "cache"],
                        help="Evidence for real dates: live retrieval (default) or on-disk "
                             "caches only. Placebo days always use live retrieval.")
    parser.add_argument("--cutoff", default=None,
                        help="Knowledge cutoff of the answering model, YYYY-MM-DD. Splits "
                             "results into dates the model could and could not have seen.")
    parser.add_argument("--providers", default=None,
                        help="Restrict the model providers, e.g. groq - so one run is one "
                             "model family, not a mix (default: the configured order)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=4.0,
                        help="Seconds to wait after each new model call (default: 4)")
    parser.add_argument("--rerun", action="store_true",
                        help="Ignore cached evidence and trials")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild the report from cached trials; call nothing")
    parser.add_argument("--no-fill", action="store_true",
                        help="Do not fetch missing news for the placebo days first; days "
                             "whose news is incomplete are then left out")
    args = parser.parse_args()

    experiments = _csv(args.experiments)
    unknown = set(experiments or []) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiment(s): {', '.join(sorted(unknown))}")
    from src.rag.explainer import PROMPT_TEMPLATES
    bad = set(_csv(args.prompt_versions) or []) - set(PROMPT_TEMPLATES) - {BLIND}
    if bad:
        parser.error(f"unknown --prompt-versions value(s): {', '.join(sorted(bad))}; "
                     f"choose from {BLIND}, {', '.join(sorted(PROMPT_TEMPLATES))}")
    if not args.report_only:
        from src.rag.explainer import check_providers
        check_providers(_csv(args.providers))
    if args.cutoff:
        try:
            datetime.strptime(args.cutoff, "%Y-%m-%d")
        except ValueError:
            parser.error(f"--cutoff takes a real date such as 2024-06-30 (the answering "
                         f"model's knowledge cutoff, from its model card); got {args.cutoff!r}")

    run(experiments=experiments, n_real=args.n_real, n_placebo=args.n_placebo,
        prompt_versions=_csv(args.prompt_versions), retrieval=args.retrieval,
        seed=args.seed, cutoff=args.cutoff, providers=_csv(args.providers),
        sleep_seconds=args.sleep, report_only=args.report_only, use_cache=not args.rerun,
        fill=not args.no_fill)


if __name__ == "__main__":
    main()
