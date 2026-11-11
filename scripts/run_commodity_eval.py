"""
The explain-or-refuse pipeline on the other commodities: crude oil, wheat,
gold and silver.

Coffee has an answer key (data/labeling/labels_by_written_rule.csv). The other
commodities do not, and writing one is a person's work. Two things can still
be measured for them without a key:

  Should refuse.   The invented-move test of src/evaluation/robustness.py
                   needs no labels: on a quiet day nothing happened, so any
                   explanation of an invented move is false. It is run for
                   each commodity on its own quiet days.

  Should explain.  Each commodity's known_events (src/config/commodities.py)
                   are documented moves with a cited cause. The days the
                   detector flags as a one-day shock inside those windows are
                   run through the pipeline. These dates are documented by
                   construction, which is weaker than a key someone has
                   checked date by date: the citation says the EVENT has a
                   cause, not that every flagged day inside its window does,
                   and "explained" is not the same as "named the cited cause".
                   A person has to read the explanations for that; the report
                   prints each one for the purpose.

Only events from 2017 on are run. The news search (GDELT's article search)
does not reach further back, so an older date would be refused for lack of
news - a fact about the archive, not about the system.

Everything is resumable: a finished date is not run again, the news cache
keeps what was answered, and the report is rebuilt from what is on disk.

Usage:
    python -m scripts.run_commodity_eval --all --providers groq
    python -m scripts.run_commodity_eval --commodity gold --providers groq
    python -m scripts.run_commodity_eval --all --report-only       # no model, no network
    python -m scripts.run_commodity_eval --all --skip-placebo      # documented dates only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.commodities import COMMODITIES, get_commodity   # noqa: E402
from src.evaluation.metrics import fmt_rate, rate                # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORT_MD = RESULTS_DIR / "multi_commodity_report.md"
REPORT_JSON = RESULTS_DIR / "multi_commodity_report.json"

NEWS_SEARCH_STARTS = "2017-01-01"     # GDELT's article search has nothing earlier
DEFAULT_PER_EVENT = 2
DEFAULT_PLACEBO_DAYS = 12
DEFAULT_SEED = 7
OTHER_COMMODITIES = [key for key in COMMODITIES if key != "coffee"]


def documented_dates(commodity: str, per_event: int = DEFAULT_PER_EVENT,
                     detections: pd.DataFrame = None) -> list:
    """[{date, event, z_score}]: for every known event from 2017 on, the
    `per_event` days inside its window that the detector flags as a one-day
    shock, largest |z| first. Trend-only days are left out: they are flagged
    for a ten-day move, and the day's own news need not mention it."""
    cfg = get_commodity(commodity)
    if detections is None:
        path = RESULTS_DIR / cfg.anomalies_file
        if not path.exists():
            return []
        detections = pd.read_csv(path)
    df = detections.copy()
    df["date"] = df["date"].astype(str).str[:10]
    kinds = df["anomaly_type"].fillna("").astype(str) if "anomaly_type" in df.columns else ""
    shocks = df[df["anomaly_flag"].fillna(False).astype(bool)
                & (kinds.str.contains("shock") if "anomaly_type" in df.columns else True)]
    picked, seen = [], set()
    for label, start, end, _source in cfg.known_events:
        if end < NEWS_SEARCH_STARTS:
            continue
        window = shocks[(shocks["date"] >= max(start, NEWS_SEARCH_STARTS)) & (shocks["date"] <= end)]
        window = window.assign(_size=window["z_score"].abs()).sort_values(
            ["_size", "date"], ascending=[False, True])
        for _, row in window.head(per_event).iterrows():
            if row["date"] not in seen:
                seen.add(row["date"])
                picked.append({"date": row["date"], "event": label,
                               "z_score": float(row["z_score"])})
    return sorted(picked, key=lambda d: d["date"])


def events_before_the_news_search(commodity: str) -> list:
    return [label for label, _s, end, _src in get_commodity(commodity).known_events
            if end < NEWS_SEARCH_STARTS]


def run_documented(commodity: str, dates: list, sleep_seconds: float = 20.0) -> None:
    """Fill the news cache for these dates, then run the pipeline for each
    one that has no current output."""
    from scripts.run_eval_set import archive_output, fill_cache, output_state, run_one
    day_list = [d["date"] for d in dates]
    pending = [d for d in day_list if output_state(d, commodity) != "current"]
    if not pending:
        print(f"  [{commodity}] all {len(day_list)} documented date(s) already run")
        return
    print(f"  [{commodity}] {len(pending)} of {len(day_list)} documented date(s) to run")
    fill_cache(pending, commodity)
    failures = 0
    for i, day in enumerate(pending, 1):
        print(f"  [{commodity}] [{i}/{len(pending)}] {day} ...", flush=True)
        archive_output(day, commodity)
        code = run_one(day, commodity)
        failures = failures + 1 if code != 0 else 0
        print(f"      {'finished' if code == 0 else f'did not finish (exit {code})'}"
              f"  [{output_state(day, commodity)}]", flush=True)
        if failures >= 2 and i < len(pending):
            print(f"  [{commodity}] two dates in a row did not finish; stopping this "
                  f"commodity. Fix the error (see data/eval_cache/logs) and run again.")
            break
        if i < len(pending):
            time.sleep(sleep_seconds)


def run_placebo(commodity: str, n_placebo: int, seed: int, providers: str = None) -> int:
    """The invented-move test for one commodity, in its own process so that
    its caches and report names cannot touch another commodity's."""
    command = [sys.executable, "-m", "src.evaluation.robustness", "--commodity", commodity,
               "--experiments", "placebo", "--n-placebo", str(n_placebo), "--seed", str(seed)]
    if providers:
        command += ["--providers", providers]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    log_dir = REPO_ROOT / "data" / "eval_cache" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"placebo_{commodity}.log"
    print(f"  [{commodity}] invented-move test on {n_placebo} quiet day(s); "
          f"output in {log_path.relative_to(REPO_ROOT)}", flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        return subprocess.call(command, cwd=REPO_ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT)


# ---------------------------------------------------------------------------
# Reading what is on disk
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def documented_outcome(commodity: str, entry: dict) -> dict:
    """What the stored run for a documented date ended in."""
    from src.pipeline import PIPELINE_VERSION
    data = _read_json(RESULTS_DIR / f"pipeline_output_{commodity}_{entry['date']}.json")
    row = {**entry, "state": "not run", "tier": None, "explanation": None, "reason": None,
           "model": None, "failed_queries": None}
    if not data:
        return row
    outcome = data.get("outcome") or {}
    result = data.get("explanation_result") or {}
    row.update(tier=outcome.get("tier"), model=result.get("model_used"),
               reason=result.get("reason"), version=str(data.get("pipeline_version", "1")),
               move=(data.get("anomaly") or {}).get("pct_move"))
    if str(data.get("pipeline_version", "1")) != PIPELINE_VERSION:
        row["state"] = "older version"
    elif outcome.get("is_fault"):
        row["state"] = "fault"
    elif result.get("decision") == "EXPLAINED":
        row["state"] = "explained"
        row["explanation"] = (result.get("explanation") or "").strip()
    else:
        row["state"] = "refused"
    meta = data.get("retrieval_meta") or {}
    row["failed_queries"] = meta.get("gdelt_failed_queries", meta.get("failed_queries"))
    return row


def placebo_summary(commodity: str) -> dict:
    """The invented-move result from results/robustness[_<commodity>].json:
    how many quiet days were explained, and what was said."""
    name = "robustness.json" if commodity == "coffee" else f"robustness_{commodity}.json"
    data = _read_json(RESULTS_DIR / name)
    if not data:
        return {"run": False}
    trials = [t for t in data.get("trials", []) if t.get("experiment") == "placebo"
              and t.get("prompt_version") == "blind"]
    scored = [t for t in trials if t.get("verdict") in ("EXPLAINED", "REFUSED")]
    explained = [t for t in scored if t.get("verdict") == "EXPLAINED"]
    run = next((t.get("run") for t in trials if isinstance(t.get("run"), dict)), {}) or {}
    return {"run": True, "n": len(scored), "explained": len(explained),
            "faults": len(trials) - len(scored),
            "waiting": (data.get("config") or {}).get("placebo_waiting") or [],
            "pipeline_version": run.get("pipeline_version"), "seed": run.get("seed"),
            "false_explanations": [{"date": t["date"],
                                    "explanation": (t.get("explanation") or "")[:400]}
                                   for t in explained]}


def coffee_key_summary() -> dict:
    """Coffee's score against its answer key, from the stored evaluation
    report - only if that report was made from current runs."""
    from src.pipeline import PIPELINE_VERSION
    data = _read_json(RESULTS_DIR / "evaluation_report_v11.json") \
        or _read_json(RESULTS_DIR / "evaluation_report.json")
    if not data:
        return {}
    labelled = (data.get("labelled") or data.get("decision_accuracy") or {})
    return {"raw": labelled, "pipeline_version": PIPELINE_VERSION}


def collect(commodities: list, per_event: int = DEFAULT_PER_EVENT) -> dict:
    out = {}
    for commodity in commodities:
        cfg = get_commodity(commodity)
        rows = [documented_outcome(commodity, e) for e in documented_dates(commodity, per_event)]
        decided = [r for r in rows if r["state"] in ("explained", "refused")]
        out[commodity] = {
            "display_name": cfg.display_name,
            "documented": rows,
            "explained": sum(r["state"] == "explained" for r in rows),
            "decided": len(decided),
            "not_decided": len(rows) - len(decided),
            "events_too_old": events_before_the_news_search(commodity),
            "placebo": placebo_summary(commodity),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(collected: dict) -> str:
    L = ["# Explain or Refuse on Other Commodities", "",
         "The same pipeline that is measured on coffee, run on crude oil, wheat, gold and "
         "silver. Produced by `python -m scripts.run_commodity_eval`. Rates carry 95% Wilson "
         "intervals; the samples are small and the intervals say so.", "",
         "**These commodities have no answer key.** What is measured instead:", "",
         "- *Explained a documented move*: the days the detector flags as a one-day shock "
         "inside each commodity's cited historical events, from 2017 on (the news search "
         "reaches no further back). A documented event is weaker than a checked label, and "
         "an explanation counted here has not been checked against the cited cause.",
         "- *Explained an invented move*: quiet days given the size and direction of a real "
         "shock. Refusing is the only right answer, so every explanation is false. This "
         "needs no labels and is the one figure here that is comparable with coffee's.",
         "",
         "Always explaining would score 100% on the first column and 100% on the second; "
         "always refusing would score 0% on both. A system is doing something only if the "
         "first figure is well above the second.", "",
         "| Commodity | Explained a documented move | Explained an invented move "
         "(right answer: none) |", "|---|---|---|"]
    for key, block in collected.items():
        doc = (fmt_rate(rate(block["explained"], block["decided"])) if block["decided"]
               else "not run yet")
        if block["not_decided"] and block["decided"]:
            doc += f"; {block['not_decided']} more not decided"
        pl = block["placebo"]
        inv = (fmt_rate(rate(pl["explained"], pl["n"])) if pl.get("run") and pl.get("n")
               else "not run yet")
        L.append(f"| {block['display_name']} | {doc} | {inv} |")
    L.append("")

    for key, block in collected.items():
        L += [f"## {block['display_name']}", ""]
        if block["events_too_old"]:
            L += [f"Not run, because the news search does not reach back that far: "
                  f"{'; '.join(block['events_too_old'])}.", ""]
        if not block["documented"]:
            L += ["No shock day inside a known event from 2017 on (or the detections file "
                  "is missing: run the detector first).", ""]
        else:
            L += ["| Date | Event it belongs to | Move | Outcome | What was said, or why not |",
                  "|---|---|---|---|---|"]
            for r in block["documented"]:
                move = f"{r['move']:+.1f}%" if r.get("move") is not None else "-"
                if r["state"] == "explained":
                    said = (r["explanation"] or "").replace("\n", " ").replace("|", "/")[:420]
                elif r["state"] == "not run":
                    said = "-"
                else:
                    said = str(r.get("reason") or r.get("tier") or "").replace("|", "/")[:200]
                L.append(f"| {r['date']} | {r['event']} | {move} | {r['state']} | {said} |")
            L.append("")
        pl = block["placebo"]
        if not pl.get("run"):
            L += ["Invented-move test: not run yet.", ""]
        else:
            L += [f"Invented-move test (pipeline version {pl.get('pipeline_version')}, seed "
                  f"{pl.get('seed')}): explained "
                  f"{fmt_rate(rate(pl['explained'], pl['n'])) if pl['n'] else 'none scored'}"
                  + (f"; {pl['faults']} fault(s) left out" if pl["faults"] else "")
                  + (f"; {len(pl['waiting'])} quiet day(s) still waiting for news"
                     if pl["waiting"] else "") + ".", ""]
            for item in pl["false_explanations"]:
                L += [f"- False explanation on {item['date']}: "
                      f"{item['explanation'].replace(chr(10), ' ')}"]
            if pl["false_explanations"]:
                L.append("")

    L += ["## Limits", "",
          "- No answer key for these commodities. The documented dates were chosen by a "
          "rule (largest shock days inside cited events), not checked one by one, and "
          "nobody has read whether each explanation names the cited cause.",
          "- The headline word lists that choose which articles are read were built and "
          "tuned on coffee. For the other commodities they were written once and not tuned "
          "(src/rag/market_report.py), so their reading is likely to be worse chosen.",
          "- The decision rule and its settings were designed on coffee dates. Every figure "
          "here is therefore out of sample for the rule, which is what makes it worth "
          "having, and also why it may be lower than coffee's.",
          "- A handful of dates per commodity. One date changes a rate by ten points or "
          "more.", ""]
    return "\n".join(L)


def write_report(commodities: list, per_event: int = DEFAULT_PER_EVENT, verbose: bool = True) -> dict:
    collected = collect(commodities, per_event)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(collected, indent=2, default=str), encoding="utf-8")
    report = generate_report(collected)
    REPORT_MD.write_text(report, encoding="utf-8")
    if verbose:
        print("\n" + report)
        print(f"\nSaved {REPORT_MD}")
    return collected


def main():
    parser = argparse.ArgumentParser(description="Explain-or-refuse on the other commodities")
    parser.add_argument("--commodity", default=None, choices=list(COMMODITIES))
    parser.add_argument("--all", action="store_true",
                        help="Crude oil, wheat, gold and silver (coffee has its own evaluation)")
    parser.add_argument("--per-event", type=int, default=DEFAULT_PER_EVENT,
                        help=f"Shock days to run per known event (default {DEFAULT_PER_EVENT})")
    parser.add_argument("--n-placebo", type=int, default=DEFAULT_PLACEBO_DAYS,
                        help=f"Quiet days for the invented-move test (default {DEFAULT_PLACEBO_DAYS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--providers", default=None,
                        help="Model provider(s), e.g. groq, so one run is one model family")
    parser.add_argument("--skip-placebo", action="store_true")
    parser.add_argument("--skip-documented", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild the report from what is on disk; call nothing")
    args = parser.parse_args()
    if not args.all and not args.commodity:
        parser.error("give --commodity NAME or --all")
    commodities = OTHER_COMMODITIES if args.all else [args.commodity]

    if not args.report_only:
        if args.providers:
            os.environ["EXPLAINER_PROVIDERS"] = args.providers
        from src.rag.explainer import check_providers
        print(f"Model provider(s): {', '.join(check_providers())}")
        for commodity in commodities:
            cfg = get_commodity(commodity)
            if not (RESULTS_DIR / cfg.anomalies_file).exists():
                print(f"  [{commodity}] no detections file; run "
                      f"python -m src.modeling.anomaly_detector --commodity {commodity}")
                continue
            if not args.skip_documented:
                run_documented(commodity, documented_dates(commodity, args.per_event))
            if not args.skip_placebo:
                code = run_placebo(commodity, args.n_placebo, args.seed, args.providers)
                if code != 0:
                    print(f"  [{commodity}] the invented-move test stopped (exit {code}); "
                          f"see its log. Run this command again to continue.")
    write_report(OTHER_COMMODITIES if args.all else commodities, args.per_event)


if __name__ == "__main__":
    main()
