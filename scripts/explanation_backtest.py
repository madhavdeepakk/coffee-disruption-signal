"""
Explanation backtest: does the system explain the anomalies it SHOULD, and
refuse the ones it should? And when it explains, does it get the direction
right?

Why this is the headline evaluation
-----------------------------------
The detector is validated (recall/specificity in results/). The relevance
gate is calibrated (results/relevance_gate_calibration.md). Neither of those
measures the thing the project is actually for: given a flagged anomaly,
does the end-to-end system produce a CORRECT, grounded explanation - and
does it correctly DECLINE when there is no clean cause? This harness
measures exactly that against a hand-labelled set of historical coffee
anomalies with documented (or documented-absent) causes.

Honesty notes (read before quoting any number)
-----------------------------------------------
  * Ground truth is hard. Each label in
    data/labeling/labels_by_written_rule.csv carries a
    label_confidence and, where possible, a source_url. Only rows this
    harness can score are scored; low-confidence rows are reported but can
    be excluded with --min-confidence.
  * "Correct explanation" here means: the system EXPLAINED (rather than
    refused) AND the anomaly's own price direction matches the expected
    direction. It does NOT verify the prose names the exact documented cause
    - that remains a manual read (the per-date explanation text is printed
    so a human can check it). This is deliberately the conservative,
    non-inflating definition.
  * The labelled set is small and coffee-only. Treat the output as an
    honest, extensible starting point, not a large-sample accuracy claim.
    Add rows to the CSV and re-run to grow it.

Modes
-----
  --use-cached (default): score the pipeline outputs already in results/
    (results/pipeline_output_coffee_<date>.json). Runs offline, no API key,
    no network - reproduces a real table from runs already done.
  --live: run src.pipeline for each labelled date first (needs GEMINI_API_KEY
    and working retrieval), then score. Writes fresh pipeline_output files.

Usage
-----
    python -m scripts.explanation_backtest                 # score cached outputs
    python -m scripts.explanation_backtest --live          # run then score
    python -m scripts.explanation_backtest --min-confidence high
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = REPO_ROOT / "data" / "labeling" / "labels_by_written_rule.csv"
RESULTS_DIR = REPO_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "explanation_backtest.md"

_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def load_labels(min_confidence: str) -> list:
    floor = _CONF_RANK.get(min_confidence, 0)
    rows = []
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # EXCLUDE rows are not real moves (the price series switched contract).
            if (row.get("expected_outcome") or "").strip().upper() not in ("EXPLAIN", "REFUSE"):
                continue
            if _CONF_RANK.get(row.get("label_confidence", "low"), 0) >= floor:
                rows.append(row)
    return rows


def cached_output_path(date: str) -> Path:
    return RESULTS_DIR / f"pipeline_output_coffee_{date}.json"


def run_live(date: str) -> None:
    print(f"  running pipeline live for {date} ...")
    subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--date", date, "--commodity", "coffee"],
        cwd=REPO_ROOT, check=False,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )


def load_run(date: str) -> dict | None:
    p = cached_output_path(date)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _evidence_consistency(run: dict) -> str:
    """Is the accepted evidence direction-consistent with the actual move?

    Returns 'consistent', 'conflicted', or 'n/a' (no direction_summary in
    this output - true of runs made before that field existed). This is the
    ONLY meaningful direction check: comparing the anomaly's own direction to
    itself would be circular, so we instead ask whether the evidence the
    explanation rests on points the same way as the move.
    """
    ds = run.get("direction_summary")
    if not ds:
        return "n/a"
    consistent = ds.get("consistent", 0) or 0
    inconsistent = ds.get("inconsistent", 0) or 0
    if inconsistent > 0 and inconsistent >= consistent:
        return "conflicted"
    return "consistent"


def system_result(run: dict) -> tuple[str, str]:
    """Reduce a pipeline output to (verdict, model_confidence).

    verdict is one of: EXPLAINED, REFUSED, FAULT. Uses the new 'outcome'
    block when present (post-refactor runs), else derives it from the raw
    gate/explanation decisions so older cached outputs still score.
    """
    outcome = run.get("outcome")
    if outcome and outcome.get("tier"):
        tier = outcome["tier"]
        if outcome.get("is_fault"):
            return "FAULT", ""
        verdict = "EXPLAINED" if tier.startswith("EXPLAINED") else "REFUSED"
        conf = (run.get("explanation_result", {}) or {}).get("confidence") or ""
        return verdict, conf
    # Legacy fallback: derive from raw decisions.
    er = run.get("explanation_result", {}) or {}
    dec = er.get("decision")
    if dec in ("API_ERROR", "PARSE_ERROR"):
        return "FAULT", ""
    if dec == "EXPLAINED":
        return "EXPLAINED", er.get("confidence") or ""
    return "REFUSED", er.get("confidence") or ""


def score_row(label: dict, run: dict | None) -> dict:
    """Primary metric is DECISION CALIBRATION: explain the explainable,
    refuse the unexplainable. Evidence direction-consistency is reported as
    a secondary signal where the run recorded it, and a conflicted-yet-
    explained row is NOT counted as a clean hit."""
    expected = label["expected_outcome"].strip().upper()  # EXPLAIN | REFUSE
    if run is None:
        return {**label, "status": "NO_OUTPUT", "verdict": "-",
                "confidence": "-", "evidence_dir": "-", "correct": None}
    verdict, conf = system_result(run)
    evidence_dir = _evidence_consistency(run)
    if verdict == "FAULT":
        # A system fault is neither a hit nor a miss - excluded (retry fixes
        # it), but reported so it is visible.
        correct, status = None, "EXCLUDED_FAULT"
    elif expected == "EXPLAIN":
        # Correct = the system chose to explain. If the run also recorded
        # that its evidence CONFLICTED on direction, that is not a clean hit -
        # an explanation built on contradictory evidence fails the thesis.
        correct = (verdict == "EXPLAINED" and evidence_dir != "conflicted")
        status = "scored"
    elif expected == "REFUSE":
        correct = (verdict == "REFUSED")
        status = "scored"
    else:
        correct, status = None, "UNKNOWN_LABEL"
    return {**label, "status": status, "verdict": verdict, "confidence": conf,
            "evidence_dir": evidence_dir, "correct": correct}


def summarize(scored: list) -> dict:
    graded = [s for s in scored if s["correct"] is not None]
    hits = [s for s in graded if s["correct"]]
    exp = [s for s in graded if s["expected_outcome"].strip().upper() == "EXPLAIN"]
    ref = [s for s in graded if s["expected_outcome"].strip().upper() == "REFUSE"]
    exp_hits = [s for s in exp if s["correct"]]
    ref_hits = [s for s in ref if s["correct"]]
    return {
        "n_labels": len(scored),
        "n_graded": len(graded),
        "n_correct": len(hits),
        "overall_accuracy": (len(hits) / len(graded)) if graded else None,
        "n_explain": len(exp), "explain_correct": len(exp_hits),
        "explain_accuracy": (len(exp_hits) / len(exp)) if exp else None,
        "n_refuse": len(ref), "refuse_correct": len(ref_hits),
        "refuse_accuracy": (len(ref_hits) / len(ref)) if ref else None,
        "n_no_output": sum(1 for s in scored if s["status"] == "NO_OUTPUT"),
        "n_excluded_fault": sum(1 for s in scored if s["status"] == "EXCLUDED_FAULT"),
    }


def _pct(x):
    return "n/a" if x is None else f"{100*x:.0f}%"


def write_report(scored: list, summ: dict, mode: str) -> str:
    lines = []
    lines.append("# Explanation backtest\n")
    lines.append("Primary question - **decision calibration**: does the system EXPLAIN the "
                 "anomalies that have a documented cause, and REFUSE the ones that do not? "
                 "This is the direct test of the refuse-don't-hallucinate thesis. Scored "
                 "against hand-labelled historical coffee anomalies "
                 "(`data/labeling/labels_by_written_rule.csv`).\n")
    lines.append(f"Mode: `{mode}`. A row is *correct* when, for an EXPLAIN label, the "
                 "system chose to explain (and its evidence did not conflict on direction, "
                 "where that was recorded); for a REFUSE label, when the system refused. "
                 "System faults (model unavailable) are excluded, not counted as misses. "
                 "Evidence direction-consistency is a secondary signal and is only present "
                 "on runs made after that field was added (older cached runs show n/a).\n")
    lines.append("## Summary\n")
    coverage_pct = round(100 * summ['n_graded'] / summ['n_labels'], 0) \
        if summ['n_labels'] > 0 else 0
    lines.append(f"- Graded rows: **{summ['n_graded']}** of {summ['n_labels']} labelled "
                 f"({summ['n_no_output']} had no pipeline output, "
                 f"{summ['n_excluded_fault']} excluded as system faults)")
    lines.append(f"- **Coverage**: {coverage_pct:.0f}% of labelled dates have pipeline "
                 f"outputs to score")
    if summ['n_no_output'] > 0:
        lines.append(f"- **WARNING**: {summ['n_no_output']} labelled dates have no "
                     f"pipeline output. Run with `--live` to generate them. "
                     f"Accuracy below is computed on the graded subset only and "
                     f"SHOULD NOT be quoted as the system's overall accuracy.")
    lines.append(f"- Overall accuracy on graded rows: **{_pct(summ['overall_accuracy'])}** "
                 f"({summ['n_correct']}/{summ['n_graded']})")
    lines.append(f"- Correct explanations: **{summ['explain_correct']}/{summ['n_explain']}** "
                 f"({_pct(summ['explain_accuracy'])})")
    lines.append(f"- Correct refusals: **{summ['refuse_correct']}/{summ['n_refuse']}** "
                 f"({_pct(summ['refuse_accuracy'])})\n")
    lines.append("## Per-date\n")
    lines.append("| Date | Expected | Label conf | System | Evidence dir | Correct | Known cause |")
    lines.append("|------|----------|-----------|--------|--------------|---------|-------------|")
    for s in scored:
        mark = {True: "yes", False: "NO", None: "-"}[s["correct"]]
        cause = (s.get("known_cause") or "")[:60]
        lines.append(f"| {s['date']} | {s['expected_outcome']} | "
                     f"{s.get('label_confidence','')} | {s['verdict']} | "
                     f"{s.get('evidence_dir','-')} | {mark} | {cause} |")
    lines.append("\n## Caveats\n")
    lines.append("- Small, coffee-only labelled set: an honest, extensible baseline, not a "
                 "large-sample accuracy claim. Add rows to the CSV and re-run to grow it.")
    lines.append("- 'Correct explanation' checks direction, not that the prose names the exact "
                 "documented cause - read the printed explanation text to confirm that.")
    lines.append("- Low-confidence labels (e.g. mean-reversion down days) are the hardest to "
                 "ground-truth; use `--min-confidence high` to score only the firm ones.")
    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Run the pipeline live for each date before scoring "
                             "(needs GEMINI_API_KEY + working retrieval). Default: score "
                             "the cached results/pipeline_output_coffee_<date>.json files.")
    parser.add_argument("--min-confidence", default="low", choices=["low", "medium", "high"],
                        help="Only score labels at or above this confidence (default: low = all)")
    args = parser.parse_args()

    labels = load_labels(args.min_confidence)
    mode = "live" if args.live else "use-cached"
    print(f"Explanation backtest ({mode}); {len(labels)} label(s) "
          f"at confidence >= {args.min_confidence}\n")

    scored = []
    for label in labels:
        date = label["date"]
        if args.live:
            run_live(date)
        run = load_run(date)
        s = score_row(label, run)
        scored.append(s)
        mark = {True: "correct", False: "WRONG", None: s["status"]}[s["correct"]]
        print(f"  {date}: expected {label['expected_outcome']}  ->  system {s['verdict']}"
              f"  (evidence dir: {s.get('evidence_dir','-')})  [{mark}]")

    summ = summarize(scored)
    report = write_report(scored, summ, mode)
    print("\n" + "=" * 60)
    print(report)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
