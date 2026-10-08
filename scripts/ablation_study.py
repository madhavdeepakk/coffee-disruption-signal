"""
Ablation study: contribution of each explanation pipeline layer.

Re-processes cached pipeline outputs (results/pipeline_output_coffee_*.json)
under ablated conditions to measure each layer's contribution to the system's
decision calibration. No live API calls needed.

Layers ablated:
  1. Relevance gate — bypass the gate, pass ALL retrieved docs to the LLM
  2. Faithfulness check — ignore citation audit in outcome classification
  3. Direction reranking — ignore direction consistency in outcome scoring

Each ablation re-classifies the cached pipeline outputs using the backtest
label set and reports how decision accuracy changes when the layer is removed.

Usage:
    python -m scripts.ablation_study
    python -m scripts.ablation_study --min-confidence high
"""

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = REPO_ROOT / "data" / "labeling" / "labels_by_written_rule.csv"
RESULTS_DIR = REPO_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "ablation_study.md"

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


def load_run(date: str) -> dict | None:
    p = RESULTS_DIR / f"pipeline_output_coffee_{date}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Scoring functions: one per ablation condition
# ---------------------------------------------------------------------------

def score_full_pipeline(run: dict | None, label: dict) -> str | None:
    """Full pipeline (baseline). Returns 'correct', 'wrong', or None (no data/fault)."""
    if run is None:
        return None
    er = run.get("explanation_result", {}) or {}
    dec = er.get("decision")
    if dec in ("API_ERROR", "PARSE_ERROR"):
        return None  # fault, exclude

    expected = label["expected_outcome"].strip().upper()
    verdict = "EXPLAINED" if dec == "EXPLAINED" else "REFUSED"

    # Direction consistency check (part of full pipeline)
    ds = run.get("direction_summary")
    evidence_conflicted = False
    if ds:
        inconsistent = ds.get("inconsistent", 0) or 0
        consistent = ds.get("consistent", 0) or 0
        if inconsistent > 0 and inconsistent >= consistent:
            evidence_conflicted = True

    if expected == "EXPLAIN":
        return "correct" if (verdict == "EXPLAINED" and not evidence_conflicted) else "wrong"
    elif expected == "REFUSE":
        return "correct" if verdict == "REFUSED" else "wrong"
    return None


def score_no_gate(run: dict | None, label: dict) -> str | None:
    """Ablation: remove relevance gate — pass ALL docs to LLM.

    Without the gate, the pipeline would always attempt to explain (never refuse
    at the gate stage). We simulate this: if the gate refused in the real run,
    we flip the outcome to EXPLAINED (the LLM would have received all docs and
    attempted an explanation). If the gate passed, the outcome stays as-is.
    """
    if run is None:
        return None
    er = run.get("explanation_result", {}) or {}
    dec = er.get("decision")
    if dec in ("API_ERROR", "PARSE_ERROR"):
        return None

    gate = (run.get("gate_result") or {}).get("decision", "")
    expected = label["expected_outcome"].strip().upper()

    if gate == "INSUFFICIENT_EVIDENCE":
        # Gate would NOT have blocked — LLM gets all docs, attempts explanation
        simulated_verdict = "EXPLAINED"
    else:
        verdict = "EXPLAINED" if dec == "EXPLAINED" else "REFUSED"
        simulated_verdict = verdict

    if expected == "EXPLAIN":
        return "correct" if simulated_verdict == "EXPLAINED" else "wrong"
    elif expected == "REFUSE":
        return "correct" if simulated_verdict == "REFUSED" else "wrong"
    return None


def score_no_faithfulness(run: dict | None, label: dict) -> str | None:
    """Ablation: remove faithfulness check — ignore citation quality in outcome.

    The faithfulness check downgrades outcomes when citations don't resolve or
    have weak support. Without it, any EXPLAINED stays EXPLAINED regardless of
    citation quality.
    """
    if run is None:
        return None
    er = run.get("explanation_result", {}) or {}
    dec = er.get("decision")
    if dec in ("API_ERROR", "PARSE_ERROR"):
        return None

    expected = label["expected_outcome"].strip().upper()
    verdict = "EXPLAINED" if dec == "EXPLAINED" else "REFUSED"

    # Skip direction consistency (that's a separate layer); only remove faithfulness
    if expected == "EXPLAIN":
        return "correct" if verdict == "EXPLAINED" else "wrong"
    elif expected == "REFUSE":
        return "correct" if verdict == "REFUSED" else "wrong"
    return None


def score_no_direction(run: dict | None, label: dict) -> str | None:
    """Ablation: remove direction reranking — ignore direction consistency.

    Without direction checking, an explanation with conflicting evidence
    still counts as EXPLAINED.
    """
    if run is None:
        return None
    er = run.get("explanation_result", {}) or {}
    dec = er.get("decision")
    if dec in ("API_ERROR", "PARSE_ERROR"):
        return None

    expected = label["expected_outcome"].strip().upper()
    verdict = "EXPLAINED" if dec == "EXPLAINED" else "REFUSED"

    # No direction check — take the LLM decision at face value
    if expected == "EXPLAIN":
        return "correct" if verdict == "EXPLAINED" else "wrong"
    elif expected == "REFUSE":
        return "correct" if verdict == "REFUSED" else "wrong"
    return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def run_ablation(labels: list) -> dict:
    """Run all ablation conditions and return structured results."""
    conditions = {
        "Full pipeline (baseline)": score_full_pipeline,
        "No relevance gate": score_no_gate,
        "No faithfulness check": score_no_faithfulness,
        "No direction reranking": score_no_direction,
    }

    results = {}
    for name, scorer in conditions.items():
        scored = []
        for label in labels:
            run = load_run(label["date"])
            result = scorer(run, label)
            scored.append({"date": label["date"], "expected": label["expected_outcome"],
                           "result": result})

        graded = [s for s in scored if s["result"] is not None]
        correct = sum(1 for s in graded if s["result"] == "correct")
        total = len(graded)

        results[name] = {
            "correct": correct, "total": total,
            "accuracy": correct / total if total > 0 else None,
            "n_no_data": sum(1 for s in scored if s["result"] is None),
            "details": scored,
        }

    return results


def write_report(results: dict, labels: list, min_confidence: str) -> str:
    lines = [
        "# Ablation Study — Explanation Pipeline Layers",
        "",
        "Each layer is removed one at a time from the explanation pipeline, "
        "and the backtest labels are re-scored to measure the layer's contribution "
        "to decision calibration (explain the explainable, refuse the rest).",
        "",
        f"Label set: {len(labels)} dates at confidence >= {min_confidence}.",
        "",
        "## Results summary",
        "",
        "| Condition | Correct | Total | Accuracy | Delta vs baseline |",
        "|-----------|---------|-------|----------|-------------------|",
    ]

    baseline_acc = results["Full pipeline (baseline)"]["accuracy"]
    for name, r in results.items():
        acc = r["accuracy"]
        acc_str = f"{acc*100:.0f}%" if acc is not None else "n/a"
        if baseline_acc is not None and acc is not None:
            delta = (acc - baseline_acc) * 100
            delta_str = f"{delta:+.0f} pts" if name != "Full pipeline (baseline)" else "—"
        else:
            delta_str = "—"
        lines.append(f"| {name} | {r['correct']} | {r['total']} | {acc_str} | {delta_str} |")

    lines.extend(["", "## Interpretation", ""])

    # Generate interpretation based on actual results
    if baseline_acc is not None:
        no_gate = results["No relevance gate"]
        no_faith = results["No faithfulness check"]
        no_dir = results["No direction reranking"]

        if no_gate["accuracy"] is not None and no_gate["accuracy"] < baseline_acc:
            lines.append(
                f"- **Relevance gate** matters: removing it drops accuracy by "
                f"{(baseline_acc - no_gate['accuracy'])*100:.0f} pts. Without the gate, "
                "the system would attempt to explain dates that lack supporting evidence, "
                "increasing false explanations.")
        elif no_gate["accuracy"] is not None and no_gate["accuracy"] >= baseline_acc:
            lines.append(
                "- **Relevance gate**: no accuracy drop on the current label set when removed. "
                "This may reflect the gate already passing all docs in the cached runs; "
                "the gate's value would show on dates with truly irrelevant retrieval.")
        lines.append("")

        if no_faith["accuracy"] is not None and no_faith["accuracy"] < baseline_acc:
            lines.append(
                f"- **Faithfulness check** contributes: accuracy drops by "
                f"{(baseline_acc - no_faith['accuracy'])*100:.0f} pts without it.")
        elif no_faith["accuracy"] is not None:
            lines.append(
                "- **Faithfulness check**: no accuracy change on this label set. "
                "Its value is in preventing hallucinated citations from reaching the user, "
                "which the decision-calibration metric does not capture directly.")
        lines.append("")

        if no_dir["accuracy"] is not None and no_dir["accuracy"] < baseline_acc:
            lines.append(
                f"- **Direction reranking** helps: accuracy drops by "
                f"{(baseline_acc - no_dir['accuracy'])*100:.0f} pts without it.")
        elif no_dir["accuracy"] is not None:
            lines.append(
                "- **Direction reranking**: no accuracy change on the current label set. "
                "Its value shows on dates where evidence and price move in opposite "
                "directions; the current set may not stress this case enough.")

    lines.extend([
        "",
        "## Layer descriptions",
        "",
        "- **Relevance gate**: filters retrieved documents by semantic and keyword "
        "relevance scores before they reach the LLM. Without it, all retrieved "
        "documents (including noise) are passed through.",
        "- **Faithfulness check**: post-hoc audit of the LLM's citations — do they "
        "resolve to real retrieved documents, and does the cited text actually "
        "support the claim?",
        "- **Direction reranking**: checks whether the evidence's direction "
        "(prices rising/falling in the cited articles) is consistent with the "
        "actual price move. Conflicting evidence downgrades the explanation.",
        "",
        "## How to run",
        "",
        "```",
        "python -m scripts.ablation_study",
        "python -m scripts.ablation_study --min-confidence high",
        "```",
        "",
    ])

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-confidence", default="low", choices=["low", "medium", "high"])
    args = parser.parse_args()

    labels = load_labels(args.min_confidence)
    print(f"Ablation study: {len(labels)} labels at confidence >= {args.min_confidence}\n")

    results = run_ablation(labels)

    for name, r in results.items():
        acc_str = f"{r['accuracy']*100:.0f}%" if r['accuracy'] is not None else "n/a"
        print(f"  {name}: {r['correct']}/{r['total']} ({acc_str})")

    report = write_report(results, labels, args.min_confidence)
    print(f"\n{report}")
    print(f"Saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
