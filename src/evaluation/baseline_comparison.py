"""
Baseline comparison: does the pipeline make better explain/refuse decisions
than doing something trivial?

The product claim is about a DECISION - explain when there is a documented
cause, decline when there is not - so that is what is compared, against the
hand-labelled dates (data/labeling/labels_by_written_rule.csv).

Systems
  always_explain    Explain every date that retrieved at least one document.
                    A headline dump with no gate and no refusal. Right on
                    every EXPLAIN label and wrong on every REFUSE label by
                    construction.
  gate_only         Explain exactly when the relevance gate passes. The
                    pipeline without the model's own judgement.
  closed_book       The same model asked with NO documents (if
                    results/robustness.json has those trials). Whatever this
                    gets right, retrieval did not contribute.
  pipeline          The stored runs, as they were made.
  pipeline_guarded  The same runs with the direction guard applied.

Scores
  explain recall      of EXPLAIN-labelled dates, how many were explained
  refusal specificity of REFUSE-labelled dates, how many were refused
  balanced accuracy   the mean of the two. Raw accuracy is also shown, but
                      the label set is mostly EXPLAIN, so a system that never
                      refuses scores well on raw accuracy while failing the
                      one thing the project claims. Balanced accuracy does
                      not reward that.

Runs offline from stored outputs: no network, no model.

This replaces an earlier comparison that scored explanation TEXT with a
regex rubric (does it contain a region name, a percent sign, the word
"because"). That rubric rewarded keyword-dense headlines, gave a refusal
zero on every dimension, and on five dates reported the pipeline 0.4 points
ahead out of 12 while scoring it below a headline dump on grounding. It
measured the rubric, not the system.

Usage:
    python -m src.evaluation.baseline_comparison
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import metrics as m   # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
ROBUSTNESS_PATH = RESULTS_DIR / "robustness.json"

EXPLAINED, REFUSED, FAULT = m.EXPLAINED, m.REFUSED, m.FAULT

SYSTEM_NOTES = {
    "always_explain": "Explains whenever anything was retrieved. No gate, no refusal.",
    "gate_only": "Explains exactly when the relevance gate passes.",
    "direction_only": "Explains every day prices rose and refuses every day they fell. "
                      "Reads no news and calls no model.",
    "closed_book": "Same model, no documents (from the robustness harness).",
    "pipeline": "Stored runs as made.",
    "pipeline_guarded": "Stored runs with the direction guard applied.",
}


def always_explain(run: dict) -> str:
    return EXPLAINED if (run.get("documents_retrieved") or 0) > 0 else REFUSED


def direction_only(run: dict) -> str:
    """Explain up days, refuse down days. A control for the direction guard:
    the retrieved news leans towards price rises, so the guard fires mostly
    on down days, and every should-refuse label so far is a down day. If this
    rule scores as well as the guarded pipeline, the labelled dates cannot
    tell the guard apart from knowing which way the price moved."""
    return EXPLAINED if (run.get("anomaly") or {}).get("direction") == "up" else REFUSED


def gate_only(run: dict) -> str:
    return EXPLAINED if (run.get("gate_result") or {}).get("decision") == "EXPLAIN" else REFUSED


def closed_book_verdicts(path: Path = ROBUSTNESS_PATH) -> dict:
    """date -> EXPLAINED/REFUSED from the robustness harness's closed-book
    trials, or {} if that experiment has not been run."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    out = {}
    for trial in data.get("trials", []):
        if (trial.get("experiment") == "closed_book" and trial.get("source") == "real"
                and trial.get("verdict") in (EXPLAINED, REFUSED)):
            out[trial["date"]] = trial["verdict"]
    return out


def score(verdicts: dict, labels: dict) -> dict:
    """verdicts: date -> EXPLAINED/REFUSED/FAULT (dates missing are skipped)."""
    exp_n = exp_k = ref_n = ref_k = faults = 0
    for date, label in labels.items():
        expected = (label.get("expected_outcome") or "").strip().upper()
        v = verdicts.get(date)
        if v is None or expected not in ("EXPLAIN", "REFUSE"):
            continue
        if v == FAULT:
            faults += 1
            continue
        if expected == "EXPLAIN":
            exp_n += 1
            exp_k += v == EXPLAINED
        else:
            ref_n += 1
            ref_k += v == REFUSED
    recall = exp_k / exp_n if exp_n else None
    specificity = ref_k / ref_n if ref_n else None
    balanced = (None if recall is None or specificity is None
                else round(100 * (recall + specificity) / 2, 1))
    return {
        "n_graded": exp_n + ref_n,
        "n_fault": faults,
        "accuracy": m.rate(exp_k + ref_k, exp_n + ref_n),
        "explain_recall": m.rate(exp_k, exp_n),
        "refusal_specificity": m.rate(ref_k, ref_n),
        "balanced_accuracy_pct": balanced,
    }


def run_comparison(verbose: bool = True) -> dict:
    labels = m.load_labels()
    if not labels:
        raise SystemExit(f"No label file at {m.LABELS_PATH}.")
    # Rows marked EXCLUDE are not real moves (the price series switched
    # contract that day) and are not scored for any system.
    labels = {d: label for d, label in labels.items()
              if (label.get("expected_outcome") or "").strip().upper() in ("EXPLAIN", "REFUSE")}
    runs = {m._run_date(r): r for r in m.load_results()}
    labelled_runs = {d: r for d, r in runs.items() if d in labels}

    # Every system is scored on the same dates: those where the pipeline run
    # did not end in a fault. Otherwise a baseline would be graded on dates
    # the pipeline is excused from.
    dates = [d for d, r in labelled_runs.items() if m.verdict(r) != FAULT]

    systems = {
        "always_explain": {d: always_explain(labelled_runs[d]) for d in dates},
        "gate_only": {d: gate_only(labelled_runs[d]) for d in dates},
        "direction_only": {d: direction_only(labelled_runs[d]) for d in dates},
        "pipeline": {d: m.verdict(labelled_runs[d]) for d in dates},
        "pipeline_guarded": {d: m.verdict_with_guard(labelled_runs[d]) for d in dates},
    }
    closed = closed_book_verdicts()
    if closed:
        systems["closed_book"] = {d: closed[d] for d in dates if d in closed}

    order = [s for s in ("always_explain", "gate_only", "direction_only", "closed_book",
                         "pipeline", "pipeline_guarded") if s in systems]
    summary = {name: score(systems[name], labels) for name in order}

    n_explain = sum(1 for d in dates if labels[d]["expected_outcome"].strip().upper() == "EXPLAIN")
    output = {
        "n_labelled": len(labels),
        "n_dates_compared": len(dates),
        "n_explain_labelled": n_explain,
        "n_refuse_labelled": len(dates) - n_explain,
        "dates": sorted(dates),
        "summary": summary,
        "per_date": [
            {"date": d, "label": labels[d]["expected_outcome"].strip().upper(),
             **{name: systems[name].get(d) for name in order}}
            for d in sorted(dates)
        ],
    }

    with open(RESULTS_DIR / "baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    report = generate_report(output, order)
    with open(RESULTS_DIR / "baseline_comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    if verbose:
        print(report)
    return output


def generate_report(output: dict, order: list) -> str:
    s = output["summary"]
    L = [
        "# Baseline Comparison",
        "",
        f"Decisions on {output['n_dates_compared']} hand-labelled dates "
        f"({output['n_explain_labelled']} labelled EXPLAIN, "
        f"{output['n_refuse_labelled']} labelled REFUSE). Dates where the pipeline run "
        f"ended in a fault are left out for every system.",
        "",
        "| System | Explain recall | Refusal specificity | Balanced accuracy | Raw accuracy |",
        "|---|---|---|---|---|",
    ]
    for name in order:
        r = s[name]
        bal = "n/a" if r["balanced_accuracy_pct"] is None else f"{r['balanced_accuracy_pct']}%"
        L.append(f"| {name} | {m.fmt_rate(r['explain_recall'])} | "
                 f"{m.fmt_rate(r['refusal_specificity'])} | {bal} | {m.fmt_rate(r['accuracy'])} |")
    L += ["", "## Systems", ""]
    L += [f"- **{name}**: {SYSTEM_NOTES[name]}" for name in order]

    L += ["", "## Reading it", ""]
    pipe, always = s["pipeline"], s["always_explain"]
    if always["accuracy"]["pct"] is not None and pipe["accuracy"]["pct"] is not None:
        if always["accuracy"]["pct"] >= pipe["accuracy"]["pct"]:
            L.append(f"On raw accuracy the pipeline as run ({pipe['accuracy']['pct']}%) does "
                     f"not beat explaining everything ({always['accuracy']['pct']}%). That is "
                     f"partly the label mix - {output['n_explain_labelled']} of "
                     f"{output['n_dates_compared']} dates are labelled EXPLAIN, so never "
                     f"refusing is right most of the time - and partly real: the pipeline "
                     f"refused dates it should have explained and explained dates it should "
                     f"have refused.")
        else:
            L.append(f"The pipeline as run ({pipe['accuracy']['pct']}%) is ahead of explaining "
                     f"everything ({always['accuracy']['pct']}%) on raw accuracy.")
        L.append("")
    if pipe["balanced_accuracy_pct"] is not None and always["balanced_accuracy_pct"] is not None:
        L.append(f"Balanced accuracy, which a never-refuse system cannot game: pipeline "
                 f"{pipe['balanced_accuracy_pct']}%, always-explain "
                 f"{always['balanced_accuracy_pct']}%.")
        L.append("")
    gate = s["gate_only"]
    if gate["refusal_specificity"]["n"]:
        L.append(f"The gate alone blocks {gate['refusal_specificity']['k']} of "
                 f"{gate['refusal_specificity']['n']} REFUSE-labelled dates. Any refusals the "
                 f"pipeline gets right beyond that come from the model's reading of the "
                 f"evidence, not from the gate.")
        L.append("")
    guarded = s["pipeline_guarded"]
    if guarded["refusal_specificity"]["k"] != pipe["refusal_specificity"]["k"]:
        L.append(f"With the direction guard applied to the same runs, refusal specificity "
                 f"goes from {pipe['refusal_specificity']['k']}/{pipe['refusal_specificity']['n']} "
                 f"to {guarded['refusal_specificity']['k']}/{guarded['refusal_specificity']['n']}. "
                 f"The guard was written after looking at these runs, so this is an in-sample "
                 f"result, not a validated one.")
        L.append("")
    trivial = s.get("direction_only")
    if (trivial and trivial["balanced_accuracy_pct"] is not None
            and guarded["balanced_accuracy_pct"] is not None):
        verb = ("does no better than" if guarded["balanced_accuracy_pct"]
                <= trivial["balanced_accuracy_pct"] else "is ahead of")
        L.append(f"The guarded pipeline (balanced accuracy {guarded['balanced_accuracy_pct']}%, "
                 f"raw {guarded['accuracy']['pct']}%) {verb} a rule that reads no news and "
                 f"simply refuses every day prices fell ({trivial['balanced_accuracy_pct']}%, "
                 f"raw {trivial['accuracy']['pct']}%). All "
                 f"{trivial['refusal_specificity']['n']} REFUSE-labelled dates are scored "
                 f"correct by that rule {trivial['refusal_specificity']['k']} time(s), so "
                 f"these labels cannot separate a guard that detects unsupported explanations "
                 f"from one that detects down days. Should-refuse dates on which prices rose "
                 f"are needed before the guard's score here means anything.")
        L.append("")
    if "closed_book" not in s:
        L.append("The closed-book baseline (same model, no documents) is missing because "
                 "that experiment has not been run: `python -m src.evaluation.robustness "
                 "--experiments closed_book`.")
        L.append("")

    L += ["## Per date", "", "| Date | Label | " + " | ".join(order) + " |",
          "|---|---|" + "---|" * len(order)]
    for row in output["per_date"]:
        L.append(f"| {row['date']} | {row['label']} | "
                 + " | ".join(str(row.get(name) or "-") for name in order) + " |")

    L += ["", "## Limits", "",
          f"- {output['n_refuse_labelled']} REFUSE-labelled dates. Every specificity figure "
          f"here rests on that many cases; the intervals say how little that pins down.",
          "- Each label rests on one round of research that nobody has checked independently.",
          "- A correct EXPLAIN decision means the system chose to explain. Whether the text "
          "names the documented cause is not scored here.",
          ""]
    return "\n".join(L)


def main():
    argparse.ArgumentParser(
        description="Decision-level comparison against trivial baselines").parse_args()
    run_comparison()


if __name__ == "__main__":
    main()
