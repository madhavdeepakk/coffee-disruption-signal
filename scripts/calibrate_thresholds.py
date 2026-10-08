"""
Reads the hand-labeled CSV produced by scripts/collect_labeling_set.py and
computes data-derived candidate thresholds for the relevance gate's
KEYWORD_THRESHOLD and SEMANTIC_THRESHOLD (src/rag/relevance_gate.py),
which are otherwise reasoned placeholders not validated against labels.
This is the Week 4 threshold calibration task.

Runs entirely offline on the labeled CSV (no network, no GDELT, no LLM), so
unlike collect_labeling_set.py it can be run anywhere once a labeled file
exists.

Method (kept simple given the sample size, typically 15-20 labeled rows):
for each score column (keyword_score, semantic_score) independently, sweep
every observed score value as a candidate threshold ("flag as relevant if
score >= threshold") and compute precision/recall/F1 against the
human_label column. Two candidates are reported per channel:
  - the threshold that maximizes F1 (the conventional "best balance" pick)
  - the highest threshold that still achieves precision >= 0.9 among
    candidates with at least 2 predicted-positive documents, per this
    project's stated principle (src/rag/relevance_gate.py,
    docs/final_evaluation_DRAFT.md): a higher refusal rate is preferable
    to an unsupported explanation, i.e. bias the choice toward precision
    over recall when the two disagree.

Caveat, also printed in the output itself: with ~15-20 labeled rows this is
illustrative calibration, not statistically robust. It checks the
thresholds against real labeled examples rather than validating a
production threshold, so treat the recommendation as informed, not final.

Held-out check: picking the threshold that gives precision 1.0 on a set of
documents and then reporting precision 1.0 on those same documents says
nothing about the next document. So the report also runs leave-one-date-out
cross-validation: for each anomaly date, the threshold is chosen from the
OTHER dates' documents only and then applied to the held-out date. The
pooled held-out precision and recall are what the threshold rule can be
expected to deliver on a date it has not seen. Whole dates are held out
(rather than single documents) because documents retrieved for the same
anomaly are not independent.

Usage:
    python -m scripts.calibrate_thresholds
    python -m scripts.calibrate_thresholds --input path/to/labeled.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import rate, fmt_rate          # noqa: E402
from src.rag.relevance_gate import KEYWORD_THRESHOLD, SEMANTIC_THRESHOLD   # noqa: E402
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "labeling" / "relevance_labeling_set.csv"
OUTPUT_MD_PATH = REPO_ROOT / "results" / "relevance_gate_calibration.md"

VALID_LABELS = {"relevant", "not_relevant"}


def load_labeled(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise SystemExit(
            f"{input_path} not found - run 'python -m scripts.collect_labeling_set' first, "
            f"then hand-label the human_label column before running this script."
        )
    df = pd.read_csv(input_path)
    # A genuinely blank human_label cell comes back as an actual NaN (not
    # the string "nan") under pandas' string dtype handling, and that NaN
    # survives .astype(str) as a real float in some pandas versions.
    # Comparing it to the literal string "nan" raises a TypeError on the
    # first blank row, which is exactly the row users are told to leave
    # blank, so normalize blanks explicitly via isna() instead.
    raw = df["human_label"]
    is_blank = raw.isna() | (raw.astype(str).str.strip() == "")
    df["human_label"] = raw.astype(str).str.strip().str.lower()
    df.loc[is_blank, "human_label"] = ""
    labeled = df[df["human_label"].isin(VALID_LABELS)].copy()
    unlabeled_count = len(df) - len(labeled)
    bad_labels = df[~df["human_label"].isin(VALID_LABELS) & (df["human_label"] != "")]
    if not bad_labels.empty:
        print(f"WARNING: {len(bad_labels)} row(s) have a human_label that isn't exactly "
              f"'relevant' or 'not_relevant' (got: {sorted(bad_labels['human_label'].unique())}) "
              f"- these are excluded, not guessed at.")
    print(f"Loaded {len(df)} rows: {len(labeled)} usable labels, {unlabeled_count} unlabeled/blank "
          f"(excluded).")
    if len(labeled) < 8:
        print(f"WARNING: only {len(labeled)} labeled rows - this is a very small sample. "
              f"Treat any threshold below as a rough signal, not a confident calibration.")
    return labeled


def sweep_thresholds(df: pd.DataFrame, score_col: str) -> dict:
    """
    Returns candidate thresholds for one score column, computed only from
    rows where that score is present (a document with no semantic_score,
    e.g. because semantic scoring was unavailable, is excluded from that
    channel's sweep rather than silently coerced to 0).
    """
    sub = df[df[score_col].notna() & (df[score_col] != "")].copy()
    sub[score_col] = sub[score_col].astype(float)
    if sub.empty:
        return {"error": f"no rows with a usable {score_col}"}

    y_true = (sub["human_label"] == "relevant").to_numpy()
    candidate_thresholds = sorted(sub[score_col].unique())

    rows = []
    for t in candidate_thresholds:
        y_pred = (sub[score_col] >= t).to_numpy()
        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1,
                      "n_flagged": int(y_pred.sum())})

    sweep_df = pd.DataFrame(rows)
    best_f1_row = sweep_df.loc[sweep_df["f1"].idxmax()]

    high_precision = sweep_df[(sweep_df["precision"] >= 0.9) & (sweep_df["n_flagged"] >= 2)]
    high_precision_row = (
        high_precision.sort_values("threshold").iloc[0] if not high_precision.empty else None
    )

    return {
        "n_labeled": len(sub),
        "n_relevant": int(y_true.sum()),
        "n_not_relevant": int((~y_true).sum()),
        "sweep": sweep_df,
        "best_f1": best_f1_row.to_dict(),
        "high_precision": high_precision_row.to_dict() if high_precision_row is not None else None,
    }


def _usable(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    sub = df[df[score_col].notna() & (df[score_col] != "")].copy()
    sub[score_col] = sub[score_col].astype(float)
    return sub


def performance_at(df: pd.DataFrame, score_col: str, threshold: float) -> dict:
    """Precision and recall of 'score >= threshold' against the labels."""
    sub = _usable(df, score_col)
    y_true = (sub["human_label"] == "relevant").to_numpy()
    y_pred = (sub[score_col] >= threshold).to_numpy()
    tp, fp = int((y_true & y_pred).sum()), int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": rate(tp, tp + fp), "recall": rate(tp, tp + fn)}


def _pick_threshold(train: pd.DataFrame, score_col: str, rule: str):
    """Apply one selection rule to a training fold. Returns None when the
    fold cannot support the rule (e.g. no threshold reaches the precision
    bar), which is itself informative and is counted."""
    result = sweep_thresholds(train, score_col)
    if "error" in result:
        return None
    if rule == "best_f1":
        return float(result["best_f1"]["threshold"])
    hp = result["high_precision"]
    return None if hp is None else float(hp["threshold"])


def leave_one_date_out(df: pd.DataFrame, score_col: str, rule: str) -> dict:
    """Choose the threshold without one anomaly date, test on that date,
    repeat for every date, and pool the held-out outcomes."""
    sub = _usable(df, score_col)
    if "anomaly_date" not in sub.columns or sub["anomaly_date"].nunique() < 3:
        return {"error": "needs an anomaly_date column with at least 3 distinct dates"}
    tp = fp = fn = 0
    thresholds, skipped = [], 0
    for date in sorted(sub["anomaly_date"].unique()):
        train, test = sub[sub["anomaly_date"] != date], sub[sub["anomaly_date"] == date]
        threshold = _pick_threshold(train, score_col, rule)
        if threshold is None:
            skipped += 1
            continue
        thresholds.append(threshold)
        y_true = (test["human_label"] == "relevant").to_numpy()
        y_pred = (test[score_col] >= threshold).to_numpy()
        tp += int((y_true & y_pred).sum())
        fp += int((~y_true & y_pred).sum())
        fn += int((y_true & ~y_pred).sum())
    return {
        "rule": rule, "n_folds": len(thresholds), "folds_skipped": skipped,
        "threshold_min": min(thresholds) if thresholds else None,
        "threshold_max": max(thresholds) if thresholds else None,
        "precision": rate(tp, tp + fp), "recall": rate(tp, tp + fn),
    }


def held_out_lines(df: pd.DataFrame, score_col: str, production_threshold: float) -> list:
    lines = []
    prod = performance_at(df, score_col, production_threshold)
    lines.append(f"- **At the production threshold ({production_threshold})**, on all labelled "
                 f"documents (in-sample if the threshold was chosen from them): precision "
                 f"{fmt_rate(prod['precision'])}, recall {fmt_rate(prod['recall'])}")
    for rule, name in (("high_precision", "precision>=0.9 rule"), ("best_f1", "best-F1 rule")):
        cv = leave_one_date_out(df, score_col, rule)
        if "error" in cv:
            lines.append(f"- Held-out ({name}): {cv['error']}")
            continue
        if not cv["n_folds"]:
            lines.append(f"- Held-out ({name}): no fold could select a threshold.")
            continue
        lines.append(
            f"- **Held-out, {name}** (leave-one-date-out, {cv['n_folds']} folds"
            + (f", {cv['folds_skipped']} skipped" if cv["folds_skipped"] else "")
            + f"; chosen thresholds ranged {cv['threshold_min']:.3f}-{cv['threshold_max']:.3f}): "
            f"precision {fmt_rate(cv['precision'])}, recall {fmt_rate(cv['recall'])}")
    lines.append("")
    return lines


def print_and_format(name: str, result: dict) -> list:
    lines = [f"### {name}", ""]
    if "error" in result:
        print(f"{name}: {result['error']}")
        lines.append(f"{result['error']}")
        return lines

    print(f"\n=== {name} ({result['n_labeled']} labeled: "
          f"{result['n_relevant']} relevant, {result['n_not_relevant']} not_relevant) ===")
    bf = result["best_f1"]
    print(f"  Best-F1 threshold:        {bf['threshold']:.3f}  "
          f"(precision={bf['precision']:.2f}, recall={bf['recall']:.2f}, f1={bf['f1']:.2f}, "
          f"n_flagged={int(bf['n_flagged'])})")
    lines.append(
        f"- **Best-F1 threshold**: {bf['threshold']:.3f} (precision={bf['precision']:.2f}, "
        f"recall={bf['recall']:.2f}, f1={bf['f1']:.2f}, n_flagged={int(bf['n_flagged'])})"
    )

    if result["high_precision"] is not None:
        hp = result["high_precision"]
        print(f"  Precision>=0.9 threshold: {hp['threshold']:.3f}  "
              f"(precision={hp['precision']:.2f}, recall={hp['recall']:.2f}, "
              f"n_flagged={int(hp['n_flagged'])}) - biases toward refusing over hallucinating")
        lines.append(
            f"- **Precision>=0.9 threshold** (biases toward refusing over hallucinating, per "
            f"this project's stated principle): {hp['threshold']:.3f} (precision={hp['precision']:.2f}, "
            f"recall={hp['recall']:.2f}, n_flagged={int(hp['n_flagged'])})"
        )
    else:
        print("  No threshold in this sweep reached precision>=0.9 with >=2 flagged documents.")
        lines.append("- No threshold reached precision>=0.9 with >=2 flagged documents in this sample.")

    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    args = parser.parse_args()

    labeled = load_labeled(args.input)

    keyword_result = sweep_thresholds(labeled, "keyword_score")
    semantic_result = sweep_thresholds(labeled, "semantic_score")

    md_lines = [
        "# Relevance Gate Threshold Calibration",
        "",
        f"Based on {len(labeled)} hand-labeled documents "
        f"(source: `{args.input.relative_to(REPO_ROOT) if args.input.is_relative_to(REPO_ROOT) else args.input}`). "
        f"Production thresholds (src/rag/relevance_gate.py): "
        f"KEYWORD_THRESHOLD={KEYWORD_THRESHOLD}, SEMANTIC_THRESHOLD={SEMANTIC_THRESHOLD}.",
        "",
        "Rates are shown with 95% Wilson intervals.",
        "",
    ]
    md_lines += print_and_format("Keyword score", keyword_result)
    md_lines += held_out_lines(labeled, "keyword_score", KEYWORD_THRESHOLD)
    md_lines += print_and_format("Semantic score", semantic_result)
    md_lines += held_out_lines(labeled, "semantic_score", SEMANTIC_THRESHOLD)
    n_dates = labeled["anomaly_date"].nunique() if "anomaly_date" in labeled.columns else 0
    md_lines += [
        "## How to read this",
        "",
        "The best-F1 and precision>=0.9 thresholds above are chosen and scored on the same "
        "documents, so their precision and recall are in-sample and optimistic. The held-out "
        "lines choose the threshold without one anomaly date and score it on that date; those "
        "are the figures to quote.",
        "",
        "## Caveat",
        "",
        f"{len(labeled)} labelled documents from {n_dates} anomaly dates, one annotator. The "
        "intervals show how little that pins down. `python -m scripts.collect_labeling_set "
        "--n 200` builds a larger sheet (existing labels are kept).",
    ]
    for line in md_lines:
        if line.startswith("- **At the production") or line.startswith("- **Held-out"):
            print("  " + line.replace("**", ""))

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nSaved full report to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
