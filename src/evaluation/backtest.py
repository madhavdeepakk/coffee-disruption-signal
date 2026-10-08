"""
Backtest: precision/recall of the anomaly detector against labeled events,
plus a significance test against a randomized/permutation baseline.

Attribution: in the team plan this is Archisha's track (backtesting
framework, event labels, significance test design). This script is a
stand-in to get a computed number into the demo - it should be reviewed and
owned by whoever is responsible for evaluation, not treated as finished
work. The event windows below are the same 4 events + 1 control period
already validated in results/anomaly_detector_validation.md - a small
hand-labeled set, not a rigorous historical event dataset.

Method:
  1. Label every trading day: "event" if it falls in one of the known
     disruption windows, "control" if in the known-quiet window, else
     "unlabeled" (excluded from scoring - we don't know the true label).
  2. Precision/recall/F1 of the actual detector's anomaly_flag against
     these labels, on the labeled subset only.
  3. Significance test: repeatedly generate a RANDOM detector that flags
     the same total number of days as the real detector, but at random
     dates, and compute its precision/recall each time. Compare the real
     detector's score against this null distribution - if the real
     detector clearly beats the random one across many trials, that's
     evidence it's not just noise.

Per the project's own rule (S8.1 "the system will produce false
positives; the report quantifies this rate rather than omitting it") and
the plan's Gate 3 rule (do not manipulate definitions/labels/thresholds
after seeing results to improve reported metrics) - this script reports
whatever it computes, does not re-tune anything based on the outcome.

Usage:
    python -m src.evaluation.backtest
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANOMALIES_PATH = REPO_ROOT / "results" / "anomaly_detections.csv"
OUTPUT_PATH = REPO_ROOT / "results" / "significance_test.csv"
SUMMARY_PATH = REPO_ROOT / "docs" / "final_evaluation_DRAFT.md"

# Same events as results/anomaly_detector_validation.md - reused, not
# re-derived, so the two documents can't silently drift apart.
EVENT_WINDOWS = [
    ("2021-06-01", "2021-08-15"),
    ("2024-08-15", "2024-10-15"),
    ("2025-08-15", "2025-09-20"),
    ("2026-07-01", "2026-07-15"),
]
CONTROL_WINDOW = ("2019-01-01", "2019-03-31")

N_PERMUTATION_TRIALS = 1000
RANDOM_SEED = 42  # fixed, so the significance test is reproducible, not a
                   # different number every run


def label_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["true_label"] = "unlabeled"
    for start, end in EVENT_WINDOWS:
        df.loc[(df["date"] >= start) & (df["date"] <= end), "true_label"] = "event"
    start, end = CONTROL_WINDOW
    df.loc[(df["date"] >= start) & (df["date"] <= end), "true_label"] = "control"
    return df


def precision_recall(y_true_event: np.ndarray, y_pred_flag: np.ndarray) -> dict:
    tp = int(np.sum(y_true_event & y_pred_flag))
    fp = int(np.sum(~y_true_event & y_pred_flag))
    fn = int(np.sum(y_true_event & ~y_pred_flag))
    tn = int(np.sum(~y_true_event & ~y_pred_flag))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def run_significance_test(labeled_df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    y_true_event = (labeled_df["true_label"] == "event").to_numpy()
    y_pred_real = labeled_df["anomaly_flag"].to_numpy().astype(bool)
    n_flagged = int(y_pred_real.sum())
    n_total = len(labeled_df)

    real_metrics = precision_recall(y_true_event, y_pred_real)

    random_precisions, random_recalls, random_f1s = [], [], []
    for _ in range(N_PERMUTATION_TRIALS):
        random_flags = np.zeros(n_total, dtype=bool)
        random_idx = rng.choice(n_total, size=n_flagged, replace=False)
        random_flags[random_idx] = True
        m = precision_recall(y_true_event, random_flags)
        random_precisions.append(m["precision"])
        random_recalls.append(m["recall"])
        random_f1s.append(m["f1"])

    def percentile_rank(real_value, null_distribution):
        return float(np.mean(np.array(null_distribution) < real_value))

    return {
        "real_detector": real_metrics,
        "random_baseline": {
            "n_trials": N_PERMUTATION_TRIALS,
            "mean_precision": round(float(np.mean(random_precisions)), 4),
            "mean_recall": round(float(np.mean(random_recalls)), 4),
            "mean_f1": round(float(np.mean(random_f1s)), 4),
        },
        "real_beats_random_baseline_pct_of_trials": {
            "precision": round(100 * percentile_rank(real_metrics["precision"], random_precisions), 1),
            "recall": round(100 * percentile_rank(real_metrics["recall"], random_recalls), 1),
            "f1": round(100 * percentile_rank(real_metrics["f1"], random_f1s), 1),
        },
        "n_days_flagged_by_real_detector": n_flagged,
        "n_labeled_days_total": n_total,
        "n_event_days": int(y_true_event.sum()),
    }


def main():
    if not ANOMALIES_PATH.exists():
        raise SystemExit(f"{ANOMALIES_PATH} not found - run src.modeling.anomaly_detector first.")

    df = pd.read_csv(ANOMALIES_PATH)
    labeled = label_days(df)
    labeled_only = labeled[labeled["true_label"] != "unlabeled"].copy()

    print(f"Labeled subset: {len(labeled_only)} days "
          f"({(labeled_only['true_label']=='event').sum()} event, "
          f"{(labeled_only['true_label']=='control').sum()} control)")

    results = run_significance_test(labeled_only)
    print(json.dumps(results, indent=2))

    pd.DataFrame([{
        "metric": "precision", "real": results["real_detector"]["precision"],
        "random_mean": results["random_baseline"]["mean_precision"],
        "real_beats_random_pct_trials": results["real_beats_random_baseline_pct_of_trials"]["precision"],
    }, {
        "metric": "recall", "real": results["real_detector"]["recall"],
        "random_mean": results["random_baseline"]["mean_recall"],
        "real_beats_random_pct_trials": results["real_beats_random_baseline_pct_of_trials"]["recall"],
    }, {
        "metric": "f1", "real": results["real_detector"]["f1"],
        "random_mean": results["random_baseline"]["mean_f1"],
        "real_beats_random_pct_trials": results["real_beats_random_baseline_pct_of_trials"]["f1"],
    }]).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(
            "# Evaluation Summary (DRAFT - stand-in, see attribution note in "
            "src/evaluation/backtest.py)\n\n"
            f"Labeled set: {len(labeled_only)} trading days across 4 real "
            f"events + 1 quiet control period (see "
            f"results/anomaly_detector_validation.md for sources).\n\n"
            f"**Real detector**: precision={results['real_detector']['precision']}, "
            f"recall={results['real_detector']['recall']}, f1={results['real_detector']['f1']} "
            f"(TP={results['real_detector']['tp']}, FP={results['real_detector']['fp']}, "
            f"FN={results['real_detector']['fn']}, TN={results['real_detector']['tn']})\n\n"
            f"**Random baseline** ({N_PERMUTATION_TRIALS} trials, same number of "
            f"flags as the real detector): mean precision="
            f"{results['random_baseline']['mean_precision']}, mean recall="
            f"{results['random_baseline']['mean_recall']}, mean f1="
            f"{results['random_baseline']['mean_f1']}\n\n"
            f"Real detector beat the random baseline in "
            f"{results['real_beats_random_baseline_pct_of_trials']['f1']}% of trials on F1.\n\n"
            "**Caveat**: this labeled set is tiny (4 events, "
            "1 control window) compared to a real evaluation dataset - not "
            "enough to make a strong statistical claim, and the event window "
            "boundaries were chosen by hand from news dates, not an "
            "independently-validated label set. This is a computed "
            "number, not a placeholder, but it should be treated as "
            "preliminary until a larger, independently-reviewed event/control "
            "set exists.\n"
        )
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
