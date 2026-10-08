"""
Quantitative evaluation of the anomaly detector against the labeled regions
available: each commodity's known_events (should be flagged) and its
control_period (should stay quiet). Turns the qualitative "4/4 events, 0
false positives" statement into reported numbers, and puts the per-day flag
distribution in context.

Note on what can and can't be computed: there is no fully per-day-labeled
price series here, so a true per-day precision/recall over the whole history
is not available. What is well-defined:
  - Event-level recall: of the known historical events, how many did the
    detector flag at least once inside their window. (An event window spans
    several days and not every day in it is an anomaly, so this is
    event-level, not day-level.)
  - Control specificity: on the control period - a stretch chosen because it
    should be quiet - what fraction of days were left unflagged, and how
    many false positives occurred.
  - Flag-rate context: the overall fraction of days flagged, and how the
    flagged days split across event windows / control / neither, so a low
    control false-positive rate can be read against the base rate.

Runs entirely offline on results/<commodity>_anomaly_detections.csv (coffee's
is anomaly_detections.csv) plus src/config/commodities.py.

Usage:
    python -m scripts.evaluate_detector                 # coffee
    python -m scripts.evaluate_detector --commodity wheat
    python -m scripts.evaluate_detector --all
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config.commodities import COMMODITIES, get_commodity

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MD_PATH = REPO_ROOT / "results" / "detector_evaluation.md"


def _in_window(df, start, end):
    return df[(df["date"] >= start) & (df["date"] <= end)]


def evaluate_commodity(commodity_key: str) -> dict:
    cfg = get_commodity(commodity_key)
    csv_path = REPO_ROOT / "results" / cfg.anomalies_file
    if not csv_path.exists():
        return {"commodity": commodity_key, "error": f"{csv_path} not found - run "
                f"'python -m src.modeling.anomaly_detector --commodity {commodity_key}' first."}
    df = pd.read_csv(csv_path)
    df["date"] = df["date"].astype(str)
    total_days = len(df)
    total_flagged = int(df["anomaly_flag"].sum())

    # Event-level recall
    event_rows = []
    detected = 0
    event_window_flagged_days = 0
    for label, start, end, _src in cfg.known_events:
        w = _in_window(df, start, end)
        flagged = w[w["anomaly_flag"]]
        hit = len(flagged) > 0
        detected += 1 if hit else 0
        event_window_flagged_days += len(flagged)
        types = sorted(set(flagged["anomaly_type"].tolist())) if "anomaly_type" in df.columns else []
        event_rows.append({
            "label": label, "window": f"{start}..{end}", "trading_days": len(w),
            "flagged_days": len(flagged), "detected": hit, "types": types,
        })
    n_events = len(cfg.known_events)
    recall = detected / n_events if n_events else 0.0

    # Control specificity
    cstart, cend, cnote = cfg.control_period
    cw = _in_window(df, cstart, cend)
    control_days = len(cw)
    control_fp = int(cw["anomaly_flag"].sum())
    specificity = (control_days - control_fp) / control_days if control_days else 0.0

    # Flag distribution: event windows vs control vs neither
    in_event = pd.Series(False, index=df.index)
    for label, start, end, _src in cfg.known_events:
        in_event |= (df["date"] >= start) & (df["date"] <= end)
    in_control = (df["date"] >= cstart) & (df["date"] <= cend)
    flagged_mask = df["anomaly_flag"].astype(bool)
    flagged_in_event = int((flagged_mask & in_event).sum())
    flagged_in_control = int((flagged_mask & in_control).sum())
    flagged_elsewhere = total_flagged - flagged_in_event - flagged_in_control

    return {
        "commodity": commodity_key,
        "display_name": cfg.display_name,
        "total_days": total_days,
        "total_flagged": total_flagged,
        "flag_rate": total_flagged / total_days if total_days else 0.0,
        "n_events": n_events,
        "events_detected": detected,
        "event_recall": recall,
        "event_rows": event_rows,
        "control_window": f"{cstart}..{cend}",
        "control_days": control_days,
        "control_false_positives": control_fp,
        "control_specificity": specificity,
        "flagged_in_event": flagged_in_event,
        "flagged_in_control": flagged_in_control,
        "flagged_elsewhere": flagged_elsewhere,
    }


def format_commodity(r: dict) -> list:
    if "error" in r:
        return [f"## {r['commodity']}", "", r["error"], ""]
    lines = [
        f"## {r['display_name']} (`{r['commodity']}`)",
        "",
        f"- **Event-level recall: {r['events_detected']}/{r['n_events']} "
        f"({r['event_recall']*100:.0f}%)** — known events flagged at least once in-window.",
        f"- **Control specificity: {r['control_specificity']*100:.1f}%** — "
        f"{r['control_false_positives']} false positive(s) across {r['control_days']} "
        f"control days ({r['control_window']}).",
        f"- Overall flag rate: {r['total_flagged']}/{r['total_days']} days "
        f"({r['flag_rate']*100:.2f}%).",
        f"- Flagged-day distribution: {r['flagged_in_event']} inside event windows, "
        f"{r['flagged_in_control']} in control, {r['flagged_elsewhere']} elsewhere "
        f"(unlabeled — may include real undocumented moves).",
        "",
        "| event | window | trading days | flagged days | detected | types |",
        "|---|---|---|---|---|---|",
    ]
    for e in r["event_rows"]:
        lines.append(
            f"| {e['label']} | {e['window']} | {e['trading_days']} | {e['flagged_days']} | "
            f"{'yes' if e['detected'] else 'NO'} | {', '.join(e['types']) or '-'} |"
        )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", default="coffee", choices=list(COMMODITIES.keys()))
    parser.add_argument("--all", action="store_true", help="Evaluate every commodity")
    args = parser.parse_args()

    keys = list(COMMODITIES.keys()) if args.all else [args.commodity]
    md = ["# Anomaly Detector — Quantitative Evaluation", "",
          "Evaluated against the labeled regions available: each commodity's known "
          "historical events (should flag) and its verified-quiet control period "
          "(should stay silent). See this script's docstring for why event-level "
          "recall + control specificity are the well-defined metrics here, rather "
          "than a per-day precision/recall over an unlabeled series.", ""]
    for k in keys:
        r = evaluate_commodity(k)
        md += format_commodity(r)
        # console
        if "error" in r:
            print(f"{k}: {r['error']}")
        else:
            print(f"{r['display_name']}: recall {r['events_detected']}/{r['n_events']}, "
                  f"control specificity {r['control_specificity']*100:.1f}% "
                  f"({r['control_false_positives']} FP / {r['control_days']} days), "
                  f"flag rate {r['flag_rate']*100:.2f}%")

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved report to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
