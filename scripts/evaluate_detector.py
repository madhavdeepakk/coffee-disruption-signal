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
  - What chance would score: the detector flags roughly one day in eight,
    and flagged days cluster, so a window of several days often holds a flag
    whether or not anything happened in it. For every event the share of ALL
    windows of the same length in the series that hold at least one flag is
    reported, and from those the number of events a detector with no skill
    would be expected to "detect". Recall is worth something only where it
    beats that. The share of each window's days that are flagged, against
    the series-wide flag rate, is the second check: a real event should be
    flagged far more densely than an ordinary stretch.

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


def chance_of_a_flag(flags: list, window_days: int) -> float:
    """The share of all windows of this many consecutive trading days that
    hold at least one flagged day: what "detected" would score for a window
    placed at random."""
    n = len(flags)
    if window_days <= 0 or n < window_days:
        return 0.0
    running = sum(flags[:window_days])
    hits = int(running > 0)
    for i in range(window_days, n):
        running += flags[i] - flags[i - window_days]
        hits += running > 0
    return hits / (n - window_days + 1)


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
    # Only days with enough history to be judged count toward the chance rate.
    judged = df[df["rolling_std"].notna()] if "rolling_std" in df.columns else df
    flag_list = [int(bool(f)) for f in judged["anomaly_flag"]]
    expected_by_chance = 0.0
    for label, start, end, _src in cfg.known_events:
        w = _in_window(df, start, end)
        flagged = w[w["anomaly_flag"]]
        hit = len(flagged) > 0
        detected += 1 if hit else 0
        event_window_flagged_days += len(flagged)
        types = sorted(set(flagged["anomaly_type"].tolist())) if "anomaly_type" in df.columns else []
        chance = chance_of_a_flag(flag_list, len(w))
        expected_by_chance += chance
        event_rows.append({
            "label": label, "window": f"{start}..{end}", "trading_days": len(w),
            "flagged_days": len(flagged), "detected": hit, "types": types,
            "chance": chance,
        })
    n_events = len(cfg.known_events)
    recall = detected / n_events if n_events else 0.0
    event_days = sum(e["trading_days"] for e in event_rows)
    event_flag_density = event_window_flagged_days / event_days if event_days else 0.0

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
        "expected_by_chance": expected_by_chance,
        "event_flag_density": event_flag_density,
        "judged_flag_rate": (sum(flag_list) / len(flag_list)) if flag_list else 0.0,
        "control_window": f"{cstart}..{cend}",
        "control_days": control_days,
        "control_false_positives": control_fp,
        "control_specificity": specificity,
        "flagged_in_event": flagged_in_event,
        "flagged_in_control": flagged_in_control,
        "flagged_elsewhere": flagged_elsewhere,
    }


SENSITIVITY_THRESHOLDS = (2.0, 2.5, 3.0)


def threshold_sensitivity(commodity_key: str) -> list:
    """The same evaluation at stricter thresholds, recomputed from the price
    series. The detector's threshold (2.0) was fixed before any validation
    and is not changed by this; the rows show what a different choice would
    have flagged. Price returns have fatter tails than a normal distribution,
    so "2 standard deviations" does not mean "1 day in 22": the share of days
    beyond each threshold is measured here instead of assumed."""
    from src.config.commodities import load_contract_switches
    from src.modeling.anomaly_detector import compute_anomalies
    cfg = get_commodity(commodity_key)
    price_path = REPO_ROOT / "data" / "raw" / cfg.price_file
    if not price_path.exists():
        return []
    raw = pd.read_csv(price_path)
    switches = load_contract_switches(cfg.key)
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        df = compute_anomalies(raw, z_threshold=threshold, reported_returns=switches)
        df["date"] = df["date"].astype(str).str[:10]
        judged = df[df["rolling_std"].notna()]
        flags = [int(bool(f)) for f in judged["anomaly_flag"]]
        detected, expected, event_days, event_flags = 0, 0.0, 0, 0
        for _label, start, end, _src in cfg.known_events:
            w = _in_window(df, start, end)
            n_flagged = int(w["anomaly_flag"].sum())
            detected += n_flagged > 0
            expected += chance_of_a_flag(flags, len(w))
            event_days += len(w)
            event_flags += n_flagged
        cstart, cend, _note = cfg.control_period
        base = sum(flags) / len(flags) if flags else 0.0
        density = event_flags / event_days if event_days else 0.0
        rows.append({
            "threshold": threshold,
            "flagged": sum(flags), "judged": len(flags), "flag_rate": base,
            "shock_rate": float((judged["z_score"].abs() >= threshold).mean()) if len(judged) else 0.0,
            "detected": int(detected), "n_events": len(cfg.known_events),
            "expected_by_chance": expected,
            "control_fp": int(_in_window(df, cstart, cend)["anomaly_flag"].sum()),
            "lift": density / base if base else 0.0,
        })
    return rows


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
        f"- **Against chance:** a window placed at random would hold a flag often enough "
        f"that a detector with no skill would be expected to 'detect' "
        f"{r['expected_by_chance']:.1f} of these {r['n_events']} events. Recall alone "
        f"therefore says little; density says more: "
        f"**{r['event_flag_density']*100:.0f}% of the days inside event windows are flagged, "
        f"against {r['judged_flag_rate']*100:.0f}% of all days** "
        f"({(r['event_flag_density']/r['judged_flag_rate'] if r['judged_flag_rate'] else 0):.1f} "
        f"times the base rate).",
        "",
        "| event | window | trading days | flagged days | detected | chance of a flag in "
        "a random window this long | types |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in r["event_rows"]:
        lines.append(
            f"| {e['label']} | {e['window']} | {e['trading_days']} | {e['flagged_days']} | "
            f"{'yes' if e['detected'] else 'NO'} | {e['chance']*100:.0f}% | "
            f"{', '.join(e['types']) or '-'} |"
        )
    lines.append("")
    sens = r.get("sensitivity") or []
    if sens:
        lines += [
            "Stricter thresholds, recomputed from the price series (the detector runs at "
            "2.0, fixed before validation; the other rows are what a different choice would "
            "flag). Under a normal distribution 4.6% of days would lie beyond 2 standard "
            "deviations, 1.2% beyond 2.5 and 0.27% beyond 3; the 'daily check alone' column "
            "is the measured share.",
            "",
            "| threshold | days flagged (either check) | daily check alone | events detected | "
            "expected by chance | control false positives | density in event windows vs base |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in sens:
            lines.append(
                f"| {row['threshold']:.1f} | {row['flagged']}/{row['judged']} "
                f"({row['flag_rate']*100:.1f}%) | {row['shock_rate']*100:.1f}% | "
                f"{row['detected']}/{row['n_events']} | {row['expected_by_chance']:.1f} | "
                f"{row['control_fp']} | {row['lift']:.1f}x |")
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
          "than a per-day precision/recall over an unlabeled series.", "",
          "**Read the recall figures with the chance column beside them.** The detector "
          "flags about one trading day in eight and flagged days cluster, so a multi-day "
          "window often holds a flag by chance. What separates the detector from chance is "
          "how densely it flags inside event windows, and how that changes at stricter "
          "thresholds (the second table for each commodity).", ""]
    for k in keys:
        r = evaluate_commodity(k)
        if "error" not in r:
            r["sensitivity"] = threshold_sensitivity(k)
        md += format_commodity(r)
        # console
        if "error" in r:
            print(f"{k}: {r['error']}")
        else:
            print(f"{r['display_name']}: recall {r['events_detected']}/{r['n_events']} "
                  f"(chance would give {r['expected_by_chance']:.1f}), "
                  f"control specificity {r['control_specificity']*100:.1f}% "
                  f"({r['control_false_positives']} FP / {r['control_days']} days), "
                  f"flag rate {r['flag_rate']*100:.2f}%")

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved report to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
