"""
Backtest: does the analog-scenario model predict price direction?

Tests the quantitative signal path — historical analog scenarios computed
at each date — against actual forward returns. This is the honest test of
whether past-pattern matching has any directional edge.

The RAG outlook's LLM synthesis cannot be backtested on historical dates
because it uses live-only signals (weather forecasts, CFTC positioning,
current news) that aren't available for the past. So this backtest
evaluates the QUANTITATIVE backbone: the analog scenario model, which
finds similar past days by price features and reports the distribution
of what the price did next.

Direction signal: share_up from the 25 nearest analog days.
  >55% of analogs went up → "upward"
  <45% of analogs went up → "downward"
  else → "balanced"

Baselines compared:
  1. Analog signal (the model being tested)
  2. Simple momentum (5-day trailing return sign)
  3. Random (50/50 up/down)

Usage:
    python -m src.evaluation.backtest_outlook
    python -m src.evaluation.backtest_outlook --dates 50
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_prices() -> pd.DataFrame:
    """Load coffee price history."""
    path = REPO_ROOT / "data" / "raw" / "coffee_prices.csv"
    if not path.exists():
        raise FileNotFoundError(f"No price data at {path}. Run fetch_price_data first.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_forward_returns(prices: pd.DataFrame, date_str: str,
                            horizons: list = None) -> dict:
    """Compute actual forward returns from a given date."""
    if horizons is None:
        horizons = [5, 20]

    idx = prices.index[prices["date"] == pd.to_datetime(date_str)]
    if len(idx) == 0:
        return {f"{h}d": None for h in horizons}

    i = idx[0]
    base_price = prices.iloc[i]["price"]
    result = {}
    for h in horizons:
        if i + h < len(prices):
            future_price = prices.iloc[i + h]["price"]
            result[f"{h}d"] = round(100 * (future_price - base_price) / base_price, 2)
        else:
            result[f"{h}d"] = None
    return result


def select_backtest_dates(prices: pd.DataFrame, n_dates: int = 50) -> list:
    """Select evenly-spaced historical dates for backtesting.
    Avoids the most recent 25 trading days (need forward returns) and the
    first 100 (need trailing data for signals)."""
    valid = prices.iloc[100:-25]
    if len(valid) < n_dates:
        indices = valid.index.tolist()
    else:
        step = len(valid) // n_dates
        indices = valid.index[::step].tolist()[:n_dates]
    return [prices.iloc[i]["date"].strftime("%Y-%m-%d") for i in indices]


def momentum_baseline(prices: pd.DataFrame, date_str: str, lookback: int = 5) -> str:
    """Simple momentum baseline: if the price went up over the last N days,
    predict 'upward', else 'downward'."""
    idx = prices.index[prices["date"] == pd.to_datetime(date_str)]
    if len(idx) == 0:
        return "balanced"
    i = idx[0]
    if i < lookback:
        return "balanced"
    past_price = prices.iloc[i - lookback]["price"]
    current_price = prices.iloc[i]["price"]
    if current_price > past_price:
        return "upward"
    elif current_price < past_price:
        return "downward"
    return "balanced"


# ---------------------------------------------------------------------------
# Analog-based directional signal (the model being tested)
# ---------------------------------------------------------------------------

SHARE_UP_THRESHOLD_HIGH = 0.55  # above this -> upward lean
SHARE_UP_THRESHOLD_LOW = 0.45   # below this -> downward lean


def analog_lean_at_date(feat: pd.DataFrame, cols: list,
                        date_str: str, horizon: int = 5) -> dict:
    """Compute the analog-scenario directional lean at a specific historical
    date. Returns {"lean": str, "share_up": float, "n_analogs": int}."""
    from src.modeling.outlook import analog_scenarios

    # Find the index of this date in the feature frame
    date_mask = feat["date"] == date_str
    if not date_mask.any():
        return {"lean": "balanced", "share_up": 0.5, "n_analogs": 0,
                "error": f"date {date_str} not in feature frame"}

    target_pos = int(np.where(date_mask.to_numpy())[0][0])

    result = analog_scenarios(feat, cols, target_pos, horizon=horizon)

    if "error" in result:
        return {"lean": "balanced", "share_up": 0.5, "n_analogs": 0,
                "error": result["error"]}

    share_up = result["share_up"]
    if share_up > SHARE_UP_THRESHOLD_HIGH:
        lean = "upward"
    elif share_up < SHARE_UP_THRESHOLD_LOW:
        lean = "downward"
    else:
        lean = "balanced"

    return {
        "lean": lean,
        "share_up": round(share_up, 3),
        "n_analogs": result["n_analogs"],
        "median_pct": round(result["median_pct"], 2),
    }


def evaluate_prediction(lean: str, actual_return: float) -> dict:
    """Evaluate whether a predicted lean matches the actual return."""
    if actual_return is None:
        return {"correct": None, "direction_match": None}

    actual_dir = ("upward" if actual_return > 0.5
                  else "downward" if actual_return < -0.5
                  else "balanced")

    if lean == "balanced":
        correct = abs(actual_return) < 0.5
    elif lean == "upward":
        correct = actual_return > 0
    elif lean == "downward":
        correct = actual_return < 0
    else:
        correct = False

    return {
        "correct": correct,
        "direction_match": (lean == actual_dir),
        "actual_direction": actual_dir,
    }


def run_backtest(n_dates: int = 50, verbose: bool = True) -> dict:
    """Run the full backtest using analog scenarios at each historical date."""
    from src.modeling.outlook import build_outlook_frame

    prices = load_prices()
    dates = select_backtest_dates(prices, n_dates)

    # Build feature frame once (covers all dates)
    feat, cols, weather_used = build_outlook_frame(prices, use_weather=True)

    if verbose:
        print(f"=== Outlook Backtest (Analog Scenarios) ===")
        print(f"Testing {len(dates)} dates from {dates[0]} to {dates[-1]}")
        print(f"Features: {cols}")
        print(f"Weather feature included: {weather_used}")
        print(f"Direction thresholds: share_up > {SHARE_UP_THRESHOLD_HIGH} = upward, "
              f"< {SHARE_UP_THRESHOLD_LOW} = downward\n")

    results = []
    for i, date_str in enumerate(dates):
        if verbose:
            print(f"[{i+1}/{len(dates)}] {date_str}...", end=" ", flush=True)

        # Forward returns (ground truth)
        fwd = compute_forward_returns(prices, date_str)

        # Analog-based lean for each horizon
        analog_5d = analog_lean_at_date(feat, cols, date_str, horizon=5)
        analog_20d = analog_lean_at_date(feat, cols, date_str, horizon=20)

        # Momentum baseline
        mom_lean = momentum_baseline(prices, date_str)

        # Random baseline
        rand_lean = random.choice(["upward", "downward"])

        # Evaluate all
        entry = {
            "date": date_str,
            "fwd_5d": fwd.get("5d"),
            "fwd_20d": fwd.get("20d"),
            "analog_lean_5d": analog_5d["lean"],
            "analog_lean_20d": analog_20d["lean"],
            "analog_share_up_5d": analog_5d["share_up"],
            "analog_share_up_20d": analog_20d["share_up"],
            "analog_median_5d": analog_5d.get("median_pct"),
            "analog_median_20d": analog_20d.get("median_pct"),
            "momentum_lean": mom_lean,
            "random_lean": rand_lean,
            "analog_5d_eval": evaluate_prediction(analog_5d["lean"], fwd.get("5d")),
            "analog_20d_eval": evaluate_prediction(analog_20d["lean"], fwd.get("20d")),
            "momentum_5d_eval": evaluate_prediction(mom_lean, fwd.get("5d")),
            "momentum_20d_eval": evaluate_prediction(mom_lean, fwd.get("20d")),
            "random_5d_eval": evaluate_prediction(rand_lean, fwd.get("5d")),
            "random_20d_eval": evaluate_prediction(rand_lean, fwd.get("20d")),
        }
        results.append(entry)

        if verbose:
            a5 = entry["analog_5d_eval"]
            m5 = entry["momentum_5d_eval"]
            print(f"analog={analog_5d['lean']}({analog_5d['share_up']:.0%}) "
                  f"5d_actual={fwd.get('5d', '?'):+.1f}% "
                  f"{'OK' if a5.get('correct') else 'X'} "
                  f"(mom: {'OK' if m5.get('correct') else 'X'})")

    # Aggregate stats
    summary = _compute_summary(results)

    output = {
        "n_dates": len(dates),
        "date_range": f"{dates[0]} to {dates[-1]}",
        "method": "analog_scenarios",
        "features": cols,
        "weather_used": weather_used,
        "thresholds": {
            "share_up_high": SHARE_UP_THRESHOLD_HIGH,
            "share_up_low": SHARE_UP_THRESHOLD_LOW,
        },
        "summary": summary,
        "results": results,
    }

    # Save results
    output_path = RESULTS_DIR / "backtest_outlook.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    # Generate markdown report
    report = _generate_report(summary, len(dates), dates[0], dates[-1],
                              cols, weather_used)
    report_path = RESULTS_DIR / "backtest_outlook_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    if verbose:
        print(f"\n{report}")
        print(f"\nSaved to {output_path}")
        print(f"Report at {report_path}")

    return output


def _compute_summary(results: list) -> dict:
    """Compute aggregate accuracy metrics."""
    summary = {}
    for model in ["analog", "momentum", "random"]:
        for horizon in ["5d", "20d"]:
            key = f"{model}_{horizon}_eval"
            evals = [r[key] for r in results if r[key].get("correct") is not None]
            correct = sum(1 for e in evals if e["correct"])
            total = len(evals)
            summary[f"{model}_{horizon}"] = {
                "correct": correct,
                "total": total,
                "accuracy": round(100 * correct / total, 1) if total > 0 else 0,
            }

    # Analog lean distribution
    lean_5d = {}
    lean_20d = {}
    for r in results:
        l5 = r["analog_lean_5d"]
        l20 = r["analog_lean_20d"]
        lean_5d[l5] = lean_5d.get(l5, 0) + 1
        lean_20d[l20] = lean_20d.get(l20, 0) + 1
    summary["lean_distribution_5d"] = lean_5d
    summary["lean_distribution_20d"] = lean_20d

    # Average share_up
    shares_5d = [r["analog_share_up_5d"] for r in results
                 if r["analog_share_up_5d"] is not None]
    shares_20d = [r["analog_share_up_20d"] for r in results
                  if r["analog_share_up_20d"] is not None]
    summary["avg_share_up_5d"] = round(np.mean(shares_5d), 3) if shares_5d else None
    summary["avg_share_up_20d"] = round(np.mean(shares_20d), 3) if shares_20d else None

    return summary


def _generate_report(summary: dict, n_dates: int, start: str, end: str,
                     features: list, weather_used: bool) -> str:
    """Generate a markdown report of backtest results."""
    lines = [
        "# Outlook Backtest Results (Analog Scenarios)",
        "",
        f"**Period**: {start} to {end} ({n_dates} dates tested)",
        f"**Method**: Historical analog matching (25 nearest neighbors by "
        f"price features)",
        f"**Features**: {', '.join(features)}",
        f"**Weather included**: {'yes' if weather_used else 'no'}",
        f"**Direction rule**: share_up > {SHARE_UP_THRESHOLD_HIGH:.0%} = upward, "
        f"< {SHARE_UP_THRESHOLD_LOW:.0%} = downward, else balanced",
        "",
        "## Direction Accuracy",
        "",
        "| Model | 5-day accuracy | 20-day accuracy |",
        "|-------|---------------|-----------------|",
    ]

    for model, label in [("analog", "Analog Scenarios"),
                          ("momentum", "Momentum (5d trailing)"),
                          ("random", "Random (50/50)")]:
        a5 = summary[f"{model}_5d"]
        a20 = summary[f"{model}_20d"]
        lines.append(f"| {label} | {a5['accuracy']}% ({a5['correct']}/{a5['total']}) "
                     f"| {a20['accuracy']}% ({a20['correct']}/{a20['total']}) |")

    lines.extend([
        "",
        "## Lean Distribution (5-day horizon)",
        "",
    ])
    dist = summary.get("lean_distribution_5d", {})
    for lean, count in sorted(dist.items()):
        lines.append(f"- {lean}: {count} ({100*count/n_dates:.0f}%)")

    lines.extend([
        "",
        "## Lean Distribution (20-day horizon)",
        "",
    ])
    dist = summary.get("lean_distribution_20d", {})
    for lean, count in sorted(dist.items()):
        lines.append(f"- {lean}: {count} ({100*count/n_dates:.0f}%)")

    avg5 = summary.get("avg_share_up_5d")
    avg20 = summary.get("avg_share_up_20d")
    lines.extend([
        "",
        "## Signal Statistics",
        "",
        f"- Average share_up (5d analogs): {avg5:.1%}" if avg5 else "",
        f"- Average share_up (20d analogs): {avg20:.1%}" if avg20 else "",
        "",
        "## Interpretation",
        "",
        "This backtest evaluates the analog scenario model — the quantitative "
        "backbone of the outlook module. It finds the 25 most similar past "
        "days (by price features) and checks whether their forward returns "
        "predict direction.",
        "",
        "The full RAG outlook adds LLM synthesis of live signals (weather "
        "forecasts, CFTC positioning, news) on top of these analogs. That "
        "layer cannot be backtested historically because those signals are "
        "not available for past dates. What this measures is whether the "
        "pattern-matching foundation has any edge.",
        "",
        "If analog accuracy is near 50%, that confirms the base finding: "
        "short-horizon commodity direction is close to random, and the "
        "system's value is in explanation (why did the price move?) not "
        "prediction (which way next?). If it's meaningfully above 50%, "
        "the feature set captures real regime information.",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the analog-scenario outlook model")
    parser.add_argument("--dates", type=int, default=50,
                        help="Number of historical dates to test (default: 50)")
    args = parser.parse_args()

    run_backtest(n_dates=args.dates)


if __name__ == "__main__":
    main()
