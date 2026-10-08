"""
Naive baseline forecast (proposal S3.2: "a naive seasonal/moving-average
forecast is built and evaluated before any transformer model. Every
subsequent model must beat this baseline... or the added complexity is
not justified").

Intentionally simple: tomorrow's predicted price = today's price (naive
persistence) and, as a second baseline, a trailing N-day moving average.
No fine-tuned HF model here - that is a separate, heavier effort scoped out
of this pass. This gives the project a computed baseline number to report,
which is what the proposal requires as a gate ("every subsequent model must
beat this"), even before the heavier model exists.

Evaluated with walk-forward validation (train on past, test on future -
never a random split, which leaks future information for time series).

Usage:
    python -m src.modeling.baseline_forecast
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = REPO_ROOT / "data" / "raw" / "coffee_prices.csv"
OUTPUT_PATH = REPO_ROOT / "results" / "baseline_metrics.json"

MOVING_AVG_WINDOW = 20  # ~1 trading month


def walk_forward_naive_and_ma(df: pd.DataFrame, ma_window: int = MOVING_AVG_WINDOW) -> pd.DataFrame:
    """
    Walk-forward: for each day t, predict price[t] using ONLY information
    available before t (price[t-1], or the moving average up to t-1).
    Never uses price[t] or later to predict price[t] - that would be
    lookahead bias, which this project treats as a hard rule elsewhere
    (data pipeline, anomaly detection) and applies here too for consistency.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["naive_pred"] = df["price"].shift(1)
    df["ma_pred"] = df["price"].rolling(ma_window).mean().shift(1)
    return df


def compute_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    valid = df.dropna(subset=[pred_col, "price"])
    errors = valid["price"] - valid[pred_col]
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "n": len(valid)}


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} not found - run src.modeling.fetch_price_data first.")

    df = pd.read_csv(INPUT_PATH)
    result_df = walk_forward_naive_and_ma(df)

    naive_metrics = compute_metrics(result_df, "naive_pred")
    ma_metrics = compute_metrics(result_df, "ma_pred")

    print("Naive (persistence) baseline:", naive_metrics)
    print(f"Moving-average ({MOVING_AVG_WINDOW}-day) baseline:", ma_metrics)

    better = "naive" if naive_metrics["mae"] < ma_metrics["mae"] else "moving_average"
    print(f"\nBetter baseline by MAE: {better}")
    print("Note: naive persistence usually wins for a random-walk-like series "
          "like commodity futures - a moving average lags real moves, which "
          "is a normal, expected result here, not a sign of a bug.")

    output = {
        "naive_persistence": naive_metrics,
        f"moving_average_{MOVING_AVG_WINDOW}day": ma_metrics,
        "better_baseline": better,
        "note": (
            "This is the required naive baseline per proposal S3.2. No "
            "fine-tuned HF forecasting model has been built - that's a "
            "separate, heavier task explicitly deferred. Any future model "
            "must beat these numbers on held-out data to justify its "
            "added complexity, per the proposal's own rule."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
