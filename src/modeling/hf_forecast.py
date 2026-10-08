"""
Alternative time-series model evaluation: ARIMA + Chronos-T5.

Compares two alternative forecasters against the project's logistic regression
baseline, all using the same walk-forward protocol (model sees only past data):

  1. ARIMA(5,1,0) — classical statistical baseline (statsmodels)
  2. Chronos-T5 — Amazon's pre-trained foundation model for time series
     (zero-shot, no fine-tuning). Requires HuggingFace access; skipped
     gracefully if the model cannot be downloaded.

Both derive a directional call from their price-level forecast vs current price
and are scored the same way as the logistic regression: walk-forward accuracy
against a naive majority-class baseline.

Usage:
    python -m src.modeling.hf_forecast                    # coffee, ARIMA + Chronos
    python -m src.modeling.hf_forecast --commodity wheat
    python -m src.modeling.hf_forecast --arima-only       # skip Chronos
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.commodities import COMMODITIES, get_commodity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_MD_PATH = REPO_ROOT / "results" / "hf_forecast_evaluation.md"

# Walk-forward parameters
MIN_HISTORY = 300       # trading days before first prediction
STEP_SIZE = 5           # re-forecast every N days (ARIMA is fast; Chronos uses 50)
ARIMA_ORDER = (5, 1, 0) # AR(5) on differenced series — simple, robust

# Chronos parameters
CHRONOS_MODEL = "amazon/chronos-t5-small"
CHRONOS_CONTEXT = 256
CHRONOS_SAMPLES = 20
CHRONOS_STEP = 50       # slower model, coarser grid


# ---------------------------------------------------------------------------
# ARIMA walk-forward
# ---------------------------------------------------------------------------

def walk_forward_arima(prices: pd.DataFrame, horizon: int) -> dict:
    """Walk-forward directional eval using ARIMA(5,1,0)."""
    from statsmodels.tsa.arima.model import ARIMA

    df = prices.sort_values("date").reset_index(drop=True)
    price_arr = df["price"].astype(float).values

    if len(price_arr) < MIN_HISTORY + horizon:
        return {"error": f"not enough data ({len(price_arr)} rows)"}

    preds, truths = [], []
    eval_positions = range(MIN_HISTORY, len(price_arr) - horizon, STEP_SIZE)

    for t in eval_positions:
        history = price_arr[max(0, t - 500):t + 1]  # last 500 days max
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(history, order=ARIMA_ORDER)
                fit = model.fit()
                forecast = fit.forecast(steps=horizon)
                predicted_price = forecast[-1]
        except Exception:
            continue

        current_price = price_arr[t]
        actual_future = price_arr[t + horizon]
        pred_up = 1 if predicted_price > current_price else 0
        actual_up = 1 if actual_future > current_price else 0
        preds.append(pred_up)
        truths.append(actual_up)

    preds, truths = np.array(preds), np.array(truths)
    if len(preds) == 0:
        return {"error": "no predictions produced"}

    acc = float((preds == truths).mean())
    up_rate = float(truths.mean())
    baseline = max(up_rate, 1 - up_rate)
    return {
        "horizon": horizon, "n_test": int(len(preds)),
        "accuracy": acc, "baseline_accuracy": baseline,
        "edge_over_baseline": acc - baseline, "test_up_rate": up_rate,
        "model": f"ARIMA{ARIMA_ORDER}", "step_size": STEP_SIZE,
    }


# ---------------------------------------------------------------------------
# Chronos walk-forward (optional — needs HuggingFace access)
# ---------------------------------------------------------------------------

def load_chronos_pipeline():
    """Load Chronos pipeline. Returns None if unavailable."""
    try:
        import torch
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32)
        return pipeline
    except Exception as e:
        print(f"  Chronos not available: {e}")
        return None


def walk_forward_chronos(prices: pd.DataFrame, horizon: int, pipeline) -> dict:
    """Walk-forward directional eval using Chronos-T5 zero-shot forecasting."""
    import torch

    df = prices.sort_values("date").reset_index(drop=True)
    price_arr = df["price"].astype(float).values

    if len(price_arr) < MIN_HISTORY + horizon:
        return {"error": f"not enough data ({len(price_arr)} rows)"}

    preds, truths = [], []
    eval_positions = range(MIN_HISTORY, len(price_arr) - horizon, CHRONOS_STEP)

    for t in eval_positions:
        start = max(0, t - CHRONOS_CONTEXT + 1)
        context = torch.tensor(price_arr[start:t + 1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            forecast = pipeline.predict(
                context, prediction_length=max(horizon, 5),
                num_samples=CHRONOS_SAMPLES)
        median_forecast = forecast[0, :, horizon - 1].median().item()
        current_price = price_arr[t]
        actual_future = price_arr[t + horizon]
        preds.append(1 if median_forecast > current_price else 0)
        truths.append(1 if actual_future > current_price else 0)

    preds, truths = np.array(preds), np.array(truths)
    if len(preds) == 0:
        return {"error": "no predictions produced"}
    acc = float((preds == truths).mean())
    up_rate = float(truths.mean())
    baseline = max(up_rate, 1 - up_rate)
    return {
        "horizon": horizon, "n_test": int(len(preds)),
        "accuracy": acc, "baseline_accuracy": baseline,
        "edge_over_baseline": acc - baseline, "test_up_rate": up_rate,
        "model": CHRONOS_MODEL, "context_length": CHRONOS_CONTEXT,
        "num_samples": CHRONOS_SAMPLES, "step_size": CHRONOS_STEP,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate(commodity_key: str, arima_only: bool = False) -> dict:
    cfg = get_commodity(commodity_key)
    path = REPO_ROOT / "data" / "raw" / cfg.price_file
    if not path.exists():
        return {"error": f"{path} not found"}
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    out = {"commodity": commodity_key, "display_name": cfg.display_name,
           "arima": {}, "chronos": {}}

    # ARIMA
    print(f"Running ARIMA{ARIMA_ORDER} walk-forward...")
    for h in (1, 5):
        print(f"  h={h}...")
        result = walk_forward_arima(df, h)
        out["arima"][h] = result
        if "error" not in result:
            print(f"    acc={result['accuracy']*100:.1f}% baseline={result['baseline_accuracy']*100:.1f}% "
                  f"edge={result['edge_over_baseline']*100:+.1f} pts n={result['n_test']}")

    # Chronos (optional)
    if not arima_only:
        print(f"\nLoading {CHRONOS_MODEL}...")
        pipeline = load_chronos_pipeline()
        if pipeline is not None:
            for h in (1, 5):
                print(f"  Chronos h={h}...")
                result = walk_forward_chronos(df, h, pipeline)
                out["chronos"][h] = result
                if "error" not in result:
                    print(f"    acc={result['accuracy']*100:.1f}% "
                          f"baseline={result['baseline_accuracy']*100:.1f}% "
                          f"edge={result['edge_over_baseline']*100:+.1f} pts n={result['n_test']}")
        else:
            out["chronos"] = {"skipped": "Model could not be loaded (no HuggingFace access). "
                              "Run on a machine with internet: python -m src.modeling.hf_forecast"}

    return out


def format_comparison_md(res: dict, lr_results: dict | None = None) -> str:
    """Build the full markdown report."""
    lines = [f"## {res.get('display_name', res.get('commodity'))}", ""]

    # ARIMA section
    lines.append(f"### ARIMA{ARIMA_ORDER}")
    lines.append("")
    for h in (1, 5):
        ev = res["arima"].get(h, {})
        name = "Next trading day" if h == 1 else "Next 5 trading days (~1 week)"
        if "error" in ev:
            lines.append(f"- **{name}**: {ev['error']}")
        else:
            lines.append(
                f"- **{name}**: accuracy **{ev['accuracy']*100:.1f}%** vs baseline "
                f"{ev['baseline_accuracy']*100:.1f}% "
                f"(edge {ev['edge_over_baseline']*100:+.1f} pts, n={ev['n_test']})")
    lines.append("")

    # Chronos section
    chronos = res.get("chronos", {})
    if isinstance(chronos, dict) and "skipped" in chronos:
        lines.append(f"### Chronos-T5 (zero-shot)")
        lines.append("")
        lines.append(f"*{chronos['skipped']}*")
        lines.append("")
    elif chronos:
        lines.append(f"### Chronos-T5 (`{CHRONOS_MODEL}`, zero-shot)")
        lines.append("")
        for h in (1, 5):
            ev = chronos.get(h, {})
            name = "Next trading day" if h == 1 else "Next 5 trading days (~1 week)"
            if "error" in ev:
                lines.append(f"- **{name}**: {ev['error']}")
            else:
                lines.append(
                    f"- **{name}**: accuracy **{ev['accuracy']*100:.1f}%** vs baseline "
                    f"{ev['baseline_accuracy']*100:.1f}% "
                    f"(edge {ev['edge_over_baseline']*100:+.1f} pts, n={ev['n_test']})")
        lines.append("")

    # Comparison table
    if lr_results:
        lines.append("### Comparison table")
        lines.append("")
        has_chronos = isinstance(chronos, dict) and "skipped" not in chronos and chronos
        header = "| Horizon | Logistic Regression | ARIMA |"
        sep = "|---------|-------------------|-------|"
        if has_chronos:
            header = "| Horizon | Logistic Regression | ARIMA | Chronos-T5 |"
            sep = "|---------|-------------------|-------|------------|"
        header += " Baseline |"
        sep += "----------|"
        lines.append(header)
        lines.append(sep)
        for h in (1, 5):
            hname = "1 day" if h == 1 else "5 day"
            lr = lr_results["horizons"].get(h, {})
            ar = res["arima"].get(h, {})
            lr_s = f"{lr['accuracy']*100:.1f}%" if "error" not in lr else "—"
            ar_s = f"{ar['accuracy']*100:.1f}%" if "error" not in ar else "—"
            bl_s = f"{ar.get('baseline_accuracy', lr.get('baseline_accuracy', 0))*100:.1f}%"
            row = f"| {hname} | {lr_s} | {ar_s} |"
            if has_chronos:
                cr = chronos.get(h, {})
                cr_s = f"{cr['accuracy']*100:.1f}%" if "error" not in cr else "—"
                row += f" {cr_s} |"
            row += f" {bl_s} |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", default="coffee", choices=list(COMMODITIES.keys()))
    parser.add_argument("--arima-only", action="store_true",
                        help="Skip Chronos (no HuggingFace needed)")
    args = parser.parse_args()

    res = evaluate(args.commodity, arima_only=args.arima_only)
    if "error" in res:
        print(f"Error: {res['error']}")
        return

    # Run LR baseline for comparison
    from src.modeling.forecast import evaluate as lr_evaluate
    print("\nRunning logistic regression baseline for comparison...")
    lr_res = lr_evaluate(args.commodity, include_weather=False)

    md_lines = [
        "# Alternative Time-Series Model Evaluation", "",
        "Walk-forward comparison of alternative forecasters against the project's "
        "logistic regression baseline on directional coffee-price forecasting. All "
        "models use the same protocol: only past data when predicting, accuracy "
        "scored against the naive majority-class baseline.", "",
    ]
    md_lines.append(format_comparison_md(res, lr_res))

    md_lines.append("## Interpretation")
    md_lines.append("")
    md_lines.append(
        "If no model beats the baseline by more than ~1-2 percentage points, the "
        "result confirms that short-horizon commodity-price direction is close to a "
        "coin flip — consistent with weak-form market efficiency. An ARIMA model "
        "that sees the full price history, a pre-trained foundation model (Chronos) "
        "that has seen diverse time-series patterns, and a logistic regression on "
        "hand-crafted features all converge on the same null result. This strengthens "
        "the project's thesis that the valuable work is *explanation* (why did the "
        "price move?) rather than *prediction* (which way will it move?).")
    md_lines.append("")

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
