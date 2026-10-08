"""
Directional price forecaster (experimental).

An extension beyond the project's core thesis. The core system detects and
explains anomalies from evidence and was scoped not to forecast, because
short-horizon commodity-price direction is close to a coin flip. This module
reports its own out-of-sample accuracy against a naive baseline alongside
every call, so the accuracy shown is the walk-forward one, not an overfit
backtest number. It is not financial advice.

What it does:
  - Features are lookahead-safe: the feature row for day t uses only prices
    up to and including t (past returns over 1/5/10 days, 10-day momentum,
    20-day realized volatility, a 75-day rolling z-score, and distance from
    the 20-day moving average).
  - Labels: 1 if the close `horizon` trading days later is higher than
    today's, else 0. Daily (horizon=1) and weekly (horizon=5) are supported.
  - Model: logistic regression implemented in numpy (no scikit-learn/scipy -
    the same dependency avoidance used for the embedding model; numpy is the
    only dependency).
  - Evaluation: walk-forward (expanding window). The model is only ever
    tested on days after the days it was trained on, never a random shuffle
    split, which would leak the future into the past for a time series.
    Accuracy is reported next to the naive baseline (always predict the
    majority direction); for daily direction that baseline is already
    ~50-55%, so beating it by a little is the realistic ceiling.

Usage:
    python -m src.modeling.forecast                 # coffee, daily + weekly
    python -m src.modeling.forecast --commodity wheat
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.commodities import COMMODITIES, get_commodity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_MD_PATH = REPO_ROOT / "results" / "forecast_evaluation.md"
GDELT_CACHE_DIR = REPO_ROOT / "data" / "gdelt_cache"

FEATURE_COLUMNS = ["ret_1", "ret_5", "ret_10", "mom_10", "vol_20", "zscore_75", "dist_ma20"]
WEATHER_FEATURE_COLUMNS = ["dryness_zscore", "frost_7d_count", "prcp_30d_sum"]
MIN_TRAIN_DAYS = 250       # ~1 trading year before the first out-of-sample prediction
RETRAIN_EVERY = 20         # walk-forward step: retrain every 20 trading days
ROLLING_Z_WINDOW = 75      # matches the anomaly detector's window

UNIFIED_TABLE_PATH = REPO_ROOT / "data" / "processed" / "unified_daily.csv"


class LogisticRegressionNP:
    """Minimal logistic regression (gradient descent) in numpy. Standardizes
    features using train-set statistics only, so no test-set leakage."""

    def __init__(self, lr: float = 0.3, n_iter: int = 800, l2: float = 1e-3):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.w = self.b = self.mu = self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            z = np.clip(Xs @ self.w + self.b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            self.w -= self.lr * (Xs.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu) / self.sd
        z = np.clip(Xs @ self.w + self.b, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))


def build_features(prices_df: pd.DataFrame, include_weather: bool = False) -> pd.DataFrame:
    """Lookahead-safe features: row t uses only prices up to t.

    When include_weather=True, merges weather columns from the unified daily
    table (data/processed/unified_daily.csv) — dryness z-score, frost count,
    and 30-day precipitation. These are same-day observations (no lookahead).
    """
    df = prices_df.sort_values("date").reset_index(drop=True).copy()
    price = df["price"].astype(float)
    ret = price.pct_change()
    df["ret_1"] = ret
    df["ret_5"] = price.pct_change(5)
    df["ret_10"] = price.pct_change(10)
    df["mom_10"] = price / price.shift(10) - 1.0
    df["vol_20"] = ret.rolling(20).std()
    ma = price.rolling(ROLLING_Z_WINDOW).mean()
    sd = price.rolling(ROLLING_Z_WINDOW).std()
    df["zscore_75"] = (price - ma) / (sd + 1e-9)
    df["dist_ma20"] = price / price.rolling(20).mean() - 1.0

    if include_weather and UNIFIED_TABLE_PATH.exists():
        unified = pd.read_csv(UNIFIED_TABLE_PATH, parse_dates=["date"])
        weather_cols = [c for c in WEATHER_FEATURE_COLUMNS if c in unified.columns]
        if weather_cols:
            df["date"] = pd.to_datetime(df["date"])
            df = df.merge(unified[["date"] + weather_cols], on="date", how="left")
    return df


def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """1 if price `horizon` trading days ahead is higher than today, else 0.
    The last `horizon` rows get NaN (no future to compare to)."""
    future = df["price"].shift(-horizon)
    return (future > df["price"]).astype(float).where(future.notna())


def walk_forward_eval(df: pd.DataFrame, horizon: int,
                      include_weather: bool = False) -> dict:
    """
    Expanding-window walk-forward evaluation. Returns out-of-sample accuracy,
    the naive majority-class baseline over the same test days, and counts.

    When include_weather=True, adds weather features alongside price features.
    """
    feat = build_features(df, include_weather=include_weather)
    feature_cols = list(FEATURE_COLUMNS)
    if include_weather:
        available = [c for c in WEATHER_FEATURE_COLUMNS if c in feat.columns]
        feature_cols = feature_cols + available
    y = make_labels(feat, horizon)
    usable = feat[feature_cols].notna().all(axis=1) & y.notna()
    idx = np.where(usable.to_numpy())[0]
    if len(idx) < MIN_TRAIN_DAYS + RETRAIN_EVERY:
        return {"error": f"not enough usable rows ({len(idx)}) for horizon {horizon}"}

    X_all = feat[feature_cols].to_numpy()
    y_all = y.to_numpy()

    preds, truths = [], []
    start = idx[MIN_TRAIN_DAYS]  # first day we start predicting
    # positions in `idx` we still need to predict, in walk-forward blocks
    pred_positions = idx[idx >= start]
    p = 0
    while p < len(pred_positions):
        block = pred_positions[p:p + RETRAIN_EVERY]
        train_cutoff = block[0]
        # train on all usable rows strictly BEFORE the first day of this block,
        # AND whose label horizon doesn't peek into/after the block (no leakage)
        train_mask = idx[idx < train_cutoff - horizon]
        if len(train_mask) < MIN_TRAIN_DAYS:
            p += RETRAIN_EVERY
            continue
        model = LogisticRegressionNP().fit(X_all[train_mask], y_all[train_mask])
        proba = model.predict_proba(X_all[block])
        preds.extend((proba >= 0.5).astype(int).tolist())
        truths.extend(y_all[block].astype(int).tolist())
        p += RETRAIN_EVERY

    preds = np.array(preds)
    truths = np.array(truths)
    if len(preds) == 0:
        return {"error": f"no out-of-sample predictions produced for horizon {horizon}"}
    acc = float((preds == truths).mean())
    up_rate = float(truths.mean())
    baseline = max(up_rate, 1 - up_rate)  # always predict the majority class
    return {
        "horizon": horizon,
        "n_test": int(len(preds)),
        "accuracy": acc,
        "baseline_accuracy": baseline,
        "edge_over_baseline": acc - baseline,
        "test_up_rate": up_rate,
        "features_used": feature_cols,
        "include_weather": include_weather,
    }


def train_and_predict_latest(df: pd.DataFrame, horizon: int) -> dict:
    """
    Train on all usable history and predict the direction for the NEXT
    `horizon` trading days from the most recent available day. For live use.
    """
    feat = build_features(df)
    y = make_labels(feat, horizon)
    usable = feat[FEATURE_COLUMNS].notna().all(axis=1) & y.notna()
    idx = np.where(usable.to_numpy())[0]
    if len(idx) < MIN_TRAIN_DAYS:
        return {"error": f"not enough history to train for horizon {horizon}"}
    X_all = feat[FEATURE_COLUMNS].to_numpy()
    y_all = y.to_numpy()
    model = LogisticRegressionNP().fit(X_all[idx], y_all[idx])

    # latest row with complete features (may be within the last `horizon` days,
    # where the label is unknown - that's exactly the day we forecast FROM)
    feat_ok = feat[FEATURE_COLUMNS].notna().all(axis=1).to_numpy()
    last_pos = np.where(feat_ok)[0][-1]
    proba_up = float(model.predict_proba(X_all[last_pos:last_pos + 1])[0])
    as_of = pd.to_datetime(feat.iloc[last_pos]["date"]).strftime("%Y-%m-%d")
    return {
        "horizon": horizon,
        "as_of_date": as_of,
        "proba_up": proba_up,
        "direction": "up" if proba_up >= 0.5 else "down",
        "confidence_pct": round(abs(proba_up - 0.5) * 200, 1),  # 0 at 50/50, 100 at 0/1
    }


def news_direction_signal(commodity_key: str, date_str: str, window_days: int = 10,
                          allow_live: bool = True, cache_dir=GDELT_CACHE_DIR) -> dict:
    """
    A separate, unvalidated context signal (not a trained model input): of the
    recent disruption-news headlines for this commodity in the lookahead-safe
    window ending on `date_str`, how many describe prices rising vs falling.

    Lightweight and cache-first: it scores GDELT article titles with the
    direction lexicon (src/rag/direction.py) - no full-text fetch, no
    embedding model - so it reuses the on-disk GDELT cache and is cheap. With
    allow_live=False it uses only cached windows and never touches the network
    (so a dashboard can show it without hitting GDELT's unreliable endpoint).

    There is no historical daily news series to train on, which is why this is
    a live overlay and not a model feature - see the module docstring.
    """
    from datetime import datetime, timedelta
    from src.rag.gdelt_client import build_query_params, cache_get, fetch_gdelt, DEFAULT_SORT
    from src.rag import direction as direction_mod

    cfg = get_commodity(commodity_key)
    end_dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
    start_dt = end_dt - timedelta(days=window_days + 1)
    startdatetime = start_dt.strftime("%Y%m%d%H%M%S")
    enddatetime = end_dt.strftime("%Y%m%d%H%M%S")

    seen, docs = set(), []
    used_live = False
    for q in cfg.gdelt_queries:
        params = build_query_params(q, 20, DEFAULT_SORT, startdatetime, enddatetime)
        raw = cache_get(cache_dir, params)
        if raw is None:
            if not allow_live:
                continue
            try:
                raw = fetch_gdelt(q, maxrecords=20, startdatetime=startdatetime,
                                  enddatetime=enddatetime, verbose=False)
                used_live = True
            except RuntimeError:
                continue
        for art in raw.get("articles", []) or []:
            url = art.get("url", "") or ""
            if url and url not in seen:
                seen.add(url)
                docs.append({"title": art.get("title", "") or "", "text": ""})
    if not docs:
        return {"available": False, "used_live": used_live,
                "reason": "no cached/retrieved headlines for this window"}
    lean = direction_mod.news_lean(docs)
    lean.update({"available": True, "used_live": used_live})
    return lean


def evaluate(commodity_key: str, include_weather: bool = False) -> dict:
    cfg = get_commodity(commodity_key)
    path = REPO_ROOT / "data" / "raw" / cfg.price_file
    if not path.exists():
        return {"error": f"{path} not found - run fetch_price_data --commodity {commodity_key} first."}
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    out = {"commodity": commodity_key, "display_name": cfg.display_name,
           "include_weather": include_weather, "horizons": {}, "latest": {}}
    for h in (1, 5):
        out["horizons"][h] = walk_forward_eval(df, h, include_weather=include_weather)
        out["latest"][h] = train_and_predict_latest(df, h)
    return out


def _format_md(res: dict, label: str = "") -> str:
    title = res.get("display_name", res.get("commodity"))
    if label:
        title = f"{title} — {label}"
    lines = [f"## {title}", ""]
    feat_info = None
    for h in (1, 5):
        ev = res["horizons"].get(h, {})
        name = "Next trading day" if h == 1 else "Next 5 trading days (~1 week)"
        if "error" in ev:
            lines.append(f"- **{name}**: {ev['error']}")
            continue
        if feat_info is None:
            feat_info = ev.get("features_used", FEATURE_COLUMNS)
        lines.append(
            f"- **{name}**: out-of-sample accuracy **{ev['accuracy']*100:.1f}%** vs naive "
            f"baseline {ev['baseline_accuracy']*100:.1f}% "
            f"(edge {ev['edge_over_baseline']*100:+.1f} pts over {ev['n_test']} walk-forward test days; "
            f"{ev['test_up_rate']*100:.0f}% of those days were up)."
        )
    if feat_info:
        lines.append(f"\nFeatures used: {', '.join(feat_info)}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", default="coffee", choices=list(COMMODITIES.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--multivariate", action="store_true",
                        help="Also run with weather features and compare (requires unified table)")
    args = parser.parse_args()

    keys = list(COMMODITIES.keys()) if args.all else [args.commodity]
    md = ["# Directional Forecaster — Out-of-Sample Evaluation", "",
          "Experimental, and not financial advice. Short-horizon commodity-price direction is "
          "near-random; these numbers are walk-forward (the model is only tested on days after "
          "the days it trained on) and shown next to the naive majority-class baseline so any "
          "edge over it is visible. See src/modeling/forecast.py.", ""]

    for k in keys:
        # Price-only evaluation
        res = evaluate(k, include_weather=False)
        if "error" in res:
            print(f"{k}: {res['error']}")
            md.append(f"## {k}\n\n{res['error']}\n")
            continue
        md.append(_format_md(res, "Price-only features"))
        for h in (1, 5):
            ev = res["horizons"][h]
            lt = res["latest"][h]
            if "error" in ev:
                print(f"{k} h={h}: {ev['error']}")
            else:
                print(f"{res['display_name']} (price-only) h={h}: acc {ev['accuracy']*100:.1f}% "
                      f"(baseline {ev['baseline_accuracy']*100:.1f}%, edge {ev['edge_over_baseline']*100:+.1f} pts, "
                      f"n={ev['n_test']})")

        # Multivariate evaluation (price + weather)
        if args.multivariate and k == "coffee":
            if not UNIFIED_TABLE_PATH.exists():
                print(f"  Skipping multivariate: {UNIFIED_TABLE_PATH} not found. "
                      "Run `python -m scripts.build_unified_table` first.")
            else:
                res_mv = evaluate(k, include_weather=True)
                md.append(_format_md(res_mv, "Price + weather features"))
                for h in (1, 5):
                    ev = res_mv["horizons"][h]
                    if "error" not in ev:
                        print(f"{res_mv['display_name']} (price+weather) h={h}: acc {ev['accuracy']*100:.1f}% "
                              f"(baseline {ev['baseline_accuracy']*100:.1f}%, "
                              f"edge {ev['edge_over_baseline']*100:+.1f} pts, n={ev['n_test']})")

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved report to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
