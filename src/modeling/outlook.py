"""
Price outlook (EXPERIMENTAL - read the honesty note).

This does NOT predict a price. Short-horizon commodity direction is close to
random (the directional forecaster in forecast.py demonstrates ~50% accuracy),
because public information is already reflected in today's price. Instead this
module gives two honest, uncertainty-first views of what could realistically
happen next, both grounded in how the price actually behaved in the past:

  A. Analog scenarios - "when conditions looked like today's in the past, what
     did the price do next?" For the target day it finds the most similar past
     days (nearest neighbours on the feature vector) and reports the
     DISTRIBUTION of their subsequent N-day returns (median, 10th-90th
     percentile, range, share that rose), plus how many analogs were found.
     This is descriptive, not a prediction: a range with a sample count.

  B. Large-move risk - "is an unusually large move (either direction) more
     likely than normal right now?" Direction is a coin flip, but volatility
     clusters, so the SIZE of the coming move is more predictable than its
     sign. A logistic-regression classifier (reused from forecast.py) is
     trained walk-forward to flag whether the next N-day absolute move lands
     in the historically large tail, and its out-of-sample accuracy is
     reported next to a naive baseline.

Inputs are lookahead-safe price features (forecast.build_features) and,
WHEN AVAILABLE, a weather-dryness feature derived from the NASA POWER file
(coffee only). News is a live-only signal (there is no historical daily news
series to train on), so it is shown as current context by the dashboard, not
baked into these historical models. NOT financial advice.

Usage:
    python -m src.modeling.outlook                 # coffee
    python -m src.modeling.outlook --commodity wheat
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.commodities import COMMODITIES, get_commodity
from src.modeling.forecast import (
    build_features, FEATURE_COLUMNS, LogisticRegressionNP,
    MIN_TRAIN_DAYS, RETRAIN_EVERY,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_MD_PATH = REPO_ROOT / "results" / "outlook_evaluation.md"
WEATHER_PATH = REPO_ROOT / "data" / "raw" / "nasa_power_weather.csv"

HORIZONS = [5, 20]          # ~1 week and ~1 month ahead
N_ANALOGS = 25              # neighbours used for the scenario distribution
LARGE_MOVE_QUANTILE = 0.75  # "large" = |N-day move| in the top quarter historically


# ---------------------------------------------------------------------------
# Optional weather feature (coffee): a dryness signal from NASA POWER
# ---------------------------------------------------------------------------

def load_weather_dryness() -> pd.DataFrame:
    """
    Return a DataFrame [date, weather_dryness] or an empty frame if no usable
    weather file exists. 'Dryness' is the negative of a 30-day rolling
    precipitation z-score: high when recent rainfall is unusually low (a
    drought signal), computed lookahead-safe. Coffee only; other commodities
    have no weather source wired up and simply run without this feature.
    """
    if not WEATHER_PATH.exists():
        return pd.DataFrame()
    try:
        w = pd.read_csv(WEATHER_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if "date" not in w.columns:
        return pd.DataFrame()
    # find a precipitation column under any of the common NASA POWER names
    precip_col = next((c for c in ("prcp_mm", "PRECTOTCORR", "precip", "prcp", "precipitation")
                       if c in w.columns), None)
    if precip_col is None:
        return pd.DataFrame()
    w = w[["date", precip_col]].copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date"]).sort_values("date")
    p = pd.to_numeric(w[precip_col], errors="coerce")
    roll = p.rolling(30).sum()
    mean = roll.rolling(365).mean().shift(1)
    std = roll.rolling(365).std().shift(1)
    w["weather_dryness"] = -((roll - mean) / (std + 1e-9))  # high = unusually dry
    w["date"] = w["date"].dt.strftime("%Y-%m-%d")
    return w[["date", "weather_dryness"]].dropna()


def build_outlook_frame(prices_df: pd.DataFrame, use_weather: bool = True):
    """Return (feature_frame, feature_columns). Adds weather_dryness when a
    usable weather file exists, otherwise falls back to price features only."""
    feat = build_features(prices_df).copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.strftime("%Y-%m-%d")
    cols = list(FEATURE_COLUMNS)
    weather_used = False
    if use_weather:
        wx = load_weather_dryness()
        if not wx.empty:
            feat = feat.merge(wx, on="date", how="left")
            # only keep the weather feature if it overlaps most of the series
            if feat["weather_dryness"].notna().mean() > 0.5:
                feat["weather_dryness"] = feat["weather_dryness"].ffill()
                cols = cols + ["weather_dryness"]
                weather_used = True
            else:
                feat = feat.drop(columns=["weather_dryness"])
    return feat, cols, weather_used


def _forward_return(price: pd.Series, h: int) -> pd.Series:
    return price.shift(-h) / price - 1.0


# ---------------------------------------------------------------------------
# A. Analog scenarios
# ---------------------------------------------------------------------------

def analog_scenarios(feat: pd.DataFrame, cols: list, target_pos: int, horizon: int,
                     n_analogs: int = N_ANALOGS) -> dict:
    """Find the most similar PAST days to the target day and report the
    distribution of what the price did over the next `horizon` days after each."""
    X = feat[cols].to_numpy(dtype=float)
    ok = ~np.isnan(X).any(axis=1)
    price = feat["price"].to_numpy(dtype=float)
    fwd = np.full(len(feat), np.nan)
    fwd[:len(feat) - horizon] = price[horizon:] / price[:len(feat) - horizon] - 1.0

    if not ok[target_pos]:
        return {"error": "target day has incomplete features"}

    mu = np.nanmean(X[ok], axis=0)
    sd = np.nanstd(X[ok], axis=0) + 1e-9
    Xs = (X - mu) / sd
    target = Xs[target_pos]

    # candidates: complete-feature PAST days whose forward window is known and
    # does not overlap the target day (>= horizon days before it)
    cand = np.where(ok & ~np.isnan(fwd) & (np.arange(len(feat)) <= target_pos - horizon))[0]
    if len(cand) < 5:
        return {"error": "not enough historical analogs"}
    dists = np.sqrt(((Xs[cand] - target) ** 2).sum(axis=1))
    nearest = cand[np.argsort(dists)[:n_analogs]]
    rets = fwd[nearest]
    return {
        "horizon": horizon,
        "n_analogs": int(len(rets)),
        "median_pct": float(np.median(rets) * 100),
        "p10_pct": float(np.percentile(rets, 10) * 100),
        "p90_pct": float(np.percentile(rets, 90) * 100),
        "min_pct": float(rets.min() * 100),
        "max_pct": float(rets.max() * 100),
        "share_up": float((rets > 0).mean()),
    }


# ---------------------------------------------------------------------------
# B. Large-move risk (walk-forward classifier)
# ---------------------------------------------------------------------------

def large_move_backtest(feat: pd.DataFrame, cols: list, horizon: int) -> dict:
    """Walk-forward: can we tell when the next `horizon`-day ABSOLUTE move will
    land in the historically large tail? Threshold is set from TRAIN data only
    in each fold (no leakage).

    Because 'large' is a rare event (~top 25%), plain accuracy is misleading
    (always saying 'not large' scores ~75%). The honest skill measure is LIFT:
    on the days the model flags as elevated, how much more often did a large
    move actually happen than the unconditional base rate. Lift > 1 means real
    skill. We also report precision (share of flagged days that were large),
    recall (share of large moves that were flagged), and how often it flags."""
    price = feat["price"].astype(float).reset_index(drop=True)
    absfwd = _forward_return(price, horizon).abs().to_numpy()
    X_all = feat[cols].to_numpy(dtype=float)
    complete = ~np.isnan(X_all).any(axis=1)
    idx = np.where(complete & ~np.isnan(absfwd))[0]
    if len(idx) < MIN_TRAIN_DAYS + RETRAIN_EVERY:
        return {"error": f"not enough usable rows ({len(idx)}) for horizon {horizon}"}

    flagged, truths = [], []
    pred_positions = idx[idx >= idx[MIN_TRAIN_DAYS]]
    p = 0
    while p < len(pred_positions):
        block = pred_positions[p:p + RETRAIN_EVERY]
        train = idx[idx < block[0] - horizon]
        if len(train) < MIN_TRAIN_DAYS:
            p += RETRAIN_EVERY
            continue
        thr = np.quantile(absfwd[train], LARGE_MOVE_QUANTILE)  # from train only
        y_train = (absfwd[train] >= thr).astype(float)
        model = LogisticRegressionNP().fit(X_all[train], y_train)
        proba = model.predict_proba(X_all[block])
        # "elevated" = model probability above the train base rate of large moves
        flag_threshold = float(y_train.mean())
        flagged.extend((proba >= flag_threshold).tolist())
        truths.extend((absfwd[block] >= thr).tolist())
        p += RETRAIN_EVERY

    flagged, truths = np.array(flagged, dtype=bool), np.array(truths, dtype=bool)
    if len(flagged) == 0:
        return {"error": f"no out-of-sample predictions for horizon {horizon}"}
    base_rate = float(truths.mean())
    n_flagged = int(flagged.sum())
    precision = float(truths[flagged].mean()) if n_flagged else 0.0
    recall = float(flagged[truths].mean()) if truths.any() else 0.0
    lift = (precision / base_rate) if base_rate > 0 else 0.0
    return {"horizon": horizon, "n_test": int(len(flagged)), "base_rate": base_rate,
            "flag_rate": float(flagged.mean()), "precision": precision,
            "recall": recall, "lift": lift, "n_flagged": n_flagged}


def latest_large_move_risk(feat: pd.DataFrame, cols: list, horizon: int) -> dict:
    """Train on all history and estimate the probability that the NEXT
    `horizon`-day absolute move is in the large tail, from the most recent day."""
    price = feat["price"].astype(float).reset_index(drop=True)
    absfwd = _forward_return(price, horizon).abs().to_numpy()
    X_all = feat[cols].to_numpy(dtype=float)
    complete = ~np.isnan(X_all).any(axis=1)
    train = np.where(complete & ~np.isnan(absfwd))[0]
    if len(train) < MIN_TRAIN_DAYS:
        return {"error": "not enough history"}
    thr = np.quantile(absfwd[train], LARGE_MOVE_QUANTILE)
    y = (absfwd[train] >= thr).astype(float)
    model = LogisticRegressionNP().fit(X_all[train], y)
    last_pos = np.where(complete)[0][-1]
    prob = float(model.predict_proba(X_all[last_pos:last_pos + 1])[0])
    base_rate = 1.0 - LARGE_MOVE_QUANTILE  # by construction ~0.25
    as_of = feat.iloc[last_pos]["date"]
    return {"horizon": horizon, "as_of_date": str(as_of), "prob_large_move": prob,
            "typical_rate": base_rate, "elevated": prob > base_rate,
            "large_move_threshold_pct": float(thr * 100)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate(commodity_key: str, use_weather: bool = True) -> dict:
    cfg = get_commodity(commodity_key)
    path = REPO_ROOT / "data" / "raw" / cfg.price_file
    if not path.exists():
        return {"error": f"{path} not found - run fetch_price_data --commodity {commodity_key} first."}
    prices = pd.read_csv(path)
    prices["date"] = pd.to_datetime(prices["date"])
    feat, cols, weather_used = build_outlook_frame(prices, use_weather=use_weather)
    last_pos = int(np.where(~feat[cols].isna().any(axis=1).to_numpy())[0][-1])
    lastrow = feat.iloc[last_pos]
    out = {"commodity": commodity_key, "display_name": cfg.display_name,
           "weather_used": weather_used, "features": cols, "horizons": {},
           "current_conditions": {
               "as_of_date": str(lastrow["date"]),
               "vol_20": float(lastrow["vol_20"]),
               "zscore_75": float(lastrow["zscore_75"]),
               "mom_10": float(lastrow["mom_10"]),
               "ret_5": float(lastrow["ret_5"]),
           }}
    for h in HORIZONS:
        out["horizons"][h] = {
            "risk_backtest": large_move_backtest(feat, cols, h),
            "risk_latest": latest_large_move_risk(feat, cols, h),
            "analog_latest": analog_scenarios(feat, cols, last_pos, h),
        }
    return out


def explain_current_context(commodity_key: str, as_of_date: str = None) -> dict:
    """
    Live RAG summary of the CURRENT news backdrop for this commodity: retrieve
    recent news up to `as_of_date` (default: the latest day in the price file),
    apply the same relevance gate, and produce a cited summary that refuses if
    the evidence is weak. This is context shown ALONGSIDE the outlook, not the
    cause of the outlook's statistical numbers. Requires network + a Gemini key,
    so it is called on demand (e.g. a dashboard button), not on every load.
    """
    from src.pipeline import retrieve_evidence, GDELT_CACHE_DIR
    from src.rag.relevance_gate import gate_documents
    from src.rag.explainer import generate_context_summary

    cfg = get_commodity(commodity_key)
    if as_of_date is None:
        prices = pd.read_csv(REPO_ROOT / "data" / "raw" / cfg.price_file)
        as_of_date = str(pd.to_datetime(prices["date"]).max().date())

    docs = retrieve_evidence(
        as_of_date, cfg.gdelt_queries, cfg.semantic_reference_query,
        cache_dir=GDELT_CACHE_DIR, commodity_key=commodity_key,
    )
    gate = gate_documents(docs)
    if gate.decision != "EXPLAIN":
        return {"as_of_date": as_of_date, "decision": "INSUFFICIENT_EVIDENCE",
                "reason": gate.reason, "explanation": None, "citations": [],
                "documents_retrieved": len(docs), "accepted": 0}
    result = generate_context_summary(as_of_date, gate.accepted_documents)
    result["as_of_date"] = as_of_date
    result["documents_retrieved"] = len(docs)
    result["accepted"] = len(gate.accepted_documents)
    return result


def _fmt_md(res: dict) -> str:
    L = [f"## {res['display_name']}", "",
         f"Features used: {', '.join(res['features'])}"
         + ("  (includes weather dryness)" if res["weather_used"] else "  (price only; no weather file found)"),
         ""]
    for h in HORIZONS:
        hz = res["horizons"][h]
        name = "week" if h == 5 else "month"
        rb, rl, an = hz["risk_backtest"], hz["risk_latest"], hz["analog_latest"]
        L.append(f"### Next ~{name} ({h} trading days)")
        if "error" not in rb:
            L.append(f"- **Large-move risk model** (walk-forward, {rb['n_test']} test days): on the days it "
                     f"flagged as elevated, a large move actually happened {rb['precision']*100:.0f}% of the "
                     f"time, vs a {rb['base_rate']*100:.0f}% base rate — a **lift of {rb['lift']:.2f}x** "
                     f"(it flagged {rb['flag_rate']*100:.0f}% of days and caught {rb['recall']*100:.0f}% of "
                     f"all large moves). Lift above 1 means real skill.")
        if "error" not in rl:
            L.append(f"- **Current large-move risk**: {'ELEVATED' if rl['elevated'] else 'normal'} "
                     f"(model probability {rl['prob_large_move']:.0%} vs a typical {rl['typical_rate']:.0%}; "
                     f"'large' means a {h}-day move bigger than "
                     f"~{rl['large_move_threshold_pct']:.1f}%). As of {rl['as_of_date']}.")
        if "error" not in an:
            L.append(f"- **Historical analogs** ({an['n_analogs']} most similar past days): the price "
                     f"then moved a median of {an['median_pct']:+.1f}% over the next {h} days, "
                     f"with a typical range of {an['p10_pct']:+.1f}% to {an['p90_pct']:+.1f}% "
                     f"(full span {an['min_pct']:+.1f}% to {an['max_pct']:+.1f}%; "
                     f"{an['share_up']*100:.0f}% rose).")
        L.append("")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", default="coffee", choices=list(COMMODITIES.keys()))
    parser.add_argument("--no-weather", action="store_true")
    args = parser.parse_args()

    res = evaluate(args.commodity, use_weather=not args.no_weather)
    md = ["# Price Outlook — Analog Scenarios + Large-Move Risk", "",
          "EXPERIMENTAL and NOT financial advice. This does not predict a price; it reports how "
          "the price behaved after historically-similar days (a range, not a guess) and whether an "
          "unusually large move is more likely than normal right now. Direction over these horizons "
          "is near-random; the value here is the honest range and risk level, with accuracy shown. "
          "See src/modeling/outlook.py.", ""]
    if "error" in res:
        print(res["error"]); md.append(res["error"])
    else:
        md.append(_fmt_md(res))
        for h in HORIZONS:
            hz = res["horizons"][h]
            rb, rl, an = hz["risk_backtest"], hz["risk_latest"], hz["analog_latest"]
            lift = f"{rb['lift']:.2f}x" if "error" not in rb else rb["error"]
            print(f"{res['display_name']} h={h}: risk lift {lift} "
                  f"(precision {rb.get('precision', float('nan'))*100:.0f}% vs base {rb.get('base_rate', float('nan'))*100:.0f}%); "
                  f"latest {'ELEVATED' if rl.get('elevated') else 'normal'} "
                  f"(p={rl.get('prob_large_move', float('nan')):.2f}); "
                  f"analogs median {an.get('median_pct', float('nan')):+.1f}% "
                  f"[{an.get('p10_pct', float('nan')):+.1f}%, {an.get('p90_pct', float('nan')):+.1f}%]")
    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved report to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
