"""
Inflation tracking and commodity pass-through for the United States, India
and a global reference.

Three questions, each answered with a number and the baseline it has to beat:

  1. Where is inflation now?  The latest year-on-year rate and the last three
     months annualised, per country and per US component.

  2. How much of a commodity price rise reaches consumer prices, and how
     late?  ("Pass-through".)  A distributed-lag regression of monthly
     inflation on this month's and the previous twelve months' commodity
     price changes. The sum of the thirteen coefficients is the share of a
     commodity price rise that has reached the consumer price index a year
     later. Standard errors allow for autocorrelation (Newey-West); with one
     row per commodity-and-index pair a few "significant" results are expected
     by chance, so p-values are also given after a Holm correction for the
     number of pairs tested.

  3. Does knowing the commodity price improve an inflation estimate that can
     be made BEFORE the official figure is published?  A month's average
     commodity price is known the day the month ends; that month's CPI comes
     out about two weeks later. The nowcast test predicts each month's
     inflation from past inflation alone (the baseline) and from past
     inflation plus commodity price changes up to that month (the model),
     refitting on only the months before it each time. The model is worth
     something only where its error is lower than the baseline's on months it
     was not fitted on. A Diebold-Mariano statistic says whether the gap is
     more than noise.

A relationship that is not there is reported as not there. The same project
found that the direction of coffee prices cannot be forecast (results/
forecast_evaluation.md); a pass-through that cannot be told from zero is the
same kind of result and is written down the same way.

What is NOT claimed: none of this is a forecast of future inflation, and a
regression on monthly aggregates does not show that a commodity move CAUSED
a change in consumer prices - only how the two have moved together, and
whether that helps estimate the figure about to be published.

Usage:
    python -m src.data.cpi              # download the series first
    python -m src.modeling.inflation    # write results/inflation_report.md and .json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORT_MD = RESULTS_DIR / "inflation_report.md"
REPORT_JSON = RESULTS_DIR / "inflation_report.json"

# Fixed before any result was looked at, and not tuned afterwards.
PASS_THROUGH_LAGS = list(range(0, 13))    # this month and the twelve before it
HAC_LAGS = 12                             # Newey-West truncation, monthly data
NOWCAST_AR_LAGS = [1, 2, 3]               # past inflation the baseline may use
NOWCAST_DRIVER_LAGS = [0, 1, 2, 3]        # commodity changes the model adds
YOY_DRIVER_LAGS = [0, 3, 6, 9, 12]        # for a year-on-year target (overlapping months)
YOY_AR_LAGS = [1, 2]
YOY_NOWCAST_DRIVER_LAGS = [0, 3]
MIN_TRAIN_MONTHS = 96                     # eight years before the first out-of-sample month
MIN_ROWS = 60                             # below this a pair is reported as too short
SIGNIFICANCE = 0.05


# ---------------------------------------------------------------------------
# Small numerical tools (numpy only, so the dashboard needs nothing extra)
# ---------------------------------------------------------------------------

def monthly(series: pd.Series) -> pd.Series:
    """The series on an unbroken month-start index; a missing month stays
    missing instead of being bridged by the months either side."""
    series = series.dropna().sort_index()
    if series.empty:
        return series
    return series.reindex(pd.date_range(series.index[0], series.index[-1], freq="MS"))


def pct_change_log(series: pd.Series) -> pd.Series:
    """Month-on-month change in percent (100 x the change in the logarithm).
    A month next to a missing month has no change and is left missing."""
    return 100.0 * np.log(monthly(series)).diff()


def yoy(series: pd.Series) -> pd.Series:
    """Year-on-year change in percent."""
    series = monthly(series)
    return 100.0 * (series / series.shift(12) - 1.0)


def ols(y: np.ndarray, X: np.ndarray):
    """Coefficients and residuals of y on X (X already holds the constant)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def newey_west(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Covariance of the OLS coefficients allowing for heteroskedasticity and
    autocorrelation up to `lags` months (Bartlett weights)."""
    n, k = X.shape
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = Xe[lag:].T @ Xe[:-lag]
        S += weight * (gamma + gamma.T)
    bread = np.linalg.pinv(X.T @ X)
    return bread @ S @ bread * (n / max(n - k, 1))


def two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0)) if math.isfinite(z) else float("nan")


def one_sided_p(z: float) -> float:
    """P(Z > z): small when z is large and positive."""
    return 0.5 * math.erfc(z / math.sqrt(2.0)) if math.isfinite(z) else float("nan")


def holm(p_values: list) -> list:
    """Holm-adjusted p-values, in the order given. None stays None."""
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None and p == p)
    adjusted, running, m = [None] * len(p_values), 0.0, len(indexed)
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def month_terms(index: pd.DatetimeIndex) -> np.ndarray:
    """Eleven month-of-year columns (January is the reference), for a series
    that is not seasonally adjusted."""
    return np.column_stack([(index.month == m).astype(float) for m in range(2, 13)])


def _frame(target: pd.Series, driver: pd.Series, ar_lags: list, driver_lags: list) -> pd.DataFrame:
    columns = {"y": target}
    for lag in ar_lags:
        columns[f"y_l{lag}"] = target.shift(lag)
    if driver is not None:
        for lag in driver_lags:
            columns[f"x_l{lag}"] = driver.shift(lag)
    return pd.DataFrame(columns).dropna()


# ---------------------------------------------------------------------------
# The two measurements
# ---------------------------------------------------------------------------

def pass_through(target: pd.Series, driver: pd.Series, driver_lags: list = None,
                 seasonal: bool = False, hac_lags: int = HAC_LAGS) -> dict:
    """Distributed-lag regression of `target` on `driver` at `driver_lags`.

    Returns the sum of the driver coefficients (per 1% rise in the driver),
    its Newey-West standard error and two-sided p-value, the R-squared, and
    the number of months used. `seasonal` adds month-of-year terms."""
    driver_lags = PASS_THROUGH_LAGS if driver_lags is None else driver_lags
    frame = _frame(target, driver, [], driver_lags)
    n = len(frame)
    if n < MIN_ROWS:
        return {"n": n, "too_short": True}
    x_cols = [c for c in frame.columns if c.startswith("x_")]
    parts = [np.ones((n, 1)), frame[x_cols].to_numpy()]
    if seasonal:
        parts.append(month_terms(frame.index))
    X, y = np.column_stack(parts), frame["y"].to_numpy()
    beta, resid = ols(y, X)
    cov = newey_west(X, resid, hac_lags)
    pick = np.zeros(X.shape[1])
    pick[1:1 + len(x_cols)] = 1.0
    total = float(pick @ beta)
    se = float(math.sqrt(max(pick @ cov @ pick, 0.0)))
    z = total / se if se > 0 else float("nan")
    centred = y - y.mean()
    r2 = 1.0 - float(resid @ resid) / float(centred @ centred) if float(centred @ centred) > 0 else 0.0
    # how much of the driver's contribution arrives within the first quarter
    early = float(sum(beta[1 + i] for i, lag in enumerate(driver_lags) if lag <= 3))
    return {"n": n, "too_short": False, "sum_beta": total, "se": se, "z": z,
            "p": two_sided_p(z), "r2": r2, "early_beta": early,
            "start": frame.index[0].strftime("%Y-%m"), "end": frame.index[-1].strftime("%Y-%m")}


def nowcast_test(target: pd.Series, driver: pd.Series, ar_lags: list = None,
                 driver_lags: list = None, seasonal: bool = False,
                 min_train: int = MIN_TRAIN_MONTHS) -> dict:
    """Walk-forward comparison on months the fit never saw.

    For each month after the first `min_train`, both models are fitted on the
    months before it only and asked for that month:
        baseline   past values of the target (and month terms if seasonal)
        model      the same, plus the driver at `driver_lags` (lag 0 is the
                   month being estimated, whose commodity price is already
                   known when the month ends)
    Returns both root-mean-square errors, the improvement in percent
    (positive: the model is better), and a Diebold-Mariano statistic with its
    one-sided p-value for "the model's squared error is lower". The two models
    are nested, for which this test is known to be conservative."""
    ar_lags = NOWCAST_AR_LAGS if ar_lags is None else ar_lags
    driver_lags = NOWCAST_DRIVER_LAGS if driver_lags is None else driver_lags
    frame = _frame(target, driver, ar_lags, driver_lags)
    n = len(frame)
    if n < min_train + 24:
        return {"n_test": 0, "too_short": True}
    y = frame["y"].to_numpy()
    ar = frame[[c for c in frame.columns if c.startswith("y_l")]].to_numpy()
    xs = frame[[c for c in frame.columns if c.startswith("x_")]].to_numpy()
    season = month_terms(frame.index) if seasonal else np.empty((n, 0))
    base_X = np.column_stack([np.ones(n), ar, season])
    model_X = np.column_stack([base_X, xs])
    base_err, model_err = [], []
    for t in range(min_train, n):
        b0, _ = ols(y[:t], base_X[:t])
        b1, _ = ols(y[:t], model_X[:t])
        base_err.append(y[t] - base_X[t] @ b0)
        model_err.append(y[t] - model_X[t] @ b1)
    base_err, model_err = np.array(base_err), np.array(model_err)
    rmse_base = float(math.sqrt(np.mean(base_err ** 2)))
    rmse_model = float(math.sqrt(np.mean(model_err ** 2)))
    d = base_err ** 2 - model_err ** 2
    m = len(d)
    centred = d - d.mean()
    variance = float(centred @ centred) / m
    for lag in range(1, min(HAC_LAGS, m - 1) + 1):
        variance += 2.0 * (1.0 - lag / (HAC_LAGS + 1.0)) * float(centred[lag:] @ centred[:-lag]) / m
    dm = float(d.mean() / math.sqrt(variance / m)) if variance > 0 else float("nan")
    return {"n_test": m, "too_short": False, "rmse_baseline": rmse_base, "rmse_model": rmse_model,
            "improvement_pct": 100.0 * (1.0 - rmse_model / rmse_base) if rmse_base > 0 else 0.0,
            "dm": dm, "p": one_sided_p(dm),
            "first_test_month": frame.index[min_train].strftime("%Y-%m"),
            "last_test_month": frame.index[-1].strftime("%Y-%m")}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def monthly_average_of_daily(price_file: str):
    """Monthly average of a daily futures file in data/raw (gold, silver),
    or None when the file is not there. The current, unfinished month is
    left out: its average is not yet known."""
    path = RAW_DIR / price_file
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        return None
    by_month = df.set_index("date")["price"].resample("MS").mean().dropna()
    last_day = df["date"].max()
    if last_day != last_day + pd.offsets.MonthEnd(0) and len(by_month):
        by_month = by_month.iloc[:-1]
    return by_month


def load_series(table_path: Path = None) -> dict:
    from src.data import cpi
    series = cpi.load_table(table_path or cpi.OUTPUT_PATH)
    for name, price_file in (("gold", "gold_prices.csv"), ("silver", "silver_prices.csv")):
        averaged = monthly_average_of_daily(price_file)
        if averaged is not None:
            series[name] = averaged
    return series


def in_rupees(dollar_price: pd.Series, inr_per_usd: pd.Series) -> pd.Series:
    joined = pd.concat([monthly(dollar_price), monthly(inr_per_usd)], axis=1, sort=True).dropna()
    return joined.iloc[:, 0] * joined.iloc[:, 1]


# (label of the consumer price series, key, seasonal?, the drivers to test)
# Every pair listed is reported, whatever it shows.
PAIRS = [
    ("United States", "us_cpi", False,
     ["oil_wti", "food_index", "wheat", "coffee", "all_commodities", "gold", "silver"]),
    ("United States", "us_energy", False, ["oil_wti"]),
    ("United States", "us_gasoline", False, ["oil_wti"]),
    ("United States", "us_food", False, ["food_index", "wheat", "oil_wti"]),
    ("United States", "us_coffee", False, ["coffee"]),
    ("United States", "us_core", False, ["oil_wti", "all_commodities"]),
    ("India", "india_cpi", True,
     ["oil_brent", "food_index", "wheat", "coffee", "all_commodities", "gold", "silver"]),
]
GLOBAL_DRIVERS = ["all_commodities", "energy_index", "food_index"]

LABELS = {
    "us_cpi": "all items", "us_core": "core (less food and energy)", "us_food": "food",
    "us_energy": "energy", "us_gasoline": "gasoline", "us_coffee": "coffee",
    "india_cpi": "all items",
    "oil_wti": "crude oil (WTI)", "oil_brent": "crude oil (Brent)", "wheat": "wheat",
    "coffee": "coffee (arabica)", "food_index": "food commodities (IMF index)",
    "energy_index": "energy commodities (IMF index)",
    "all_commodities": "all commodities (IMF index)", "gold": "gold", "silver": "silver",
}


def tracking(series: dict) -> list:
    """Where each consumer price series stands: latest year-on-year rate, the
    last three months annualised, and the month the figure is for."""
    rows = []
    for country, key in (("United States", "us_cpi"), ("United States", "us_core"),
                         ("United States", "us_food"), ("United States", "us_energy"),
                         ("United States", "us_gasoline"), ("United States", "us_coffee"),
                         ("India", "india_cpi")):
        if key not in series:
            continue
        level = monthly(series[key])
        rate = yoy(level).dropna()
        if rate.empty:
            continue
        last = level.dropna()
        three = (100.0 * ((last.iloc[-1] / last.iloc[-4]) ** 4 - 1.0)
                 if len(last) >= 4 and (last.index[-1] - last.index[-4]).days < 100 else None)
        rows.append({"country": country, "series": key, "label": LABELS[key],
                     "month": rate.index[-1].strftime("%Y-%m"), "yoy": float(rate.iloc[-1]),
                     "three_month_annualised": three,
                     "yoy_a_year_earlier": float(rate.iloc[-13]) if len(rate) > 13 else None})
    if "g20_inflation" in series:
        g20 = series["g20_inflation"].dropna()
        rows.append({"country": "G20", "series": "g20_inflation", "label": "all items",
                     "month": g20.index[-1].strftime("%Y-%m"), "yoy": float(g20.iloc[-1]),
                     "three_month_annualised": None,
                     "yoy_a_year_earlier": float(g20.iloc[-13]) if len(g20) > 13 else None})
    for key, country in (("world_inflation_annual", "World (annual)"),
                         ("us_inflation_annual", "United States (annual)"),
                         ("india_inflation_annual", "India (annual)")):
        if key in series and len(series[key].dropna()):
            s = series[key].dropna()
            rows.append({"country": country, "series": key, "label": "all items, calendar year",
                         "month": s.index[-1].strftime("%Y"), "yoy": float(s.iloc[-1]),
                         "three_month_annualised": None,
                         "yoy_a_year_earlier": float(s.iloc[-2]) if len(s) > 1 else None})
    return rows


def analyse(series: dict) -> dict:
    rows = []
    for country, target_key, seasonal, drivers in PAIRS:
        if target_key not in series:
            continue
        target = pct_change_log(series[target_key])
        for driver_key in drivers:
            if driver_key not in series:
                continue
            price = series[driver_key]
            currency = "USD"
            if country == "India" and "inr_per_usd" in series:
                price, currency = in_rupees(price, series["inr_per_usd"]), "INR"
            driver = pct_change_log(price)
            rows.append({
                "country": country, "target": target_key, "target_label": LABELS[target_key],
                "driver": driver_key, "driver_label": LABELS[driver_key], "currency": currency,
                "kind": "monthly", "seasonal_terms": seasonal,
                "pass_through": pass_through(target, driver, seasonal=seasonal),
                "nowcast": nowcast_test(target, driver, seasonal=seasonal)})
    if "g20_inflation" in series:
        target = monthly(series["g20_inflation"])
        for driver_key in GLOBAL_DRIVERS:
            if driver_key not in series:
                continue
            driver = yoy(series[driver_key])
            rows.append({
                "country": "G20", "target": "g20_inflation", "target_label": "all items",
                "driver": driver_key, "driver_label": LABELS[driver_key], "currency": "USD",
                "kind": "yoy", "seasonal_terms": False,
                "pass_through": pass_through(target, driver, driver_lags=YOY_DRIVER_LAGS),
                "nowcast": nowcast_test(target, driver, ar_lags=YOY_AR_LAGS,
                                        driver_lags=YOY_NOWCAST_DRIVER_LAGS)})
    adjusted = holm([None if r["pass_through"].get("too_short") else r["pass_through"]["p"]
                     for r in rows])
    adjusted_now = holm([None if r["nowcast"].get("too_short") else r["nowcast"]["p"]
                         for r in rows])
    for row, p_adj, p_now in zip(rows, adjusted, adjusted_now):
        row["pass_through"]["p_holm"] = p_adj
        row["nowcast"]["p_holm"] = p_now
        row["verdict"] = verdict(row)
    return {"tracking": tracking(series), "pairs": rows, "n_pairs": len(rows),
            "settings": {"pass_through_lags": PASS_THROUGH_LAGS, "hac_lags": HAC_LAGS,
                         "nowcast_ar_lags": NOWCAST_AR_LAGS,
                         "nowcast_driver_lags": NOWCAST_DRIVER_LAGS,
                         "min_train_months": MIN_TRAIN_MONTHS, "significance": SIGNIFICANCE}}


def verdict(row: dict) -> str:
    """One of four plain outcomes, decided by the corrected p-values only."""
    pt, nc = row["pass_through"], row["nowcast"]
    if pt.get("too_short"):
        return "too few months to say"
    linked = pt.get("p_holm") is not None and pt["p_holm"] < SIGNIFICANCE and pt["sum_beta"] > 0
    helps = (not nc.get("too_short") and nc.get("p_holm") is not None
             and nc["p_holm"] < SIGNIFICANCE and nc["improvement_pct"] > 0)
    if linked and helps:
        return "pass-through measured, and it improves the estimate"
    if linked:
        return "pass-through measured, but it does not improve the estimate"
    if helps:
        return "improves the estimate, but no clear pass-through over a year"
    return "no relationship that can be told from chance"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _p(value) -> str:
    if value is None or value != value:
        return "-"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def generate_report(result: dict) -> str:
    L = ["# Inflation Tracking and Commodity Pass-Through", "",
         "United States, India and a global reference (the G20), against the commodities "
         "the anomaly detector watches. Produced by `python -m src.modeling.inflation` from "
         "`data/raw/inflation_series.csv` (see `src/data/cpi.py` for each source and its "
         "limits).", "",
         "None of this is a forecast, and a regression on monthly figures does not show "
         "that a commodity move caused a change in consumer prices. It measures how the "
         "two have moved together and whether that helps estimate a figure before it is "
         "published.", ""]

    L += ["## 1. Where inflation stands", "",
          "| | Series | Latest month | Year on year | Last 3 months, annualised | "
          "Year on year, 12 months earlier |", "|---|---|---|---|---|---|"]
    for t in result["tracking"]:
        three = f"{t['three_month_annualised']:+.1f}%" if t["three_month_annualised"] is not None else "-"
        before = f"{t['yoy_a_year_earlier']:+.1f}%" if t["yoy_a_year_earlier"] is not None else "-"
        L.append(f"| {t['country']} | {t['label']} | {t['month']} | {t['yoy']:+.1f}% | "
                 f"{three} | {before} |")
    L += ["", "India's monthly index and the G20 series come from an OECD database that "
          "stopped updating, so they end where the table says; the annual World Bank rows "
          "are the most recent figures available for them without a registered key.", ""]

    pairs = result["pairs"]
    usable = [r for r in pairs if not r["pass_through"].get("too_short")]
    L += ["## 2. Pass-through: how much of a commodity price rise reaches consumer prices",
          "",
          f"{len(pairs)} commodity-and-index pairs were chosen before any result was seen and "
          f"all are reported. \"Pass-through\" is the sum of the coefficients on this month's "
          f"and the previous twelve months' commodity price change: after a year, a 10% rise "
          f"in the commodity goes with the stated change in the price index. The p-value "
          f"tests that sum against zero with Newey-West errors; the Holm column corrects it "
          f"for the {len(usable)} pairs tested, and only that column decides the verdict.",
          "",
          "| Consumer prices | Commodity | Months | After a 10% rise, index moves | "
          "of which within 3 months | p | p (Holm) | R-squared |",
          "|---|---|---|---|---|---|---|---|"]
    for r in pairs:
        pt = r["pass_through"]
        name = f"{r['country']}: {r['target_label']}"
        drv = r["driver_label"] + (" in rupees" if r["currency"] == "INR" else "")
        if r["kind"] == "yoy":
            drv += " (year on year)"
        if pt.get("too_short"):
            L.append(f"| {name} | {drv} | {pt['n']} | too few months | - | - | - | - |")
            continue
        L.append(f"| {name} | {drv} | {pt['n']} ({pt['start']} to {pt['end']}) | "
                 f"{10 * pt['sum_beta']:+.2f}% | {10 * pt['early_beta']:+.2f}% | "
                 f"{_p(pt['p'])} | {_p(pt['p_holm'])} | {pt['r2']:.2f} |")
    L += ["", "For the G20 row the target is already a year-on-year rate and the commodity "
          "change is year on year too, at lags of 0, 3, 6, 9 and 12 months; the months "
          "overlap, so read that row as an association, not as a share passed through.", ""]

    L += ["## 3. Does the commodity price improve the estimate made before publication?",
          "",
          "Each month from the first test month on, inflation is estimated twice from the "
          "months before it only: from past inflation alone (the baseline), and from past "
          "inflation plus the commodity's price changes up to that month, which are known "
          "when the month ends and before the CPI is published. A model is better only if "
          "its error on those months is lower. Positive improvement means lower error.",
          "",
          "| Consumer prices | Commodity | Test months | Baseline error (RMSE, pp) | "
          "With the commodity | Improvement | p | p (Holm) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in pairs:
        nc = r["nowcast"]
        name = f"{r['country']}: {r['target_label']}"
        drv = r["driver_label"] + (" in rupees" if r["currency"] == "INR" else "")
        if nc.get("too_short"):
            L.append(f"| {name} | {drv} | too few months | - | - | - | - | - |")
            continue
        L.append(f"| {name} | {drv} | {nc['n_test']} ({nc['first_test_month']} to "
                 f"{nc['last_test_month']}) | {nc['rmse_baseline']:.3f} | {nc['rmse_model']:.3f} | "
                 f"{nc['improvement_pct']:+.1f}% | {_p(nc['p'])} | {_p(nc['p_holm'])} |")
    L += ["", "The baseline here is not a strawman: it is the same autoregression with the "
          "commodity terms removed, refitted every month. The test is one-sided (is the "
          "model's squared error lower?) and is known to be conservative when one model "
          "contains the other, as here.", ""]

    L += ["## 4. What the numbers say", ""]
    groups = {}
    for r in pairs:
        groups.setdefault(r["verdict"], []).append(r)
    order = ["pass-through measured, and it improves the estimate",
             "pass-through measured, but it does not improve the estimate",
             "improves the estimate, but no clear pass-through over a year",
             "no relationship that can be told from chance",
             "too few months to say"]
    for key in order:
        if key not in groups:
            continue
        names = "; ".join(
            f"{r['driver_label']}{' (rupees)' if r['currency'] == 'INR' else ''} -> "
            f"{r['country']} {r['target_label']}" for r in groups[key])
        L.append(f"- **{key.capitalize()}** ({len(groups[key])} of {len(pairs)}): {names}.")
    L += ["",
          "Verdicts use the Holm-corrected p-values at 5%, which is strict with this many "
          "pairs: several nowcast rows show a large improvement with an uncorrected p-value "
          "under 0.05 and are still not counted. A pair listed under \"no relationship\" is "
          "a result, not a gap: for that commodity and that price index the monthly record "
          "shows nothing that chance alone would not.", "",
          "## 5. Limits", "",
          "- India: the all-items index only, not seasonally adjusted (month-of-year terms "
          "are included), ending March 2025. No food or fuel sub-index, which is where "
          "pass-through would be expected to show first.",
          "- Fuel prices in India were administered for much of the sample, which weakens "
          "the link between the world oil price and what a household paid.",
          "- Monthly averages hide timing within the month; the daily anomaly detector and "
          "this monthly analysis are separate instruments.",
          "- Gold and silver start in 2012 and 2010 (monthly averages of the daily futures "
          "series), so their rows rest on fewer months.",
          "- The lag lengths, the training window and the pairs were fixed in the code "
          "before the results were produced and were not adjusted afterwards.", ""]
    return "\n".join(L)


def run(table_path: Path = None, verbose: bool = True) -> dict:
    series = load_series(table_path)
    result = analyse(series)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    report = generate_report(result)
    REPORT_MD.write_text(report, encoding="utf-8")
    if verbose:
        print(report)
        print(f"\nSaved {REPORT_MD}\nSaved {REPORT_JSON}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Inflation tracking and commodity pass-through")
    parser.add_argument("--table", default=None,
                        help="Series table to read (default data/raw/inflation_series.csv)")
    args = parser.parse_args()
    table = Path(args.table) if args.table else None
    from src.data import cpi
    if not (table or cpi.OUTPUT_PATH).exists():
        raise SystemExit(f"{table or cpi.OUTPUT_PATH} not found. Run: python -m src.data.cpi")
    run(table)


if __name__ == "__main__":
    main()
