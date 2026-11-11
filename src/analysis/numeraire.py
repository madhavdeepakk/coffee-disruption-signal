"""
Did the commodity move, or did the measuring stick?

Commodities are quoted in US dollars. When the dollar falls, the dollar
price of everything priced in dollars rises, without anything happening to
any commodity. A detector watching a dollar price cannot tell the two
apart, and an explanation pipeline asked "why did gold rise 4%" will go and
find a gold story, because a gold story is what it was asked for. It will
usually find one: there is always a central bank doing something.

The test is simple and decisive, and it is the cheapest strong check in the
project:

    Price the commodity in euro, yen, yuan and rupees as well as dollars.

    Rose in dollars only, by roughly the size of the dollar's fall
        -> the move is mostly the numeraire. Nothing happened to the
           commodity.
    Rose in every currency
        -> something specific to the commodity is moving it.

Between those is the usual case, and the output is the share, not a verdict
word: how much of the dollar move survives translation into other
currencies.

Two ways of splitting the move are reported, because they answer different
questions and disagreeing with each other is informative:

  Translation. The local-currency return, arithmetic and exact. This asks
  "what did a buyer in Mumbai experience?". It needs no model and no
  estimation, so it is an OBSERVED quantity.

  Regression. The commodity's return is regressed on the dollar's return
  over a trailing window that excludes the day, and the move is split into
  a dollar-explained part and a residual. This asks "how much of this is
  what the dollar usually does to this commodity?". It is a TESTED
  quantity, with the window and its R-squared reported.

They differ because a commodity's dollar beta is not mechanically -1. If
the two disagree sharply, the honest reading is that the commodity's usual
relationship with the dollar did not hold on this day, which is itself the
finding.

What this does NOT do: decide a cause. A move that survives translation is
specific to the commodity; it says nothing about which commodity story is
right. That is the explanation pipeline's job, and this test tells it
whether the question is worth asking at all.

Relationships change. The module reports the trailing-window dollar beta
and R-squared alongside every result, because a textbook link applied
without checking is exactly the error this project is built to avoid: gold
tracked US real yields closely from about 2005 to 2021 and much less so
afterwards, so a decomposition that assumed the old relationship would
confidently mis-assign recent moves.

Usage:
    python -m src.analysis.numeraire --commodity gold --date 2026-01-30
    python -m src.analysis.numeraire --commodity gold --flagged
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.commodities import COMMODITIES

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = REPO_ROOT / "data" / "raw"
FACTORS_PATH = RAW_DIR / "market_factors.csv"

# Foreign units per US dollar, as src/data/market_factors.py normalises them.
CURRENCIES = {"EUR": "eur_per_usd", "JPY": "jpy_per_usd",
              "CNY": "cny_per_usd", "INR": "inr_per_usd"}

WINDOW_DAYS = 75          # matches the detector and the factor model
MIN_WINDOW_OBS = 40

# Above this share of the dollar move surviving translation, the move is
# plainly about the commodity; below the lower bound it is plainly the
# numeraire. Between them the number is reported without a verdict word,
# because a split that lands there is genuinely mixed and a label would
# claim more than the arithmetic supports.
MOSTLY_COMMODITY = 0.80
MOSTLY_NUMERAIRE = 0.35


def load_factors(path: Path = FACTORS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.market_factors")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def load_prices(commodity_key: str) -> pd.Series:
    cfg = COMMODITIES[commodity_key]
    path = RAW_DIR / cfg.price_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.modeling.fetch_price_data "
            f"--commodity {commodity_key}")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    col = "price" if "price" in df.columns else df.columns[0]
    return df[col].astype(float)


def local_currency_returns(usd_returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """The commodity's return as a buyer in each currency experienced it.

    In log terms a local return is the dollar return plus the change in
    foreign units per dollar: if the dollar buys more rupees today, a
    rupee buyer paid relatively more. Exact arithmetic, no estimation."""
    out = pd.DataFrame({"USD": usd_returns})
    for code, col in CURRENCIES.items():
        if col in factors.columns:
            out[code] = usd_returns + factors[col]
    return out


def dollar_beta(usd_returns: pd.Series, dollar_returns: pd.Series, upto,
                window: int = WINDOW_DAYS, min_obs: int = MIN_WINDOW_OBS):
    """Beta and R-squared on the `window` trading days BEFORE `upto`.

    The day being explained is excluded: including it pulls the line
    towards itself and shrinks the residual the test is looking for."""
    joined = pd.concat([usd_returns.rename("y"), dollar_returns.rename("x")],
                       axis=1).dropna()
    prior = joined[joined.index < upto].tail(window)
    if len(prior) < min_obs:
        return None, None, len(prior)
    x, y = prior["x"].to_numpy(), prior["y"].to_numpy()
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    var_y = float(np.var(y, ddof=1))
    r2 = 1.0 - float(np.var(resid, ddof=1)) / var_y if var_y > 0 else None
    return float(coef[1]), r2, len(prior)


def analyse_day(commodity_key: str, day, factors: pd.DataFrame = None) -> dict:
    factors = load_factors() if factors is None else factors
    prices = load_prices(commodity_key)
    usd = np.log(prices).diff()
    day = pd.Timestamp(day)
    if day not in usd.index or pd.isna(usd.get(day)):
        raise SystemExit(f"No price change for {commodity_key} on {day.date()} "
                         "(not a trading day, or outside the series).")

    locals_ = local_currency_returns(usd, factors)
    row = locals_.loc[day]
    usd_move = float(row["USD"])

    per_currency = {c: (None if pd.isna(row.get(c)) else float(row[c]))
                    for c in ["USD", *CURRENCIES] if c in locals_.columns}
    others = [v for c, v in per_currency.items() if c != "USD" and v is not None]

    # Survival: how much of the dollar move is still there once the
    # numeraire is changed. The median across currencies, so one thinly
    # traded or pegged rate cannot drive the answer. The yuan is managed
    # against the dollar, which is why a single currency is never used.
    survival = (float(np.median(others)) / usd_move) if others and usd_move else None

    dollar_col = "dollar" if "dollar" in factors.columns else "dollar_broad"
    dollar_move = (None if dollar_col not in factors.columns or day not in factors.index
                   or pd.isna(factors.loc[day, dollar_col])
                   else float(factors.loc[day, dollar_col]))
    beta, r2, n_obs = (dollar_beta(usd, factors[dollar_col], day)
                       if dollar_col in factors.columns else (None, None, 0))
    explained = (beta * dollar_move) if (beta is not None and dollar_move is not None) else None
    residual = (usd_move - explained) if explained is not None else None

    if survival is None:
        reading = "not enough currency data"
    elif survival >= MOSTLY_COMMODITY:
        reading = "specific to the commodity"
    elif survival <= MOSTLY_NUMERAIRE:
        reading = "mostly the numeraire"
    else:
        reading = "mixed"

    return {
        "commodity": commodity_key,
        "date": str(day.date()),
        "usd_move_pct": 100 * (np.exp(usd_move) - 1),
        "per_currency_pct": {c: (None if v is None else 100 * (np.exp(v) - 1))
                             for c, v in per_currency.items()},
        "dollar_move_pct": (None if dollar_move is None
                            else 100 * (np.exp(dollar_move) - 1)),
        "survival": survival,
        "reading": reading,
        "regression": {
            "dollar_beta": beta, "window_r2": r2, "window_obs": n_obs,
            "explained_pct": (None if explained is None
                              else 100 * (np.exp(explained) - 1)),
            "unexplained_pct": (None if residual is None
                                else 100 * (np.exp(residual) - 1)),
        },
    }


def report(results: list) -> str:
    L = ["# Numeraire Test: did the commodity move, or the dollar?", ""]
    L.append("Commodities are quoted in dollars, so a fall in the dollar raises the dollar "
             "price of everything without anything happening to any commodity. Each move "
             "below is re-priced in euro, yen, yuan and rupees. **Survival** is the share "
             "of the dollar move still there after translation, taken as the median across "
             "the four currencies so one managed rate cannot drive it.")
    L.append("")
    L.append("| Commodity | Date | USD | EUR | JPY | CNY | INR | Dollar index | Survival | Reading |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        pc = r["per_currency_pct"]
        cells = " | ".join(
            "-" if pc.get(c) is None else f"{pc[c]:+.2f}%" for c in ("USD", "EUR", "JPY", "CNY", "INR"))
        dm = "-" if r["dollar_move_pct"] is None else f"{r['dollar_move_pct']:+.2f}%"
        sv = "-" if r["survival"] is None else f"{100 * r['survival']:.0f}%"
        L.append(f"| {r['commodity']} | {r['date']} | {cells} | {dm} | {sv} | {r['reading']} |")
    L.append("")
    L.append("## Split against the commodity's usual dollar sensitivity")
    L.append("")
    L.append("Beta fitted on the 75 trading days before each date, excluding it. The "
             "unexplained column is reported whatever its size and is never assigned to a "
             "cause: it is what the dollar does not account for, not evidence for any "
             "particular story.")
    L.append("")
    L.append("| Commodity | Date | Dollar beta | Window R² | Explained by the dollar | Unexplained |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        g = r["regression"]
        b = "-" if g["dollar_beta"] is None else f"{g['dollar_beta']:+.2f}"
        r2 = "-" if g["window_r2"] is None else f"{g['window_r2']:.2f}"
        ex = "-" if g["explained_pct"] is None else f"{g['explained_pct']:+.2f}%"
        un = "-" if g["unexplained_pct"] is None else f"{g['unexplained_pct']:+.2f}%"
        L.append(f"| {r['commodity']} | {r['date']} | {b} | {r2} | {ex} | {un} |")
    L.append("")
    L.append("A low window R² means the dollar explains little of this commodity's "
             "day-to-day variation in that period, so the split carries little "
             "information and the translation column is the one to read. The two methods "
             "disagreeing is itself a finding: the commodity's usual relationship with "
             "the dollar did not hold that day.")
    return "\n".join(L)


def run(commodity_key: str, dates=None, flagged: bool = False,
        verbose: bool = True) -> list:
    factors = load_factors()
    if flagged:
        cfg = COMMODITIES[commodity_key]
        det = pd.read_csv(RESULTS_DIR / cfg.anomalies_file, parse_dates=["date"])
        if "contract_switch" in det.columns:
            det = det[~det["contract_switch"].fillna(False).astype(bool)]
        dates = det.loc[det["anomaly_flag"] == True, "date"].tolist()  # noqa: E712
    results = []
    for d in dates or []:
        try:
            results.append(analyse_day(commodity_key, d, factors))
        except SystemExit:
            continue
    if verbose and results:
        print(report(results))
    return results


def main():
    p = argparse.ArgumentParser(description="Separate a commodity move from a dollar move")
    p.add_argument("--commodity", required=True, choices=sorted(COMMODITIES))
    p.add_argument("--date", action="append", default=[])
    p.add_argument("--flagged", action="store_true",
                   help="every day the base detector flagged for this commodity")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    results = run(args.commodity, dates=args.date, flagged=args.flagged)
    if args.save and results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{args.commodity}_numeraire.md").write_text(
            report(results), encoding="utf-8")
        with open(RESULTS_DIR / f"{args.commodity}_numeraire.json", "w",
                  encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results/{args.commodity}_numeraire.md and .json")


if __name__ == "__main__":
    main()
