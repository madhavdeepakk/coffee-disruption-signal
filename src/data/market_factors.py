"""
The shared drivers: the dollar, the market, the complex, and four exchange
rates.

Two modules need these and neither can work without them:

  src/modeling/factor_model.py  removes the part of a commodity's move that
                                the dollar, equities and the broad complex
                                explain, so a flagged day can be called
                                commodity-specific or macro-driven.
  src/analysis/numeraire.py     prices a commodity in euro, yen, yuan and
                                rupees as well as dollars, to separate "the
                                commodity moved" from "the measuring stick
                                moved".

Sources, all keyless. The daily series come from FRED's public CSV
endpoint, the same one src/data/cpi.py uses:

    DTWEXBGS   broad trade-weighted US dollar index (the Fed's own)
    DFII10     10-year US real yield (TIPS)
    T10YIE     10-year breakeven inflation
    DGS10      10-year nominal Treasury yield
    DEXUSEU    US dollars per euro
    DEXJPUS    yen per US dollar
    DEXCHUS    yuan per US dollar
    DEXINUS    rupees per US dollar

Two series FRED does not carry daily and far enough back come from Yahoo
Finance through yfinance, which the project already uses for futures:

    ^GSPC      S&P 500, the risk-appetite factor
    DBC        Invesco DB Commodity Index Fund, the broad-complex factor

Why the broad dollar index and not DXY. DXY is roughly 58% euro by weight,
so a move in it is substantially a move in one currency pair; using it
would make "the dollar" mean "the euro" in every result. DTWEXBGS is
trade-weighted across the currencies the US actually trades with, which is
what the numeraire argument needs. Both are recorded if available so the
choice can be checked, but the factor model uses the broad index.

Why DBC rather than a spot index. It is an investable fund, so it exists at
daily frequency over the whole period; a spot index like the S&P GSCI is
not freely available at that frequency without a key. DBC holds energy
heavily, which means that for crude oil the "broad complex" factor is
partly the commodity itself - the factor model's report prints the window
R-squared so that overlap is visible rather than hidden, and crude's
residual should be read with it in mind.

Exchange rate direction. FRED quotes these inconsistently: DEXUSEU is
dollars per euro, the other three are foreign units per dollar. This module
normalises every one to FOREIGN UNITS PER US DOLLAR, so multiplying a
dollar price by the rate always gives the local price. The conversion is
done once, here, rather than in each caller, because getting it backwards
inverts the sign of every pass-through result downstream.

Usage:
    python -m src.data.market_factors
    python -m src.data.market_factors --start 2005-01-01
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "market_factors.csv"
LEVELS_PATH = REPO_ROOT / "data" / "raw" / "market_factor_levels.csv"

DEFAULT_START = "2005-01-01"

# series id -> (column name, what it is, invert?)
# invert=True means FRED quotes it as US dollars per foreign unit and it has
# to be flipped to reach foreign units per US dollar.
FRED_SERIES = {
    "DTWEXBGS": ("dollar_broad", "Broad trade-weighted US dollar index", False),
    "DFII10": ("real_yield_10y", "10-year US real yield (TIPS), percent", False),
    "T10YIE": ("breakeven_10y", "10-year breakeven inflation, percent", False),
    "DGS10": ("nominal_yield_10y", "10-year Treasury yield, percent", False),
    "DEXUSEU": ("eur_per_usd", "Euro per US dollar (FRED quotes USD per EUR)", True),
    "DEXJPUS": ("jpy_per_usd", "Yen per US dollar", False),
    "DEXCHUS": ("cny_per_usd", "Yuan per US dollar", False),
    "DEXINUS": ("inr_per_usd", "Rupee per US dollar", False),
}

YAHOO_SERIES = {
    "^GSPC": ("equity", "S&P 500 close"),
    "DBC": ("broad_commodity", "Invesco DB Commodity Index Fund close"),
}

# Columns the factor model needs as RETURNS. Everything else is carried at
# its level, because a yield is already a rate and differencing it twice is
# meaningless.
RETURN_COLUMNS = ("dollar_broad", "equity", "broad_commodity",
                  "eur_per_usd", "jpy_per_usd", "cny_per_usd", "inr_per_usd")
LEVEL_COLUMNS = ("real_yield_10y", "breakeven_10y", "nominal_yield_10y")


def fetch_fred(series_id: str, start: str) -> list:
    """Reuses cpi.py's fetcher so there is one FRED code path, not two."""
    from src.data.cpi import fetch_series
    return fetch_series(series_id, start)


def fetch_yahoo(ticker: str, start: str) -> list:
    import yfinance as yf
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if hasattr(close, "columns"):          # MultiIndex for a single ticker
        close = close.iloc[:, 0]
    return [(d.date().isoformat(), float(v)) for d, v in close.items() if v == v]


def build_levels(start: str = DEFAULT_START, verbose: bool = True) -> dict:
    """{column: {date: level}} with every exchange rate already normalised
    to foreign units per US dollar."""
    out, failed = {}, []
    for series_id, (col, what, invert) in FRED_SERIES.items():
        try:
            rows = fetch_fred(series_id, start)
        except Exception as exc:  # noqa: BLE001 - report each, decide at the end
            failed.append(series_id)
            if verbose:
                print(f"  {series_id:<10} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        values = {d: (1.0 / v if invert and v else v) for d, v in rows if v}
        out[col] = values
        if verbose:
            span = f"{rows[0][0]} to {rows[-1][0]}" if rows else "no rows"
            print(f"  {series_id:<10} {len(values):>5} rows  {span}  {what}")

    for ticker, (col, what) in YAHOO_SERIES.items():
        try:
            rows = fetch_yahoo(ticker, start)
        except Exception as exc:  # noqa: BLE001
            failed.append(ticker)
            if verbose:
                print(f"  {ticker:<10} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        out[col] = dict(rows)
        if verbose:
            span = f"{rows[0][0]} to {rows[-1][0]}" if rows else "no rows"
            print(f"  {ticker:<10} {len(rows):>5} rows  {span}  {what}")

    if failed:
        # A factor table with a column missing would silently drop that
        # factor from every regression that reads it, and the result would
        # look fine. Refuse instead.
        raise SystemExit(
            f"\n{len(failed)} series could not be downloaded ({', '.join(failed)}). "
            f"The existing {OUTPUT_PATH.name} was left alone: a table with a factor "
            "missing gives regressions that run and are wrong.")
    return out


def to_returns(levels: dict):
    """Daily log returns for the price-like columns, levels for the rates.

    Returns are taken over consecutive OBSERVED days, not calendar days, and
    nothing is forward-filled: holidays differ between the exchanges and the
    FRED series, and filling a gap invents a zero-return day that the
    regression would treat as real."""
    import numpy as np
    import pandas as pd

    frame = {}
    for col, values in levels.items():
        s = pd.Series(values)
        s.index = pd.to_datetime(s.index)
        frame[col] = s.sort_index()
    df = pd.DataFrame(frame).sort_index()

    out = pd.DataFrame(index=df.index)
    for col in RETURN_COLUMNS:
        if col in df.columns:
            out[col] = np.log(df[col].astype(float)).diff()
    for col in LEVEL_COLUMNS:
        if col in df.columns:
            out[col] = df[col].astype(float)
            out[col + "_change"] = df[col].astype(float).diff()

    # The factor model's three columns, under the names it expects.
    if "dollar_broad" in out.columns:
        out["dollar"] = out["dollar_broad"]
    return df, out


def write(levels: dict, path: Path = OUTPUT_PATH, levels_path: Path = LEVELS_PATH) -> int:
    lv, rt = to_returns(levels)
    path.parent.mkdir(parents=True, exist_ok=True)
    rt.index.name = "date"
    rt.to_csv(path)
    lv.index.name = "date"
    lv.to_csv(levels_path)
    return len(rt)


def main():
    p = argparse.ArgumentParser(description="Download the shared market drivers")
    p.add_argument("--start", default=DEFAULT_START)
    args = p.parse_args()
    levels = build_levels(args.start)
    n = write(levels)
    print(f"\nSaved {OUTPUT_PATH}  ({n} rows, daily returns)")
    print(f"Saved {LEVELS_PATH}  (the underlying levels, for checking)")
    print("\nExchange rates are normalised to FOREIGN UNITS PER US DOLLAR: multiply a "
          "dollar price by the rate to get the local price.")


if __name__ == "__main__":
    main()
