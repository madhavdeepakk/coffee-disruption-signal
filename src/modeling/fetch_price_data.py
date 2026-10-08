"""
Fetch daily futures price history from Yahoo Finance for any configured
commodity (src/config/commodities.py). Coffee (KC=F), crude oil, and wheat
use the same fetch/validate code path, per the proposal's "Build One
Deeply, Architect for Any" idea (S4). Coffee's file paths
(data/raw/coffee_prices.csv) are unchanged so its validated results are not
disturbed.

Why Yahoo Finance, not FRED/World Bank: the anomaly detector needs a
60-90 day rolling volatility window per the proposal's design (z-score
against trailing volatility, not a fixed % threshold). FRED's series and
World Bank's Pink Sheet are both monthly - a 60-90 day rolling window over
monthly data is only 2-3 points, statistically meaningless. Yahoo Finance
gives daily close prices for all three commodities' futures contracts,
which is what the design requires.

The fetch start date lives in each commodity's config
(history_start_date in src/config/commodities.py) rather than as one global
constant. A single global start date correct for coffee (earliest event
2021) would leave wheat's 2010 and 2012 events, and both new commodities'
2017 control period, with zero trading days in-window; those empty windows
would then report as an event MISS or a "clean" control pass, neither a
real test.

Usage:
    python -m src.modeling.fetch_price_data                  # coffee (default)
    python -m src.modeling.fetch_price_data --commodity crude_oil
    python -m src.modeling.fetch_price_data --commodity wheat
    python -m src.modeling.fetch_price_data --all             # fetch all configured commodities
"""

import argparse
from pathlib import Path

import yfinance as yf

from src.config.commodities import COMMODITIES, get_commodity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def fetch_prices(ticker: str, start_date: str) -> "pd.DataFrame":
    print(f"Fetching {ticker} daily price history from {start_date}...")
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    if df.empty:
        raise SystemExit(
            f"yfinance returned no data for {ticker} - check your internet connection, "
            f"whether Yahoo Finance is reachable from this network, or whether the ticker "
            f"symbol is still valid."
        )
    # yfinance can return a MultiIndex column structure for a single ticker
    # depending on version - flatten defensively rather than assume a shape.
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["date", "price"]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def fetch_and_save(commodity_key: str) -> None:
    cfg = get_commodity(commodity_key)
    df = fetch_prices(cfg.ticker, cfg.history_start_date)
    output_path = REPO_ROOT / "data" / "raw" / cfg.price_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[{cfg.display_name}] Saved {len(df)} rows to {output_path}")
    print(f"[{cfg.display_name}] Date range: {df['date'].min()} to {df['date'].max()}")
    print(df.tail())
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", default="coffee", choices=list(COMMODITIES.keys()))
    parser.add_argument("--all", action="store_true", help="Fetch every configured commodity")
    args = parser.parse_args()

    if args.all:
        for key in COMMODITIES:
            fetch_and_save(key)
    else:
        fetch_and_save(args.commodity)


if __name__ == "__main__":
    main()
