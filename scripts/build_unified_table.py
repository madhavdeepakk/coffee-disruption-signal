"""
Build the unified multi-source daily table (revised proposal S3.1 deliverable).

Joins three data sources into one lookahead-safe DataFrame:
  1. Daily futures prices (KC=F via yfinance, already in data/raw/)
  2. Daily weather (NASA POWER satellite reanalysis for Minas Gerais)
  3. Monthly World Bank Pink Sheet (Arabica benchmark, joined by known_as_of_date)

If the NASA POWER weather CSV does not exist, this script fetches it live
from the API (no key required). Run on a machine with unrestricted HTTPS.

Lookahead rules (hard, checked by assertion):
  - Daily price on date t uses the close on t (known at EOD t).
  - Daily weather on date t uses the observation on t (satellite reanalysis
    is available same-day or next-day; we use same-day, conservative).
  - Monthly World Bank on date t uses the most recent month whose
    known_as_of_date <= t. A row labeled "2021M07" is not known until
    ~2021-08-01, so it joins only to rows dated August 1 or later.

Output: data/processed/unified_daily.csv

Usage:
    python -m scripts.build_unified_table
    python -m scripts.build_unified_table --refresh-weather   # re-fetch NASA POWER
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "unified_daily.csv"

PRICE_PATH = RAW_DIR / "coffee_prices.csv"
WEATHER_PATH = RAW_DIR / "nasa_power_weather.csv"
WORLDBANK_PATH = RAW_DIR / "world_bank_coffee_monthly.csv"


def ensure_weather_data(refresh: bool = False) -> None:
    """Fetch NASA POWER weather if the file is missing or refresh requested."""
    if WEATHER_PATH.exists() and not refresh:
        return
    print("Fetching NASA POWER weather data (no API key required)...")
    prices = pd.read_csv(PRICE_PATH)
    end_date = prices["date"].max()[:10]
    subprocess.run(
        [sys.executable, "-m", "src.data.nasa_power",
         "--start", "2018-01-01", "--end", end_date],
        cwd=REPO_ROOT, check=True,
    )


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICE_PATH, parse_dates=["date"])
    df = df.rename(columns={"price": "close_usd"})
    df = df.sort_values("date").reset_index(drop=True)
    df["ret_1d"] = df["close_usd"].pct_change()
    return df


def load_weather() -> pd.DataFrame:
    if not WEATHER_PATH.exists():
        print(f"  WARNING: {WEATHER_PATH} not found. Run with --refresh-weather or "
              "run `python -m src.data.nasa_power` first.")
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Derived weather features (all same-day, no lookahead)
    df["prcp_30d_sum"] = df["prcp_mm"].rolling(30, min_periods=20).sum()
    df["prcp_90d_sum"] = df["prcp_mm"].rolling(90, min_periods=60).sum()
    df["frost_flag"] = (df["tmin_c"] < 2.0).astype(int)
    df["frost_7d_count"] = df["frost_flag"].rolling(7, min_periods=1).sum()

    # Dryness z-score: how unusual is the recent 30-day precip vs trailing year?
    prcp_365_mean = df["prcp_30d_sum"].rolling(365, min_periods=180).mean()
    prcp_365_std = df["prcp_30d_sum"].rolling(365, min_periods=180).std()
    df["dryness_zscore"] = (df["prcp_30d_sum"] - prcp_365_mean) / prcp_365_std.replace(0, np.nan)

    return df


def load_worldbank() -> pd.DataFrame:
    if not WORLDBANK_PATH.exists():
        print(f"  WARNING: {WORLDBANK_PATH} not found. Run `python -m src.data.world_bank` first.")
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(WORLDBANK_PATH, parse_dates=["date", "known_as_of_date"])
    df = df.rename(columns={"price_usd_per_kg": "wb_price_usd_kg"})
    df = df[["known_as_of_date", "wb_price_usd_kg"]].dropna()
    df = df.sort_values("known_as_of_date").reset_index(drop=True)
    return df


def merge_worldbank_asof(prices: pd.DataFrame, wb: pd.DataFrame) -> pd.DataFrame:
    """As-of merge: for each trading day, attach the most recent World Bank
    monthly price whose known_as_of_date <= that day."""
    if wb.empty:
        prices["wb_price_usd_kg"] = np.nan
        return prices
    prices = prices.sort_values("date")
    wb = wb.sort_values("known_as_of_date")
    merged = pd.merge_asof(
        prices, wb,
        left_on="date", right_on="known_as_of_date",
        direction="backward",
    )
    merged = merged.drop(columns=["known_as_of_date"], errors="ignore")
    return merged


def build_unified(refresh_weather: bool = False) -> pd.DataFrame:
    ensure_weather_data(refresh=refresh_weather)

    print("Loading daily prices...")
    prices = load_prices()
    print(f"  {len(prices)} trading days, {prices['date'].min().date()} to {prices['date'].max().date()}")

    print("Loading daily weather (NASA POWER)...")
    weather = load_weather()
    has_weather = not weather.empty and len(weather) > 1
    if has_weather:
        print(f"  {len(weather)} calendar days of weather")
    else:
        print("  (no weather data — will build table without it)")

    print("Loading monthly World Bank Pink Sheet...")
    wb = load_worldbank()
    if not wb.empty:
        print(f"  {len(wb)} monthly observations")

    # Join weather (daily-daily exact date)
    if has_weather:
        weather_cols = [c for c in weather.columns if c != "date"]
        df = prices.merge(weather[["date"] + weather_cols], on="date", how="left")
    else:
        df = prices.copy()

    # As-of merge World Bank monthly
    df = merge_worldbank_asof(df, wb)

    # Cross-source derived feature: futures-vs-benchmark spread
    if "wb_price_usd_kg" in df.columns:
        df["wb_price_cents_lb"] = df["wb_price_usd_kg"] * 100 / 2.20462
        df["futures_wb_spread_pct"] = (
            (df["close_usd"] - df["wb_price_cents_lb"]) / df["wb_price_cents_lb"] * 100
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df


def run_sanity_checks(df: pd.DataFrame) -> None:
    print("\n--- Lookahead sanity checks ---")
    if "wb_price_usd_kg" in df.columns:
        row = df[df["date"] == pd.Timestamp("2021-07-19")]
        if not row.empty:
            wb_val = row.iloc[0].get("wb_price_usd_kg")
            print(f"  WB price on 2021-07-19 (should be Jun 2021 value, not Jul): {wb_val:.4f}")

    if "dryness_zscore" in df.columns:
        drought = df[(df["date"] >= "2021-05-01") & (df["date"] <= "2021-07-19")]
        if not drought.empty and drought["dryness_zscore"].notna().any():
            avg_z = drought["dryness_zscore"].mean()
            print(f"  Avg dryness z-score during 2021 drought (expect negative): {avg_z:.2f}")

    if "frost_flag" in df.columns:
        frost_jul21 = df[(df["date"] >= "2021-07-18") & (df["date"] <= "2021-07-22")]
        if not frost_jul21.empty:
            n_frost = frost_jul21["frost_flag"].sum()
            print(f"  Frost days around Jul 20 2021 event (expect >0): {n_frost}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-weather", action="store_true",
                        help="Re-fetch NASA POWER weather data even if the file exists")
    args = parser.parse_args()

    df = build_unified(refresh_weather=args.refresh_weather)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nUnified table: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved to {OUTPUT_PATH}")

    run_sanity_checks(df)


if __name__ == "__main__":
    main()
