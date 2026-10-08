"""
World Bank Pink Sheet ingestion: monthly Arabica coffee price (a distinct
macro/benchmark price series, separate from the daily KC=F futures used by
the anomaly detector) - part of the proposal's 5-source data plan (S3.1).

Auth: none required - the Pink Sheet is a free public download, no API key,
no signup. Source: https://www.worldbank.org/en/research/commodity-markets
(direct link: CMO-Historical-Data-Monthly.xlsx under "Pink Sheet").

Why this series matters here: it is monthly, not daily, so it cannot detect
the day-level anomalies that src/modeling/anomaly_detector.py finds in the
daily KC=F futures series. Its value in this project is (a) a slower,
exchange-independent cross-check that a detected price move shows up in a
broader benchmark too, not just one futures contract, and (b) as one input
into the "inflation detection and analysis" objective (comparing commodity
price trends against broader price-level data), which the team has
deprioritized for later. This script does the cadence-aware join correctly
(see lookahead note below) but does not treat monthly data as capable of
daily-anomaly detection.

Lookahead note: the Pink Sheet publishes a given month's average AFTER
that month ends (typically early the following month), so a row labeled
"2021M07" is not known until roughly early August 2021. This script keeps
the raw period label and computes a conservative "known_as_of_date" as the
first day of the FOLLOWING month, so any code that joins this against
daily data can do it without lookahead bias - this project's data pipeline
treats that as a hard rule (see src/pipeline.py, src/modeling/*).

Usage:
    python -m src.data.world_bank --input data/raw/CMO-Historical-Data-Monthly.xlsx
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "raw" / "CMO-Historical-Data-Monthly.xlsx"
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "world_bank_coffee_monthly.csv"

SHEET_NAME = "Monthly Prices"
HEADER_ROW = 4  # 0-indexed pandas row for the "Coffee, Arabica ($/kg)" header row (row 5 in Excel's 1-indexed view)
DATE_COL_INDEX = 0
ARABICA_COL_NAME = "Coffee, Arabica"  # matched via startswith below - the raw header has a unit suffix appended


def parse_period_label(label: str) -> str:
    """
    Pink Sheet period labels look like "2021M07" - convert to an ISO date
    for the first day of that month (the period the price DESCRIBES, not
    when it was known - see known_as_of_date for the lookahead-safe field).
    """
    year = int(label[:4])
    month = int(label[5:7])
    return f"{year:04d}-{month:02d}-01"


def compute_known_as_of_date(period_date_str: str) -> str:
    """
    Conservative assumption: a given month's Pink Sheet average is not
    published until sometime in the following month. We use the 1st of
    the month AFTER the period as the "known as of" date - conservative
    (the real publish date is often a few days into that month, so this
    errs on the side of not overclaiming how early the data was known).
    """
    dt = datetime.strptime(period_date_str, "%Y-%m-%d")
    if dt.month == 12:
        return f"{dt.year + 1:04d}-01-01"
    return f"{dt.year:04d}-{dt.month + 1:02d}-01"


def load_world_bank_coffee(input_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(input_path, sheet_name=SHEET_NAME, header=HEADER_ROW)
    date_col = raw.columns[DATE_COL_INDEX]

    arabica_col = None
    for col in raw.columns:
        if isinstance(col, str) and col.strip().startswith(ARABICA_COL_NAME):
            arabica_col = col
            break
    if arabica_col is None:
        raise SystemExit(
            f"Could not find an '{ARABICA_COL_NAME}...' column in sheet '{SHEET_NAME}'. "
            f"Columns found: {list(raw.columns)[:20]}"
        )

    df = raw[[date_col, arabica_col]].copy()
    df.columns = ["period_label", "price_usd_per_kg"]
    df = df.dropna(subset=["period_label"])
    df = df[df["period_label"].astype(str).str.match(r"^\d{4}M\d{2}$")]

    df["date"] = df["period_label"].astype(str).apply(parse_period_label)
    df["known_as_of_date"] = df["date"].apply(compute_known_as_of_date)
    df["price_usd_per_kg"] = pd.to_numeric(df["price_usd_per_kg"], errors="coerce")
    df["source"] = "world_bank_pink_sheet"
    df["commodity"] = "coffee_arabica"

    df = df[["date", "known_as_of_date", "price_usd_per_kg", "commodity", "source", "period_label"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"{input_path} not found.")

    df = load_world_bank_coffee(input_path)
    print(f"Loaded {len(df)} monthly rows from {input_path.name}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing prices: {df['price_usd_per_kg'].isna().sum()}")
    print("\nMost recent 5 rows:")
    print(df.tail(5).to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(
        "\nReminder: this is MONTHLY data, kept as a cross-check/macro series - the "
        "day-level anomaly detector (src/modeling/anomaly_detector.py) runs on the "
        "daily KC=F futures series in data/raw/coffee_prices.csv, not this file."
    )


if __name__ == "__main__":
    main()
