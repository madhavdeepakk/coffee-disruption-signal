"""
Consumer prices and monthly commodity prices for the inflation objective:
the United States, India and a global reference, next to the commodities the
anomaly detector watches.

Source. Every series is downloaded from FRED (Federal Reserve Bank of St.
Louis) through its public CSV download, which needs no key and no account:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>&cosd=<START>

and answers with two columns, `observation_date,<SERIES>`. FRED republishes
the statistics offices' own figures (the US Bureau of Labor Statistics, the
OECD's Main Economic Indicators for India and the G20, the World Bank's
annual inflation, the IMF's Primary Commodity Prices), so one keyless source
covers both countries and the commodities at one frequency. Two sources that
were checked and not used: the World Bank API has consumer prices only once a
year, and India's own portal (MOSPI / data.gov.in) needs a registered key.

What the series can and cannot say:

  - India's monthly index (INDCPIALLMINMEI) is the OECD's copy of the national
    CPI. The OECD stopped updating that database, so the series ends in March
    2025; the annual World Bank figure (FPCPITOTLZGIND) runs a little later.
    India results therefore stop there and say so. It is not seasonally
    adjusted, which the analysis allows for with month-of-year terms.
  - India has only the all-items index here: no food or fuel sub-index.
  - The "global" monthly series is the G20's year-on-year inflation, which
    also ends in March 2025; the World Bank's world figure is annual.
  - Commodity prices are the IMF's monthly averages in US dollars. For India
    they are converted to rupees with the monthly average exchange rate
    (EXINUS), because a household in India pays in rupees.

When a figure is known. A month's CPI is published about two weeks after the
month ends; a month's average commodity price is known the day the month
ends. The analysis in src/modeling/inflation.py relies on exactly that gap
and on nothing later, and each row written here carries a `known_from` date
(the first day of the following month for prices, the fifteenth of the
following month for consumer prices) so that any other use can respect it.

Usage:
    python -m src.data.cpi                 # download every series, write the table
    python -m src.data.cpi --start 2000-01-01
"""

import argparse
import csv
import io
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "inflation_series.csv"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_START = "2000-01-01"
REQUEST_TIMEOUT_SECONDS = 30

# series id -> (short name used in the code, what it is, kind)
# kind: "cpi" (a price index of consumer prices), "cpi_yoy" (already a
# year-on-year % rate), "commodity" (a monthly average price or price index),
# "fx" (currency units per US dollar).
SERIES = {
    # United States - Bureau of Labor Statistics, seasonally adjusted
    "CPIAUCSL": ("us_cpi", "US consumer prices, all items", "cpi"),
    "CPILFESL": ("us_core", "US consumer prices, all items less food and energy", "cpi"),
    "CPIUFDSL": ("us_food", "US consumer prices, food", "cpi"),
    "CPIENGSL": ("us_energy", "US consumer prices, energy", "cpi"),
    "CUSR0000SETB01": ("us_gasoline", "US consumer prices, gasoline", "cpi"),
    "CUSR0000SEFP01": ("us_coffee", "US consumer prices, coffee", "cpi"),
    # India - OECD Main Economic Indicators (not seasonally adjusted; ends 2025-03)
    "INDCPIALLMINMEI": ("india_cpi", "India consumer prices, all items", "cpi"),
    "EXINUS": ("inr_per_usd", "Indian rupees per US dollar, monthly average", "fx"),
    # Global reference
    "G20CPALTT01GYM": ("g20_inflation", "G20 consumer price inflation, year on year", "cpi_yoy"),
    "FPCPITOTLZGWLD": ("world_inflation_annual", "World consumer price inflation, annual", "cpi_yoy"),
    "FPCPITOTLZGUSA": ("us_inflation_annual", "US consumer price inflation, annual", "cpi_yoy"),
    "FPCPITOTLZGIND": ("india_inflation_annual", "India consumer price inflation, annual", "cpi_yoy"),
    # Commodities - IMF Primary Commodity Prices, monthly averages, US dollars
    "POILWTIUSDM": ("oil_wti", "Crude oil, WTI, $ per barrel", "commodity"),
    "POILBREUSDM": ("oil_brent", "Crude oil, Brent, $ per barrel", "commodity"),
    "PWHEAMTUSDM": ("wheat", "Wheat, $ per tonne", "commodity"),
    "PCOFFOTMUSDM": ("coffee", "Coffee, other mild arabicas, cents per pound", "commodity"),
    "PFOODINDEXM": ("food_index", "IMF food price index", "commodity"),
    "PNRGINDEXM": ("energy_index", "IMF energy price index", "commodity"),
    "PALLFNFINDEXM": ("all_commodities", "IMF all-commodity price index", "commodity"),
}
NAME_TO_ID = {name: series_id for series_id, (name, _what, _kind) in SERIES.items()}


def parse_fred_csv(text: str, series_id: str) -> list:
    """[(YYYY-MM-DD, value)] from FRED's CSV download. Rows with no value
    (an empty field, or the '.' FRED writes for a missing observation) are
    left out rather than read as zero. Raises ValueError when the text is not
    the two-column table expected - an HTML error page, for instance - so a
    failed download is never mistaken for an empty series."""
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    header = next(reader, None)
    if (not header or len(header) < 2
            or header[0].strip().lower() not in ("observation_date", "date")):
        raise ValueError(f"{series_id}: not a FRED CSV (first line: {str(header)[:80]})")
    rows = []
    for line in reader:
        if len(line) < 2:
            continue
        day, raw = line[0].strip(), line[1].strip()
        if raw in ("", "."):
            continue
        try:
            rows.append((date.fromisoformat(day).isoformat(), float(raw)))
        except ValueError:
            continue
    return rows


def fetch_series(series_id: str, start: str = DEFAULT_START) -> list:
    import requests
    response = requests.get(FRED_CSV_URL, params={"id": series_id, "cosd": start},
                            timeout=REQUEST_TIMEOUT_SECONDS,
                            headers={"User-Agent": "Mozilla/5.0 (research capstone)"})
    response.raise_for_status()
    return parse_fred_csv(response.text, series_id)


def known_from(period: str, kind: str) -> str:
    """The first day a figure for this month can be used without looking
    ahead: the first of the next month for a price average, the fifteenth of
    the next month for consumer prices (a deliberately late round figure)."""
    year, month = int(period[:4]), int(period[5:7])
    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return f"{year:04d}-{month:02d}-{'15' if kind in ('cpi', 'cpi_yoy') else '01'}"


def write_table(rows_by_id: dict, path: Path = OUTPUT_PATH) -> int:
    """One tidy table: series_id, name, kind, date, value, known_from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["series_id", "name", "kind", "date", "value", "known_from"])
        for series_id in SERIES:
            name, _what, kind = SERIES[series_id]
            for day, value in rows_by_id.get(series_id, []):
                writer.writerow([series_id, name, kind, day, value, known_from(day, kind)])
                n += 1
    return n


def load_table(path: Path = OUTPUT_PATH):
    """{name: pandas Series indexed by month-start Timestamp}."""
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["date"])
    return {name: group.set_index("date")["value"].sort_index()
            for name, group in df.groupby("name")}


def main():
    parser = argparse.ArgumentParser(description="Download consumer and commodity price series")
    parser.add_argument("--start", default=DEFAULT_START)
    args = parser.parse_args()

    rows_by_id, failed = {}, []
    for series_id, (name, what, _kind) in SERIES.items():
        try:
            rows = fetch_series(series_id, args.start)
        except Exception as exc:  # noqa: BLE001 - report every failure, then decide
            failed.append(series_id)
            print(f"  {series_id:<16} FAILED: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        rows_by_id[series_id] = rows
        span = f"{rows[0][0]} to {rows[-1][0]}" if rows else "no rows"
        print(f"  {series_id:<16} {len(rows):>4} rows  {span}  {what}")
    if failed:
        # A table with some series missing would silently drop those rows
        # from every later result. Keep the previous table instead.
        print(f"\n{len(failed)} series could not be downloaded ({', '.join(failed)}). "
              f"Nothing was written; {OUTPUT_PATH.name} is unchanged. Run this again.")
        sys.exit(1)
    n = write_table(rows_by_id)
    print(f"\nSaved {n} rows to {OUTPUT_PATH}")
    print("Next:\n    python -m src.modeling.inflation")


if __name__ == "__main__":
    main()
