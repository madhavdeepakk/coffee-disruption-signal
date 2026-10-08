"""
NOAA CDO (Climate Data Online) ingestion: weather/precipitation data for
Brazil's main coffee-growing regions (Minas Gerais), used as a candidate
explanatory signal alongside the news-based RAG evidence (per proposal
S3.1's 5-source data plan).

Auth: NOAA CDO requires only a free token (no approval wait), obtained by
email at https://www.ncdc.noaa.gov/cdo-web/token. Read from NOAA_TOKEN in
.env - never hardcode it in this file or commit it anywhere.

Scope note: this pulls GHCND (Global Historical Climatology Network -
Daily) station data for a station in/near Minas Gerais, Brazil (the
dominant Arabica-growing region referenced in the known frost/drought
events in results/anomaly_detector_validation.md). Station selection below
is a reasonable default, not a rigorously chosen "best" station - flag if
someone with domain knowledge of Brazilian coffee geography wants to
swap it for a better one.

Usage:
    python -m src.data.noaa --start 2018-01-01 --end 2026-08-28
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "noaa_weather.csv"

NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

# NOTE: there is no verified default station ID. An earlier version of this
# script guessed "GHCND:BRXX0021" - that ID does not exist (confirmed: the
# API returned 0 records for every year 2018-2026, which is what a bad
# station ID looks like, not what a network/rate-limit problem looks like).
# Run with --list-stations first to find a REAL, active station in Minas
# Gerais and pass its id via --station. Do not guess IDs.
DEFAULT_STATION_ID = None

# Bounding box roughly covering Minas Gerais, Brazil (the main Arabica belt
# referenced in the 2021 frost / 2024 & 2026 drought events), used by
# --list-stations to search for real GHCND stations in that area.
MINAS_GERAIS_EXTENT = "-22.5,-51.0,-14.0,-39.5"  # south,west,north,east

# NOAA CDO paginates in max 1-year chunks per request for daily data, and
# caps at 1000 records per request - handled below by chunking by year.
DATATYPES = "PRCP,TMAX,TMIN"  # precipitation, max/min temp (frost signal)

REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 5


def list_stations(token: str, extent: str, datasetid: str = "GHCND",
                   min_recent_year: int = 2018, verbose: bool = True) -> list:
    """
    Search for real stations in a bounding box. Use this BEFORE picking a
    --station value - guessing a station ID silently returns 0 records
    (looks identical to "no data" rather than "invalid id"), which is
    exactly what happened with the earlier placeholder ID.

    NOAA paginates at 25 results/page by default even when limit=1000 is
    requested for some datasets - this fetches every page (offset-based)
    rather than trusting the first page, since a Minas Gerais search can
    have 100+ stations and most Brazilian COOP stations stopped reporting
    in the 1990s. Results are sorted so stations with recent coverage
    (maxdate >= min_recent_year) are shown first and highlighted, since
    those are the only ones actually useful for this project's 2018-2026
    date range.
    """
    headers = {"token": token}
    all_results = []
    offset = 1
    page_size = 1000
    while True:
        params = {"datasetid": datasetid, "extent": extent, "limit": page_size, "offset": offset}
        resp = requests.get(
            "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations",
            headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        body = resp.json()
        page = body.get("results", []) or []
        all_results.extend(page)
        total_count = body.get("metadata", {}).get("resultset", {}).get("count", len(all_results))
        if len(all_results) >= total_count or not page:
            break
        offset += page_size

    def maxdate_year(s):
        md = s.get("maxdate") or ""
        try:
            return int(md[:4])
        except ValueError:
            return 0

    recent = [s for s in all_results if maxdate_year(s) >= min_recent_year]
    stale = [s for s in all_results if maxdate_year(s) < min_recent_year]

    if verbose:
        print(f"Found {len(all_results)} total station(s) in extent={extent} for dataset={datasetid}")
        print(f"  -> {len(recent)} with coverage through {min_recent_year} or later "
              f"(usable for this project), {len(stale)} stale/historical only\n")
        if recent:
            print(f"RECENT STATIONS (maxdate >= {min_recent_year}):")
            for s in sorted(recent, key=maxdate_year, reverse=True):
                print(f"  {s.get('id'):25s} {s.get('name','')[:40]:40s} "
                      f"coverage {s.get('mindate','?')}..{s.get('maxdate','?')}")
        else:
            print(
                f"No stations in this bounding box have data through {min_recent_year} or "
                "later. Most Brazilian COOP-network GHCND stations stopped reporting in the "
                "1990s - this is a real, common gap in Brazil's GHCND coverage, not a bug in "
                "this search. Options: (1) widen the search area (edit MINAS_GERAIS_EXTENT to "
                "cover a larger part of Brazil / nearby states), (2) search for an airport "
                "METAR-derived station instead of a COOP station (these tend to have live "
                "coverage) - try https://www.ncdc.noaa.gov/cdo-web/search directly and filter "
                "to 'Daily Summaries' + Brazil, or (3) treat weather/drought data as a gap "
                "in this project rather than force a bad-fit station."
            )
    return all_results


def _year_chunks(start_date: str, end_date: str):
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(datetime(cur.year, 12, 31), end)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end.replace(year=chunk_end.year + 1, month=1, day=1)
    return chunks


def fetch_noaa_chunk(token: str, station_id: str, start_date: str, end_date: str, verbose: bool = True) -> list:
    headers = {"token": token}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": DATATYPES,
        "units": "metric",
        "limit": 1000,
    }
    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if verbose:
                print(f"  fetching {start_date}..{end_date} (attempt {attempt}/{MAX_ATTEMPTS})...")
            resp = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 429:
                # rate limited - NOAA CDO allows 5 req/sec, 10000 req/day
                if verbose:
                    print("    rate limited, backing off...")
                time.sleep(BACKOFF_SECONDS)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", []) or []
        except (requests.exceptions.RequestException, ValueError) as exc:
            last_exc = exc
            if verbose:
                print(f"    failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS)
    raise RuntimeError(f"All {MAX_ATTEMPTS} attempts failed for {start_date}..{end_date}") from last_exc


def fetch_noaa_range(token: str, station_id: str, start_date: str, end_date: str, verbose: bool = True) -> list:
    """
    NOAA CDO limits daily-data requests to 1-year spans and 1000 records
    per call, so we chunk by calendar year and concatenate.
    """
    all_results = []
    for chunk_start, chunk_end in _year_chunks(start_date, end_date):
        results = fetch_noaa_chunk(token, station_id, chunk_start, chunk_end, verbose=verbose)
        if verbose:
            print(f"    got {len(results)} records")
        all_results.extend(results)
        time.sleep(0.25)  # stay well under the 5 req/sec limit
    return all_results


def reshape_to_wide(raw_records: list) -> "pd.DataFrame":
    """
    NOAA returns one row per (date, datatype) - pivot to one row per date
    with prcp/tmax/tmin columns, matching the team's tabular-per-day schema.
    """
    import pandas as pd

    if not raw_records:
        return pd.DataFrame(columns=["date", "prcp_mm", "tmax_c", "tmin_c", "station_id"])

    df = pd.DataFrame(raw_records)
    df["date"] = df["date"].str[:10]  # NOAA returns "YYYY-MM-DDT00:00:00"
    wide = df.pivot_table(index=["date", "station"], columns="datatype", values="value", aggfunc="first").reset_index()
    wide = wide.rename(columns={
        "PRCP": "prcp_mm", "TMAX": "tmax_c", "TMIN": "tmin_c", "station": "station_id",
    })
    for col in ["prcp_mm", "tmax_c", "tmin_c"]:
        if col not in wide.columns:
            wide[col] = None
    return wide[["date", "prcp_mm", "tmax_c", "tmin_c", "station_id"]].sort_values("date").reset_index(drop=True)


def main():
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("NOAA_TOKEN")
    if not token:
        raise SystemExit(
            "NOAA_TOKEN not found. Add NOAA_TOKEN=<your token> to your .env file "
            f"at {REPO_ROOT / '.env'}."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-08-28")
    parser.add_argument("--station", default=DEFAULT_STATION_ID)
    parser.add_argument("--list-stations", action="store_true",
                         help="Search for real GHCND stations in Minas Gerais and exit "
                              "(no data fetched). Run this first to get a real --station id.")
    args = parser.parse_args()

    if args.list_stations:
        print(f"Searching for GHCND stations in Minas Gerais (extent={MINAS_GERAIS_EXTENT})...\n")
        stations = list_stations(token, MINAS_GERAIS_EXTENT)
        if not stations:
            print("\nNo stations found in this bounding box for GHCND. Try widening "
                  "MINAS_GERAIS_EXTENT in this file, or search manually at "
                  "https://www.ncdc.noaa.gov/cdo-web/search")
        else:
            print("\nPick a station id above (one with coverage overlapping 2018-2026 and "
                  "a real Minas Gerais airport/city name), then run:\n"
                  "  python -m src.data.noaa --station <id>")
        return

    if not args.station:
        raise SystemExit(
            "No --station given and no verified default exists. Run "
            "'python -m src.data.noaa --list-stations' first to find a real station id."
        )

    print(f"Fetching NOAA GHCND data for station {args.station}, {args.start}..{args.end}")

    raw = fetch_noaa_range(token, args.station, args.start, args.end)
    print(f"\nTotal raw records: {len(raw)}")

    df = reshape_to_wide(raw)
    print(f"Reshaped to {len(df)} daily rows")

    if len(df) == 0:
        print(
            "\nWARNING: zero rows returned. Common causes: (1) wrong/inactive station ID - "
            "search https://www.ncdc.noaa.gov/cdo-web/search for an active GHCND station in "
            "Minas Gerais and pass it via --station, or (2) date range outside the station's "
            "coverage. Not writing an empty file."
        )
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Date range in data: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing values: prcp={df['prcp_mm'].isna().sum()}, tmax={df['tmax_c'].isna().sum()}, tmin={df['tmin_c'].isna().sum()}")


if __name__ == "__main__":
    main()
