"""
NASA POWER ingestion: daily precipitation and temperature for the coffee
anomaly detector's weather/drought signal. This addresses NOAA GHCND's
data-sparsity problem (src/data/noaa.py).

Why this replaces NOAA as the primary weather source: NOAA GHCND depends
on physical ground stations, and Brazil's station network in that database
is real but mostly dead - almost every station found stopped reporting in
the 1990s, and the few still "active" (Araxa, tested) return only a handful
of days total and no precipitation at all. That is a limitation of that
dataset for this region, documented in results/.

NASA POWER is a different kind of dataset: it is a satellite/reanalysis-
derived GRID product (not individual physical stations), so it has no
"station went offline in 1997" problem - it has dense daily coverage
worldwide, including anywhere in Brazil, from 1981 to near-present. It is
free, requires NO signup or API key, is run by NASA (a defensible,
citable government source for an academic report), and is widely used in
exactly this kind of agriculture/climate application (the API's own
"community=AG" parameter is literally the agroclimatology preset).

Location choice: this pulls data for a single representative point in
Patrocinio, Minas Gerais (lat -18.9439, lon -46.9925), a municipality in the
"Cerrado Mineiro" region, a documented high-altitude arabica coffee growing
area. A single point is a simplification (real disruptions are regional, not
a single pixel), noted here rather than presented as regional precision.

Usage:
    python -m src.data.nasa_power --start 2018-01-01 --end 2026-08-28
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "nasa_power_weather.csv"

POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Patrocinio, MG - a real municipality in the "Cerrado Mineiro" arabica
# coffee region (Minas Gerais). A single representative point, not a
# regional average - see module docstring.
DEFAULT_LATITUDE = -18.9439
DEFAULT_LONGITUDE = -46.9925

# PRECTOTCORR = bias-corrected precipitation (mm/day)
# T2M_MAX / T2M_MIN = 2-meter max/min air temperature (C) - a frost signal
PARAMETERS = "PRECTOTCORR,T2M_MAX,T2M_MIN"
COMMUNITY = "AG"  # agroclimatology preset - appropriate for a crop-weather signal

REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 5

# NASA POWER's fill value for missing/unavailable data - must be converted
# to NaN, not left as a real-looking number.
FILL_VALUE = -999.0


def fetch_power_data(latitude: float, longitude: float, start_date: str, end_date: str,
                      verbose: bool = True) -> dict:
    start_compact = start_date.replace("-", "")
    end_compact = end_date.replace("-", "")
    params = {
        "parameters": PARAMETERS,
        "community": COMMUNITY,
        "longitude": longitude,
        "latitude": latitude,
        "start": start_compact,
        "end": end_compact,
        "format": "JSON",
    }

    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if verbose:
                print(f"  fetching {start_date}..{end_date} for ({latitude}, {longitude}) "
                      f"(attempt {attempt}/{MAX_ATTEMPTS})...")
            resp = requests.get(POWER_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            last_exc = exc
            if verbose:
                print(f"    failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                import time
                time.sleep(BACKOFF_SECONDS)
    raise RuntimeError(f"All {MAX_ATTEMPTS} attempts failed for {start_date}..{end_date}") from last_exc


def reshape_to_dataframe(raw_response: dict) -> pd.DataFrame:
    props = raw_response.get("properties", {})
    param_data = props.get("parameter", {})
    if not param_data:
        raise RuntimeError(
            f"No 'properties.parameter' in NASA POWER response - full response: {raw_response}"
        )

    frames = {}
    for param_name, day_values in param_data.items():
        s = pd.Series(day_values, name=param_name)
        frames[param_name] = s
    df = pd.DataFrame(frames)
    df.index.name = "date_raw"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date_raw"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["date_raw"])

    df = df.rename(columns={
        "PRECTOTCORR": "prcp_mm",
        "T2M_MAX": "tmax_c",
        "T2M_MIN": "tmin_c",
    })
    for col in ["prcp_mm", "tmax_c", "tmin_c"]:
        if col in df.columns:
            df.loc[df[col] == FILL_VALUE, col] = pd.NA

    cols = ["date"] + [c for c in ["prcp_mm", "tmax_c", "tmin_c"] if c in df.columns]
    df = df[cols].sort_values("date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-08-28")
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE)
    args = parser.parse_args()

    print(f"Fetching NASA POWER daily weather for ({args.lat}, {args.lon}) "
          f"[Patrocinio, MG - Cerrado Mineiro coffee region], {args.start}..{args.end}")
    print("No API key required - this is a public NASA dataset.\n")

    raw = fetch_power_data(args.lat, args.lon, args.start, args.end)
    df = reshape_to_dataframe(raw)

    print(f"\nReceived {len(df)} daily rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing values: prcp={df['prcp_mm'].isna().sum()}, "
          f"tmax={df['tmax_c'].isna().sum()}, tmin={df['tmin_c'].isna().sum()}")

    if len(df) == 0:
        print("\nWARNING: zero rows returned - not writing an empty file. Check the date "
              "range and coordinates.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(
        "\nThis is a single-point satellite/reanalysis series (not a physical weather "
        "station), which is what makes it dense and reliable for this region - see this "
        "file's module docstring for the tradeoff a single point involves."
    )


if __name__ == "__main__":
    main()
