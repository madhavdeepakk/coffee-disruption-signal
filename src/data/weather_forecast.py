"""
Fetch 16-day weather forecast for the Minas Gerais coffee-growing region in Brazil.

Uses the Open-Meteo free forecast API (https://open-meteo.com/) which requires
no API key or authentication. Default coordinates target Patrocínio, Minas Gerais
(-18.9439, -46.9925), a major arabica coffee production area.

The module caches responses to disk and respects a 6-hour TTL to avoid
unnecessary API calls.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = REPO_ROOT / "data" / "raw" / "weather_forecast.json"

DEFAULT_LATITUDE = -18.9439
DEFAULT_LONGITUDE = -46.9925
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours

API_URL = "https://api.open-meteo.com/v1/forecast"


def _load_cache() -> dict | None:
    """Return cached forecast data if the cache exists and is less than 6 hours old."""
    if not CACHE_PATH.exists():
        return None

    try:
        with open(CACHE_PATH, "r") as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    cached_ts = cached.get("_cached_at")
    if cached_ts is None:
        return None

    age = time.time() - cached_ts
    if age > CACHE_TTL_SECONDS:
        return None

    return cached


def _save_cache(data: dict) -> None:
    """Write forecast data to the cache file with a timestamp."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["_cached_at"] = time.time()
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # non-critical — we can always re-fetch


def fetch_forecast(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> dict:
    """
    Fetch a 16-day daily weather forecast from Open-Meteo.

    Returns the raw JSON response as a dict. Uses a local file cache with a
    6-hour TTL so repeated calls don't hit the API unnecessarily.

    Parameters
    ----------
    latitude : float
        Latitude of the target location (default: Minas Gerais coffee region).
    longitude : float
        Longitude of the target location.

    Returns
    -------
    dict
        Raw API response with daily forecast arrays, or a dict containing
        ``_error`` and ``_reason`` keys on failure.
    """
    cached = _load_cache()
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "forecast_days": 16,
        "timezone": "America/Sao_Paulo",
    }

    verify = os.environ.get("REQUESTS_CA_BUNDLE", True)

    try:
        resp = requests.get(API_URL, params=params, timeout=30, verify=verify)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        return {"_error": True, "_reason": str(exc)}

    data["_from_cache"] = False
    _save_cache(data)
    return data


def parse_forecast(raw: dict) -> dict:
    """
    Parse raw Open-Meteo response into an actionable summary.

    Parameters
    ----------
    raw : dict
        Raw response from :func:`fetch_forecast`.

    Returns
    -------
    dict
        Summary containing precipitation totals, frost-day counts,
        dry-day counts, and a per-day breakdown.
    """
    if raw.get("_error"):
        return {"error": True, "reason": raw.get("_reason", "unknown")}

    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    tmax_list = daily.get("temperature_2m_max", [])
    tmin_list = daily.get("temperature_2m_min", [])
    precip_list = daily.get("precipitation_sum", [])
    prob_list = daily.get("precipitation_probability_max", [])

    n = len(dates)

    per_day = []
    for i in range(n):
        per_day.append(
            {
                "date": dates[i],
                "tmax": tmax_list[i] if i < len(tmax_list) else None,
                "tmin": tmin_list[i] if i < len(tmin_list) else None,
                "precip": precip_list[i] if i < len(precip_list) else None,
                "precip_probability": prob_list[i] if i < len(prob_list) else None,
            }
        )

    def _safe(val, default=0.0):
        return val if val is not None else default

    total_precip_7d = sum(_safe(precip_list[i]) for i in range(min(7, n)))
    total_precip_14d = sum(_safe(precip_list[i]) for i in range(min(14, n)))

    temps_7d = [tmin_list[i] for i in range(min(7, n)) if i < len(tmin_list) and tmin_list[i] is not None]
    min_temp_7d = min(temps_7d) if temps_7d else None

    frost_days_7d = sum(
        1 for i in range(min(7, n))
        if i < len(tmin_list) and tmin_list[i] is not None and tmin_list[i] < 2.0
    )
    frost_days_14d = sum(
        1 for i in range(min(14, n))
        if i < len(tmin_list) and tmin_list[i] is not None and tmin_list[i] < 2.0
    )

    dry_days_7d = sum(
        1 for i in range(min(7, n))
        if i < len(precip_list) and precip_list[i] is not None and precip_list[i] < 1.0
    )

    return {
        "total_precip_7d": round(total_precip_7d, 1),
        "total_precip_14d": round(total_precip_14d, 1),
        "min_temp_7d": round(min_temp_7d, 1) if min_temp_7d is not None else None,
        "frost_days_7d": frost_days_7d,
        "frost_days_14d": frost_days_14d,
        "dry_days_7d": dry_days_7d,
        "daily": per_day,
    }


def get_forecast_observation() -> dict:
    """
    Fetch and parse the forecast, returning a clean summary dict.

    Returns
    -------
    dict
        Forecast summary with keys: ``status``, ``as_of``, precipitation totals,
        frost counts, dry-day counts, and a per-day ``daily`` list.
        ``status`` is one of ``"ok"``, ``"cached"``, or ``"error"``.
    """
    raw = fetch_forecast()

    if raw.get("_error"):
        return {
            "status": "error",
            "reason": raw.get("_reason", "unknown"),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    from_cache = raw.get("_from_cache", False)
    parsed = parse_forecast(raw)

    if parsed.get("error"):
        return {
            "status": "error",
            "reason": parsed.get("reason", "parse failure"),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    parsed["status"] = "cached" if from_cache else "ok"
    parsed["as_of"] = datetime.now(timezone.utc).isoformat()
    return parsed


if __name__ == "__main__":
    summary = get_forecast_observation()

    print(f"Weather Forecast — Minas Gerais Coffee Region")
    print(f"Coordinates: {DEFAULT_LATITUDE}, {DEFAULT_LONGITUDE}")
    print(f"Status: {summary.get('status')}")
    print(f"As of: {summary.get('as_of')}")
    print()

    if summary.get("status") == "error":
        print(f"Error: {summary.get('reason')}")
    else:
        print(f"Precipitation (7d):  {summary['total_precip_7d']} mm")
        print(f"Precipitation (14d): {summary['total_precip_14d']} mm")
        print(f"Min temperature (7d): {summary['min_temp_7d']} °C")
        print(f"Frost days (7d):  {summary['frost_days_7d']}")
        print(f"Frost days (14d): {summary['frost_days_14d']}")
        print(f"Dry days (7d):    {summary['dry_days_7d']}")
        print()
        print("Daily breakdown:")
        print(f"{'Date':<12} {'Tmax':>6} {'Tmin':>6} {'Precip':>8} {'Prob':>5}")
        print("-" * 42)
        for day in summary.get("daily", []):
            tmax = f"{day['tmax']:.1f}" if day["tmax"] is not None else "  n/a"
            tmin = f"{day['tmin']:.1f}" if day["tmin"] is not None else "  n/a"
            precip = f"{day['precip']:.1f}" if day["precip"] is not None else "  n/a"
            prob = f"{day['precip_probability']}%" if day["precip_probability"] is not None else " n/a"
            print(f"{day['date']:<12} {tmax:>6} {tmin:>6} {precip:>7}mm {prob:>5}")
