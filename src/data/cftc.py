"""
CFTC Commitment of Traders (COT) data for coffee futures: weekly
speculative positioning from the disaggregated futures-only report.

The CFTC publishes COT data every Friday (covering positions as of the
prior Tuesday). This module fetches the current-year report, filters for
coffee (KC) rows (CFTC contract code 083731), and computes net speculative
positioning metrics that help gauge whether speculators are crowded long
or short in coffee futures.

Primary source (disaggregated futures-only, tab-separated):
    https://www.cftc.gov/dea/newcot/f_disagg.txt

Fallback source (legacy combined report, tab-separated):
    https://www.cftc.gov/dea/newcot/deacom.txt

The disaggregated report separates traders into Producer/Merchant,
Swap Dealers, Managed Money, and Other Reportables. For net speculative
positioning we use Managed Money (the closest proxy for speculative
positioning in the disaggregated format). The legacy format uses the
traditional Noncommercial category instead.

Usage:
    python -m src.data.cftc
"""

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = REPO_ROOT / "data" / "raw" / "cftc_cot.csv"

DISAGG_URL = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
LEGACY_URL = "https://www.cftc.gov/dea/newcot/deacom.txt"

COFFEE_CONTRACT_CODE = "083731"
REQUEST_TIMEOUT = 60
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 3

# cftc.gov returns 403 to the default python-requests User-Agent. A realistic
# browser UA is required or every fetch fails and the source reads "unavailable".
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/plain,text/csv,*/*",
}


def _get_session() -> requests.Session:
    """Build a requests session with a browser UA and proxy CA bundle."""
    session = requests.Session()
    session.headers.update(HEADERS)
    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE")
    if ca_bundle and Path(ca_bundle).exists():
        session.verify = ca_bundle
    return session


def _get_with_retries(session: requests.Session, url: str) -> str | None:
    """GET with retries. Returns response text, or None after all attempts."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception:
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS * attempt)
    return None


def _merge_cache(new_df: pd.DataFrame) -> pd.DataFrame:
    """Accumulate weekly reports into the cache so a real history builds up.

    The f_disagg.txt / deacom.txt endpoints only carry the latest week, so the
    percentile/z-score would be computed on a single row and be meaningless.
    Merging each fetch with the cached CSV (deduped by report_date) grows a
    genuine multi-week history over successive runs."""
    if CACHE_PATH.exists():
        try:
            old = pd.read_csv(CACHE_PATH, parse_dates=["report_date"])
            new_df = pd.concat([old, new_df], ignore_index=True)
        except Exception:
            pass
    new_df = new_df.dropna(subset=["report_date"])
    new_df = new_df.drop_duplicates(subset=["report_date"], keep="last")
    return new_df.sort_values("report_date").reset_index(drop=True)


def _cache_is_fresh(cache_hours: float) -> bool:
    """Return True if the cache file exists and is younger than cache_hours."""
    if not CACHE_PATH.exists():
        return False
    age_seconds = time.time() - CACHE_PATH.stat().st_mtime
    return age_seconds < cache_hours * 3600


def _parse_disaggregated(text: str) -> pd.DataFrame:
    """
    Parse the CFTC disaggregated futures-only report (f_disagg.txt).

    The file is comma-separated despite the .txt extension. Key columns
    (by header name) for coffee:
        Market_and_Exchange_Names, CFTC_Contract_Market_Code,
        Report_Date_as_YYYY-MM-DD, Open_Interest_All,
        M_Money_Positions_Long_All, M_Money_Positions_Short_All,
        Prod_Merc_Positions_Long_All, Prod_Merc_Positions_Short_All
    """
    from io import StringIO

    df = pd.read_csv(StringIO(text), low_memory=False)

    # Normalise column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Filter for coffee rows
    code_col = None
    for candidate in ["CFTC_Contract_Market_Code", "CFTC Contract Market Code"]:
        if candidate in df.columns:
            code_col = candidate
            break

    name_col = None
    for candidate in ["Market_and_Exchange_Names", "Market and Exchange Names"]:
        if candidate in df.columns:
            name_col = candidate
            break

    if code_col is not None:
        df[code_col] = df[code_col].astype(str).str.strip()
        mask = df[code_col] == COFFEE_CONTRACT_CODE
    elif name_col is not None:
        mask = df[name_col].str.upper().str.contains("COFFEE", na=False)
    else:
        return pd.DataFrame()

    # If code filter yields nothing, try name-based fallback
    if mask.sum() == 0 and name_col is not None:
        mask = df[name_col].str.upper().str.contains("COFFEE", na=False)

    coffee = df.loc[mask].copy()
    if coffee.empty:
        return pd.DataFrame()

    # Locate the columns we need (disaggregated format)
    col_map = {}
    for target, candidates in {
        "report_date": [
            "Report_Date_as_YYYY-MM-DD",
            "Report Date as YYYY-MM-DD",
            "As_of_Date_In_Form_YYMMDD",
        ],
        "open_interest": [
            "Open_Interest_All",
            "Open Interest (All)",
            "Open_Interest",
        ],
        "noncommercial_long": [
            "M_Money_Positions_Long_All",
            "M Money Positions Long All",
            "Money_Manager_Longs",
        ],
        "noncommercial_short": [
            "M_Money_Positions_Short_All",
            "M Money Positions Short All",
            "Money_Manager_Shorts",
        ],
        "commercial_long": [
            "Prod_Merc_Positions_Long_All",
            "Prod Merc Positions Long All",
            "Producer_Merchant_Longs",
        ],
        "commercial_short": [
            "Prod_Merc_Positions_Short_All",
            "Prod Merc Positions Short All",
            "Producer_Merchant_Shorts",
        ],
    }.items():
        for c in candidates:
            if c in coffee.columns:
                col_map[target] = c
                break

    required = [
        "report_date",
        "open_interest",
        "noncommercial_long",
        "noncommercial_short",
    ]
    if not all(k in col_map for k in required):
        return pd.DataFrame()

    result = pd.DataFrame()
    for new_name, orig_name in col_map.items():
        result[new_name] = coffee[orig_name].values

    result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")

    for num_col in [
        "open_interest",
        "noncommercial_long",
        "noncommercial_short",
        "commercial_long",
        "commercial_short",
    ]:
        if num_col in result.columns:
            result[num_col] = pd.to_numeric(result[num_col], errors="coerce")

    # Fill commercial columns with 0 if missing
    for col in ["commercial_long", "commercial_short"]:
        if col not in result.columns:
            result[col] = 0

    result = result.dropna(subset=["report_date"]).sort_values("report_date")
    result = result.reset_index(drop=True)
    return result


def _parse_legacy(text: str) -> pd.DataFrame:
    """
    Parse the CFTC legacy combined report (deacom.txt).

    The legacy format uses traditional Noncommercial / Commercial
    categories directly. Columns (comma- or tab-separated):
        Market_and_Exchange_Names, CFTC_Contract_Market_Code,
        As_of_Date_In_Form_YYMMDD, Open_Interest_All,
        NonComm_Positions_Long_All, NonComm_Positions_Short_All,
        Comm_Positions_Long_All, Comm_Positions_Short_All
    """
    from io import StringIO

    # Try comma first, then tab
    df = pd.read_csv(StringIO(text), low_memory=False)
    if len(df.columns) < 5:
        df = pd.read_csv(StringIO(text), sep="\t", low_memory=False)

    df.columns = [c.strip() for c in df.columns]

    # Filter for coffee
    code_col = None
    for candidate in ["CFTC_Contract_Market_Code", "CFTC Contract Market Code"]:
        if candidate in df.columns:
            code_col = candidate
            break

    name_col = None
    for candidate in ["Market_and_Exchange_Names", "Market and Exchange Names"]:
        if candidate in df.columns:
            name_col = candidate
            break

    if code_col is not None:
        df[code_col] = df[code_col].astype(str).str.strip()
        mask = df[code_col] == COFFEE_CONTRACT_CODE
    elif name_col is not None:
        mask = df[name_col].str.upper().str.contains("COFFEE", na=False)
    else:
        return pd.DataFrame()

    if mask.sum() == 0 and name_col is not None:
        mask = df[name_col].str.upper().str.contains("COFFEE", na=False)

    coffee = df.loc[mask].copy()
    if coffee.empty:
        return pd.DataFrame()

    col_map = {}
    for target, candidates in {
        "report_date": [
            "As_of_Date_In_Form_YYMMDD",
            "Report_Date_as_YYYY-MM-DD",
            "As of Date in Form YYMMDD",
        ],
        "open_interest": [
            "Open_Interest_All",
            "Open Interest (All)",
        ],
        "noncommercial_long": [
            "NonComm_Positions_Long_All",
            "NonComm Positions-Long (All)",
            "Noncommercial Long",
        ],
        "noncommercial_short": [
            "NonComm_Positions_Short_All",
            "NonComm Positions-Short (All)",
            "Noncommercial Short",
        ],
        "commercial_long": [
            "Comm_Positions_Long_All",
            "Comm Positions-Long (All)",
            "Commercial Long",
        ],
        "commercial_short": [
            "Comm_Positions_Short_All",
            "Comm Positions-Short (All)",
            "Commercial Short",
        ],
    }.items():
        for c in candidates:
            if c in coffee.columns:
                col_map[target] = c
                break

    required = [
        "report_date",
        "open_interest",
        "noncommercial_long",
        "noncommercial_short",
    ]
    if not all(k in col_map for k in required):
        return pd.DataFrame()

    result = pd.DataFrame()
    for new_name, orig_name in col_map.items():
        result[new_name] = coffee[orig_name].values

    result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")

    for num_col in [
        "open_interest",
        "noncommercial_long",
        "noncommercial_short",
        "commercial_long",
        "commercial_short",
    ]:
        if num_col in result.columns:
            result[num_col] = pd.to_numeric(result[num_col], errors="coerce")

    for col in ["commercial_long", "commercial_short"]:
        if col not in result.columns:
            result[col] = 0

    result = result.dropna(subset=["report_date"]).sort_values("report_date")
    result = result.reset_index(drop=True)
    return result


def fetch_cot_data(cache_hours: float = 24) -> pd.DataFrame:
    """
    Fetch CFTC COT data for coffee futures.

    Checks a local CSV cache first. If the cache is stale or missing,
    downloads the disaggregated report (primary) or the legacy combined
    report (fallback), filters for coffee rows, and caches the result.

    Parameters
    ----------
    cache_hours : float
        Maximum age of cached data in hours before re-downloading.

    Returns
    -------
    pd.DataFrame
        Columns: report_date, open_interest, noncommercial_long,
        noncommercial_short, commercial_long, commercial_short.
        Empty DataFrame on failure.
    """
    if _cache_is_fresh(cache_hours):
        try:
            df = pd.read_csv(CACHE_PATH, parse_dates=["report_date"])
            if not df.empty:
                return df
        except Exception:
            pass

    session = _get_session()

    # Try disaggregated report first, then the legacy combined report.
    df = pd.DataFrame()
    text = _get_with_retries(session, DISAGG_URL)
    if text:
        df = _parse_disaggregated(text)
    if df.empty:
        text = _get_with_retries(session, LEGACY_URL)
        if text:
            df = _parse_legacy(text)

    if df.empty:
        # Network failed, but a prior cache may still hold usable history.
        if CACHE_PATH.exists():
            try:
                cached = pd.read_csv(CACHE_PATH, parse_dates=["report_date"])
                if not cached.empty:
                    return cached
            except Exception:
                pass
        return df

    # Accumulate into the cache so percentile history builds over time.
    df = _merge_cache(df)
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
    except Exception:
        pass

    return df


def analyze_positioning(df: pd.DataFrame) -> dict:
    """
    Compute speculative positioning metrics from COT data.

    Parameters
    ----------
    df : pd.DataFrame
        COT data as returned by fetch_cot_data().

    Returns
    -------
    dict
        Positioning summary including net_speculative, percentile rank,
        week-over-week change, and a text interpretation.
    """
    if df.empty or len(df) < 1:
        return {
            "error": "No COT data available for analysis",
        }

    df = df.sort_values("report_date").reset_index(drop=True)

    latest = df.iloc[-1]

    net_spec = int(latest["noncommercial_long"] - latest["noncommercial_short"])

    # Week-over-week change
    if len(df) >= 2:
        prev = df.iloc[-2]
        prev_net = int(prev["noncommercial_long"] - prev["noncommercial_short"])
        net_spec_change = net_spec - prev_net
    else:
        net_spec_change = 0

    # Percentile vs trailing 52 weeks (or all available data)
    lookback = min(len(df), 52)
    recent = df.tail(lookback).copy()
    recent["net_spec"] = recent["noncommercial_long"] - recent["noncommercial_short"]
    if recent["net_spec"].std() == 0:
        percentile = 50.0
    else:
        rank = (recent["net_spec"] < net_spec).sum()
        percentile = round(100.0 * rank / len(recent), 1)

    open_interest = int(latest["open_interest"])
    spec_long_pct = round(
        100.0 * latest["noncommercial_long"] / open_interest, 1
    ) if open_interest > 0 else 0.0

    commercial_net = int(latest["commercial_long"] - latest["commercial_short"])

    # Interpretation
    if percentile > 80:
        interpretation = (
            "Speculative longs are at historically high levels — crowded trade"
        )
    elif percentile < 20:
        interpretation = (
            "Speculators are heavily short — potential for short-covering rally"
        )
    else:
        interpretation = (
            "Speculative positioning is within a normal range"
        )

    report_date = latest["report_date"]
    if hasattr(report_date, "strftime"):
        report_date = report_date.strftime("%Y-%m-%d")
    else:
        report_date = str(report_date)

    return {
        "report_date": report_date,
        "open_interest": open_interest,
        "net_speculative": net_spec,
        "net_speculative_change": net_spec_change,
        "net_spec_percentile": percentile,
        "spec_long_pct": spec_long_pct,
        "commercial_net": commercial_net,
        "interpretation": interpretation,
    }


def get_cot_observation() -> dict:
    """
    Orchestrate COT data fetch and analysis into a single summary.

    Returns
    -------
    dict
        Keys: status ('ok', 'cached', or 'error'), plus all fields from
        analyze_positioning() on success.
    """
    try:
        from_cache = _cache_is_fresh(24)
        df = fetch_cot_data(cache_hours=24)

        if df.empty:
            return {
                "status": "error",
                "error": "Could not fetch or parse CFTC COT data for coffee",
            }

        result = analyze_positioning(df)

        if "error" in result:
            return {"status": "error", **result}

        result["status"] = "cached" if from_cache else "ok"
        result["num_weeks"] = len(df)
        return result

    except Exception as exc:
        return {
            "status": "error",
            "error": f"COT observation failed: {exc}",
        }


if __name__ == "__main__":
    obs = get_cot_observation()
    print("=== CFTC COT Coffee Positioning ===\n")

    if obs.get("status") == "error":
        print(f"Error: {obs.get('error', 'unknown')}")
    else:
        print(f"Status:                {obs['status']}")
        print(f"Report date:           {obs['report_date']}")
        print(f"Weeks of data:         {obs['num_weeks']}")
        print(f"Open interest:         {obs['open_interest']:,}")
        print(f"Net speculative:       {obs['net_speculative']:,}")
        print(f"  Week-over-week chg:  {obs['net_speculative_change']:+,}")
        print(f"  52-wk percentile:    {obs['net_spec_percentile']:.1f}%")
        print(f"Spec long % of OI:     {obs['spec_long_pct']:.1f}%")
        print(f"Commercial net:        {obs['commercial_net']:,}")
        print(f"\nInterpretation: {obs['interpretation']}")
