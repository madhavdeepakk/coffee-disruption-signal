"""
BRL/USD exchange rate ingestion via Yahoo Finance (yfinance).

The Brazilian real is a key macro driver for coffee prices: Brazil is the
world's largest coffee producer, and a weaker real (higher BRL/USD) makes
Brazilian exports cheaper in dollar terms, which tends to put downward
pressure on global coffee benchmarks. Conversely, a stronger real raises
the effective cost of Brazilian coffee and can support prices.

This module fetches the BRL=X (BRLUSD) cross rate and computes summary
statistics that feed into the multi-factor analysis pipeline.

Usage:
    python -m src.data.currency
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE_FILE = RAW_DIR / "brl_usd.csv"
CACHE_TTL_HOURS = 24


def _cache_is_fresh() -> bool:
    """Return True if the cached CSV exists and is less than CACHE_TTL_HOURS old."""
    if not CACHE_FILE.exists():
        return False
    mtime = dt.datetime.fromtimestamp(CACHE_FILE.stat().st_mtime, tz=dt.timezone.utc)
    age = dt.datetime.now(dt.timezone.utc) - mtime
    return age.total_seconds() < CACHE_TTL_HOURS * 3600


def fetch_brl_usd(period: str = "1y") -> pd.DataFrame:
    """
    Fetch BRL/USD exchange rate history from Yahoo Finance.

    Parameters
    ----------
    period : str
        yfinance period string, e.g. "1y", "6mo", "2y".

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime), rate (float, BRL per USD close).
    """
    if _cache_is_fresh():
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        return df

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for currency data. Install with: pip install yfinance"
        ) from exc

    ticker = yf.Ticker("BRL=X")
    hist = ticker.history(period=period)

    if hist.empty:
        raise ValueError(
            f"No data returned for BRL=X with period={period}. "
            "Yahoo Finance may be temporarily unavailable."
        )

    df = (
        hist[["Close"]]
        .reset_index()
        .rename(columns={"Date": "date", "Close": "rate"})
    )
    # Normalize timezone-aware index to plain dates
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)

    return df


def analyze_currency(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics from a BRL/USD rate DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``date`` and ``rate``.

    Returns
    -------
    dict
        Keys: current_rate, change_1w_pct, change_1m_pct, percentile_1y,
        high_1y, low_1y, interpretation.
    """
    df = df.sort_values("date").reset_index(drop=True)
    current_rate = float(df["rate"].iloc[-1])
    latest_date = df["date"].iloc[-1]

    # Week-over-week
    week_ago = latest_date - pd.Timedelta(days=7)
    mask_week = df["date"] <= week_ago
    if mask_week.any():
        rate_1w = float(df.loc[mask_week, "rate"].iloc[-1])
        change_1w_pct = round((current_rate - rate_1w) / rate_1w * 100, 2)
    else:
        change_1w_pct = None

    # Month-over-month
    month_ago = latest_date - pd.Timedelta(days=30)
    mask_month = df["date"] <= month_ago
    if mask_month.any():
        rate_1m = float(df.loc[mask_month, "rate"].iloc[-1])
        change_1m_pct = round((current_rate - rate_1m) / rate_1m * 100, 2)
    else:
        change_1m_pct = None

    # 1-year range and percentile
    high_1y = float(df["rate"].max())
    low_1y = float(df["rate"].min())
    rate_range = high_1y - low_1y
    if rate_range > 0:
        percentile_1y = round((current_rate - low_1y) / rate_range * 100, 1)
    else:
        percentile_1y = 50.0

    # Interpretation
    interpretation = _build_interpretation(
        current_rate, change_1m_pct, percentile_1y
    )

    return {
        "current_rate": round(current_rate, 4),
        "change_1w_pct": change_1w_pct,
        "change_1m_pct": change_1m_pct,
        "percentile_1y": percentile_1y,
        "high_1y": round(high_1y, 4),
        "low_1y": round(low_1y, 4),
        "interpretation": interpretation,
    }


def _build_interpretation(
    current_rate: float,
    change_1m_pct: float | None,
    percentile_1y: float,
) -> str:
    """
    Generate a human-readable interpretation of the BRL/USD rate for the
    coffee price context.
    """
    parts = []

    # Describe the level relative to the 1-year range
    if percentile_1y >= 80:
        parts.append(
            f"The real is near its weakest level in the past year "
            f"(BRL/USD at {current_rate:.2f}, {percentile_1y:.0f}th percentile)."
        )
        parts.append(
            "A weak real makes Brazilian coffee exports cheaper in dollar terms, "
            "which tends to put downward pressure on global coffee prices."
        )
    elif percentile_1y <= 20:
        parts.append(
            f"The real is near its strongest level in the past year "
            f"(BRL/USD at {current_rate:.2f}, {percentile_1y:.0f}th percentile)."
        )
        parts.append(
            "A strong real raises the effective cost of Brazilian exports, "
            "which can support or push coffee prices higher."
        )
    else:
        parts.append(
            f"The real is trading mid-range for the past year "
            f"(BRL/USD at {current_rate:.2f}, {percentile_1y:.0f}th percentile)."
        )
        parts.append(
            "Currency is not an extreme factor in either direction right now."
        )

    # Describe recent momentum
    if change_1m_pct is not None:
        if change_1m_pct > 2:
            parts.append(
                f"The real has weakened {change_1m_pct:.1f}% over the past month, "
                "increasing headwinds for coffee prices."
            )
        elif change_1m_pct < -2:
            parts.append(
                f"The real has strengthened {abs(change_1m_pct):.1f}% over the past month, "
                "a modest tailwind for coffee prices."
            )

    return " ".join(parts)


def get_currency_observation() -> dict:
    """
    Orchestrate fetch + analysis. Returns a summary dict suitable for
    inclusion in a multi-factor report.

    Returns
    -------
    dict
        Keys: status ("ok" or "error"), plus analysis fields on success
        or an error_message on failure.
    """
    try:
        df = fetch_brl_usd(period="1y")
        analysis = analyze_currency(df)
        return {"status": "ok", **analysis}
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Currency data unavailable: {exc}",
        }


if __name__ == "__main__":
    result = get_currency_observation()
    if result["status"] == "ok":
        print("=== BRL/USD Exchange Rate Summary ===")
        print(f"  Current rate:      {result['current_rate']:.4f} BRL per USD")
        print(f"  1-week change:     {result['change_1w_pct']}%")
        print(f"  1-month change:    {result['change_1m_pct']}%")
        print(f"  1-year percentile: {result['percentile_1y']}%")
        print(f"  1-year range:      {result['low_1y']:.4f} - {result['high_1y']:.4f}")
        print(f"\n  {result['interpretation']}")
    else:
        print(f"Error: {result['error_message']}")
