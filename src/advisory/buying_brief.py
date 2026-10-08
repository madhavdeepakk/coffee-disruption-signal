"""
Coffee market intelligence briefing.

Reads every data source the system has — price feeds, satellite weather,
weather forecasts, speculative positioning, currency, news, seasonal
calendar, historical analogs — and writes a layered situation report.

Layer 1: Headline — the top 2-3 things that matter right now.
Layer 2: Full situation by category — supply side, demand/market, events.
Layer 3: Context — historical analogs, seasonal calendar, background.

No buy/sell signals. No predictions. No scores. Just: here is what the
data says, structured so a procurement analyst can make their own call.

Usage:
    python -m src.advisory.buying_brief
    python -m src.advisory.buying_brief --json
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, date as date_type, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """One factual observation from a data source."""
    category: str       # "supply", "demand", "events", "context"
    topic: str          # e.g. "Weather (observed)", "Speculative positioning"
    text: str           # short plain-English summary (shown by default)
    source: str         # data attribution
    importance: int     # 1 = headline-worthy, 2 = notable, 3 = background
    data: dict = field(default_factory=dict)
    detail: str = ""    # longer analysis (shown on expand)


@dataclass
class MarketBrief:
    as_of_date: str
    current_price: float
    observations: list   # list[Observation]
    generated_at: str = ""
    outlook: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    @property
    def headline_obs(self):
        return sorted([o for o in self.observations if o.importance == 1],
                      key=lambda o: o.category)

    @property
    def supply_obs(self):
        return [o for o in self.observations if o.category == "supply"]

    @property
    def demand_obs(self):
        return [o for o in self.observations if o.category == "demand"]

    @property
    def event_obs(self):
        return [o for o in self.observations if o.category == "events"]

    @property
    def context_obs(self):
        return [o for o in self.observations if o.category == "context"]

    def to_dict(self) -> dict:
        return {
            "as_of_date": self.as_of_date,
            "current_price": self.current_price,
            "generated_at": self.generated_at,
            "headline": [_obs_dict(o) for o in self.headline_obs],
            "supply": [_obs_dict(o) for o in self.supply_obs],
            "demand": [_obs_dict(o) for o in self.demand_obs],
            "events": [_obs_dict(o) for o in self.event_obs],
            "context": [_obs_dict(o) for o in self.context_obs],
        }


def _obs_dict(o: Observation) -> dict:
    d = {"topic": o.topic, "text": o.text, "source": o.source,
         "importance": o.importance, "data": o.data}
    if o.detail:
        d["detail"] = o.detail
    return d


# ---------------------------------------------------------------------------
# Data loaders (shared)
# ---------------------------------------------------------------------------

def _load_prices() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "raw" / "coffee_prices.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _refresh_unified_if_stale(max_age_hours: float = 6.0) -> None:
    """Re-fetch NASA POWER weather and rebuild the unified table when stale.

    Staleness check: if the unified CSV is missing or its last row is more
    than *max_age_hours* old, we pull fresh NASA POWER data and rebuild.
    Failures are silently ignored so the dashboard still works from cache.
    """
    import time as _time
    path = REPO_ROOT / "data" / "processed" / "unified_daily.csv"

    needs_refresh = False
    if not path.exists():
        needs_refresh = True
    else:
        age_hours = (_time.time() - path.stat().st_mtime) / 3600
        if age_hours > max_age_hours:
            needs_refresh = True

    if not needs_refresh:
        return

    try:
        from src.data.nasa_power import (
            fetch_power_data, reshape_to_dataframe,
            DEFAULT_LATITUDE, DEFAULT_LONGITUDE, OUTPUT_PATH as WEATHER_PATH,
        )
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        raw = fetch_power_data(
            DEFAULT_LATITUDE, DEFAULT_LONGITUDE,
            "2018-01-01", today_str, verbose=False,
        )
        weather_df = reshape_to_dataframe(raw)
        WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
        weather_df.to_csv(WEATHER_PATH, index=False)

        from scripts.build_unified_table import build_unified, OUTPUT_PATH as UNI_PATH
        unified = build_unified(refresh_weather=False)
        UNI_PATH.parent.mkdir(parents=True, exist_ok=True)
        unified.to_csv(UNI_PATH, index=False)
    except Exception:
        pass  # fall back to whatever file already exists


def _load_unified() -> pd.DataFrame:
    _refresh_unified_if_stale()
    path = REPO_ROOT / "data" / "processed" / "unified_daily.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ===================================================================
# SUPPLY SIDE observations
# ===================================================================

def observe_weather_observed(unified: pd.DataFrame) -> list[Observation]:
    """Current weather conditions in Brazil's coffee belt (satellite data)."""
    if unified.empty or "dryness_zscore" not in unified.columns:
        return []

    recent = unified.dropna(subset=["dryness_zscore"]).tail(30)
    if recent.empty:
        return []

    latest = recent.iloc[-1]
    dryness = latest["dryness_zscore"]
    frost_count = int(latest.get("frost_7d_count", 0) or 0)
    prcp_30d = latest.get("prcp_30d_sum", None)
    as_of = pd.to_datetime(latest["date"]).strftime("%d %b %Y")

    # Contextualize with seasonal calendar
    try:
        from src.data.seasonal import contextualize_weather
        ref_date = pd.to_datetime(latest["date"]).date()
        seasonal_context = contextualize_weather(dryness, frost_count, ref_date)
    except Exception:
        seasonal_context = None

    obs = []

    # Determine importance: drought/frost during critical phase = headline
    importance = 3  # background by default
    if frost_count > 0:
        importance = 1  # frost is always headline
    elif dryness > 1.5:
        importance = 1  # significant drought
    elif abs(dryness) > 0.5:
        importance = 2  # notable

    # Short summary (shown by default)
    if dryness > 1.5:
        text = "Significant drought stress in Brazil's coffee belt."
    elif dryness > 0.5:
        text = "Drier than normal conditions in Brazil's coffee belt."
    elif dryness < -1.0:
        text = "Rainfall in Brazil's coffee belt is well above average."
    else:
        text = "Weather in Brazil's coffee belt is near normal."

    # Detailed analysis (shown on expand)
    detail_parts = []
    if dryness > 1.5:
        detail_parts.append(
            f"Rainfall has been well below normal (dryness index: {dryness:+.1f} "
            f"standard deviations from average). This level of dryness can stress "
            f"coffee plants, especially during flowering or fruit development.")
    elif dryness > 0.5:
        detail_parts.append(
            f"Rainfall is somewhat below the seasonal average (dryness index: "
            f"{dryness:+.1f} standard deviations). Not yet at critical levels "
            f"but worth monitoring.")
    elif dryness < -1.0:
        detail_parts.append(
            f"Rainfall is well above the seasonal average (dryness index: "
            f"{dryness:+.1f} standard deviations). Good moisture levels for "
            f"the coffee crop.")
    else:
        detail_parts.append(
            f"Rainfall is close to the seasonal average (dryness index: "
            f"{dryness:+.1f} standard deviations). No weather concerns.")

    if prcp_30d is not None:
        detail_parts.append(f"Total rainfall over the past 30 days: {prcp_30d:.0f} mm.")

    if frost_count > 0:
        detail_parts.append(
            f"Frost detected: {frost_count} day(s) with freezing temperatures "
            f"in the past week. Frost is one of the biggest risks to coffee "
            f"production and can destroy entire harvests.")

    if seasonal_context:
        detail_parts.append(seasonal_context)

    detail = " ".join(detail_parts)

    obs.append(Observation(
        "supply", "Weather (observed)", text,
        f"NASA POWER satellite ({as_of})",
        importance,
        {"dryness_zscore": round(dryness, 2), "frost_7d": frost_count,
         "prcp_30d_mm": round(prcp_30d, 1) if prcp_30d is not None else None},
        detail=detail,
    ))

    if frost_count > 0:
        obs.append(Observation(
            "supply", "Frost alert",
            f"Freezing temperatures recorded on {frost_count} day(s) this past week.",
            "NASA POWER satellite",
            1,
            {"frost_7d_count": frost_count},
            detail=(
                f"Satellite data shows {frost_count} day(s) with freezing or near-freezing "
                f"temperatures in Brazil's main coffee-growing region (Minas Gerais). "
                f"Frost is the single biggest weather risk for coffee. A severe frost event "
                f"can destroy an entire season's crop and send prices sharply higher. "
                f"The famous 1975 'Black Frost' wiped out most of Brazil's crop and tripled "
                f"world coffee prices. Even a mild frost can damage flowering and reduce yields."
            ),
        ))

    return obs


def observe_weather_forecast() -> list[Observation]:
    """16-day weather forecast for Brazil's coffee belt."""
    try:
        from src.data.weather_forecast import get_forecast_observation
        fc = get_forecast_observation()
    except Exception as exc:
        return [Observation(
            "supply", "Weather (forecast)",
            f"Weather forecast unavailable ({type(exc).__name__}).",
            "Open-Meteo API", 3,
        )]

    if fc.get("status") == "error":
        reason = fc.get("reason", "unknown error")
        # Truncate long HTTP error messages for readability
        if len(reason) > 80:
            reason = "API unreachable (network or proxy error)"
        return [Observation(
            "supply", "Weather (forecast)",
            f"Weather forecast unavailable: {reason}.",
            "Open-Meteo API", 3,
        )]

    summary = fc.get("summary", {})
    if not summary:
        return []

    precip_7d = summary.get("total_precip_7d", 0)
    precip_14d = summary.get("total_precip_14d", 0)
    min_temp = summary.get("min_temp_7d", 10)
    frost_7d = summary.get("frost_days_7d", 0)
    frost_14d = summary.get("frost_days_14d", 0)
    dry_days = summary.get("dry_days_7d", 0)

    importance = 3

    # Short summary
    if frost_7d > 0 or frost_14d > 0:
        importance = 1
        text = f"Frost risk ahead: freezing temperatures forecast in the next {7 if frost_7d else 14} days."
    elif min_temp < 5:
        importance = 2
        text = "Cold weather ahead. Temperatures could drop close to freezing."
    elif precip_7d < 5 and dry_days >= 5:
        importance = 2
        text = "Very dry conditions expected over the coming week."
    elif precip_7d > 50:
        text = "Heavy rainfall expected this week."
    else:
        text = "Normal weather expected over the coming weeks."

    # Detailed analysis
    detail_parts = []
    if frost_7d > 0 or frost_14d > 0:
        detail_parts.append(
            f"The 16-day forecast shows {frost_7d} day(s) in the next week and "
            f"{frost_14d} day(s) in the next two weeks where overnight lows could "
            f"drop below 2°C in Brazil's coffee belt. The coldest forecast night "
            f"is {min_temp:.1f}°C. Frost during the growing season can severely "
            f"damage coffee plants and typically triggers sharp price increases."
        )
    elif min_temp < 5:
        detail_parts.append(
            f"The lowest overnight temperature forecast in the next week is "
            f"{min_temp:.1f}°C. While not a frost event, temperatures this close "
            f"to freezing warrant close monitoring, as conditions can shift quickly."
        )

    detail_parts.append(
        f"Expected rainfall: {precip_7d:.0f} mm over the next 7 days, "
        f"{precip_14d:.0f} mm over 14 days. "
        f"Number of dry days in the next week: {dry_days} out of 7."
    )

    if precip_7d < 5 and dry_days >= 5:
        detail_parts.append(
            "Very little rain is expected, which could worsen any existing "
            "drought stress on the coffee crop."
        )
    elif precip_7d > 50:
        detail_parts.append(
            "This is a significant amount of rain. While good for soil moisture, "
            "excessive rain during harvest can damage drying coffee cherries."
        )

    detail = " ".join(detail_parts)
    return [Observation(
        "supply", "Weather (forecast)", text,
        "Open-Meteo 16-day forecast (Minas Gerais)",
        importance,
        {"precip_7d_mm": round(precip_7d, 1), "precip_14d_mm": round(precip_14d, 1),
         "min_temp_7d": round(min_temp, 1), "frost_days_7d": frost_7d,
         "frost_days_14d": frost_14d, "dry_days_7d": dry_days},
        detail=detail,
    )]


def observe_supply_benchmark(unified: pd.DataFrame) -> list[Observation]:
    """Futures vs World Bank physical-market benchmark."""
    if unified.empty or "futures_wb_spread_pct" not in unified.columns:
        return []

    recent = unified.dropna(subset=["futures_wb_spread_pct"]).tail(5)
    if recent.empty:
        return []

    latest = recent.iloc[-1]
    spread = latest["futures_wb_spread_pct"]
    wb_price = latest.get("wb_price_cents_lb", None)
    futures_price = latest.get("close_usd", None)

    importance = 2 if abs(spread) > 10 else 3

    # Short summary
    if abs(spread) < 5:
        text = f"Exchange-traded and physical-market prices are closely aligned."
    elif spread > 0:
        text = f"Exchange-traded price is running {spread:.0f}% above the physical-market reference."
    else:
        text = f"Exchange-traded price is running {abs(spread):.0f}% below the physical-market reference."

    # Detail
    detail = (
        f"The futures price on the exchange ({futures_price:.0f} cents/lb) vs the "
        f"World Bank's physical-market benchmark ({wb_price:.0f} cents/lb), "
        f"a spread of {spread:+.0f}%. "
    )
    if abs(spread) > 10:
        detail += (
            "A large gap between these two prices can signal that the exchange "
            "is pricing in supply risks or demand shifts that haven't yet shown up "
            "in physical transactions. "
        )
    else:
        detail += (
            "The two prices being close together suggests the market is in "
            "a relatively stable state with no major disconnect between "
            "traded expectations and physical supply. "
        )
    detail += (
        "Note: the World Bank price is published monthly with about a one-month "
        "delay, so some gap is expected."
    )

    return [Observation(
        "supply", "Futures vs physical benchmark", text,
        "World Bank Pink Sheet + Yahoo Finance",
        importance,
        {"spread_pct": round(spread, 1)},
        detail=detail,
    )]


# ===================================================================
# DEMAND / MARKET observations
# ===================================================================

def observe_price_trend(prices: pd.DataFrame) -> list[Observation]:
    """Current price, direction, range position."""
    if prices.empty or len(prices) < 50:
        return []

    obs = []
    latest = prices.iloc[-1]
    price = latest["price"]
    date = latest["date"]

    # Week and month returns
    week_ret = (price / prices.iloc[-6]["price"] - 1) * 100 if len(prices) >= 6 else 0
    month_ret = (price / prices.iloc[-21]["price"] - 1) * 100 if len(prices) >= 21 else 0

    # 1-year range
    one_year = prices[prices["date"] >= date - pd.Timedelta(days=365)]
    hi = one_year["price"].max()
    lo = one_year["price"].min()
    pct = (one_year["price"] < price).mean() * 100

    # Importance: big moves are notable
    importance = 2 if abs(week_ret) > 5 else 3

    direction_w = "up" if week_ret > 0.5 else "down" if week_ret < -0.5 else "roughly flat"
    direction_m = "risen" if month_ret > 1 else "fallen" if month_ret < -1 else "held steady"

    # Translate percentile into plain position language
    if pct >= 80:
        range_desc = "near the top"
    elif pct >= 60:
        range_desc = "in the upper half"
    elif pct >= 40:
        range_desc = "in the middle"
    elif pct >= 20:
        range_desc = "in the lower half"
    else:
        range_desc = "near the bottom"

    # Short summary
    text = (f"Coffee is at {price:.2f} cents/lb, {direction_w} {abs(week_ret):.1f}% "
            f"this week and {range_desc} of its 1-year range.")

    # Detail
    detail = (
        f"Current price: {price:.2f} US cents per pound. "
        f"Weekly change: {week_ret:+.1f}%. Monthly change: {month_ret:+.1f}%. "
        f"Over the past year, coffee has traded between {lo:.0f} and {hi:.0f} "
        f"cents/lb. The current price sits at the {pct:.0f}th percentile of "
        f"that range, meaning it has been lower than this {pct:.0f}% of the "
        f"time over the past year."
    )

    obs.append(Observation(
        "demand", "Price trend", text,
        "Yahoo Finance (KC=F)",
        importance,
        {"price": round(price, 2), "week_ret_pct": round(week_ret, 1),
         "month_ret_pct": round(month_ret, 1), "pct_1y": round(pct, 1),
         "high_1y": round(hi, 2), "low_1y": round(lo, 2)},
        detail=detail,
    ))

    # Volatility
    rets = prices["price"].pct_change().dropna()
    recent_vol = rets.tail(20).std() * np.sqrt(252) * 100
    hist_vol = rets.tail(252).std() * np.sqrt(252) * 100
    if hist_vol > 0:
        ratio = recent_vol / hist_vol
        if ratio > 1.5:
            vol_text = "The market has been unusually jumpy lately."
            vol_detail = (
                f"Daily price swings over the past month are about {ratio:.1f}x "
                f"larger than the past year's average. In numbers: recent "
                f"annualised volatility is {recent_vol:.0f}% vs the 1-year "
                f"average of {hist_vol:.0f}%. High volatility often reflects "
                f"uncertainty. Traders are unsure about the near-term direction, "
                f"which can lead to larger-than-normal price moves in either direction."
            )
            vol_imp = 2
        elif ratio < 0.7:
            vol_text = "The market has been calm, with smaller swings than usual."
            vol_detail = (
                f"Daily price swings are well below the 1-year average "
                f"({recent_vol:.0f}% vs {hist_vol:.0f}% annualised). Low "
                f"volatility suggests the market is in a settled period with "
                f"no major surprises moving prices."
            )
            vol_imp = 3
        else:
            vol_text = "Day-to-day price swings are in the normal range."
            vol_detail = (
                f"Recent volatility ({recent_vol:.0f}% annualised) is in line "
                f"with the 1-year average ({hist_vol:.0f}%). Nothing unusual about "
                f"the size of daily price movements."
            )
            vol_imp = 3

        obs.append(Observation(
            "demand", "Volatility", vol_text,
            "Yahoo Finance (KC=F)", vol_imp,
            {"recent_vol_pct": round(recent_vol, 1), "ratio": round(ratio, 2)},
            detail=vol_detail,
        ))

    return obs


def observe_cftc_positioning() -> list[Observation]:
    """CFTC speculative positioning in coffee futures."""
    try:
        from src.data.cftc import get_cot_observation
        cot = get_cot_observation()
    except Exception as exc:
        return [Observation(
            "demand", "Speculative positioning",
            f"CFTC data unavailable ({type(exc).__name__}).",
            "CFTC Commitment of Traders", 3,
        )]

    if cot.get("status") == "error":
        reason = cot.get("error", "fetch failed")
        if len(reason) > 80:
            reason = "API unreachable (network or proxy error)"
        return [Observation(
            "demand", "Speculative positioning",
            f"CFTC data unavailable: {reason}.",
            "CFTC Commitment of Traders", 3,
        )]

    # CFTC returns flat dict, not nested under "analysis"
    net_spec = cot.get("net_speculative", 0)
    pct = cot.get("net_spec_percentile", 50)
    change = cot.get("net_speculative_change", 0)
    report_date = cot.get("report_date", "unknown")
    interp = cot.get("interpretation", "")

    importance = 1 if (pct > 85 or pct < 15) else 2 if (pct > 70 or pct < 30) else 3

    direction = "betting on higher prices" if net_spec > 0 else "betting on lower prices"
    if pct > 85:
        crowd_short = "an unusually large bullish bet"
    elif pct > 70:
        crowd_short = "a bigger-than-average bullish bet"
    elif pct < 15:
        crowd_short = "an unusually bearish stance"
    elif pct < 30:
        crowd_short = "a more bearish stance than usual"
    else:
        crowd_short = "a roughly average position"

    # Short summary
    text = f"Large traders are {direction}, {crowd_short}."

    # Detail
    detail_parts = [
        f"The CFTC (Commodity Futures Trading Commission) publishes weekly data "
        f"on how large traders are positioned in futures markets. ",
        f"Current net position: {net_spec:+,.0f} contracts "
        f"({'long/bullish' if net_spec > 0 else 'short/bearish'}). "
        f"This is at the {pct:.0f}th percentile of the past year's range, ",
    ]
    if pct > 85:
        detail_parts.append(
            "meaning traders are more bullish than they have been almost all year. "
            "Extreme positioning can be a contrarian signal: when everyone is already "
            "long, there may be fewer new buyers to push prices higher."
        )
    elif pct < 15:
        detail_parts.append(
            "meaning traders are more bearish than they have been almost all year. "
            "Extreme short positioning can set up a 'short squeeze' if prices "
            "start rising and shorts rush to cover."
        )
    else:
        detail_parts.append("within the normal range for the past year.")

    if abs(change) > 0:
        change_dir = "increased their bullish" if change > 0 else "reduced their"
        detail_parts.append(
            f" Week-over-week, traders {change_dir} position by "
            f"{abs(change):,.0f} contracts."
        )
    if interp:
        detail_parts.append(f" {interp}")

    detail = "".join(detail_parts)

    return [Observation(
        "demand", "Speculative positioning", text,
        f"CFTC Commitment of Traders ({report_date})",
        importance,
        {"net_speculative": net_spec, "percentile": round(pct, 1),
         "change_wow": change},
        detail=detail,
    )]


def observe_currency() -> list[Observation]:
    """BRL/USD exchange rate and its implications for Brazilian exports."""
    try:
        from src.data.currency import get_currency_observation
        fx = get_currency_observation()
    except Exception as exc:
        return [Observation(
            "demand", "BRL/USD exchange rate",
            f"Currency data unavailable ({type(exc).__name__}).",
            "Yahoo Finance (BRL=X)", 3,
        )]

    if fx.get("status") == "error":
        reason = fx.get("error_message", "fetch failed")
        if len(reason) > 80:
            reason = "API unreachable (network or proxy error)"
        return [Observation(
            "demand", "BRL/USD exchange rate",
            f"Currency data unavailable: {reason}.",
            "Yahoo Finance (BRL=X)", 3,
        )]

    # Currency returns flat dict, not nested under "analysis"
    rate = fx.get("current_rate", 0)
    change_1m = fx.get("change_1m_pct", 0)
    pct = fx.get("percentile_1y", 50)
    interp = fx.get("interpretation", "")

    importance = 2 if abs(change_1m) > 5 else 3

    # Plain-language range position
    if pct >= 75:
        range_note = "The real is relatively weak compared to the past year"
    elif pct <= 25:
        range_note = "The real is relatively strong compared to the past year"
    else:
        range_note = "The real is in its mid-range for the past year"

    if abs(change_1m) < 1:
        change_note = "little changed over the past month"
    elif change_1m > 0:
        change_note = f"the real has weakened {abs(change_1m):.1f}% over the past month"
    else:
        change_note = f"the real has strengthened {abs(change_1m):.1f}% over the past month"

    # Short summary
    text = f"Brazilian real at {rate:.2f} per dollar; {change_note}."

    # Detail
    detail = (
        f"The Brazilian real is at {rate:.2f} per US dollar. "
        f"Over the past month it has moved {change_1m:+.1f}%. "
        f"{range_note} (at the {pct:.0f}th percentile of its 1-year range). "
        f"Why this matters for coffee: Brazil is the world's largest coffee "
        f"producer. When the real weakens against the dollar, Brazilian farmers "
        f"earn more in local currency for their coffee, which encourages them "
        f"to sell more, increasing global supply and putting downward pressure "
        f"on prices. Conversely, a stronger real discourages selling."
    )
    if interp:
        detail += f" {interp}"

    return [Observation(
        "demand", "BRL/USD exchange rate", text,
        "Yahoo Finance (BRL=X)",
        importance,
        {"rate": round(rate, 4), "change_1m_pct": round(change_1m, 1),
         "percentile_1y": round(pct, 1)},
        detail=detail,
    )]


# ===================================================================
# EVENT observations
# ===================================================================

def observe_anomalies(prices: pd.DataFrame) -> list[Observation]:
    """Unusual price moves in the past 7 trading days."""
    if prices.empty or len(prices) < 80:
        return []
    try:
        from src.modeling.anomaly_detector import compute_anomalies
        anom = compute_anomalies(prices[["date", "price"]].copy())
        anom["date"] = pd.to_datetime(anom["date"])
    except Exception:
        return []

    recent = anom.tail(7)
    flagged = recent[recent["anomaly_flag"] == True]  # noqa: E712

    if len(flagged) == 0:
        return [Observation(
            "events", "Unusual moves",
            "No unusual price moves in the past 7 trading days.",
            "Statistical anomaly detection", 3,
            detail="The system checks each day's price change against its recent "
                   "history (75-day rolling window). A move is flagged when it "
                   "exceeds 2 standard deviations from the recent average, roughly "
                   "the kind of move you would expect less than 5% of the time.",
        )]

    parts = []
    detail_parts = []
    for _, row in flagged.iterrows():
        d = row["date"].strftime("%d %b")
        direction = "jump up" if row["z_score"] > 0 else "drop"
        magnitude = abs(row["z_score"])
        if magnitude > 3:
            size = "very large"
        elif magnitude > 2.5:
            size = "large"
        else:
            size = "notable"
        parts.append(f"{size} {direction} on {d}")
        detail_parts.append(
            f"{d}: {row['z_score']:+.1f} standard deviations from the 75-day "
            f"average daily change (price: {row['price']:.2f} cents/lb)"
        )

    text = f"{len(flagged)} unusual move(s) in the past week."

    detail = (
        f"The system compares each day's price change to the past 75 trading "
        f"days. Moves larger than 2 standard deviations are flagged as unusual; "
        f"this happens less than 5% of the time in normal markets.\n\n"
        f"Flagged days: {'; '.join(detail_parts)}."
    )

    return [Observation(
        "events", "Unusual moves", text,
        "Statistical anomaly detection",
        1 if any(abs(row["z_score"]) > 3 for _, row in flagged.iterrows()) else 2,
        {"n_flagged": len(flagged),
         "dates": flagged["date"].dt.strftime("%Y-%m-%d").tolist(),
         "z_scores": [round(z, 2) for z in flagged["z_score"].tolist()]},
        detail=detail,
    )]


def observe_news(commodity_key: str = "coffee") -> list[Observation]:
    """Recent headline context from GDELT."""
    try:
        from src.modeling.forecast import news_direction_signal
        prices = _load_prices()
        if prices.empty:
            return []
        latest_date = prices.iloc[-1]["date"].strftime("%Y-%m-%d")
        news = news_direction_signal(commodity_key, latest_date,
                                     window_days=14, allow_live=False)
    except Exception:
        return []

    if not news.get("available", False):
        return []

    n_up = news.get("n_up", 0)
    n_down = news.get("n_down", 0)
    n_total = news.get("n_scored", 0)
    lean = news.get("lean", "neutral")

    if lean == "up":
        text = "Recent news coverage leans toward rising prices or supply worries."
        importance = 2
    elif lean == "down":
        text = "Recent news coverage leans toward easing prices or improving supply."
        importance = 2
    else:
        text = "Recent news coverage is mixed, with no clear direction."
        importance = 3

    detail = (
        f"The system scanned {n_total} news articles from the past 2 weeks "
        f"and scored each for whether it suggests rising or falling prices. "
        f"Result: {n_up} articles leaning bullish (prices up), "
        f"{n_down} leaning bearish (prices down). "
        f"Overall lean: {lean}. "
        f"This is a rough directional indicator based on keyword analysis. "
        f"It captures broad sentiment but not the nuance of individual stories."
    )

    return [Observation(
        "events", "News", text,
        "GDELT news archive",
        importance,
        {"n_up": n_up, "n_down": n_down, "lean": lean},
        detail=detail,
    )]


# ===================================================================
# CONTEXT observations
# ===================================================================

def observe_seasonal_context() -> list[Observation]:
    """Where we are in the coffee growing calendar."""
    try:
        from src.data.seasonal import current_season
        season = current_season()
    except Exception:
        return []

    obs = []

    # Short summary for calendar
    text = f"Brazil: {season['brazil_phase']}. Vietnam: {season['vietnam_phase']}."

    detail = (
        f"Brazil (world's largest producer): {season['brazil_phase']}. "
        f"{season['brazil_note']} "
        f"Vietnam (second-largest, mainly robusta): {season['vietnam_phase']}. "
        f"Colombia (third-largest, high-quality arabica): {season['colombia_phase']}. "
        f"The coffee growing cycle drives seasonal price patterns. Harvest periods "
        f"tend to increase supply and pressure prices, while growing and flowering "
        f"periods are when weather risks are highest."
    )

    obs.append(Observation(
        "context", "Seasonal calendar", text,
        "Coffee agricultural calendar",
        2,
        {"brazil_phase": season["brazil_phase"],
         "vietnam_phase": season["vietnam_phase"],
         "colombia_phase": season["colombia_phase"]},
        detail=detail,
    ))

    obs.append(Observation(
        "context", "Key seasonal risk",
        season["key_risk"],
        "Coffee agricultural calendar",
        2,
        detail=(
            f"Each time of year carries different risks for coffee production. "
            f"The current key risk: {season['key_risk']} "
            f"Understanding where we are in the crop cycle helps anticipate "
            f"which types of disruptions could affect supply."
        ),
    ))

    if "transitioning" in season.get("upcoming", "").lower():
        obs.append(Observation(
            "context", "Upcoming transitions",
            season["upcoming"],
            "Coffee agricultural calendar",
            3,
            detail=(
                f"What is coming next in the growing calendar: {season['upcoming']} "
                f"Transitions between crop phases often shift which risks matter "
                f"most. For example, moving from harvest into flowering means "
                f"weather risk increases."
            ),
        ))

    return obs


def observe_historical_analogs(prices: pd.DataFrame) -> list[Observation]:
    """What happened after similar conditions in the past."""
    try:
        from src.modeling.outlook import build_outlook_frame, analog_scenarios
        feat, cols, _ = build_outlook_frame(prices, use_weather=True)
        last_pos = int(np.where(~feat[cols].isna().any(axis=1).to_numpy())[0][-1])
        week = analog_scenarios(feat, cols, last_pos, horizon=5, n_analogs=25)
        month = analog_scenarios(feat, cols, last_pos, horizon=20, n_analogs=25)
    except Exception:
        return []

    if "error" in week or "error" in month:
        return []

    share_up_w = week['share_up'] * 100
    share_up_m = month.get('share_up', 0.5) * 100

    week_dir = "rose" if week['median_pct'] > 0 else "fell"
    month_dir = "rose" if month['median_pct'] > 0 else "fell"

    # Short summary
    text = (
        f"In similar past situations, the price typically {week_dir} over "
        f"the following week and {month_dir} over the following month."
    )

    # Detail
    detail = (
        f"The system found {week['n_analogs']} past trading days with similar "
        f"conditions (price trend, volatility, and weather patterns). Here is "
        f"what happened next in those historical cases:\n\n"
        f"Next week: the price moved a median of {week['median_pct']:+.1f}% "
        f"(range: {week['p10_pct']:+.1f}% to {week['p90_pct']:+.1f}%). "
        f"It went up {share_up_w:.0f}% of the time.\n\n"
        f"Next month: the price moved a median of {month['median_pct']:+.1f}% "
        f"(range: {month['p10_pct']:+.1f}% to {month['p90_pct']:+.1f}%). "
        f"It went up {share_up_m:.0f}% of the time.\n\n"
        f"Past patterns do not guarantee the same result this time, but they "
        f"show the realistic range of outcomes from a similar starting point."
    )

    return [Observation(
        "context", "Historical analogs", text,
        "Historical pattern matching (price and weather)",
        3,
        {"week": week, "month": month},
        detail=detail,
    )]


# ===================================================================
# Headline generation
# ===================================================================

def _generate_headline(observations: list[Observation]) -> str:
    """Write a 1-2 sentence headline from the most important observations."""
    top = sorted(
        [o for o in observations if o.importance <= 2],
        key=lambda o: (o.importance, o.category),
    )

    if not top:
        return "Conditions are quiet, with no major signals across supply, demand or events."

    # Take the top 3 most important
    headline_parts = []
    for o in top[:3]:
        # Extract the key sentence — use regex to split on sentence-ending periods
        # (not decimal points like "312.60")
        sentences = re.split(r'(?<=[a-zA-Z\)%])\.\s', o.text)
        first_sentence = sentences[0].rstrip(".") + "."
        headline_parts.append(first_sentence)

    return " ".join(headline_parts)


# ===================================================================
# Price outlook (signal balance)
# ===================================================================

def _obs_data(observations: list, topic: str) -> dict:
    for o in observations:
        if o.topic == topic and o.data:
            return o.data
    return {}


def build_outlook(observations: list, current_price: float) -> dict:
    """Combine the current signals into one near-term price outlook.

    Each factor that has a known, direct link to coffee prices adds a vote
    between -2 (downward pressure) and +2 (upward pressure). The votes are
    summed into an overall lean, and the historical analogs supply a realistic
    price range. Short-term momentum is deliberately left out: our forecasting
    evaluation found it has no predictive value for direction.
    """
    drivers = []

    def add(factor, score, why, phrase=""):
        drivers.append({"factor": factor, "score": score, "why": why,
                        "phrase": phrase or factor.lower()})

    # Observed weather (NASA POWER)
    w = _obs_data(observations, "Weather (observed)")
    if w:
        dry, frost = w.get("dryness_zscore", 0) or 0, w.get("frost_7d", 0) or 0
        if frost > 0:
            add("Recent frost", 2, f"{frost} frost day(s) in Minas Gerais this past week can damage the crop.", "recent frost in Brazil")
        elif dry > 1.5:
            add("Drought", 2, "Rainfall in the coffee belt is far below normal, which threatens yields.", "drought in Brazil's coffee belt")
        elif dry > 0.5:
            add("Dry conditions", 1, "Rainfall is below normal in the coffee belt.", "dry weather in Brazil")
        elif dry < -1.0:
            add("Good rainfall", -1, "Above-average rain supports the next crop.", "good rainfall in Brazil")
        else:
            add("Weather (recent)", 0, "Rainfall in the coffee belt is close to normal.")

    # Weather forecast (Open-Meteo)
    f = _obs_data(observations, "Weather (forecast)")
    if f:
        if (f.get("frost_days_7d", 0) or 0) > 0 or (f.get("frost_days_14d", 0) or 0) > 0:
            add("Frost forecast", 2, "Freezing nights are forecast for the coffee belt.", "a frost forecast for Brazil")
        elif (f.get("min_temp_7d", 10) or 10) < 5:
            add("Cold forecast", 1, "Night temperatures are forecast close to freezing.", "a cold-weather forecast")
        elif (f.get("precip_7d_mm", 10) or 0) < 5 and (f.get("dry_days_7d", 0) or 0) >= 5:
            add("Dry forecast", 1, "Very little rain is expected over the next week.", "a dry forecast")
        else:
            add("Weather (forecast)", 0, "No frost or drought risk in the 16-day forecast.")

    # Brazilian real
    fx = _obs_data(observations, "BRL/USD exchange rate")
    if fx:
        ch = fx.get("change_1m_pct", 0) or 0
        if ch > 3:
            add("Weaker real", -1, f"The real fell {ch:.1f}% this month, which encourages Brazilian farmers to sell.", "a weaker Brazilian real")
        elif ch < -3:
            add("Stronger real", 1, f"The real rose {abs(ch):.1f}% this month, which makes Brazilian farmers hold back.", "a stronger Brazilian real")
        else:
            add("Brazilian real", 0, "The real is broadly stable against the dollar.")

    # Speculative positioning (CFTC)
    c = _obs_data(observations, "Speculative positioning")
    if c:
        pct = c.get("percentile", 50) or 50
        if pct > 85:
            add("Crowded long bets", -1, "Speculators are already heavily long, so there are few new buyers and a risk of selling.", "crowded speculative buying")
        elif pct < 15:
            add("Crowded short bets", 1, "Speculators are heavily short, so a rally could force them to buy back.", "heavy speculative short selling")
        else:
            add("Speculative positioning", 0, "Large traders hold an average-sized position.")

    # News tone (GDELT headlines)
    n = _obs_data(observations, "News")
    if n:
        lean = n.get("lean", "neutral")
        if lean == "up":
            add("News tone", 1, "Recent headlines lean toward supply worries and rising prices.", "worried news coverage")
        elif lean == "down":
            add("News tone", -1, "Recent headlines lean toward easing supply concerns.", "calmer news coverage")
        else:
            add("News tone", 0, "Recent headlines are mixed.")

    # Crop calendar
    sea = _obs_data(observations, "Seasonal calendar")
    if sea:
        phase = (sea.get("brazil_phase") or "").lower()
        if "harvest" in phase:
            add("Brazil harvest", -1, "New-crop coffee is coming to market, adding supply.", "the Brazilian harvest")
        elif "flowering" in phase:
            add("Brazil flowering", 0, "Flowering makes the crop sensitive to frost and dry spells, so weather news can move prices quickly.")
        else:
            add("Crop calendar", 0, sea.get("brazil_phase", ""))

    # Historical analogs
    an = _obs_data(observations, "Historical analogs")
    month = an.get("month", {}) if an else {}
    week = an.get("week", {}) if an else {}
    if month:
        med, up = month.get("median_pct", 0), month.get("share_up", 0.5)
        if med > 2 and up > 0.55:
            add("Similar past periods", 1, f"After similar past days the price rose over the next month {up*100:.0f}% of the time.", "how prices moved after similar past days")
        elif med < -2 and up < 0.45:
            add("Similar past periods", -1, f"After similar past days the price fell over the next month {(1-up)*100:.0f}% of the time.", "how prices moved after similar past days")
        else:
            add("Similar past periods", 0, "Similar past days were followed by mixed price moves.")

    total = sum(d["score"] for d in drivers)
    ups = [d for d in drivers if d["score"] > 0]
    downs = [d for d in drivers if d["score"] < 0]

    if total >= 2:
        lean, label = "up", "Upward pressure"
    elif total <= -2:
        lean, label = "down", "Downward pressure"
    else:
        lean, label = "balanced", "Balanced"

    def names(lst):
        lst = sorted(lst, key=lambda d: -abs(d["score"]))
        return " and ".join(d["phrase"] for d in lst[:2])

    if lean == "up":
        summary = f"Current signals point to upward pressure on coffee prices over the coming weeks, driven mainly by {names(ups)}."
        if downs:
            summary += f" This is partly offset by {names(downs)}."
    elif lean == "down":
        summary = f"Current signals point to downward pressure on coffee prices over the coming weeks, driven mainly by {names(downs)}."
        if ups:
            summary += f" This is partly offset by {names(ups)}."
    elif ups and downs:
        summary = (f"Signals are mixed: {names(ups)} would push prices up, while "
                   f"{names(downs)} would push them down. No clear direction for now.")
    else:
        summary = "No strong signals in either direction. Prices are likely to be driven by new weather or market news."

    price_range = {}
    for key, h in (("week", week), ("month", month)):
        if h and "p10_pct" in h:
            price_range[key] = {
                "low": current_price * (1 + h["p10_pct"] / 100),
                "high": current_price * (1 + h["p90_pct"] / 100),
                "median": current_price * (1 + h["median_pct"] / 100),
                "median_pct": h["median_pct"],
                "n": h.get("n_analogs", 0),
            }

    return {"lean": lean, "label": label, "score": total, "summary": summary,
            "drivers": drivers, "range": price_range}


# ===================================================================
# Assemble
# ===================================================================

def generate_brief(commodity_key: str = "coffee") -> MarketBrief:
    """Collect all observations into a layered intelligence briefing."""
    prices = _load_prices()
    unified = _load_unified()

    if prices.empty:
        raise RuntimeError(
            "No coffee price data found. "
            "Run: python -m src.modeling.fetch_price_data --commodity coffee"
        )

    latest = prices.iloc[-1]
    as_of = latest["date"].strftime("%Y-%m-%d")
    current_price = float(latest["price"])

    observations = []

    # Supply side
    observations.extend(observe_weather_observed(unified))
    observations.extend(observe_weather_forecast())
    observations.extend(observe_supply_benchmark(unified))

    # Demand / market
    observations.extend(observe_price_trend(prices))
    observations.extend(observe_cftc_positioning())
    observations.extend(observe_currency())

    # Events
    observations.extend(observe_anomalies(prices))
    observations.extend(observe_news(commodity_key))

    # Context
    observations.extend(observe_seasonal_context())
    observations.extend(observe_historical_analogs(prices))

    return MarketBrief(
        as_of_date=as_of,
        current_price=current_price,
        observations=observations,
        outlook=build_outlook(observations, current_price),
    )


# ===================================================================
# Text formatting
# ===================================================================

def format_brief_text(brief: MarketBrief) -> str:
    """Format as a structured intelligence report."""
    lines = [
        f"COFFEE MARKET INTELLIGENCE, {brief.as_of_date}",
        f"Price: {brief.current_price:.2f} US cents/lb",
        "",
    ]

    # Headline
    headline = _generate_headline(brief.observations)
    lines.append(f"  {headline}")
    lines.append("")
    if brief.outlook:
        lines.append(f"  OUTLOOK: {brief.outlook['label']}. {brief.outlook['summary']}")
        lines.append("")
    lines.append("=" * 70)

    # Layer 2: Full situation
    sections = [
        ("SUPPLY SIDE", brief.supply_obs),
        ("DEMAND & MARKET", brief.demand_obs),
        ("EVENTS", brief.event_obs),
        ("CONTEXT", brief.context_obs),
    ]

    for title, obs_list in sections:
        if not obs_list:
            continue
        lines.append(f"\n  [{title}]")
        for o in sorted(obs_list, key=lambda x: x.importance):
            tag = ""
            if o.importance == 1:
                tag = " **"
            lines.append(f"    {o.topic}{tag}: {o.text}")
            if o.detail:
                lines.append(f"      Detail: {o.detail}")
            lines.append(f"      Source: {o.source}")

    lines.extend([
        "",
        "-" * 70,
    ])

    return "\n".join(lines)


# ===================================================================
# CLI
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Coffee market intelligence briefing.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    brief = generate_brief("coffee")

    if args.json:
        print(json.dumps(brief.to_dict(), indent=2, default=str))
    else:
        print(format_brief_text(brief))


if __name__ == "__main__":
    main()
