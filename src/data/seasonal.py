"""
Coffee agricultural calendar and seasonal context engine.

Hard-coded agronomic calendar for the three largest coffee-producing origins
(Brazil, Vietnam, Colombia). No network calls or external data — this is
pure domain knowledge encoded as a lookup so the analysis pipeline can
contextualize weather anomalies and price movements against crop phenology.

Key insight: the *same* weather event (e.g. frost, drought) has radically
different price implications depending on the crop phase. Frost during
flowering is catastrophic; frost during harvest is a nuisance. This module
makes that distinction explicit.

Usage:
    python -m src.data.seasonal
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PhaseEntry:
    """A single phenological phase for one origin."""
    phase: str
    note: str
    months: tuple[int, ...]


# ── Brazil Arabica ──────────────────────────────────────────────────────
# Minas Gerais / Sao Paulo / Parana belt (southern hemisphere cycle).
BRAZIL_PHASES: list[PhaseEntry] = [
    PhaseEntry(
        phase="Harvest (main crop)",
        note=(
            "Peak harvest period. Large volumes hitting the market can weigh "
            "on prices. Weather disruptions now affect logistics more than yield."
        ),
        months=(5, 6, 7, 8, 9),
    ),
    PhaseEntry(
        phase="Flowering: frost risk is critical",
        note=(
            "Flowering sets next year's crop potential. Frost events in the "
            "southern Brazil coffee belt during this window can destroy flowers "
            "and drastically reduce the following season's output. This is "
            "historically the highest-impact period for weather-driven price spikes."
        ),
        months=(9, 10, 11),
    ),
    PhaseEntry(
        phase="Fruit development",
        note=(
            "Cherries are filling and maturing. Adequate rainfall is essential; "
            "drought stress now reduces cherry size and can cut yields. Frost is "
            "less likely but still damaging if it occurs."
        ),
        months=(12, 1, 2, 3, 4),
    ),
]

# ── Vietnam Robusta ─────────────────────────────────────────────────────
# Central Highlands (Dak Lak, Lam Dong). Tropical monsoon cycle.
VIETNAM_PHASES: list[PhaseEntry] = [
    PhaseEntry(
        phase="Harvest",
        note=(
            "Robusta cherry picking. Vietnam is the world's largest Robusta "
            "producer; harvest pace and quality affect the global Robusta market."
        ),
        months=(10, 11, 12, 1, 2, 3),
    ),
    PhaseEntry(
        phase="Flowering and fruit set",
        note=(
            "Flowering triggered by early rains after the dry season. "
            "Insufficient rainfall delays flowering and reduces fruit set."
        ),
        months=(1, 2, 3),
    ),
    PhaseEntry(
        phase="Fruit development and dry season stress",
        note=(
            "Cherries developing through the dry months. Irrigation availability "
            "is important; prolonged drought can hurt yields."
        ),
        months=(4, 5, 6, 7, 8, 9),
    ),
]

# ── Colombia ────────────────────────────────────────────────────────────
# Dual-harvest system thanks to the equatorial latitude and varied altitude.
COLOMBIA_PHASES: list[PhaseEntry] = [
    PhaseEntry(
        phase="Main harvest (cosecha principal)",
        note=(
            "Primary harvest, especially in southern and central departments. "
            "Colombia supplies high-quality washed Arabica year-round, but "
            "the main crop is the larger of the two harvests."
        ),
        months=(10, 11, 12),
    ),
    PhaseEntry(
        phase="Mitaca harvest (fly crop)",
        note=(
            "Secondary harvest, mainly from northern departments. Smaller volume "
            "but important for continuity of supply."
        ),
        months=(4, 5, 6),
    ),
    PhaseEntry(
        phase="Flowering and fruit development (main crop)",
        note="Flowering for the main October-December harvest.",
        months=(1, 2, 3),
    ),
    PhaseEntry(
        phase="Flowering and fruit development (mitaca crop)",
        note="Flowering for the April-June mitaca harvest.",
        months=(7, 8, 9),
    ),
]

COFFEE_CALENDAR = {
    "brazil": BRAZIL_PHASES,
    "vietnam": VIETNAM_PHASES,
    "colombia": COLOMBIA_PHASES,
}


def _get_phase(phases: list[PhaseEntry], month: int) -> PhaseEntry:
    """Return the phase entry whose month range includes ``month``."""
    for entry in phases:
        if month in entry.months:
            return entry
    # Fallback (should not happen with complete calendars)
    return PhaseEntry(phase="Off-season", note="No major activity.", months=())


def _next_month(month: int, offset: int = 1) -> int:
    """Return the month number ``offset`` months ahead, wrapping at 12."""
    return (month - 1 + offset) % 12 + 1


def _get_upcoming(month: int) -> str:
    """Describe what's coming 1-2 months ahead across all origins."""
    upcoming_parts = []
    m1 = _next_month(month, 1)
    m2 = _next_month(month, 2)

    for origin, phases in COFFEE_CALENDAR.items():
        current = _get_phase(phases, month)
        future = _get_phase(phases, m2)
        if future.phase != current.phase:
            label = origin.replace("_", " ").title()
            upcoming_parts.append(f"{label}: transitioning to {future.phase}")

    if not upcoming_parts:
        return "No major phase transitions in the next 1-2 months."
    return "; ".join(upcoming_parts) + "."


def _determine_key_risk(month: int) -> str:
    """Identify the single most important seasonal risk factor right now."""
    # September: end of the Brazilian harvest overlaps with the first flowering
    if month == 9:
        return (
            "End of the Brazilian harvest and start of flowering: frost or a "
            "long dry spell now can damage next year's crop."
        )
    # October-November: Brazilian flowering + frost risk dominates
    if month in (10, 11):
        return (
            "Brazilian frost risk during flowering: the highest-impact "
            "seasonal factor for global coffee prices."
        )
    # December-April: Brazilian fruit development, drought matters
    if month in (12, 1, 2, 3, 4):
        return (
            "Brazilian drought during fruit development: insufficient rain "
            "reduces cherry size and can cut next harvest's yield."
        )
    # May-September: harvest logistics + early frost watch
    if month in (5, 6, 7, 8):
        return (
            "Brazilian harvest pace and quality: weather disruptions affect "
            "logistics, and late-season frost can damage remaining crop."
        )
    return "No single dominant risk factor identified."


def current_season(date: Optional[dt.date] = None) -> dict:
    """
    Return the current coffee seasonal context for all three major origins.

    Parameters
    ----------
    date : datetime.date, optional
        Reference date. Defaults to today.

    Returns
    -------
    dict
        Keys: brazil_phase, brazil_note, vietnam_phase, colombia_phase,
        key_risk, upcoming.
    """
    if date is None:
        date = dt.date.today()
    month = date.month

    brazil = _get_phase(BRAZIL_PHASES, month)
    vietnam = _get_phase(VIETNAM_PHASES, month)
    colombia = _get_phase(COLOMBIA_PHASES, month)

    return {
        "brazil_phase": brazil.phase,
        "brazil_note": brazil.note,
        "vietnam_phase": vietnam.phase,
        "colombia_phase": colombia.phase,
        "key_risk": _determine_key_risk(month),
        "upcoming": _get_upcoming(month),
    }


def contextualize_weather(
    dryness_zscore: float,
    frost_count: int,
    date: Optional[dt.date] = None,
) -> str:
    """
    Interpret weather readings in the context of the current crop phase.

    Parameters
    ----------
    dryness_zscore : float
        Standardized dryness anomaly. Positive = drier than normal.
    frost_count : int
        Number of frost events detected in the recent observation window.
    date : datetime.date, optional
        Reference date. Defaults to today.

    Returns
    -------
    str
        A sentence placing the weather data in agronomic context.
    """
    if date is None:
        date = dt.date.today()

    month = date.month
    brazil = _get_phase(BRAZIL_PHASES, month)
    phase_lower = brazil.phase.lower()

    parts = []

    # Frost interpretation depends heavily on the phase
    if frost_count > 0:
        if "flowering" in phase_lower:
            parts.append(
                f"Frost detected during flowering ({frost_count} event(s)): "
                "this is the highest-risk period for frost damage to the crop. "
                "Flower destruction can drastically reduce next season's yield."
            )
        elif "harvest" in phase_lower:
            parts.append(
                f"Frost detected during harvest ({frost_count} event(s)): "
                "less damaging than during flowering, but can still affect "
                "cherry quality and complicate field operations."
            )
        elif "fruit development" in phase_lower:
            parts.append(
                f"Frost detected during fruit development ({frost_count} event(s)): "
                "developing cherries are vulnerable; frost can cause cell damage "
                "and reduce both yield and quality."
            )
        else:
            parts.append(
                f"Frost detected ({frost_count} event(s)) during {brazil.phase}."
            )

    # Drought interpretation
    if dryness_zscore > 1.5:
        severity = "severe" if dryness_zscore > 2.5 else "notable"
        if "fruit development" in phase_lower:
            parts.append(
                f"Drought stress during fruit development (z-score: {dryness_zscore:.1f}, "
                f"{severity}): can reduce cherry size and next year's yield."
            )
        elif "flowering" in phase_lower:
            parts.append(
                f"Drought stress during flowering (z-score: {dryness_zscore:.1f}, "
                f"{severity}): insufficient moisture can impair flower "
                "development and reduce fruit set."
            )
        elif "harvest" in phase_lower:
            parts.append(
                f"Dry conditions during harvest (z-score: {dryness_zscore:.1f}, "
                f"{severity}): generally favorable for harvest logistics, "
                "though extreme drought can stress trees for the next cycle."
            )
        else:
            parts.append(
                f"Drought stress detected (z-score: {dryness_zscore:.1f}, "
                f"{severity}) during {brazil.phase}."
            )
    elif dryness_zscore < -1.5:
        parts.append(
            f"Excess rainfall detected (z-score: {dryness_zscore:.1f}): "
            "can hamper harvest operations and promote fungal diseases."
        )

    if not parts:
        return (
            f"Weather conditions are within normal range during {brazil.phase}. "
            "No significant stress factors detected."
        )

    return " ".join(parts)


if __name__ == "__main__":
    today = dt.date.today()
    season = current_season(today)

    print(f"=== Coffee Seasonal Context for {today.strftime('%B %d, %Y')} ===\n")
    print(f"  Brazil:   {season['brazil_phase']}")
    print(f"            {season['brazil_note']}\n")
    print(f"  Vietnam:  {season['vietnam_phase']}")
    print(f"  Colombia: {season['colombia_phase']}\n")
    print(f"  Key risk: {season['key_risk']}")
    print(f"  Upcoming: {season['upcoming']}")

    print("\n--- Example weather contextualization ---")
    print(f"  Drought (z=2.0): {contextualize_weather(2.0, 0, today)}")
    print(f"  Frost (n=1):     {contextualize_weather(0.0, 1, today)}")
    print(f"  Both:            {contextualize_weather(2.5, 2, today)}")
