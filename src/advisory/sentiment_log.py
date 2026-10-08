"""
Sentiment time series: persist daily outlook/briefing snapshots to a JSON-lines
log so the dashboard can plot how market sentiment evolves over time.

Each entry records:
  - date (as-of date for the signals)
  - lean (upward / downward / balanced)
  - confidence (high / medium / low)
  - headline (from the daily briefing)
  - n_upward, n_downward, n_neutral (signal counts)
  - sources_used (number of news articles)
  - timestamp (when the entry was logged)

The log lives at results/sentiment_log.jsonl — one JSON object per line,
append-only. The dashboard reads it to render a sentiment timeline.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = REPO_ROOT / "results" / "sentiment_log.jsonl"


def log_sentiment(outlook: dict = None, briefing: dict = None,
                  as_of_date: str = None):
    """Append a sentiment snapshot to the log. Accepts either an outlook
    result, a briefing result, or both. Deduplicates by date — if an entry
    for this date already exists, it's updated (last write wins)."""

    if as_of_date is None:
        # Try to get the date from signals
        if outlook:
            as_of_date = (outlook.get("signals_collected", {})
                          .get("price", {}).get("as_of", ""))
        if not as_of_date and briefing:
            as_of_date = (briefing.get("signals_collected", {})
                          .get("price", {}).get("as_of", ""))
        if not as_of_date:
            as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    entry = {
        "date": as_of_date,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if outlook:
        entry.update({
            "lean": outlook.get("lean", "balanced"),
            "confidence": outlook.get("confidence", "low"),
            "n_upward": outlook.get("n_upward", 0),
            "n_downward": outlook.get("n_downward", 0),
            "n_neutral": outlook.get("n_neutral", 0),
            "outlook_label": outlook.get("label", ""),
            "outlook_summary": (outlook.get("summary", "") or "")[:200],
        })

    if briefing:
        entry.update({
            "briefing_headline": (briefing.get("headline", "") or "")[:200],
            "briefing_sources": briefing.get("sources_used", 0),
            "briefing_model_generated": briefing.get("model_generated", False),
        })
        # If outlook wasn't provided, use briefing's lean if available
        if not outlook and "lean" not in entry:
            entry["lean"] = "balanced"
            entry["confidence"] = "low"

    # Read existing entries, update or append
    existing = load_sentiment_log()
    updated = False
    for i, e in enumerate(existing):
        if e.get("date") == as_of_date:
            existing[i] = entry
            updated = True
            break
    if not updated:
        existing.append(entry)

    # Sort by date and write back
    existing.sort(key=lambda e: e.get("date", ""))

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for e in existing:
            f.write(json.dumps(e, default=str) + "\n")

    return entry


def load_sentiment_log() -> list:
    """Load all sentiment entries from the log."""
    if not LOG_PATH.exists():
        return []
    entries = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []
    return entries


def get_sentiment_streak(entries: list = None) -> dict:
    """Analyze the most recent sentiment streak.
    Returns {"direction": "bearish"/"bullish"/"mixed", "days": N, "since": date}."""
    if entries is None:
        entries = load_sentiment_log()
    if not entries:
        return {"direction": "mixed", "days": 0, "since": ""}

    # Work backwards from most recent
    entries_sorted = sorted(entries, key=lambda e: e.get("date", ""), reverse=True)
    current_lean = entries_sorted[0].get("lean", "balanced")
    streak = 1
    for e in entries_sorted[1:]:
        if e.get("lean", "balanced") == current_lean:
            streak += 1
        else:
            break

    direction_map = {"upward": "bullish", "downward": "bearish", "balanced": "mixed"}
    return {
        "direction": direction_map.get(current_lean, "mixed"),
        "days": streak,
        "since": entries_sorted[min(streak - 1, len(entries_sorted) - 1)].get("date", ""),
        "lean": current_lean,
    }


if __name__ == "__main__":
    entries = load_sentiment_log()
    print(f"Sentiment log: {len(entries)} entries")
    for e in entries[-5:]:
        print(f"  {e.get('date')}: {e.get('lean', '?')} "
              f"({e.get('confidence', '?')} confidence) "
              f"up={e.get('n_upward', 0)} down={e.get('n_downward', 0)}")
    streak = get_sentiment_streak(entries)
    print(f"\nCurrent streak: {streak['direction']} for {streak['days']} day(s)")
