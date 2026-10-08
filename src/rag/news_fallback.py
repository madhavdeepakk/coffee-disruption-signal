"""
Local news-fallback provider: rescue a date GDELT keeps starving.

The on-disk GDELT cache (src/rag/gdelt_client.py) makes re-runs of a date
reliable once it has succeeded even once. But a date whose queries never get
past GDELT's 429 rate limiter even a single time can't be cached, because
there is nothing to cache. The production answer is a historical news source
that isn't rate-limited the way GDELT's free public endpoint is - GDELT's
own corpus is available un-throttled on Google BigQuery (the `gdelt-bq`
public dataset), the recommended path for a real deployment. Most free news
APIs are not a substitute here: they don't offer historical, date-windowed
search, which this project requires for lookahead-safe retrieval.

This module is the fallback that needs no new credentials: a place to drop
in articles gathered by hand (or exported from BigQuery/another tool) for a
stubborn date, which the pipeline then merges into retrieval like GDELT
results. It is:
  - Opt-in by existence: nothing happens unless a file exists at
    data/news_fallback/<commodity>/<YYYY-MM-DD>.json, so it never changes a
    run that didn't have one (the validated runs are unaffected).
  - Lookahead-safe: any supplied article dated after the anomaly date is
    dropped with a warning, the same discipline GDELT's date window enforces.

File format (a JSON list; only `url` and `title` are required):
  [
    {"url": "https://...", "title": "...", "seendate": "20250815T120000Z",
     "text": "optional full text"},
    ...
  ]
`seendate` may be GDELT's YYYYMMDDTHHMMSSZ form or a plain YYYY-MM-DD.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def _parse_date(seendate: str) -> Optional[datetime]:
    if not seendate:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(seendate, fmt)
        except ValueError:
            continue
    return None


def fallback_path(fallback_dir, commodity_key: str, anomaly_date: str) -> Path:
    return Path(fallback_dir) / commodity_key / f"{anomaly_date}.json"


def load_fallback_articles(fallback_dir, commodity_key: str, anomaly_date: str,
                           verbose: bool = True) -> list:
    """
    Return a list of article dicts (GDELT-article shape: url/title/seendate)
    from the fallback file for this commodity+date, or [] if none exists.
    Drops any article published after the anomaly date (lookahead-safety).
    """
    if not fallback_dir:
        return []
    path = fallback_path(fallback_dir, commodity_key, anomaly_date)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (ValueError, OSError) as exc:
        if verbose:
            print(f"  fallback file {path} unreadable, skipping: {exc}")
        return []
    if not isinstance(raw, list):
        if verbose:
            print(f"  fallback file {path} is not a JSON list, skipping")
        return []

    cutoff = _parse_date(anomaly_date)
    out = []
    dropped_future = 0
    dropped_bad = 0
    for art in raw:
        if not isinstance(art, dict) or not art.get("url"):
            dropped_bad += 1
            continue
        pub = _parse_date(art.get("seendate", "") or "")
        if cutoff is not None and pub is not None and pub.date() > cutoff.date():
            dropped_future += 1
            continue
        out.append({
            "url": art.get("url", ""),
            "title": art.get("title", "") or "",
            "seendate": art.get("seendate", "") or "",
            "text": art.get("text", "") or "",
            "_matched_query": "local_fallback",
            "_source": "local_fallback",
        })
    if verbose and (out or dropped_future or dropped_bad):
        msg = f"  local fallback: loaded {len(out)} article(s) from {path.name}"
        if dropped_future:
            msg += f", dropped {dropped_future} dated after the anomaly (lookahead-safe)"
        if dropped_bad:
            msg += f", skipped {dropped_bad} malformed"
        print(msg)
    return out


def _self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = fallback_path(d, "coffee", "2025-08-15")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([
            {"url": "http://a", "title": "on time", "seendate": "20250810T000000Z"},
            {"url": "http://b", "title": "future - must drop", "seendate": "2025-08-20"},
            {"title": "no url - must drop"},
        ]), encoding="utf-8")
        arts = load_fallback_articles(d, "coffee", "2025-08-15")
        assert len(arts) == 1 and arts[0]["url"] == "http://a", arts
        assert load_fallback_articles(d, "coffee", "2099-01-01") == []
        print("news_fallback self-test OK")


if __name__ == "__main__":
    _self_test()
