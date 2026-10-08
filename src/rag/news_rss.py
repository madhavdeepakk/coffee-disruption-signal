"""
Second retrieval source: Google News RSS (stdlib only, no new dependency).

Why this exists
---------------
The primary evidence source (GDELT DOC 2.0) has a high observed failure
rate - it rate-limits (429) and 503s frequently. When it returns nothing,
the pipeline previously had only the on-disk cache and an opt-in local
fallback file to fall back on, so GDELT being down effectively dictated the
system's refusal rate. That conflates "the news API was unavailable" with
"there was no relevant news" - the exact distinction src/rag/outcome.py now
keeps separate. A second, independent source narrows that gap.

Honest scope (important, state this in any write-up)
----------------------------------------------------
Google News RSS returns only *recent* articles - it is a recency-limited
feed, not a historical archive. So this source meaningfully helps dates at
or near the present (where GDELT was simply down), and for a deep-historical
anomaly (e.g. a 2021 date) it will correctly return nothing, because the
feed does not reach that far back. That is handled honestly: it is a
best-effort supplement that never fabricates and returns an empty list
rather than out-of-window articles. It is NOT a replacement for the
production-grade answer (GDELT on BigQuery), which is the only free way to
query the full historical archive reliably.

Lookahead safety
-----------------
Every returned item is filtered to publication dates within
[anomaly_date - window_days, anomaly_date] inclusive. Items with no parseable
date, or dated after the anomaly date, are dropped - the same lookahead
discipline the GDELT path enforces. An item's date must be known to be kept.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

# Per-commodity search query for the RSS feed. Kept deliberately simple and
# English - this is a supplementary source; the multilingual coverage is the
# GDELT + semantic-scoring path's job.
RSS_QUERIES = {
    "coffee": ("coffee prices OR coffee futures OR arabica OR coffee harvest "
               "OR coffee tariff OR coffee supply OR coffee export"),
    "crude_oil": "crude oil prices OR oil futures OR OPEC OR oil supply",
    "wheat": "wheat prices OR wheat futures OR grain harvest OR wheat export",
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
REQUEST_TIMEOUT_SECONDS = 12
USER_AGENT = "Mozilla/5.0 (research capstone; news retrieval fallback)"


def _to_gdelt_seendate(dt: datetime) -> str:
    """Format a datetime as GDELT's seendate (YYYYMMDDTHHMMSSZ) so the
    pipeline's existing gdelt_seendate_to_iso_date() resolves it unchanged."""
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_pubdate(raw: str):
    """RFC-822 pubDate -> aware UTC datetime, or None if unparseable."""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_rss_articles(commodity_key: str, anomaly_date: str,
                       window_days: int, timeout: int = REQUEST_TIMEOUT_SECONDS) -> list:
    """Return GDELT-shaped article dicts from Google News RSS within the
    lookahead-safe window. Never raises for an empty/failed feed - returns
    [] so the caller treats it as 'this source found nothing', which is the
    correct behaviour for a supplementary source.

    Returned dicts carry the keys the pipeline's document builder reads:
    title, url, seendate (GDELT format), and a _source marker.
    """
    query = RSS_QUERIES.get(commodity_key)
    if not query:
        return []

    end_dt = datetime.strptime(anomaly_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) \
        + timedelta(days=1)
    start_dt = end_dt - timedelta(days=window_days + 1)

    # Scope the search to the lookahead-safe window with Google News' date
    # operators, so this works as a HISTORICAL fallback (not just for recent
    # dates). "before" is exclusive, so use the day after the anomaly. Items
    # are still date-filtered locally below as a second guard.
    dated_query = (f"({query}) "
                   f"after:{start_dt.strftime('%Y-%m-%d')} "
                   f"before:{end_dt.strftime('%Y-%m-%d')}")
    params = {"q": dated_query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = f"{GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except Exception:  # noqa: BLE001 - network/HTTP errors are expected; stay quiet
        return []

    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []

    articles = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = item.findtext("pubDate") or ""
        pub_dt = _parse_pubdate(pub_raw)
        if not title or not link or pub_dt is None:
            continue
        # Lookahead-safe: keep only items published within the window and on
        # or before the anomaly date. Unknown/after-date items are dropped.
        if not (start_dt <= pub_dt <= end_dt):
            continue
        source = ""
        src_el = item.find("source")
        if src_el is not None and src_el.text:
            source = src_el.text.strip()
        articles.append({
            "title": f"{title} - {source}" if source else title,
            "url": link,
            "seendate": _to_gdelt_seendate(pub_dt),
            "_source": "google_news_rss",
        })
    return articles


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "coffee"
    date = sys.argv[2] if len(sys.argv) > 2 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    arts = fetch_rss_articles(key, date, 10)
    print(f"RSS fallback for {key} @ {date} (10-day lookahead window): {len(arts)} article(s)")
    for a in arts[:10]:
        print(f"  [{a['seendate'][:8]}] {a['title'][:90]}")
