"""
Multi-source news aggregator: replaces GDELT as the primary retrieval source.

Why this exists
---------------
GDELT DOC 2.0 has an 87% observed failure rate (429/503 errors) and 22-25s
response times. That made the system's ability to explain anomalies
contingent on whether a single, unreliable API happened to be up. This
module replaces that dependency with a multi-source approach that is faster,
more reliable, and covers the same ground.

Sources (in priority order)
----------------------------
1. Google News RSS — broad coverage, fast, no API key needed. Already proven
   in the project's news_rss.py fallback (now promoted to primary).
2. Coffee industry RSS feeds — Daily Coffee News, Perfect Daily Grind,
   Reuters commodities. These carry expert market analysis that general news
   misses (harvest reports, ICO data, trade policy).
3. GDELT cache — existing cached responses are still used when available.
   GDELT live queries are demoted to a last-resort fallback.

Lookahead safety
-----------------
Every source is filtered to [anomaly_date - window_days, anomaly_date].
Articles with no parseable date or dated after the anomaly date are dropped.
This is the same discipline the original GDELT path enforces.

Architecture
------------
Each source returns GDELT-shaped article dicts (title, url, seendate, _source)
so the downstream pipeline (text_fetch, keyword scoring, semantic scoring,
relevance gate, LLM explanation) works unchanged. Deduplication by URL
happens across all sources before returning.
"""

import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration per commodity
# ---------------------------------------------------------------------------

# Google News RSS queries per commodity (broader than GDELT's narrow queries
# because RSS search is more forgiving and we want recall)
NEWS_QUERIES = {
    "coffee": [
        "coffee prices OR coffee futures OR arabica OR robusta",
        "coffee harvest OR coffee supply OR coffee drought OR coffee frost",
        "coffee export OR coffee trade OR Brazil coffee",
        "coffee market outlook OR coffee demand",
    ],
    "crude_oil": [
        "crude oil prices OR oil futures OR OPEC",
        "oil supply disruption OR oil production",
        "oil geopolitical OR oil sanctions",
    ],
    "wheat": [
        "wheat prices OR wheat futures OR grain harvest",
        "wheat export ban OR wheat supply OR wheat drought",
    ],
}

# Industry-specific RSS feeds (these carry deep market analysis that
# Google News misses: harvest reports, ICO data, trade flows)
INDUSTRY_FEEDS = {
    "coffee": [
        {
            "url": "https://dailycoffeenews.com/feed/",
            "name": "Daily Coffee News",
        },
        {
            "url": "https://perfectdailygrind.com/feed/",
            "name": "Perfect Daily Grind",
        },
    ],
    "crude_oil": [],
    "wheat": [],
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = "Mozilla/5.0 (research capstone; multi-source news aggregator)"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _to_gdelt_seendate(dt: datetime) -> str:
    """Format a datetime as GDELT's seendate (YYYYMMDDTHHMMSSZ) so the
    pipeline's existing gdelt_seendate_to_iso_date() resolves it unchanged."""
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_rfc822(raw: str) -> Optional[datetime]:
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


def _make_window(anomaly_date: str, window_days: int):
    """Return (start_dt, end_dt) as aware UTC datetimes for the lookahead-safe
    retrieval window: [anomaly_date - window_days, anomaly_date] inclusive."""
    end_dt = datetime.strptime(anomaly_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) \
        + timedelta(days=1)
    start_dt = end_dt - timedelta(days=window_days + 1)
    return start_dt, end_dt


def _in_window(pub_dt: Optional[datetime], start_dt: datetime, end_dt: datetime) -> bool:
    """Return True if pub_dt is within the lookahead-safe window. end_dt is
    midnight at the START of the day after the anomaly, so it is exclusive:
    an article stamped exactly 00:00 the next day is next-day news."""
    if pub_dt is None:
        return False
    return start_dt <= pub_dt < end_dt


def _date_scoped_query(query: str, start_dt: datetime, end_dt: datetime) -> str:
    """Restrict a Google News search to the retrieval window with its
    after:/before: operators.

    Without them the feed returns only the newest ~100 items for the query,
    and every one of those is then thrown away by the local date filter for
    any anomaly more than a few days old - so for a historical date this
    source silently contributed nothing and retrieval rested on the GDELT
    cache alone. (src/rag/news_rss.py already scoped its query this way; the
    aggregator that replaced it did not.) "before" is exclusive, so end_dt -
    the day after the anomaly - is the right bound. Items are still
    date-checked locally as a second guard.
    """
    return (f"({query}) after:{start_dt.strftime('%Y-%m-%d')} "
            f"before:{end_dt.strftime('%Y-%m-%d')}")


# ---------------------------------------------------------------------------
# Source 1: Google News RSS
# ---------------------------------------------------------------------------

def _fetch_google_news(query: str, start_dt: datetime, end_dt: datetime,
                       timeout: int = REQUEST_TIMEOUT_SECONDS) -> list:
    """Fetch articles from Google News RSS for a single query, filtered to
    the lookahead-safe window. Returns GDELT-shaped dicts."""
    params = {"q": _date_scoped_query(query, start_dt, end_dt),
              "hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = f"{GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except Exception:
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
        pub_dt = _parse_rfc822(pub_raw)
        if not title or not link or not _in_window(pub_dt, start_dt, end_dt):
            continue
        source = ""
        src_el = item.find("source")
        if src_el is not None and src_el.text:
            source = src_el.text.strip()
        # Google News titles normally already end in " - Publisher"; only add
        # the publisher when it is not there, so it is never doubled.
        if source and not title.lower().endswith(f"- {source.lower()}"):
            title = f"{title} - {source}"
        articles.append({
            "title": title,
            "url": link,
            "seendate": _to_gdelt_seendate(pub_dt),
            "_source": "google_news_rss",
        })
    return articles


# ---------------------------------------------------------------------------
# Source 2: Industry RSS feeds
# ---------------------------------------------------------------------------

def _fetch_industry_feed(feed_url: str, feed_name: str,
                         start_dt: datetime, end_dt: datetime,
                         timeout: int = REQUEST_TIMEOUT_SECONDS) -> list:
    """Fetch articles from an industry RSS feed, filtered to the window."""
    req = urllib.request.Request(feed_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except Exception:
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
        pub_dt = _parse_rfc822(pub_raw)
        if not title or not link or not _in_window(pub_dt, start_dt, end_dt):
            continue
        articles.append({
            "title": f"{title} - {feed_name}",
            "url": link,
            "seendate": _to_gdelt_seendate(pub_dt),
            "_source": f"industry_rss:{feed_name.lower().replace(' ', '_')}",
        })
    return articles


# ---------------------------------------------------------------------------
# Source 3: GDELT cache (read-only, no live queries by default)
# ---------------------------------------------------------------------------

def _fetch_gdelt_cached(queries: list, start_dt: datetime, end_dt: datetime,
                        cache_dir, allow_live: bool = False) -> tuple:
    """Read GDELT cache for previously-fetched articles. Optionally attempt
    live queries as a last resort. Returns (articles, cache_hits, live_attempts,
    live_failures)."""
    if cache_dir is None:
        return [], 0, 0, 0

    from src.rag.gdelt_client import (
        build_query_params, cache_get, fetch_gdelt,
        gdelt_seendate_to_iso_date, DEFAULT_SORT,
    )

    startdatetime = start_dt.strftime("%Y%m%d%H%M%S")
    enddatetime = end_dt.strftime("%Y%m%d%H%M%S")
    max_per_query = 20

    articles = []
    cache_hits = 0
    live_attempts = 0
    live_failures = 0

    for query in queries:
        params = build_query_params(
            query, max_per_query, DEFAULT_SORT, startdatetime, enddatetime,
        )
        cached = cache_get(cache_dir, params)
        if cached is not None:
            cache_hits += 1
            for art in cached.get("articles", []) or []:
                articles.append(art)
        elif allow_live:
            live_attempts += 1
            try:
                import time
                if live_attempts > 1:
                    time.sleep(3)
                raw = fetch_gdelt(
                    query, maxrecords=max_per_query,
                    startdatetime=startdatetime, enddatetime=enddatetime,
                )
                for art in raw.get("articles", []) or []:
                    articles.append(art)
                # Cache the successful response
                from src.rag.gdelt_client import cache_put
                cache_put(cache_dir, params, raw)
            except RuntimeError:
                live_failures += 1

    return articles, cache_hits, live_attempts, live_failures


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

def aggregate_news(commodity_key: str, anomaly_date: str,
                   window_days: int = 10,
                   gdelt_queries: list = None,
                   cache_dir=None,
                   allow_gdelt_live: bool = False) -> tuple:
    """Aggregate news from all sources for a commodity and date window.

    Returns (articles, stats) where articles is a deduplicated list of
    GDELT-shaped dicts and stats is a dict of retrieval health counters.

    The articles list is ready for the pipeline's text_fetch + scoring steps.
    """
    start_dt, end_dt = _make_window(anomaly_date, window_days)

    stats = {
        "google_news_count": 0,
        "industry_rss_count": 0,
        "gdelt_cache_count": 0,
        "gdelt_live_count": 0,
        "total_before_dedupe": 0,
        "total_after_dedupe": 0,
        "sources_attempted": 0,
        "sources_succeeded": 0,
    }

    all_articles = []

    # Source 1: Google News RSS (primary)
    queries = NEWS_QUERIES.get(commodity_key, [])
    stats["sources_attempted"] += 1
    google_articles = []
    for q in queries:
        google_articles.extend(_fetch_google_news(q, start_dt, end_dt))
    stats["google_news_count"] = len(google_articles)
    if google_articles:
        stats["sources_succeeded"] += 1
    all_articles.extend(google_articles)
    print(f"  Google News RSS: {len(google_articles)} articles from {len(queries)} queries")

    # Source 2: Industry RSS feeds
    feeds = INDUSTRY_FEEDS.get(commodity_key, [])
    if feeds:
        stats["sources_attempted"] += 1
        industry_count = 0
        for feed in feeds:
            arts = _fetch_industry_feed(feed["url"], feed["name"], start_dt, end_dt)
            industry_count += len(arts)
            all_articles.extend(arts)
            print(f"  {feed['name']}: {len(arts)} articles")
        stats["industry_rss_count"] = industry_count
        if industry_count:
            stats["sources_succeeded"] += 1

    # Source 3: GDELT (cache first, live as last resort)
    if gdelt_queries and cache_dir:
        stats["sources_attempted"] += 1
        gdelt_arts, cache_hits, live_attempts, live_failures = _fetch_gdelt_cached(
            gdelt_queries, start_dt, end_dt, cache_dir, allow_live=allow_gdelt_live,
        )
        stats["gdelt_cache_count"] = len([a for a in gdelt_arts])  # count before adding
        stats["gdelt_cache_hits"] = cache_hits
        stats["gdelt_live_attempts"] = live_attempts
        stats["gdelt_live_failures"] = live_failures
        if gdelt_arts:
            stats["sources_succeeded"] += 1
        all_articles.extend(gdelt_arts)
        print(f"  GDELT: {len(gdelt_arts)} articles ({cache_hits} cache hits, "
              f"{live_attempts} live attempts, {live_failures} failures)")

    stats["total_before_dedupe"] = len(all_articles)

    # Deduplicate by URL across all sources
    seen_urls = set()
    deduped = []
    for art in all_articles:
        url = art.get("url", "")
        if not url:
            continue
        # Normalize URL for deduplication (strip trailing slashes, fragments)
        norm_url = url.rstrip("/").split("#")[0]
        if norm_url in seen_urls:
            continue
        seen_urls.add(norm_url)
        deduped.append(art)

    stats["total_after_dedupe"] = len(deduped)
    print(f"  Total: {stats['total_before_dedupe']} -> {stats['total_after_dedupe']} after dedupe")

    return deduped, stats


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "coffee"
    date = sys.argv[2] if len(sys.argv) > 2 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print(f"=== News aggregator test: {key} @ {date} (window={window}d) ===\n")
    arts, stats = aggregate_news(key, date, window)
    print(f"\nStats: {stats}")
    print(f"\nTop 10 articles:")
    for a in arts[:10]:
        print(f"  [{a.get('seendate', '')[:8]}] [{a.get('_source', '')}] {a.get('title', '')[:90]}")
