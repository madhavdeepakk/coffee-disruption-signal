"""
Low-level GDELT DOC 2.0 API client: retry/timeout/UA handling only.

Extracted from scripts/test_gdelt.py so both the RAG retriever
(src/rag/retriever.py) and the raw ingestion script (src/data/gdelt.py) can
reuse the same request logic instead of duplicating it.

This module does one thing: call the API and return the raw parsed JSON
response for a query. It does not know about the team schema or the RAG
schema — that mapping happens in the callers.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Union

import requests

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Upper bound on how long to honor a server-provided Retry-After (see the
# 429 handling in fetch_gdelt). A misbehaving header should not be able to
# make a run hang for minutes - GDELT's successful responses come back in
# ~22-25s, so a cap in this range keeps a retry roughly in line with a
# normal request's cost rather than blowing the run's total time budget.
MAX_RETRY_AFTER_SECONDS = 30

# GDELT's public API has an observed ~87% failure rate and 22-25s successful
# response times in testing. These settings are intentionally generous.
REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 6   # never sooner than GDELT's one request every 5 seconds

# The GDELT relevance sort used everywhere. Defined once so fetch_gdelt's
# default and any caller that rebuilds the params for the cache key (see
# build_query_params / src/pipeline.py) can't silently disagree about it -
# a mismatch there would make every cache lookup miss.
DEFAULT_SORT = "hybridrel"

# A flat 5s backoff between attempts is sometimes not enough: a run can get
# 429 (Too Many Requests) on all 3 attempts for all 3 queries, ending in "no
# documents retrieved" even on a real anomaly date where nearby dates
# succeed. A 429 means rate-limited right now, so waiting longer (not just
# retrying sooner) helps - the backoff grows with the attempt number rather
# than staying flat. This does not fix the endpoint's underlying
# unreliability; it just gives each retry a better chance of landing outside
# the current rate-limit window.

# Missing a realistic browser User-Agent is a common cause of silent
# connection resets on GDELT's endpoint.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
# Why this exists: GDELT's public endpoint 429s (Too Many Requests) on the
# large majority of individual attempts. The pipeline and the labeling
# collector both re-query the same handful of historical anomaly dates
# repeatedly, and a fixed historical window (startdatetime..enddatetime)
# returns the same articles every time, so re-hitting the live endpoint for
# it just burns through the rate limit again. Caching each successful
# response on disk, keyed on the exact query params (window included), means
# every (query, window) pair only has to get past the rate limiter once;
# every later run for that pair reads from disk.
#
# Lookahead-safe: the cache key includes startdatetime and enddatetime, and
# those are already restricted to on-or-before the anomaly date by the caller
# (see src/pipeline.py). A cache entry for anomaly date T therefore only
# holds articles published <= T, and is only reused for that same window - no
# post-anomaly article can leak in via the cache.
#
# Only successes are cached: a 429/timeout/failure is never written, so a
# failed query is retried fresh on the next run rather than remembered as if
# it were a real empty result.


def _cache_key(params: dict) -> str:
    """Stable hash of the request params that determine the response."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def cache_get(cache_dir: Union[str, Path, None], params: dict,
              verbose: bool = True) -> Optional[dict]:
    """
    Return the cached response dict for these params, or None on a miss
    (including a missing/corrupt cache file - a corrupt entry is treated as
    a miss so a bad write can never wedge a date permanently).
    """
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"{_cache_key(params)}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        return entry.get("response")
    except (ValueError, OSError):
        return None


def cache_put(cache_dir: Union[str, Path, None], params: dict, response: dict) -> None:
    """
    Persist a SUCCESSFUL response. Stores the params alongside the response
    (human-readable, for debugging which window a file corresponds to). Best
    effort: a cache-write failure must never break an otherwise-good run, so
    OSErrors are swallowed.
    """
    if not cache_dir:
        return
    cache_path = Path(cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        path = cache_path / f"{_cache_key(params)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"params": params, "response": response}, f)
    except OSError:
        pass


def build_query_params(query: str, maxrecords: int, sort: str,
                       startdatetime: str = None, enddatetime: str = None) -> dict:
    """
    The exact set of fields that determine a GDELT response, used both as the
    request params and as the cache key. Kept in one place so the cache key
    can never silently drift out of sync with what's actually sent.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(maxrecords),
        "sort": sort,
    }
    if startdatetime:
        params["startdatetime"] = startdatetime
    if enddatetime:
        params["enddatetime"] = enddatetime
    return params


def _parse_retry_after(exc: Exception) -> Optional[int]:
    """
    If exc is an HTTP error carrying a numeric Retry-After header, return that
    many seconds (capped at MAX_RETRY_AFTER_SECONDS). Returns None if there's
    no response, no header, or the header isn't a plain integer count of
    seconds (the HTTP-date form is intentionally not handled - GDELT doesn't
    send it, and guessing wrong is worse than falling back to our own backoff).
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    value = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
    if not value:
        return None
    try:
        seconds = int(str(value).strip())
    except (ValueError, TypeError):
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def fetch_gdelt(
    query: str,
    maxrecords: int = 75,
    sort: str = DEFAULT_SORT,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_seconds: int = BACKOFF_SECONDS,
    startdatetime: str = None,
    enddatetime: str = None,
    verbose: bool = True,
) -> dict:
    """
    Fetch articles from the GDELT DOC 2.0 API for a single query, with
    retry + fixed backoff. Raises the last exception if all attempts fail
    so the caller decides explicitly how to handle a fully-failed query
    (skip it, log it, etc.) rather than silently returning partial data.

    startdatetime/enddatetime: optional, format YYYYMMDDHHMMSS. Only valid
    within GDELT's rolling 3-month window (see docs/rag_design.md). Used by
    src/pipeline.py to enforce a lookahead-safe retrieval window - evidence
    for an anomaly on date T should not include articles published after T,
    which matters for a system whose whole premise is same-day detection,
    not after-the-fact narrative construction.
    """
    params = build_query_params(query, maxrecords, sort, startdatetime, enddatetime)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            if verbose:
                print(f'  attempt {attempt}/{max_attempts} for query="{query}" ...')
            start = time.monotonic()
            resp = requests.get(
                GDELT_BASE_URL, params=params, headers=HEADERS, timeout=timeout
            )
            elapsed = time.monotonic() - start
            if verbose:
                print(f"    HTTP {resp.status_code} in {elapsed:.1f}s")
            resp.raise_for_status()

            try:
                return resp.json()
            except ValueError as parse_err:
                raise RuntimeError(
                    f"Response was not valid JSON (first 200 chars): "
                    f"{resp.text[:200]!r}"
                ) from parse_err

        except (requests.exceptions.RequestException, RuntimeError) as exc:
            last_exc = exc
            if verbose:
                print(f"    failed: {exc}")
            if attempt < max_attempts:
                # Default: backoff grows with attempt number (6s, then 12s,
                # for the default 3-attempt/5s-base settings) rather than
                # staying flat - see the module-level comment on why, for
                # 429s especially.
                sleep_seconds = backoff_seconds * attempt
                # If the server sent a Retry-After header (the 429 case,
                # when it knows best), honor that instead - but never less
                # than the computed backoff, so a tiny "Retry-After: 1"
                # can't make the client hit the endpoint faster than planned.
                retry_after = _parse_retry_after(exc)
                if retry_after is not None:
                    sleep_seconds = max(sleep_seconds, retry_after)
                    if verbose:
                        print(f"    server sent Retry-After={retry_after}s")
                if verbose:
                    print(f"    retrying in {sleep_seconds}s...")
                time.sleep(sleep_seconds)

    raise RuntimeError(
        f'All {max_attempts} attempts failed for query="{query}"'
    ) from last_exc


def gdelt_seendate_to_iso_date(seendate: str) -> str:
    """
    Convert GDELT's seendate (YYYYMMDDTHHMMSSZ) to YYYY-MM-DD.
    Returns "" (never "N/A") if missing or unparseable.
    """
    from datetime import datetime

    if not seendate:
        return ""
    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d")
    except ValueError:
        return ""
