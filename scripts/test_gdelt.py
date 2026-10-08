"""
Week 1 GDELT DOC 2.0 API verification script.

Purpose: confirm the API is reachable and returns usable article data for
three candidate queries, using retry/timeout/UA settings tuned for GDELT's
unreliable public endpoint (observed ~87% failure rate via check-host.net;
successful responses take 22-25s). The failures are on GDELT's side, so the
script uses generous timeouts and backoff to tolerate them.

Output: raw GDELT results saved to data/raw/gdelt_raw.json, matching the
team's shared raw-data schema:
    source, title, url, publication_date (YYYY-MM-DD), reference_date/
    known_as_of_date where applicable. Missing values are always blank,
    never "N/A".

Usage:
    python scripts/test_gdelt.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT's public API has been observed with a high failure rate and slow
# successful responses (22-25s) in testing, so these settings are more
# generous than the defaults for a typical API.
REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 5

# Missing a realistic browser User-Agent is a common cause of silent
# connection resets on GDELT's endpoint.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

QUERIES = [
    "coffee drought Brazil",
    "Brazil coffee frost",
    "coffee supply disruption",
]

MAX_RECORDS = 75  # keep well under the 250 cap; plenty for manual review

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "gdelt_raw.json"


def fetch_gdelt(query: str, maxrecords: int = MAX_RECORDS) -> dict:
    """
    Fetch articles from the GDELT DOC 2.0 API for a single query, with
    retry + exponential-ish backoff. Raises the last exception if all
    attempts fail, so the caller can log/skip that query explicitly rather
    than silently continuing with partial data.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(maxrecords),
        "sort": "hybridrel",  # relevance-ranked, not just newest-first
    }

    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f'  attempt {attempt}/{MAX_ATTEMPTS} for query="{query}" ...')
            start = time.monotonic()
            resp = requests.get(
                GDELT_BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            elapsed = time.monotonic() - start
            print(f"    HTTP {resp.status_code} in {elapsed:.1f}s")
            resp.raise_for_status()

            # GDELT sometimes returns HTML error pages with a 200 status
            # (e.g. rate-limit or malformed-query messages) instead of JSON.
            # Treat a JSON parse failure as a retryable failure, not a
            # silent empty result.
            try:
                data = resp.json()
            except ValueError as parse_err:
                raise RuntimeError(
                    f"Response was not valid JSON (first 200 chars): "
                    f"{resp.text[:200]!r}"
                ) from parse_err

            return data

        except (requests.exceptions.RequestException, RuntimeError) as exc:
            last_exc = exc
            print(f"    failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                print(f"    retrying in {BACKOFF_SECONDS}s...")
                time.sleep(BACKOFF_SECONDS)

    raise RuntimeError(
        f'All {MAX_ATTEMPTS} attempts failed for query="{query}"'
    ) from last_exc


def gdelt_seendate_to_iso_date(seendate: str) -> str:
    """
    Convert GDELT's seendate format (YYYYMMDDTHHMMSSZ) to YYYY-MM-DD per
    the team schema. Returns "" (never "N/A") if the field is missing or
    unparseable.
    """
    if not seendate:
        return ""
    try:
        dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return ""


def normalize_articles(query: str, raw_response: dict) -> list[dict]:
    """
    Map GDELT's raw article records onto the team's shared raw-data schema:
    source, title, url, publication_date, reference_date/known_as_of_date
    where applicable. Missing values are blank strings, never "N/A".
    """
    articles = raw_response.get("articles", []) or []
    normalized = []
    for art in articles:
        pub_date = gdelt_seendate_to_iso_date(art.get("seendate", ""))
        normalized.append(
            {
                "source": "GDELT",
                "title": art.get("title", "") or "",
                "url": art.get("url", "") or "",
                "publication_date": pub_date,
                # GDELT's seendate is when GDELT's crawler indexed/saw the
                # article, which we treat as our best available proxy for
                # both "publication_date" and "known_as_of_date" here.
                # This is a Week 1 simplification, flagged for revisit once
                # Yashika's schema conventions are finalized.
                "known_as_of_date": pub_date,
                "domain": art.get("domain", "") or "",
                "language": art.get("language", "") or "",
                "sourcecountry": art.get("sourcecountry", "") or "",
                "query": query,
            }
        )
    return normalized


def main():
    print("GDELT DOC 2.0 API verification script")
    print(f"Timeout: {REQUEST_TIMEOUT_SECONDS}s | Max attempts: {MAX_ATTEMPTS} | "
          f"Backoff: {BACKOFF_SECONDS}s\n")

    all_articles = []
    run_log = []

    for query in QUERIES:
        print(f'Query: "{query}"')
        try:
            raw = fetch_gdelt(query)
            articles = normalize_articles(query, raw)
            print(f"  -> {len(articles)} articles retrieved\n")
            all_articles.extend(articles)
            run_log.append(
                {"query": query, "status": "success", "count": len(articles)}
            )
        except Exception as exc:  # noqa: BLE001 - intentionally broad; log and continue
            print(f"  -> FAILED after all retries: {exc}\n")
            run_log.append(
                {"query": query, "status": "failed", "error": str(exc)}
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GDELT",
        "endpoint": GDELT_BASE_URL,
        "run_log": run_log,
        "articles": all_articles,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Total articles saved: {len(all_articles)}")
    print(f"Output written to: {OUTPUT_PATH}")
    print("Run log:")
    for entry in run_log:
        print(f"  {entry}")

    if all(entry["status"] == "failed" for entry in run_log):
        print(
            "\nWARNING: every query failed. This is consistent with GDELT's "
            "unreliable public endpoint (~87% failure rate observed). "
            "Re-running the script often succeeds."
        )


if __name__ == "__main__":
    main()
