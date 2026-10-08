"""
Raw GDELT ingestion for the shared team data pipeline (data/raw/).

This is kept separate from src/rag/retriever.py's output:
  - src/rag/retriever.py produces data/sample_articles.json in the RAG-
    specific schema (document_id, retrieval_score, relevance_score, text,
    etc.) - for Madhav's own retrieval/relevance-gating work.
  - This script produces data/raw/gdelt_raw.json in the shared team schema
    that ALL FOUR data-source scripts use (world_bank.py, gdelt.py, fred.py,
    noaa.py/freight.py), per the schema agreed Day 1 of Week 1:
        source, title, url, publication_date (YYYY-MM-DD),
        reference_date/known_as_of_date where applicable.
        Missing values: always blank, never "N/A".
  - It reuses src/rag/gdelt_client.py's retry/timeout/UA logic rather than
    duplicating it - GDELT's reliability problems apply equally to both
    use cases, so the fix belongs in one place.
  - No article-text fetch here. That's a RAG-track-specific enrichment step
    (src/rag/text_fetch.py); the shared raw data source doesn't need it and
    keeping this script's only external dependency as GDELT itself keeps it
    simple for the rest of the team to run/audit.

Usage:
    python -m src.data.gdelt
    python -m src.data.gdelt --queries "coffee drought Brazil" "coffee price"
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.rag.gdelt_client import fetch_gdelt, gdelt_seendate_to_iso_date

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "gdelt_raw.json"

# Same default queries as Week 1's verification script - this is the
# GDELT-as-shared-data-source use case, not the RAG retrieval use case,
# but Week 1's queries are still the team's agreed starting point for what
# "coffee disruption" coverage looks like.
DEFAULT_QUERIES = [
    "coffee drought Brazil",
    "Brazil coffee frost",
    "coffee supply disruption",
]

MAX_RECORDS_PER_QUERY = 75


def normalize_to_team_schema(query: str, raw_response: dict) -> list:
    """
    Map GDELT's raw article records onto the team's shared raw-data schema.
    Missing values are "" (never "N/A"), per the team-wide convention.
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
                # GDELT's seendate is crawl/index time, used here as the
                # best available proxy for known_as_of_date too - see the
                # reliability caveat in docs/data_source_report_gdelt_contribution.md.
                "known_as_of_date": pub_date,
                "domain": art.get("domain", "") or "",
                "language": art.get("language", "") or "",
                "sourcecountry": art.get("sourcecountry", "") or "",
                "query": query,
            }
        )
    return normalized


def run_ingestion(queries: list) -> dict:
    all_articles = []
    run_log = []

    for query in queries:
        print(f'Query: "{query}"')
        try:
            raw = fetch_gdelt(query, maxrecords=MAX_RECORDS_PER_QUERY)
            normalized = normalize_to_team_schema(query, raw)
            print(f"  -> {len(normalized)} articles retrieved\n")
            all_articles.extend(normalized)
            run_log.append({"query": query, "status": "success", "count": len(normalized)})
        except Exception as exc:  # noqa: BLE001
            print(f"  -> FAILED after all retries: {exc}\n")
            run_log.append({"query": query, "status": "failed", "error": str(exc)})

    return {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GDELT",
        "run_log": run_log,
        "articles": all_articles,
    }


def main():
    parser = argparse.ArgumentParser(description="Raw GDELT ingestion for the shared team pipeline")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES,
                         help="Queries to run (default: the Week 1 coffee-disruption set)")
    args = parser.parse_args()

    output = run_ingestion(args.queries)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Total articles saved: {len(output['articles'])}")
    print(f"Output written to: {OUTPUT_PATH}")
    for entry in output["run_log"]:
        print(f"  {entry}")


if __name__ == "__main__":
    main()
