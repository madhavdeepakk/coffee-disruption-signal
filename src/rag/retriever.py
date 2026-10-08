"""
Week 2: RAG retriever.

Orchestrates: GDELT query -> dedupe -> article text fetch -> keyword
relevance scoring -> save to the frozen schema (docs/rag_design.md) ->
compare scores against Week 1's manual labels.

Design choices (see docs/rag_design.md for full reasoning):
  - Reuses data/raw/gdelt_raw.json from Week 1 by default instead of
    re-querying GDELT's unreliable endpoint again, unless --refresh is
    passed. The article-text fetch step is a new, separate network
    dependency (src/rag/text_fetch.py) and doesn't need a second round of
    GDELT flakiness stacked on top of it.
  - relevance_score here is Week 2's BASIC keyword score, not the Week 6
    relevance-gating score. The schema field is the same column; the
    methodology will change in Week 6, and that's expected/documented.

Usage:
    python -m src.rag.retriever                  # use existing gdelt_raw.json
    python -m src.rag.retriever --refresh         # re-query GDELT live
    python -m src.rag.retriever --sample-size 75  # default is 75 (50-100 range)
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from src.rag.gdelt_client import fetch_gdelt, gdelt_seendate_to_iso_date
from src.rag.text_fetch import fetch_texts_for_documents

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_GDELT_PATH = REPO_ROOT / "data" / "raw" / "gdelt_raw.json"
SAMPLE_OUTPUT_PATH = REPO_ROOT / "data" / "sample_articles.json"
MANUAL_LABELS_PATH = REPO_ROOT / "data" / "week1_manual_labels.json"
COMPARISON_OUTPUT_PATH = REPO_ROOT / "docs" / "week2_keyword_scoring_comparison.md"

DEFAULT_QUERIES = [
    "coffee drought Brazil",
    "Brazil coffee frost",
    "coffee supply disruption",
]

# Terms suggestive of a disruption/causal narrative, distinct from the raw
# query terms themselves. A small, documented starting list, not tuned
# against a large labeled set yet (that's Week 4). It exists so scoring
# isn't purely "did the query words appear" (which the Week 1 classification
# showed produces false positives from festivals, broker ads, etc. that
# also contain the literal query words).
DISRUPTION_VOCAB = [
    "drought", "frost", "flood", "heatwave", "el nino", "el niño", "la nina",
    "la niña", "shortage", "disrupt", "supply", "harvest", "crop damage",
    "export", "tariff", "shipping", "price spike", "surge", "plunge",
    "spike", "soar", "tumble", "crisis", "shortfall", "damage", "storm",
    "hurricane", "wildfire", "yield loss", "rainfall", "dry spell",
]


def make_document_id(url: str) -> str:
    """Stable ID derived from url (not row index) so re-retrieval of the
    same article across different queries dedupes to one ID."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def compute_keyword_relevance_score(title: str, text: str, query: str) -> float:
    """
    Basic keyword relevance score (Week 2 baseline - NOT the Week 6
    relevance gate). Combines:
      - fraction of query terms found in title (weighted 2x) and text (1x)
      - fraction of DISRUPTION_VOCAB terms found in title (weighted 2x) and text (1x)
    Both title and full text contribute; title matches count more because a
    query term or disruption term appearing in the headline is stronger
    evidence than the same term appearing once in a long article body.

    Returns a score in [0, 1]. This is intentionally simple and will
    disagree with human judgment in predictable ways (see the comparison
    report) - documenting those disagreements is the actual Week 2
    deliverable, not a high score in isolation.
    """
    title_tokens = tokenize(title)
    text_tokens = tokenize(text)
    query_terms = [t for t in re.findall(r"[a-zA-Z]+", query.lower())]

    if not query_terms:
        return 0.0

    query_title_hits = sum(1 for t in query_terms if t in title_tokens)
    query_text_hits = sum(1 for t in query_terms if t in text_tokens)
    query_title_frac = query_title_hits / len(query_terms)
    query_text_frac = query_text_hits / len(query_terms)

    vocab_terms = DISRUPTION_VOCAB
    vocab_title_hits = sum(
        1 for term in vocab_terms if all(w in title_tokens for w in term.split())
    )
    vocab_text_hits = sum(
        1 for term in vocab_terms if all(w in text_tokens for w in term.split())
    )
    vocab_title_frac = vocab_title_hits / len(vocab_terms)
    vocab_text_frac = vocab_text_hits / len(vocab_terms)

    raw_score = (
        2 * query_title_frac + 1 * query_text_frac
        + 2 * vocab_title_frac + 1 * vocab_text_frac
    )
    max_possible = 2 + 1 + 2 + 1  # = 6, when every fraction is 1.0
    return round(min(raw_score / max_possible, 1.0), 4)


def load_or_fetch_raw_articles(refresh: bool, queries: list) -> list:
    if not refresh and RAW_GDELT_PATH.exists():
        print(f"Using existing {RAW_GDELT_PATH} (pass --refresh to re-query GDELT live)")
        with open(RAW_GDELT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data["articles"]

    print("Querying GDELT live...")
    all_articles = []
    for query in queries:
        try:
            raw = fetch_gdelt(query)
            for art in raw.get("articles", []) or []:
                all_articles.append(
                    {
                        "source": "GDELT",
                        "title": art.get("title", "") or "",
                        "url": art.get("url", "") or "",
                        "publication_date": gdelt_seendate_to_iso_date(art.get("seendate", "")),
                        "domain": art.get("domain", "") or "",
                        "query": query,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(f'  query "{query}" failed after retries: {exc}')
    return all_articles


def dedupe_by_url(articles: list) -> list:
    seen = set()
    deduped = []
    for a in articles:
        url = a.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(a)
    return deduped


def build_sample_articles(sample_size: int, refresh: bool, queries: list) -> list:
    raw_articles = load_or_fetch_raw_articles(refresh, queries)
    deduped = dedupe_by_url(raw_articles)
    print(f"{len(raw_articles)} raw articles -> {len(deduped)} after url dedupe")

    sample = deduped[:sample_size]
    print(f"Using {len(sample)} documents for this retrieval run "
          f"(target range 50-100)")

    print("Fetching article body text (this hits many different domains, "
          "sequentially, with politeness delay)...")
    fetch_stats = fetch_texts_for_documents(sample, verbose=True)
    print(f"Text fetch stats: {fetch_stats}")

    documents = []
    for a in sample:
        title = a.get("title", "")
        text = a.get("text", "")
        query = a.get("query", "")
        score = compute_keyword_relevance_score(title, text, query)
        documents.append(
            {
                "document_id": make_document_id(a["url"]),
                "title": title,
                "url": a["url"],
                "publication_date": a.get("publication_date", ""),
                "source": a.get("source", "GDELT"),
                "text": text,
                "query": query,
                "retrieval_score": score,
                "relevance_score": "",  # reserved for Week 6 gate, intentionally blank
                "text_fetch_error": a.get("text_fetch_error", ""),
            }
        )
    return documents, fetch_stats


def compare_to_manual_labels(documents: list) -> str:
    """
    Match retrieved documents to Week 1's manual labels by url, compare the
    Week 2 keyword score's implied classification against the human label,
    and write out a report of agreements/false positives/false negatives.
    Returns the report as a markdown string.
    """
    if not MANUAL_LABELS_PATH.exists():
        return "(No manual labels file found - skipping comparison.)"

    with open(MANUAL_LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    labels_by_url = {entry["url"]: entry["manual_label"] for entry in labels}

    matched = [d for d in documents if d["url"] in labels_by_url]
    if not matched:
        return (
            "(No overlap between this run's retrieved documents and the "
            "Week 1 manually-labeled set - comparison skipped. This can "
            "happen because GDELT's 3-month rolling window shifts which "
            "articles a query returns over time.)"
        )

    lines = [
        "# Week 2 — Keyword Relevance Score vs Week 1 Manual Labels",
        "",
        f"{len(matched)} of {len(documents)} retrieved documents overlap "
        f"with the Week 1 manually-labeled set (matched by url).",
        "",
        "Manual labels: R = Relevant, P = Partially relevant, I = Irrelevant.",
        "Score-implied label uses simple thresholds for illustration only "
        "(>=0.5 = 'R-like', 0.25-0.5 = 'P-like', <0.25 = 'I-like') - these "
        "thresholds are NOT the frozen Week 4 threshold, just a way to make "
        "disagreements visible now.",
        "",
        "| url (truncated) | manual label | keyword score | score-implied | agree? |",
        "|---|---|---|---|---|",
    ]

    def implied_label(score):
        if score >= 0.5:
            return "R"
        if score >= 0.25:
            return "P"
        return "I"

    false_positives = []  # keyword scored high, human said Irrelevant
    false_negatives = []  # keyword scored low, human said Relevant
    agreements = 0

    for d in matched:
        manual = labels_by_url[d["url"]]
        score = d["retrieval_score"]
        implied = implied_label(score)
        agree = "yes" if implied == manual else "no"
        if implied == manual:
            agreements += 1
        if manual == "I" and implied in ("R", "P"):
            false_positives.append((d["url"], d["title"], score))
        if manual == "R" and implied in ("I",):
            false_negatives.append((d["url"], d["title"], score))

        lines.append(
            f"| {d['url'][:60]}... | {manual} | {score:.3f} | {implied} | {agree} |"
        )

    lines += [
        "",
        f"**Agreement rate: {agreements}/{len(matched)} "
        f"({100*agreements/len(matched):.0f}%)**",
        "",
        f"**False positives ({len(false_positives)})** — keyword score rated "
        f"'relevant-ish' but manually labeled Irrelevant:",
    ]
    for url, title, score in false_positives:
        lines.append(f"- [{score:.3f}] {title} — {url}")
    if not false_positives:
        lines.append("- (none in this overlap)")

    lines += [
        "",
        f"**False negatives ({len(false_negatives)})** — keyword score rated "
        f"'irrelevant-ish' but manually labeled Relevant:",
    ]
    for url, title, score in false_negatives:
        lines.append(f"- [{score:.3f}] {title} — {url}")
    if not false_negatives:
        lines.append("- (none in this overlap)")

    lines += [
        "",
        "## Interpretation",
        "",
        "A basic keyword/vocabulary overlap score is expected to disagree "
        "with human judgment in specific, explainable ways - not randomly. "
        "False positives are the more dangerous failure mode for this "
        "project (per the relevance-gating principle: an over-eager score "
        "feeding a confident-but-wrong LLM explanation), so they matter "
        "more here than false negatives. Any false positives above should "
        "be inspected for *why* they scored high - the most likely cause is "
        "surface keyword overlap without the disruption-vocabulary term "
        "appearing in the true causal sense (e.g. a broker product page "
        "mentioning 'Robusta' or a festival page mentioning 'Brazil' and "
        "'coffee'). This is exactly the failure mode Week 6's relevance "
        "gate needs to catch, and this comparison is preliminary evidence "
        "for why keyword-only scoring alone (without semantic retrieval or "
        "a proper gate) is not sufficient.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Week 2 RAG retriever")
    parser.add_argument("--refresh", action="store_true",
                         help="Re-query GDELT live instead of reusing data/raw/gdelt_raw.json")
    parser.add_argument("--sample-size", type=int, default=75,
                         help="Number of documents to retrieve/save (target range 50-100)")
    args = parser.parse_args()

    documents, fetch_stats = build_sample_articles(
        sample_size=args.sample_size, refresh=args.refresh, queries=DEFAULT_QUERIES
    )

    SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "week2_v1",
                "document_count": len(documents),
                "text_fetch_stats": fetch_stats,
                "documents": documents,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved {len(documents)} documents to {SAMPLE_OUTPUT_PATH}")

    report = compare_to_manual_labels(documents)
    COMPARISON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Comparison report written to {COMPARISON_OUTPUT_PATH}")
    print("\n" + "=" * 60)
    print(report[:2000])


if __name__ == "__main__":
    main()
