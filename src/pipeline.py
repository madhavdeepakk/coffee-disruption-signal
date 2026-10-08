"""
End-to-end pipeline: anomaly -> GDELT retrieval -> keyword scoring ->
relevance gate -> LLM explanation.

Takes a flagged anomaly from src/modeling/anomaly_detector.py's output and
runs it through the full RAG chain against freshly retrieved GDELT
articles.

Lookahead discipline: retrieval is restricted to articles published on or
before the anomaly date (see gdelt_client.fetch_gdelt's startdatetime/
enddatetime params). The system is built for same-day detection and
explanation, so evidence published after the anomaly date is explicitly
excluded rather than incidentally missing.

Each document is scored two ways: the keyword score (src/rag/retriever.py)
and, when the embedding model is available, a semantic score
(src/rag/vector_store.py). The gate accepts a document if either score
clears its own threshold. The semantic channel covers non-English articles
that the English-only keyword scorer misses (a known coverage gap), and
uses a higher threshold because embedding similarity for topically related
text sits in a compressed, generally-high band. See relevance_gate.py for
the thresholds.

Works for any commodity in src/config/commodities.py: GDELT queries, the
semantic reference query, and the anomaly-detections file all come from
that commodity's CommodityConfig, following the proposal's "Build One
Deeply, Architect for Any" idea (S4).

GDELT's public endpoint has a high observed failure rate and rate-limits
(429) frequently, so successful responses are cached on disk (see
gdelt_client.cache_get/cache_put and GDELT_CACHE_DIR below). Each anomaly
date is a fixed historical date queried over a fixed, lookahead-safe
window, so its articles never change: once a (query, window) pair gets
past the rate limiter once, later runs read it from disk. Pass --no-cache
to force a fresh fetch. Only successful responses are cached; a
failed/empty query is retried live next time.

Usage:
    python -m src.pipeline --date 2026-07-07                        # coffee (default)
    python -m src.pipeline --date 2022-03-01 --commodity crude_oil
    python -m src.pipeline --date 2022-03-01 --commodity wheat
    python -m src.pipeline --date 2026-07-07 --no-cache             # force fresh GDELT fetch
"""

import argparse
import json
import sys
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

# Windows terminals default to the cp1252 encoding, which cannot encode the
# accented characters common in article titles and explanations (é, ñ, ç, ...).
# Printing those there raises UnicodeEncodeError and would crash a run. Force
# UTF-8 output and replace anything that still can't be shown, so printing is
# always safe regardless of the console's code page.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - older/odd streams may lack reconfigure
        pass

import pandas as pd

from src.rag.gdelt_client import (
    fetch_gdelt, gdelt_seendate_to_iso_date,
    build_query_params, cache_get, cache_put, DEFAULT_SORT,
)
from src.rag.retriever import make_document_id, compute_keyword_relevance_score
from src.rag.text_fetch import fetch_article_text, cached_article_text
from src.rag.relevance_gate import gate_documents
from src.rag.explainer import generate_explanation
from src.rag import direction as direction_mod
from src.rag import faithfulness as faithfulness_mod
from src.rag.news_fallback import load_fallback_articles
from src.rag.outcome import (
    GUARD_RULE, RetrievalMeta, classify as classify_outcome, apply_direction_guard,
)
from src.rag.evidence import (
    annotate_recency, collapse_syndicated, filter_to_window, recency_summary,
)
from src.rag.rerank import rerank_by_anomaly_query
from src.rag import blind_evidence
from src.rag import market_report
from src.config.commodities import COMMODITIES, get_commodity

# Semantic scoring (src/rag/vector_store.py) covers the keyword scorer's
# non-English blind spot: on-topic Portuguese/Spanish articles score low on
# the English-only keyword scorer and would otherwise be rejected by the
# gate. It uses onnxruntime + tokenizers directly (not sentence-transformers
# or transformers) to avoid a scipy/scikit-learn dependency that a
# machine-level security policy can block at import time - see
# src/rag/vector_store.py's module docstring.
# The import is soft: if the packages aren't installed or fail to load, the
# pipeline runs on keyword scoring alone rather than crashing.
# _SEMANTIC_IMPORT_ERROR is initialized here (not only in the except) so the
# name is always defined for the "semantic scoring unavailable" warning
# branch in retrieve_evidence.
_SEMANTIC_IMPORT_ERROR = None
try:
    from src.rag.vector_store import score_documents_against_query
    SEMANTIC_SCORING_AVAILABLE = True
except ImportError as _exc:
    SEMANTIC_SCORING_AVAILABLE = False
    _SEMANTIC_IMPORT_ERROR = _exc

REPO_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_WINDOW_DAYS_BEFORE = 10
# The short window used by the "recent" queries: the day of the move and the
# five calendar days before it, which always covers the three trading days
# before (a Wednesday move reaches back to the Friday). A query over the full
# ten-day window returns its twenty most relevant articles from anywhere in
# it, and most of those are too old for the decision rule to use; the same
# query over this window spends all twenty on the days that matter.
RECENT_WINDOW_DAYS_BEFORE = 5
# Results asked for per recent query. The full-window queries keep
# MAX_ARTICLES_PER_QUERY: their job is topic coverage. A recent query's job is
# to find the coverage of the move itself, and the decision can only use what
# was retrieved, so it asks for more.
RECENT_MAX_ARTICLES = 50
# Results asked for by a combined search (one request standing in for several
# queries): the most GDELT returns for one request.
COMBINED_MAX_ARTICLES = 250
# In blind mode a recent document is read if it clears the relevance gate OR
# scores at least this on the semantic channel. The gate's own bar (0.85) was
# set to keep off-topic material out of an explanation the model wrote
# freely. The blind reading is itself a filter - a document that says nothing
# about coffee prices gets no direction and no reason, and counts for nothing
# - so near the bar it is cheaper to read a document than to miss the one
# report of the day. Every reading records whether its document cleared the
# gate proper, so the two can be compared afterwards.
RECENT_SEMANTIC_FLOOR = 0.80

# Written into every output file. Bump it whenever a change alters what the
# pipeline retrieves, gates or sends to the model, so evaluation can tell
# which runs are comparable. Outputs with no pipeline_version field predate
# versioning (treated as "1").
#   2: out-of-window documents filtered locally, syndicated copies collapsed,
#      evidence packed to the provider's request limit, prompt v2,
#      temperature 0, direction guard, Google News queries date-scoped.
#   3: document age given to the model in trading days and prompt rule 5
#      reworded accordingly (version 2 called Friday's news "3 days before" a
#      Monday move and the model refused on that basis).
#   4: three GDELT queries added that are not about supply shocks (the six
#      before were all frost, drought, tariffs and shipping - stories about
#      prices rising - so on a day prices fell the evidence started out
#      against the move); the explain/refuse decision moved from the model
#      to a rule over a direction-blind reading of the recent documents
#      (src/rag/blind_evidence.py). The earlier decision path is kept as
#      --decision legacy.
#   5: the direction-neutral queries, and one aimed at daily market reports,
#      are sent over the last five days only instead of the ten-day window,
#      and the news feed is asked once more for just those days. The decision
#      rule uses the day of the move and the two trading days before it, and
#      a ten-day search spent most of its results on older days.
#   6: recent queries ask for 50 results; in blind mode recent documents just
#      under the gate's bar are read as well (and marked as such); the reading
#      also records the size and timing of the move each document reports;
#      article text is fetched only for documents recent enough to be read.
#   7: GDELT is asked for two combined searches per date (one over the full
#      window, one over the recent days) instead of ten separate ones; the
#      separate queries are read from the cache only. GDELT answered about one
#      request in five whatever the pacing, so the number of requests was the
#      thing to cut. The pause between requests went from 3 to 6 seconds:
#      GDELT asks for at most one every 5.
#   8: no change to what is retrieved or decided. Version 7's larger document
#      sets made the relevance model run out of memory (it scored every
#      document of a date in one batch), the gate fell back to keyword scores
#      alone, and every date was refused without a reading. The model now
#      runs in small batches, and a run whose semantic scoring fails ends in
#      an error instead of saving a refusal. The version is raised so that no
#      output from that fault is taken for a result.
#   9: what the blind reading is shown. The dozen documents read are chosen
#      market reports first (src/rag/market_report.py) instead of by
#      similarity to one phrase about supply shocks; each is shown from its
#      opening and the passages that mention the commodity instead of from its
#      first character; a document first seen before the exchange opened is
#      counted with the session before; and a report of a move from an earlier
#      day, or a forecast, no longer counts as a report of the day's move.
#      Made after the version 8 run on the labelled dates, where the cause was
#      retrieved and not read.
#  10: three changes that came out of the version 9 run on the labelled
#      dates. (a) Documents the reading's answer left out are sent again, on
#      their own, to the model that answered: the answer reached its output
#      limit on 7 of 18 dates and 29 of the 196 documents shown got no
#      reading - always the last ones in reading order. Nothing a model
#      already answered is asked again. (b) A third GDELT request per date,
#      over the day of the move alone (see gdelt_query_plan): on three of
#      the four dates still refused, no report of the day with a reason had
#      been retrieved. (c) The headline scorer of src/rag/market_report.py
#      reads Arabic, Korean and Turkish: on the fourth, a report of the
#      day's rise was retrieved and not read because its headline was in
#      Arabic. (b) and (c) were made with those four dates in view.
#  11: what can support an explanation. Version 10 was first run on quiet
#      days given an invented move (src/evaluation/robustness.py) and
#      explained 5 of the first 10: each time from one routine report of the
#      day, twice one that gave "technical adjustments" as the reason, once
#      one about robusta, once one stating a move of 0.6% against a claimed
#      3.9%. The reading now also records how big each document's words make
#      the move, whether the reason is an event or only a description of
#      trading, and whether the move is of this market; a report supports an
#      explanation only if it is about this market, gives an event as the
#      reason and describes a move of this size (blind_evidence.decide). The
#      reading is still never told the move or its size. Made with those ten
#      trials in view, so it has to be confirmed on other quiet days.
PIPELINE_VERSION = "11"

# How the explain/refuse decision is made.
#   "blind":  the model reads the recent documents without being told the
#             move; a fixed rule decides (src/rag/blind_evidence.py).
#   "legacy": the model is told the move and decides, then the direction
#             guard may withhold its explanation. Kept for comparison.
DECISION_MODE = "blind"
DECISION_MODES = ("blind", "legacy")

# On-disk cache of successful GDELT responses (see gdelt_client.cache_get/
# cache_put for the rationale). Anomaly dates are fixed historical dates
# queried over a fixed, lookahead-safe window, so their retrieved articles
# never change: caching means each (query, window) pair only has to get past
# GDELT's rate limiter once, and later runs read from disk. Pass --no-cache
# to force fresh retrieval.
GDELT_CACHE_DIR = REPO_ROOT / "data" / "gdelt_cache"

# Local news-fallback directory (see src/rag/news_fallback.py). Opt-in by
# existence: a run only uses it if a file exists at
# data/news_fallback/<commodity>/<date>.json, so it never changes a run that
# doesn't have one. It rescues a date GDELT keeps rate-limiting to zero; the
# production-grade answer is GDELT-on-BigQuery.
FALLBACK_DIR = REPO_ROOT / "data" / "news_fallback"

# GDELT queries and the semantic reference query come from each commodity's
# CommodityConfig (src/config/commodities.py), passed into
# retrieve_evidence() below, so this file works unchanged across commodities.
# See that module's docstring for why narrow, multi-query GDELT retrieval
# (rather than one broad AND-of-many-terms query) is the design.
MAX_ARTICLES_PER_QUERY = 20
FETCH_TEXT = True  # slower, but gives the scorer/explainer real content, not just titles

# Pause between GDELT requests in retrieve_evidence() below. GDELT's own
# rejection message asks for at most one request every 5 seconds. This was 3
# until pipeline version 7, which was under that after every answered request.
# (In 764 logged requests the answer rate was the same one in five at 3, 5 and
# 10 seconds, so the short pause was a fault to fix, not the cause of the
# rejections.)
INTER_QUERY_DELAY_SECONDS = 6


def get_anomaly(date_str: str, anomalies_file: str) -> dict:
    df = pd.read_csv(REPO_ROOT / "results" / anomalies_file)
    row = df[df["date"] == date_str]
    if row.empty:
        raise SystemExit(f"No row for {date_str} in results/{anomalies_file} - "
                          f"run src.modeling.anomaly_detector first.")
    row = row.iloc[0]
    pct_move = None
    prior_idx = df.index[df["date"] == date_str][0] - 1
    if prior_idx >= 0:
        prior_price = df.iloc[prior_idx]["price"]
        pct_move = 100 * (row["price"] - prior_price) / prior_price

    def _num(col):
        value = row[col] if col in row.index else None
        return None if value is None or pd.isna(value) else float(value)

    anomaly_type = row["anomaly_type"] if "anomaly_type" in row.index else None
    if anomaly_type is None or pd.isna(anomaly_type):
        anomaly_type = None
    cumulative_z = _num("cumulative_z_score")
    cumulative_return = _num("cumulative_return")

    # Direction of the move that was actually flagged. A "trend" anomaly is
    # flagged on the multi-day cumulative return, not the day's own move, and
    # the two can point opposite ways: 41 of the 151 trend-only coffee
    # anomalies have a daily z-score whose sign disagrees with the cumulative
    # one (2026-06-18: flagged for an +11.3% ten-day rise on a day the price
    # slipped). Taking the daily sign there asks the model to explain a move
    # in the wrong direction.
    if anomaly_type == "trend" and cumulative_z is not None:
        direction = "up" if cumulative_z > 0 else "down"
    else:
        direction = "up" if row["z_score"] > 0 else "down"

    return {
        "date": date_str,
        "price": float(row["price"]),
        "z_score": float(row["z_score"]),
        "anomaly_flag": bool(row["anomaly_flag"]),
        "anomaly_type": anomaly_type,
        "pct_move": pct_move,
        "cumulative_pct_move": None if cumulative_return is None else 100 * cumulative_return,
        "cumulative_z_score": cumulative_z,
        "direction": direction,
    }


# One GDELT request for one date. `live` requests are sent when they are not
# in the cache; the others are only ever read from it. `label` is what a
# document found by this request records as its source.
GdeltRequest = namedtuple(
    "GdeltRequest", "query params start_dt startdatetime enddatetime maxrecords live label")


def gdelt_query_plan(anomaly_date: str, queries: list, recent_queries: list = None,
                     combined_query: str = None, combined_recent_query: str = None,
                     combined_day_query: str = None) -> list:
    """Every GDELT request for one date, as GdeltRequest tuples, in the order
    they are read.

    `queries` cover the full retrieval window; `recent_queries` cover only
    the last few days (RECENT_WINDOW_DAYS_BEFORE) and ask for more results
    each. A combined query is one request standing in for a whole group
    (`combined_query` for `queries`, `combined_recent_query` for
    `recent_queries`): when it is given, it is the group's only live request
    and comes first, and the group's separate queries are read from the cache
    only - answers an earlier run already has are still used, and nothing is
    spent asking for them again.

    One function builds these for both the pipeline and the cache-filling
    pass in scripts/run_eval_set.py, so the two can never ask for (and cache)
    different requests.

    `combined_day_query` is one more live request, over the day of the move
    alone. Each combined request returns at most COMBINED_MAX_ARTICLES results
    and both of the others reach that limit on nearly every date, so the day's
    own market reports compete for places with five or ten days of coverage;
    on three of the four labelled dates version 9 still refused, no report of
    the day that gave a reason was among what came back. It is read last, so
    a document counts as found by it only if neither other request returned
    it."""
    end_dt = datetime.strptime(anomaly_date, "%Y-%m-%d") + timedelta(days=1)  # include the day itself
    enddatetime = end_dt.strftime("%Y%m%d%H%M%S")

    def window(days):
        start_dt = end_dt - timedelta(days=days + 1)
        return start_dt, start_dt.strftime("%Y%m%d%H%M%S")

    def request(query, days, maxrecords, live, label):
        start_dt, startdatetime = window(days)
        params = build_query_params(query, maxrecords, DEFAULT_SORT, startdatetime, enddatetime)
        return GdeltRequest(query, params, start_dt, startdatetime, enddatetime, maxrecords,
                            live, label)

    groups = (
        (queries or [], combined_query, RETRIEVAL_WINDOW_DAYS_BEFORE, MAX_ARTICLES_PER_QUERY,
         "combined search, full window"),
        (recent_queries or [], combined_recent_query, RECENT_WINDOW_DAYS_BEFORE,
         RECENT_MAX_ARTICLES, "combined search, recent days"),
    )
    plan = [request(combined, days, COMBINED_MAX_ARTICLES, True, label)
            for _, combined, days, _, label in groups if combined]
    if combined_day_query:
        plan.append(request(combined_day_query, 0, COMBINED_MAX_ARTICLES, True,
                            "combined search, day of the move"))
    for group, combined, days, maxrecords, _ in groups:
        plan += [request(query, days, maxrecords, not combined, query) for query in group]
    return plan


def semantic_scoring_failed(documents: list) -> bool:
    """True if the relevance model is installed and yet no document carries a
    semantic score: the model failed while running (out of memory, a broken
    download), and the gate has only keyword scores to go on."""
    return bool(documents) and SEMANTIC_SCORING_AVAILABLE and not any(
        d.get("semantic_score") is not None for d in documents)


def search_arguments(cfg) -> dict:
    """The search settings of a commodity as retrieve_evidence's keyword
    arguments, so every caller asks for the same requests."""
    return {"recent_queries": cfg.gdelt_recent_queries,
            "combined_query": cfg.gdelt_combined_query or None,
            "combined_recent_query": cfg.gdelt_combined_recent_query or None,
            "combined_day_query": cfg.gdelt_combined_day_query or None}


def retrieve_evidence(anomaly_date: str, queries: list, semantic_reference_query: str,
                      cache_dir=GDELT_CACHE_DIR, commodity_key: str = None,
                      fallback_dir=FALLBACK_DIR, return_meta: bool = False,
                      gdelt_live: bool = True, fetch_text=None,
                      enforce_window: bool = True, collapse_duplicates: bool = True,
                      recent_queries: list = None, fetch_text_max_age: int = None,
                      combined_query: str = None, combined_recent_query: str = None,
                      combined_day_query: str = None):
    """Retrieve lookahead-safe news evidence for an anomaly date.

    fetch_text_max_age: if set, article text is downloaded only for documents
    published at most this many trading days before the anomaly; older ones
    keep their headline. Downloading text is the slowest part of a run, and a
    decision that only reads recent documents has no use for the body of a
    week-old one.

    recent_queries: GDELT queries sent over the short recent window only (see
    RECENT_WINDOW_DAYS_BEFORE), and when given the news aggregator is also
    asked once more for just those days. These are the searches for coverage
    of the move itself.

    combined_query / combined_recent_query: one request standing in for all
    of `queries` / `recent_queries` (see gdelt_query_plan). When given, it is
    the only request sent for its group; the separate queries are read from
    the cache if an earlier run has them and are otherwise left alone.
    combined_day_query: one more request, over the day of the move alone.

    Returns the list of scored documents. With return_meta=True, returns
    (documents, RetrievalMeta) instead - the meta records whether retrieval
    itself actually ran, which is what lets the caller tell a retrieval
    OUTAGE (RETRIEVAL_FAILED) apart from a genuine "no relevant news"
    refusal. The default (return_meta=False) keeps the original signature so
    existing callers (labeling script, cache tests) are unchanged.

    gdelt_live: if False, GDELT queries only read the on-disk cache — no
    live API calls. This avoids the 429 rate-limit delays that dominate
    evaluation runtime. The news aggregator (Google News RSS + industry
    feeds) still runs normally. Default True preserves existing behavior.

    fetch_text: whether to download article body text from each URL. None
    (default) uses the module-level FETCH_TEXT flag. Pass False to skip text
    fetching entirely — keyword scoring will use titles only. This eliminates
    the minutes spent on dead/paywalled URLs in evaluation runs where full
    text is not needed. Pass "cache" to use body text only where it is already
    in the on-disk text cache (no network): the offline evaluations use this
    so they score the same text a live run saw without re-fetching anything.

    enforce_window / collapse_duplicates: the two clean-up steps applied to
    the retrieved articles (see src/rag/evidence.py). Both default to on;
    they are switchable only so the ablation study can measure what each one
    changes.
    """
    text_from_cache_only = (fetch_text == "cache")
    do_fetch_text = FETCH_TEXT if fetch_text is None else bool(fetch_text)
    plan = gdelt_query_plan(anomaly_date, queries, recent_queries,
                            combined_query, combined_recent_query, combined_day_query)
    live_requests = sum(1 for r in plan if r.live)

    seen_urls = set()
    raw_articles = []
    # Retrieval-health counters (see src/rag/outcome.RetrievalMeta). These make
    # the difference between "the news API was down" and "the news was quiet"
    # observable instead of both collapsing to an empty result.
    network_attempts = 0    # queries that actually hit the network (not cache)
    network_failures = 0    # of those, how many raised after retries
    network_successes = 0   # of those, how many returned a response
    cache_hits = 0
    reused_cache_hits = 0   # separate queries read from the cache, never requested
    fallback_added = 0
    for query, params, start_dt, startdatetime, enddatetime, maxrecords, live, label in plan:
        cached = cache_get(cache_dir, params, verbose=live) if cache_dir else None
        if cached is None and not live:
            continue
        if cached is not None:
            print(f"Querying GDELT: \"{query}\" restricted to "
                  f"{start_dt:%Y-%m-%d} .. {anomaly_date} (cache hit)")
            raw = cached
            if live:
                cache_hits += 1
            else:
                reused_cache_hits += 1
        elif not gdelt_live:
            print(f"Querying GDELT: \"{query}\" restricted to "
                  f"{start_dt:%Y-%m-%d} .. {anomaly_date} (not cached, live disabled — skipping)")
            continue
        else:
            if network_attempts > 0:
                time.sleep(INTER_QUERY_DELAY_SECONDS)
            print(f"Querying GDELT: \"{query}\" restricted to "
                  f"{start_dt:%Y-%m-%d} .. {anomaly_date} (lookahead-safe window)")
            try:
                raw = fetch_gdelt(
                    query,
                    maxrecords=maxrecords,
                    startdatetime=startdatetime,
                    enddatetime=enddatetime,
                )
            except RuntimeError as exc:
                print(f"  query failed after retries, skipping: {exc}")
                network_attempts += 1
                network_failures += 1
                continue
            network_attempts += 1
            network_successes += 1
            if cache_dir:
                cache_put(cache_dir, params, raw)
        articles = raw.get("articles", []) or []
        print(f"  -> {len(articles)} articles")
        for art in articles:
            url = art.get("url", "") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                art["_matched_query"] = query
                art["_found_by"] = label
                raw_articles.append(art)

    print(f"Retrieved {len(raw_articles)} unique articles from {live_requests} GDELT request(s)"
          + (f" and {reused_cache_hits} earlier cached answer(s)" if reused_cache_hits else "")
          + " in window")

    # Multi-source aggregator: Google News RSS + industry feeds run in parallel
    # with GDELT. This replaces the old "RSS only when GDELT returns nothing"
    # approach — the aggregator is now the primary source, and GDELT supplements
    # it (mostly via its cache). See src/rag/news_aggregator.py.
    if commodity_key:
        try:
            from src.rag.news_aggregator import aggregate_news
            agg_arts = []
            # Once over the full window, and when recent queries are in use
            # once more over just the recent days: the feed returns a fixed
            # number of items per search, so a search limited to those days
            # returns more of that day's coverage.
            windows = [RETRIEVAL_WINDOW_DAYS_BEFORE] + (
                [RECENT_WINDOW_DAYS_BEFORE] if recent_queries or combined_recent_query else [])
            for window_days in windows:
                arts, agg_stats = aggregate_news(
                    commodity_key, anomaly_date,
                    window_days=window_days,
                    gdelt_queries=None,  # GDELT is handled above; don't double-query
                    cache_dir=None,
                    allow_gdelt_live=False,
                )
                agg_arts.extend(arts)
            agg_used = 0
            for art in agg_arts:
                url = art.get("url", "") or ""
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    art["_matched_query"] = art.get("_matched_query", queries[0] if queries else "")
                    art["_found_by"] = "news feed"
                    raw_articles.append(art)
                    agg_used += 1
            if agg_used:
                network_successes += 1
                print(f"Merged {agg_used} article(s) from the news aggregator "
                      f"(Google News RSS + industry feeds); {len(raw_articles)} total")
        except Exception as exc:  # noqa: BLE001 - supplementary source must never crash
            print(f"  News aggregator unavailable: {exc}")

    # Merge any user-supplied local-fallback articles for this date (opt-in by
    # file existence; lookahead-safe filtering happens in load_fallback_articles).
    if commodity_key and fallback_dir:
        added = 0
        for art in load_fallback_articles(fallback_dir, commodity_key, anomaly_date):
            url = art.get("url", "") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                raw_articles.append(art)
                added += 1
        if added:
            fallback_added += added
            print(f"Merged {added} local-fallback article(s); {len(raw_articles)} total")

    # Build the candidate documents (no body text yet).
    candidates = []
    for art in raw_articles:
        url = art.get("url", "") or ""
        candidates.append({
            "document_id": make_document_id(url),
            "title": art.get("title", "") or "",
            "url": url,
            "publication_date": gdelt_seendate_to_iso_date(art.get("seendate", "")),
            # The trading day the document can be reporting on: the day before
            # its date if it was first seen before the exchange opened.
            "session_date": market_report.session_date(art.get("seendate", "")),
            "seen_at": art.get("seendate", "") or "",
            "text": art.get("text", "") or "",   # the local fallback file can carry text
            "_matched_query": art.get("_matched_query", queries[0] if queries else ""),
            # Which search returned this article: kept so that searches that
            # never produce a document the decision used can be dropped on
            # evidence rather than on a hunch.
            "found_by": art.get("_found_by", "local file"),
        })

    # Enforce the retrieval window locally. The window is sent to GDELT as
    # startdatetime/enddatetime, but the API does not hold to it strictly (it
    # returns articles dated the day after), so the claim that nothing
    # published after the anomaly date is used has to be checked here.
    out_of_window = []
    if enforce_window:
        candidates, out_of_window = filter_to_window(
            candidates, anomaly_date, RETRIEVAL_WINDOW_DAYS_BEFORE)
    if out_of_window:
        n_after = sum(1 for d in out_of_window if d["reason"] == "published_after_anomaly")
        print(f"Dropped {len(out_of_window)} article(s) dated outside the window "
              f"({n_after} published after {anomaly_date})")

    # Collapse the same headline republished on several sites BEFORE fetching
    # text, so a wire story carried by ten outlets costs one fetch, not ten.
    n_before = len(candidates)
    if collapse_duplicates:
        candidates = collapse_syndicated(candidates, use_text=False)

    documents = list(candidates)
    if do_fetch_text:
        to_fetch = documents
        if fetch_text_max_age is not None:
            from src.rag.evidence import trading_days_before

            def recent_enough(doc):
                age = trading_days_before(doc.get("publication_date", ""), anomaly_date)
                return age is not None and 0 <= age <= fetch_text_max_age
            to_fetch = [d for d in documents if recent_enough(d)]
            print(f"Article text wanted for {len(to_fetch)} of {len(documents)} documents "
                  f"(those from the last {fetch_text_max_age} trading days)")
        _fetch_texts(to_fetch, cache_only=text_from_cache_only)

    # Second pass now that body text is available: catches copies that were
    # given a different headline by the republishing site.
    if collapse_duplicates:
        documents = collapse_syndicated(documents, use_text=True)
    collapsed = n_before - len(documents)
    if collapsed:
        print(f"Collapsed {collapsed} syndicated copy/copies; "
              f"{len(documents)} distinct stories remain")

    for doc in documents:
        matched_query = doc.pop("_matched_query", queries[0] if queries else "")
        doc["retrieval_score"] = compute_keyword_relevance_score(
            doc["title"], doc["text"], matched_query)
    annotate_recency(documents, anomaly_date)

    if SEMANTIC_SCORING_AVAILABLE and documents:
        print(f"\nScoring {len(documents)} document(s) semantically "
              f"(multilingual-e5-small) against: \"{semantic_reference_query}\"")
        # The import succeeding does not mean the model can be loaded: the
        # first call downloads it, and that fails without network access to
        # the model hub. Treat a load failure like a missing package - warn
        # and gate on the keyword score alone - rather than ending the run in
        # a traceback.
        try:
            semantic_scores = score_documents_against_query(documents, semantic_reference_query)
        except Exception as exc:  # noqa: BLE001 - any model/download failure
            semantic_scores = None
            print(f"WARNING: semantic scoring failed ({type(exc).__name__}: {str(exc)[:120]}). "
                  f"Gate is running on the keyword score alone for this run.")
        if semantic_scores is not None:
            for doc, sem_score in zip(documents, semantic_scores):
                doc["semantic_score"] = sem_score
    elif documents:
        print(f"\nWARNING: semantic scoring unavailable ({_SEMANTIC_IMPORT_ERROR}). "
              f"Gate is running on the keyword score alone, which has a known "
              f"non-English blind spot - install with: "
              f"pip install onnxruntime tokenizers huggingface_hub")

    meta = RetrievalMeta(
        queries_attempted=live_requests,
        network_attempts=network_attempts,
        network_failures=network_failures,
        network_successes=network_successes,
        cache_hits=cache_hits,
        reused_cache_hits=reused_cache_hits,
        fallback_added=fallback_added,
        documents_returned=len(documents),
        dropped_out_of_window=len(out_of_window),
        collapsed_duplicates=collapsed,
    )
    if return_meta:
        return documents, meta
    return documents


# Trading days in the cumulative-return check (mirrors
# src.modeling.anomaly_detector.CUMULATIVE_WINDOW_DAYS; repeated here so the
# pipeline does not import the plotting stack just for one constant).
CUMULATIVE_WINDOW_DAYS = 10


# Hosts whose article URLs never yield body text. Google News RSS links are
# redirect pages: every one of them fetches "OK" and extracts nothing, so
# requesting them only adds a second or two each (and there can be ninety of
# them per date). Those documents are scored on their headline.
NO_TEXT_HOSTS = ("news.google.com",)
TEXT_FETCH_WORKERS = 8


def _has_fetchable_text(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return bool(host) and not any(host == h or host.endswith("." + h) for h in NO_TEXT_HOSTS)


def _fetch_texts(documents: list, cache_only: bool = False) -> None:
    """Fill in doc["text"] for documents that have a URL and no text yet.

    Fetches run in a small thread pool: the articles are on many different
    sites, and text_fetch already spaces out requests to the same site, so
    this is as polite as the old one-at-a-time loop and several times faster
    (roughly half the URLs are dead and each used to block for its whole
    timeout).
    """
    todo = [d for d in documents if d["url"] and not d["text"]]
    skipped = [d for d in todo if not _has_fetchable_text(d["url"])]
    todo = [d for d in todo if _has_fetchable_text(d["url"])]
    if cache_only:
        for doc in todo:
            doc["text"] = cached_article_text(doc["url"]) or ""
        return
    if not todo:
        return

    def fetch(doc):
        try:
            return fetch_article_text(doc["url"])
        except Exception as exc:  # noqa: BLE001 - one bad URL must not end the run
            return "", f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=min(TEXT_FETCH_WORKERS, len(todo))) as pool:
        results = list(pool.map(fetch, todo))

    failures = []
    for doc, (text, err) in zip(todo, results):
        doc["text"] = text or ""
        if err:
            failures.append((doc["url"], err))
    print(f"Article text: fetched {len(todo) - len(failures)} of {len(todo)}"
          + (f"; {len(skipped)} Google News link(s) kept as headline only" if skipped else ""))
    for url, err in failures[:5]:
        print(f"  text fetch failed for {url[:60]}: {str(err)[:90]}")
    if len(failures) > 5:
        print(f"  ... and {len(failures) - 5} more")


# Where each run keeps the exact text its model was shown (git-ignored: it is
# other publishers' text). scripts/compare_models.py gives the same text to
# other models, so a difference between models is a difference in reading
# and not in what was read.
SHOWN_DIR = REPO_ROOT / "data" / "eval_cache" / "shown"


def save_shown_documents(directory, date: str, commodity: str, result: dict,
                         documents: list) -> None:
    """Write the documents a blind reading was shown, as shown: in prompt
    order, with the excerpt cut to the length that was sent. Best effort - a
    failure to write this must not end the run."""
    by_id = {d.get("document_id"): d for d in documents}
    excerpt = (result.get("evidence_packing") or {}).get("excerpt_chars") or 1500
    shown = [{"document_id": doc_id,
              "title": by_id[doc_id].get("title", ""),
              "publication_date": by_id[doc_id].get("publication_date", ""),
              "url": by_id[doc_id].get("url", ""),
              "passed_gate": bool(by_id[doc_id].get("passed_gate", True)),
              "text": blind_evidence.as_shown(by_id[doc_id])["text"][:excerpt]}
             for doc_id in result["evidence_document_ids"] if doc_id in by_id]
    try:
        path = Path(directory) / f"{commodity.replace(' ', '_')}_{date}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"date": date, "commodity": commodity,
                                    "excerpt_chars": excerpt, "documents": shown}),
                        encoding="utf-8")
    except OSError:
        pass


def reading_candidates(documents: list, accepted: list, anomaly_date: str) -> list:
    """The documents a blind reading may be shown: everything the relevance
    gate accepted, plus recent documents that score at least
    RECENT_SEMANTIC_FLOOR on the semantic channel (see that constant for why).
    Each document is marked with whether it cleared the gate proper."""
    accepted_ids = {d.get("document_id") for d in accepted}
    out = []
    for doc in documents:
        passed = doc.get("document_id") in accepted_ids
        if not passed:
            age = blind_evidence._age(doc, anomaly_date)
            near = (doc.get("semantic_score") or 0.0) >= RECENT_SEMANTIC_FLOOR
            recent = age is not None and 0 <= age <= blind_evidence.MAX_AGE_TRADING_DAYS
            if not (near and recent):
                continue
        doc["passed_gate"] = passed
        out.append(doc)
    return out


def preview_reading(anomaly: dict, documents: list) -> list:
    """Print, and return, the documents the blind reading would be shown for
    this date, in order - without calling a model. What gets read decides
    what can be found, and this is the cheap way to look at it."""
    gate_result = gate_documents(documents)
    candidates = reading_candidates(documents, gate_result.accepted_documents, anomaly["date"])
    order = blind_evidence.recent_documents(candidates, anomaly["date"])
    shown = order[:blind_evidence.MAX_DOCUMENTS_READ]
    reports = sum(1 for d in order
                  if market_report.market_report_score(d) >= market_report.REPORT_SCORE)
    print(f"\n--- Reading preview for {anomaly['date']} (no model call, nothing saved) ---")
    print(f"{len(documents)} retrieved, {len(candidates)} relevant enough to read, "
          f"{len(order)} of those recent, {reports} that look like market reports. "
          f"The first {len(shown)} would be read:")
    for i, doc in enumerate(shown, 1):
        focused = blind_evidence.as_shown(doc)["text"]
        print(f"  {i:>2}. session -{doc.get('trading_days_before_anomaly')}  "
              f"report score {market_report.market_report_score(doc)}  "
              f"seen {doc.get('seen_at') or '?'}  text {len(doc.get('text') or '')} chars"
              f"{' (passages chosen)' if focused != (doc.get('text') or '') else ''}"
              f"\n      {doc.get('title', '')[:110]}")
    return shown


def describe_move(anomaly: dict) -> str:
    """The one-line description of the move given to the model.

    A shock anomaly is described by its day-over-day move, as before. A
    trend anomaly is described by the multi-day move it was flagged for, with
    the day's own move given as context - otherwise the model is asked why
    the price moved 1% on a day whose daily move was unremarkable.
    """
    pct = anomaly.get("pct_move")
    daily = (f"{'+' if pct > 0 else ''}{pct:.1f}% day-over-day move "
             f"(z-score {anomaly['z_score']:.2f})"
             if pct is not None else f"z-score {anomaly['z_score']:.2f}")

    cum = anomaly.get("cumulative_pct_move")
    cum_z = anomaly.get("cumulative_z_score")
    kind = anomaly.get("anomaly_type")
    if kind == "trend" and cum is not None and cum_z is not None:
        return (f"{'+' if cum > 0 else ''}{cum:.1f}% over the last "
                f"{CUMULATIVE_WINDOW_DAYS} trading days (cumulative z-score {cum_z:.2f}); "
                f"the move on the day itself was "
                f"{'+' if (pct or 0) > 0 else ''}{(pct or 0):.1f}%")
    if kind == "shock+trend" and cum is not None:
        return (f"{daily}, on top of {'+' if cum > 0 else ''}{cum:.1f}% over the last "
                f"{CUMULATIVE_WINDOW_DAYS} trading days")
    return daily


def _source_record(doc: dict) -> dict:
    return {
        "document_id": doc.get("document_id"),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "publication_date": doc.get("publication_date", ""),
        "days_before_anomaly": doc.get("days_before_anomaly"),
        "trading_days_before_anomaly": doc.get("trading_days_before_anomaly"),
        "publication_date_unknown": bool(doc.get("publication_date_unknown")),
        "text_chars": len(doc.get("text") or ""),   # 0 = the model saw the headline only
        "seen_at": doc.get("seen_at"),
        "session_date": doc.get("session_date"),
        "market_report_score": market_report.market_report_score(doc),
        "found_by": doc.get("found_by"),
        "passed_gate": bool(doc.get("passed_gate", True)),
        "n_copies": doc.get("n_copies", 1),
        "syndication_domains": doc.get("syndication_domains", []),
        "retrieval_score": doc.get("retrieval_score"),
        "semantic_score": doc.get("semantic_score"),
        "direction_consistency": doc.get("direction_consistency"),
        "anomaly_query_score": doc.get("anomaly_query_score"),
    }


def explain_from_documents(anomaly: dict, documents: list, retrieval_meta=None, *,
                           price_description: str = None,
                           strict_direction: bool = False,
                           direction_guard: bool = True,
                           prompt_version: str = None,
                           providers: list = None,
                           rerank: bool = False,
                           decision_mode: str = None,
                           commodity_name: str = "coffee",
                           reading_cache_dir=None,
                           shown_dir=None,
                           verbose: bool = True) -> dict:
    """Everything after retrieval: gate -> decision -> citation audit ->
    outcome. The decision is the blind-evidence rule by default, or with
    decision_mode="legacy" the earlier path (direction ordering -> model
    explanation -> direction guard).

    Split out of main() so the evaluation harness runs the SAME code path the
    CLI does. The robustness tests work by changing one input and holding the
    rest fixed (a flipped price move, another date's evidence), which is only
    meaningful if nothing else about the path differs.

    Returns the dict that main() saves as the pipeline output.
    """
    def say(*a, **k):
        if verbose:
            print(*a, **k)

    if price_description is None:
        price_description = describe_move(anomaly)
    mode = decision_mode or DECISION_MODE
    if mode not in DECISION_MODES:
        raise ValueError(f"unknown decision_mode {mode!r}; expected one of {DECISION_MODES}")

    say("\n--- Relevance gate ---")
    gate_result = gate_documents(documents)
    say(f"Decision: {gate_result.decision}  reason: {gate_result.reason}  "
        f"best_score: {gate_result.best_score:.3f}  "
        f"accepted: {len(gate_result.accepted_documents)}/{len(documents)}")

    # Direction-aware re-ranking of the accepted evidence: order documents so
    # those describing a price move in the same direction as this anomaly
    # lead, and (only with --strict-direction) drop contradictory ones. This
    # changes the order evidence is handed to the LLM, not which documents the
    # gate accepted, so it cannot flip an EXPLAIN/refuse decision on its own.
    # See src/rag/direction.py for the rationale.
    accepted = gate_result.accepted_documents
    direction_summary = {}
    if accepted and rerank and SEMANTIC_SCORING_AVAILABLE:
        # Optional, off by default: second ranking against a query built from
        # this move's direction. See src/rag/rerank.py.
        try:
            accepted = rerank_by_anomaly_query(accepted, anomaly, score_documents_against_query)
        except Exception as exc:  # noqa: BLE001 - an optional step must not end the run
            say(f"  rerank skipped: {type(exc).__name__}: {str(exc)[:100]}")
    if accepted:
        accepted = direction_mod.rerank(accepted, anomaly.get("direction"),
                                        strict=strict_direction)
        direction_summary = direction_mod.summarize(accepted)
        say(f"Direction vs anomaly ({anomaly.get('direction')}): "
            f"{direction_summary.get('consistent', 0)} consistent, "
            f"{direction_summary.get('neutral', 0)} neutral, "
            f"{direction_summary.get('inconsistent', 0)} inconsistent"
            + ("  (strict: inconsistent dropped)" if strict_direction else ""))

    # The word-list tally is kept under its own name in every output; in
    # blind mode the tally that goes into the outcome is the blind reading's.
    wordlist_summary = direction_summary

    say("\n--- Explanation generation ---")
    candidates = (reading_candidates(documents, gate_result.accepted_documents, anomaly["date"])
                  if mode == "blind" else [])
    if candidates:
        # The documents go in as the gate left them, not in the
        # direction-ranked order above: that order depends on which way the
        # move went, and nothing about the reading may.
        say(f"Reading candidates: {len(candidates)} "
            f"({sum(1 for d in candidates if d['passed_gate'])} cleared the gate, "
            f"{sum(1 for d in candidates if not d['passed_gate'])} recent and near it)")
        result = blind_evidence.blind_decision(
            anomaly, price_description, candidates,
            providers=providers, commodity=commodity_name, cache_dir=reading_cache_dir)
        if shown_dir and result.get("evidence_document_ids"):
            save_shown_documents(shown_dir, anomaly["date"], commodity_name, result, candidates)
        if "blind_evidence" in result:
            direction_summary = blind_evidence.tally(result)
            blind = result["blind_evidence"]
            say(f"Blind reading: {len(blind['readings'])} recent document(s) read; "
                f"deciding evidence {len(blind['supporting'])} with the move, "
                f"{len(blind['opposing'])} against, "
                f"{len(blind.get('set_aside') or [])} set aside (trading reason, or "
                f"not a move of this size).")
    elif mode == "legacy" and gate_result.decision == "EXPLAIN" and accepted:
        result = generate_explanation(anomaly["date"], price_description, accepted,
                                      prompt_version=prompt_version, providers=providers)
    else:
        result = {"decision": "INSUFFICIENT_EVIDENCE", "explanation": None,
                   "citations": [], "confidence": None,
                   "reason": f"gate_rejected: {gate_result.reason}"}

    # Direction guard: a deterministic check applied AFTER the model answers.
    # If the model explained the move but more accepted documents point
    # against it than with it, the explanation is withheld and the run is a
    # refusal (rule and its history: outcome.apply_direction_guard).
    # Both should-refuse dates that were wrongly explained in the stored runs
    # (2021-07-19: 2 consistent vs 13 inconsistent; 2023-09-20: 0 vs 5) fit
    # this pattern. The model's text is kept under withheld_explanation so the
    # decision can be inspected and the guard's cost measured.
    if direction_guard and mode == "legacy":
        guarded = apply_direction_guard(result, direction_summary)
        if guarded is not result:
            say("Direction guard: explanation withheld - "
                f"{direction_summary.get('inconsistent', 0)} accepted document(s) point "
                f"against the move vs {direction_summary.get('consistent', 0)} with it.")
        result = guarded

    say(json.dumps(result, indent=2))

    # Citation-faithfulness audit: verify every citation resolves to a
    # document the model was actually shown and that its claim overlaps that
    # document's text. Uses the multilingual embedder when available (a claim
    # can be English while its source is Portuguese), else a lexical +
    # shared-numbers check.
    faithfulness_report = None
    if result.get("citations"):
        shown_ids = result.get("evidence_document_ids")
        pool = candidates if mode == "blind" else accepted
        shown = ([d for d in pool if d.get("document_id") in set(shown_ids)]
                 if shown_ids else pool)
        scorers = {}
        if SEMANTIC_SCORING_AVAILABLE:
            scorers["semantic_fn"] = lambda claim, doc_text: \
                score_documents_against_query([{"text": doc_text}], claim)[0]
            scorers["semantic_batch_fn"] = lambda claim, doc_texts: \
                score_documents_against_query([{"text": t} for t in doc_texts], claim)
        faithfulness_report = faithfulness_mod.check_citations(result, shown, **scorers)
        say("\n--- Citation faithfulness ---")
        say(faithfulness_mod.format_report(faithfulness_report))

    # Classify the run into one labelled outcome + confidence tier. This is
    # the single place "could not retrieve" (an outage) is separated from
    # "retrieved but weak/conflicting" (the safety mechanism working) - see
    # src/rag/outcome.py. Both the CLI and the dashboard read this.
    outcome = classify_outcome(
        retrieval_meta=retrieval_meta,
        gate_result=gate_result,
        explanation_result=result,
        direction_summary=direction_summary,
        faithfulness_report=faithfulness_report,
    )
    say("\n--- Outcome ---")
    say(f"[{outcome.tier}] {outcome.label}")
    say(f"  {outcome.headline}")
    say(f"  {outcome.detail}")

    meta_dict = None
    if retrieval_meta is not None:
        meta_dict = (retrieval_meta if isinstance(retrieval_meta, dict)
                     else retrieval_meta.to_dict())

    return {
        "pipeline_version": PIPELINE_VERSION,
        "anomaly": anomaly,
        "price_description": price_description,
        "outcome": outcome.to_dict(),
        "retrieval_meta": meta_dict,
        "gate_result": gate_result.to_dict(),
        "documents_retrieved": len(documents),
        # How many retrieved documents are recent enough for the decision rule
        # to use, before the gate: tells a date nothing recent was found for
        # apart from one where recent documents were found and filtered out.
        "documents_retrieved_recent": sum(
            1 for d in documents
            if d.get("trading_days_before_anomaly") is not None
            and 0 <= d["trading_days_before_anomaly"] <= blind_evidence.FRESH_TRADING_DAYS),
        # In blind mode: every document that could be read (those that cleared
        # the gate and those read as recent and near it; "passed_gate" says
        # which). In legacy mode: the documents the gate accepted.
        "sources": [_source_record(d) for d in ((candidates if mode == "blind" else accepted)
                                                 or [])],
        "decision_mode": mode,
        "direction_summary": direction_summary,
        "direction_summary_wordlist": wordlist_summary,
        "guard_rule": GUARD_RULE if (direction_guard and mode == "legacy") else None,
        "evidence_recency": recency_summary(accepted or []),
        "explanation_result": result,
        "faithfulness_report": faithfulness_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Anomaly date, YYYY-MM-DD")
    parser.add_argument("--commodity", default="coffee", choices=list(COMMODITIES.keys()))
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the GDELT response cache and force fresh retrieval "
                             "(default: reuse cached responses for previously-seen query windows)")
    parser.add_argument("--strict-direction", action="store_true",
                        help="Additionally DROP evidence whose described price direction "
                             "contradicts the anomaly's direction (default: keep it but rank it "
                             "last and let the LLM judge). See src/rag/direction.py.")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Ignore any local news-fallback file for this date "
                             "(default: merge data/news_fallback/<commodity>/<date>.json if present)")
    parser.add_argument("--no-direction-guard", action="store_true",
                        help="Do not withhold an explanation when more of the accepted "
                             "evidence points against the move than with it "
                             "(default: guard on). "
                             "See explain_from_documents().")
    parser.add_argument("--rerank", action="store_true",
                        help="Also rank accepted evidence against a query built from this "
                             "move's direction (experimental, off by default; see "
                             "src/rag/rerank.py).")
    parser.add_argument("--prompt-version", default=None, choices=["v1", "v2"],
                        help="Explanation prompt version (default: the current one, v2). "
                             "v1 is the original four-rule prompt, kept for comparison.")
    parser.add_argument("--decision", default=None, choices=list(DECISION_MODES),
                        help="How explain/refuse is decided (default: blind). 'blind': the "
                             "model reads the recent documents without being told the move "
                             "and a fixed rule decides. 'legacy': the model is told the move "
                             "and decides, then the direction guard applies.")
    parser.add_argument("--preview-reading", action="store_true",
                        help="Retrieve and score as usual, print the documents the blind "
                             "reading would be shown, in order, and stop: no model is called "
                             "and nothing is saved.")
    args = parser.parse_args()

    cfg = get_commodity(args.commodity)
    from src.rag.explainer import check_providers
    check_providers()      # a bad provider name or missing key stops here, not after retrieval
    print(f"=== Pipeline run for {cfg.display_name}, anomaly date {args.date} ===\n")

    anomaly = get_anomaly(args.date, cfg.anomalies_file)
    print(f"Anomaly: price={anomaly['price']:.2f}  z_score={anomaly['z_score']:.2f}  "
          f"flagged={anomaly['anomaly_flag']}")
    if not anomaly["anomaly_flag"]:
        print("WARNING: this date was not flagged as an anomaly by the detector. "
              "Proceeding because it was requested explicitly, but this is not a "
              "detected event.")

    documents, retrieval_meta = retrieve_evidence(
        args.date, cfg.gdelt_queries, cfg.semantic_reference_query,
        cache_dir=None if args.no_cache else GDELT_CACHE_DIR,
        commodity_key=cfg.key,
        fallback_dir=None if args.no_fallback else FALLBACK_DIR,
        return_meta=True,
        **search_arguments(cfg),
        fetch_text_max_age=(blind_evidence.MAX_AGE_TRADING_DAYS
                            if (args.decision or DECISION_MODE) == "blind" else None),
    )

    # A run where the relevance model was meant to score the documents and
    # did not is a fault, not a finding: the gate would decide on keyword
    # scores alone, which is not the system being evaluated, and what it then
    # "refuses" says nothing about the evidence. End with an error and save
    # nothing, so the date is run again. (A machine without the model
    # installed at all is the documented keyword-only mode and carries on.)
    if semantic_scoring_failed(documents):
        raise SystemExit(
            "ERROR: the relevance model failed to score this date's documents (see the "
            "warning above). Nothing was saved.")

    if documents:
        print("\n--- Per-document scores (top 5 by best score) ---")
        ranked = sorted(
            documents,
            key=lambda d: max(d.get("retrieval_score", 0.0) or 0.0, d.get("semantic_score", 0.0) or 0.0),
            reverse=True,
        )
        for d in ranked[:5]:
            kw = d.get("retrieval_score", 0.0) or 0.0
            sem = d.get("semantic_score")
            sem_str = f"{sem:.3f}" if sem is not None else "n/a"
            print(f"  kw={kw:.3f}  sem={sem_str}  {d['title'][:70]}")

    if args.preview_reading:
        preview_reading(anomaly, documents)
        return

    output = explain_from_documents(
        anomaly, documents, retrieval_meta,
        strict_direction=args.strict_direction,
        direction_guard=not args.no_direction_guard,
        prompt_version=args.prompt_version,
        rerank=args.rerank,
        decision_mode=args.decision,
        commodity_name=cfg.key.replace("_", " "),
        shown_dir=SHOWN_DIR,
    )

    output_path = REPO_ROOT / "results" / f"pipeline_output_{cfg.key}_{args.date}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved full pipeline output to {output_path}")


if __name__ == "__main__":
    main()
