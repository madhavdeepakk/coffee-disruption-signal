"""
RAG-powered daily market briefing: synthesizes all available data sources
into a comprehensive "what's happening today" narrative.

This is distinct from the Price Outlook (outlook_rag.py) which gives a
directional lean. The daily briefing answers a broader question: "What
should I know about the coffee market right now?" It pulls together:

  - Today's price action and recent trend
  - Weather conditions in key growing regions
  - Speculative positioning (CFTC)
  - Currency movements (BRL/USD)
  - Recent news from multiple sources (via news_aggregator.py)
  - Seasonal context

The briefing is generated fresh each time the dashboard opens (cached for
1 hour via Streamlit) and uses the same multi-provider LLM fallback chain
as the anomaly explainer and price outlook.

Usage:
    from src.advisory.daily_briefing import generate_daily_briefing
    briefing = generate_daily_briefing()
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# News retrieval via the multi-source aggregator
# ---------------------------------------------------------------------------

BRIEFING_WINDOW_DAYS = 7  # look back 7 days for the daily briefing

def _retrieve_briefing_news(as_of_date: str, commodity_key: str = "coffee") -> tuple:
    """Retrieve recent news for the daily briefing using the multi-source
    aggregator. Returns (accepted_documents, retrieval_info_str)."""
    from src.rag.news_aggregator import aggregate_news
    from src.rag.text_fetch import fetch_article_text
    from src.rag.retriever import make_document_id, compute_keyword_relevance_score
    from src.config.commodities import get_commodity

    cfg = get_commodity(commodity_key)

    # Use the aggregator (Google News RSS + industry feeds + GDELT cache)
    raw_articles, agg_stats = aggregate_news(
        commodity_key, as_of_date,
        window_days=BRIEFING_WINDOW_DAYS,
        gdelt_queries=cfg.gdelt_queries,
        cache_dir=REPO_ROOT / "data" / "gdelt_cache",
        allow_gdelt_live=False,  # never block the briefing on GDELT
    )

    # Fetch text and score (same pipeline as anomaly explainer)
    documents = []
    for art in raw_articles[:20]:  # cap at 20 to keep latency sane
        title = art.get("title", "") or ""
        url = art.get("url", "") or ""
        text = ""
        if url:
            text, err = fetch_article_text(url)
        score = compute_keyword_relevance_score(
            title, text, cfg.semantic_reference_query
        )
        documents.append({
            "document_id": make_document_id(url),
            "title": title,
            "url": url,
            "publication_date": art.get("seendate", "")[:8] if art.get("seendate") else "",
            "text": text,
            "retrieval_score": score,
            "_source": art.get("_source", "unknown"),
        })

    # Semantic scoring if available
    try:
        from src.rag.vector_store import score_documents_against_query
        if documents:
            scores = score_documents_against_query(documents, cfg.semantic_reference_query)
            for doc, sem_score in zip(documents, scores):
                doc["semantic_score"] = sem_score
    except ImportError:
        pass

    # Gate with a lower threshold for the briefing (we want broader coverage)
    try:
        from src.rag.relevance_gate import gate_documents
        gate = gate_documents(documents, keyword_threshold=0.3, semantic_threshold=0.65)
        accepted = gate.accepted_documents if gate.decision == "EXPLAIN" else documents[:8]
    except Exception:
        accepted = documents[:8]

    # Cap to top 10 by best score
    MAX_BRIEFING_DOCS = 10
    if len(accepted) > MAX_BRIEFING_DOCS:
        accepted.sort(
            key=lambda d: max(d.get("semantic_score", 0) or 0,
                              d.get("retrieval_score", 0) or 0),
            reverse=True,
        )
        accepted = accepted[:MAX_BRIEFING_DOCS]

    info = (f"Aggregated {agg_stats.get('total_after_dedupe', 0)} articles "
            f"(Google News: {agg_stats.get('google_news_count', 0)}, "
            f"Industry: {agg_stats.get('industry_rss_count', 0)}, "
            f"GDELT cache: {agg_stats.get('gdelt_cache_count', 0)}), "
            f"{len(accepted)} sent to LLM")

    return accepted, info


# ---------------------------------------------------------------------------
# Signal collection (reuses outlook_rag's collector)
# ---------------------------------------------------------------------------

def _collect_briefing_signals() -> dict:
    """Collect all quantitative signals for the briefing."""
    try:
        from src.advisory.outlook_rag import _collect_signals
        return _collect_signals()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# LLM synthesis prompt
# ---------------------------------------------------------------------------

BRIEFING_PROMPT = """You are a coffee market analyst writing a daily intelligence briefing. Your audience is a commodity trader or procurement manager who needs to know what's happening RIGHT NOW in the coffee market.

You have two types of evidence: structured quantitative data and recent news articles.

MARKET DATA:
{signals_block}

RECENT NEWS (past 7 days, relevance-filtered):
{evidence_block}

TASK — Write a daily market briefing following this structure:

1. HEADLINE: One sentence summarizing the single most important thing happening in coffee markets today.

2. PRICE ACTION: Current price, recent trend (up/down/flat over past week and month), where it sits in its historical range, and whether volatility is elevated. Use the specific numbers from the market data.

3. KEY DEVELOPMENTS: 2-4 bullet points on the most important developments from the news. Each must cite its source by document_id, e.g. "(source: doc_abc123)". Focus on what MATTERS — supply threats, demand shifts, trade policy, weather — not generic market commentary.

4. WEATHER & CROP: What's happening in Brazil's coffee regions (observed weather + forecast). Flag any frost risk, drought stress, or unusual conditions. Use the satellite data and forecast numbers provided.

5. POSITIONING: What speculators are doing (CFTC data) and what it implies.

6. WATCH LIST: 2-3 things to monitor in the coming days/week, based on the evidence above.

RULES:
- Base EVERY claim on the data or documents provided. No outside knowledge.
- Cite news sources by document_id wherever used.
- Use specific numbers (prices, percentages, z-scores) — not vague language.
- If a data category is missing, skip it — do not guess.
- Keep the total briefing under 500 words.
- Write for a professional audience. No disclaimers about AI.

Respond with ONLY a JSON object:
{{
  "headline": "...",
  "price_action": "...",
  "key_developments": ["...", "..."],
  "weather_crop": "...",
  "positioning": "...",
  "watch_list": ["...", "..."],
  "citations": [{{"document_id": "...", "supports": "..."}}],
  "sources_used": <int>
}}
"""


def _format_signals_for_briefing(signals: dict) -> str:
    """Format signals for the briefing prompt. Reuses the outlook format."""
    try:
        from src.advisory.outlook_rag import _format_signals_block
        return _format_signals_block(signals)
    except Exception:
        return "(No market data available)"


def _format_evidence_for_briefing(documents: list) -> str:
    """Format news articles for the briefing prompt."""
    if not documents:
        return "(No recent news articles available)"
    lines = []
    for doc in documents:
        lines.append(
            f"- document_id: {doc.get('document_id')}\n"
            f"  title: {doc.get('title', '')}\n"
            f"  source_type: {doc.get('_source', 'unknown')}\n"
            f"  text: {(doc.get('text') or doc.get('title') or '')[:500]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_daily_briefing(commodity_key: str = "coffee",
                            allow_live: bool = True) -> dict:
    """Generate a RAG-powered daily market briefing.

    Returns a dict with:
      - headline: single most important development
      - price_action: price trend narrative
      - key_developments: list of bullet points
      - weather_crop: weather/crop conditions
      - positioning: speculative positioning summary
      - watch_list: what to monitor
      - citations: cited news documents
      - retrieval_info: what happened during news retrieval
      - signals_collected: raw signal data
      - accepted_documents: news docs sent to LLM
    """
    # 1. Collect quantitative signals
    signals = _collect_briefing_signals()

    # 2. Retrieve recent news
    accepted_docs = []
    retrieval_info = "News retrieval skipped (allow_live=False)"
    if allow_live:
        try:
            as_of = signals.get("price", {}).get("as_of")
            if not as_of:
                as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            accepted_docs, retrieval_info = _retrieve_briefing_news(as_of, commodity_key)
        except Exception as exc:
            retrieval_info = f"News retrieval failed: {type(exc).__name__}: {exc}"

    # 3. Build prompt
    signals_block = _format_signals_for_briefing(signals)
    evidence_block = _format_evidence_for_briefing(accepted_docs)

    prompt = BRIEFING_PROMPT.format(
        signals_block=signals_block,
        evidence_block=evidence_block,
    )

    # allow_live=False means no network and no model: the public dashboard
    # runs this way so that opening the page cannot spend the project's
    # model quota or add to the news source's rate limiting. Skipping only
    # the news retrieval was not enough - the model was still called with an
    # empty evidence block, which cost quota and produced a briefing resting
    # on no news at all.
    if not allow_live:
        return _build_fallback_briefing(signals, accepted_docs, retrieval_info,
                                        "live access is off")

    # 4. Call LLM (same fallback chain as outlook and explainer)
    try:
        from src.rag.explainer import _call_model, log_call
        response, model_used = _call_model(prompt)
        parsed = json.loads(response.text)
        log_call("daily_briefing", getattr(response, "usage_metadata", None),
                 "BRIEFING", model=model_used)
    except Exception as exc:
        # Fallback: return what we can without LLM
        return _build_fallback_briefing(signals, accepted_docs, retrieval_info, str(exc))

    # 5. Validate and fill missing fields
    parsed = _postprocess_briefing(parsed)

    return {
        "headline": parsed.get("headline", ""),
        "price_action": parsed.get("price_action", ""),
        "key_developments": parsed.get("key_developments", []),
        "weather_crop": parsed.get("weather_crop", ""),
        "positioning": parsed.get("positioning", ""),
        "watch_list": parsed.get("watch_list", []),
        "citations": parsed.get("citations", []),
        "sources_used": parsed.get("sources_used", len(accepted_docs)),
        "retrieval_info": retrieval_info,
        "signals_collected": signals,
        "accepted_documents": accepted_docs,
        "model_generated": True,
    }


def _postprocess_briefing(parsed: dict) -> dict:
    """Fix common LLM output issues."""
    # Ensure all expected fields exist
    for key in ("headline", "price_action", "weather_crop", "positioning"):
        if key not in parsed or not parsed[key]:
            parsed[key] = ""
    for key in ("key_developments", "watch_list", "citations"):
        if key not in parsed or not isinstance(parsed.get(key), list):
            parsed[key] = []
    if "sources_used" not in parsed:
        parsed["sources_used"] = 0
    return parsed


def _build_fallback_briefing(signals: dict, docs: list, retrieval_info: str,
                              error: str) -> dict:
    """When the LLM fails, build a basic briefing from raw data."""
    headline = "Market data available but LLM synthesis unavailable."
    price_action = ""
    if "price" in signals:
        p = signals["price"]
        price_action = (
            f"Coffee is at {p['current_cents_lb']} cents/lb, "
            f"{p['week_change_pct']:+.1f}% this week, "
            f"{p['month_change_pct']:+.1f}% this month. "
            f"Sitting at the {p['percentile_1y']}th percentile of its 1-year range. "
            f"Recent volatility: {p['recent_vol_annualized']:.0f}% annualized."
        )

    weather = ""
    if "weather_observed" in signals:
        w = signals["weather_observed"]
        weather = (
            f"Dryness z-score: {w['dryness_zscore']:+.2f}. "
            f"Frost days (past week): {w['frost_7d_count']}. "
            f"30-day rainfall: {w['prcp_30d_mm']} mm."
        )
    if "weather_forecast" in signals:
        wf = signals["weather_forecast"]
        weather += (
            f" Forecast: {wf['precip_7d_mm']} mm rain expected, "
            f"min temp {wf['min_temp_7d']}°C, "
            f"{wf['frost_days_7d']} frost risk days."
        )

    positioning = ""
    if "cftc" in signals:
        c = signals["cftc"]
        positioning = (
            f"Net speculative: {c['net_speculative']:+,.0f} contracts "
            f"({c['net_spec_percentile']}th percentile). "
            f"Week-over-week: {c['net_speculative_change']:+,.0f}."
        )

    key_devs = [f"{d.get('title', '')}" for d in docs[:3]] if docs else []

    return {
        "headline": headline,
        "price_action": price_action,
        "key_developments": key_devs,
        "weather_crop": weather,
        "positioning": positioning,
        "watch_list": [],
        "citations": [],
        "sources_used": len(docs),
        "retrieval_info": retrieval_info,
        "signals_collected": signals,
        "accepted_documents": docs,
        "model_generated": False,
        "error": error,
    }


if __name__ == "__main__":
    import sys
    allow = "--no-live" not in sys.argv
    result = generate_daily_briefing(allow_live=allow)
    print(json.dumps(result, indent=2, default=str))
