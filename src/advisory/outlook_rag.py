"""
RAG-powered price outlook: collects market signals, retrieves recent news,
gates it, and asks Gemini to synthesize a directional assessment with
chain-of-thought reasoning, confidence tiers, and cited evidence.

This is the SAME architecture as the anomaly explainer (retrieve -> gate ->
LLM with citations -> faithfulness audit), applied to a different question:
instead of "why did the price move?" it answers "what do current conditions
suggest about near-term direction?"

The LLM never predicts a price. It reads the structured signal data and
gated news evidence, evaluates each factor independently (bullish / bearish /
neutral), then synthesizes a composite lean with a confidence tier based on
signal agreement.

Usage:
    from src.advisory.outlook_rag import generate_outlook
    outlook = generate_outlook()
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Signal collection (reuses existing data modules)
# ---------------------------------------------------------------------------

def _collect_signals() -> dict:
    """Gather all available quantitative signals into a structured dict.
    Each signal carries its raw data and source attribution."""
    signals = {}

    # --- Seasonal calendar ---
    try:
        from src.data.seasonal import current_season
        season = current_season()
        signals["seasonal"] = {
            "brazil_phase": season.get("brazil_phase", "unknown"),
            "vietnam_phase": season.get("vietnam_phase", "unknown"),
            "colombia_phase": season.get("colombia_phase", "unknown"),
            "key_risk": season.get("key_risk", ""),
            "source": "Coffee agricultural calendar",
        }
    except Exception:
        pass

    # --- Weather (observed) ---
    try:
        from src.advisory.buying_brief import _load_unified
        unified = _load_unified()
        if not unified.empty and "dryness_zscore" in unified.columns:
            recent = unified.dropna(subset=["dryness_zscore"]).tail(7)
            if not recent.empty:
                latest = recent.iloc[-1]
                signals["weather_observed"] = {
                    "dryness_zscore": round(float(latest["dryness_zscore"]), 2),
                    "frost_7d_count": int(latest.get("frost_7d_count", 0) or 0),
                    "prcp_30d_mm": round(float(latest.get("prcp_30d_sum", 0) or 0), 1),
                    "as_of": str(latest["date"])[:10],
                    "source": "NASA POWER satellite data",
                }
    except Exception:
        pass

    # --- Weather (forecast) ---
    try:
        from src.data.weather_forecast import get_forecast_observation
        fc = get_forecast_observation()
        if fc.get("status") != "error":
            summary = fc.get("summary", {})
            if summary:
                signals["weather_forecast"] = {
                    "precip_7d_mm": round(summary.get("total_precip_7d", 0), 1),
                    "min_temp_7d": round(summary.get("min_temp_7d", 10), 1),
                    "frost_days_7d": summary.get("frost_days_7d", 0),
                    "frost_days_14d": summary.get("frost_days_14d", 0),
                    "dry_days_7d": summary.get("dry_days_7d", 0),
                    "source": "Open-Meteo 16-day forecast (Minas Gerais)",
                }
    except Exception:
        pass

    # --- CFTC positioning ---
    try:
        from src.data.cftc import get_cot_observation
        cot = get_cot_observation()
        if cot.get("status") != "error":
            signals["cftc"] = {
                "net_speculative": cot.get("net_speculative", 0),
                "net_spec_percentile": round(cot.get("net_spec_percentile", 50), 1),
                "net_speculative_change": cot.get("net_speculative_change", 0),
                "report_date": cot.get("report_date", "unknown"),
                "source": "CFTC Commitment of Traders",
            }
    except Exception:
        pass

    # --- BRL/USD ---
    try:
        from src.data.currency import get_currency_observation
        fx = get_currency_observation()
        if fx.get("status") != "error":
            signals["currency"] = {
                "brl_usd_rate": round(fx.get("current_rate", 0), 4),
                "change_1m_pct": round(fx.get("change_1m_pct", 0), 1),
                "percentile_1y": round(fx.get("percentile_1y", 50), 1),
                "source": "Yahoo Finance (BRL=X)",
            }
    except Exception:
        pass

    # --- Price context ---
    try:
        from src.advisory.buying_brief import _load_prices
        import numpy as np
        prices = _load_prices()
        if not prices.empty and len(prices) >= 50:
            p = prices["price"].to_numpy()
            now = float(p[-1])
            wk = (now / p[-6] - 1) * 100 if len(p) > 5 else 0.0
            mo = (now / p[-21] - 1) * 100 if len(p) > 20 else 0.0
            one_yr = prices[prices["date"] >= prices["date"].iloc[-1] - __import__("pandas").Timedelta(days=365)]
            pct_1y = float((one_yr["price"] < now).mean() * 100)
            rets = prices["price"].pct_change().dropna()
            vol_20 = float(rets.tail(20).std() * np.sqrt(252) * 100)
            vol_252 = float(rets.tail(252).std() * np.sqrt(252) * 100)
            signals["price"] = {
                "current_cents_lb": round(now, 2),
                "week_change_pct": round(wk, 1),
                "month_change_pct": round(mo, 1),
                "percentile_1y": round(pct_1y, 1),
                "recent_vol_annualized": round(vol_20, 1),
                "hist_vol_annualized": round(vol_252, 1),
                "as_of": str(prices.iloc[-1]["date"])[:10],
                "source": "Yahoo Finance (KC=F)",
            }
    except Exception:
        pass

    # --- Historical analogs ---
    try:
        from src.advisory.buying_brief import _load_prices
        from src.modeling.outlook import build_outlook_frame, analog_scenarios
        import numpy as np
        prices = _load_prices()
        if not prices.empty:
            feat, cols, _ = build_outlook_frame(prices, use_weather=True)
            last_pos = int(np.where(~feat[cols].isna().any(axis=1).to_numpy())[0][-1])
            week = analog_scenarios(feat, cols, last_pos, horizon=5, n_analogs=25)
            month = analog_scenarios(feat, cols, last_pos, horizon=20, n_analogs=25)
            if "error" not in week and "error" not in month:
                signals["historical_analogs"] = {
                    "week": {
                        "n_analogs": week["n_analogs"],
                        "median_pct": round(week["median_pct"], 1),
                        "p10_pct": round(week["p10_pct"], 1),
                        "p90_pct": round(week["p90_pct"], 1),
                        "share_up_pct": round(week["share_up"] * 100, 0),
                    },
                    "month": {
                        "n_analogs": month["n_analogs"],
                        "median_pct": round(month["median_pct"], 1),
                        "p10_pct": round(month["p10_pct"], 1),
                        "p90_pct": round(month["p90_pct"], 1),
                        "share_up_pct": round(month["share_up"] * 100, 0),
                    },
                    "source": "Historical pattern matching (price + weather features)",
                }
    except Exception:
        pass

    return signals


# ---------------------------------------------------------------------------
# News retrieval (reuses the same GDELT/RSS + gate pipeline)
# ---------------------------------------------------------------------------

OUTLOOK_QUERIES = [
    "coffee price forecast outlook",
    "coffee supply harvest Brazil",
    "arabica futures market",
    "coffee export tariff trade",
]


def _retrieve_outlook_news(as_of_date: str) -> tuple:
    """Retrieve and gate recent news for the outlook.
    Returns (accepted_documents, retrieval_info_str).

    Uses a LOWER semantic threshold (0.70) than the anomaly explainer (0.85)
    because the outlook needs broader market-context articles — supply
    forecasts, trade policy, harvest reports — not just disruption-specific
    evidence. The anomaly gate was calibrated on disruption content where
    loosely-related articles scored 0.75-0.84; those same articles are
    *exactly* what the outlook wants. The keyword threshold stays at 0.4."""
    from src.pipeline import retrieve_evidence, GDELT_CACHE_DIR
    from src.rag.relevance_gate import gate_documents, KEYWORD_THRESHOLD
    from src.config.commodities import get_commodity

    OUTLOOK_SEMANTIC_THRESHOLD = 0.70

    cfg = get_commodity("coffee")

    docs = retrieve_evidence(
        as_of_date, OUTLOOK_QUERIES, cfg.semantic_reference_query,
        cache_dir=GDELT_CACHE_DIR, commodity_key="coffee",
    )

    gate = gate_documents(docs, keyword_threshold=KEYWORD_THRESHOLD,
                          semantic_threshold=OUTLOOK_SEMANTIC_THRESHOLD)
    accepted = gate.accepted_documents if gate.decision == "EXPLAIN" else []

    # Cap to top 12 by semantic score to keep the LLM prompt within token
    # limits (Groq free tier caps at 8K TPM; 42 articles blew past that).
    # Sort by semantic_score descending, fall back to keyword retrieval_score.
    MAX_OUTLOOK_DOCS = 12
    if len(accepted) > MAX_OUTLOOK_DOCS:
        accepted.sort(
            key=lambda d: max(d.get("semantic_score", 0) or 0,
                              d.get("retrieval_score", 0) or 0),
            reverse=True,
        )
        accepted = accepted[:MAX_OUTLOOK_DOCS]

    info = (f"Retrieved {len(docs)} articles, {gate.accepted_documents.__len__()} passed relevance gate, "
            f"top {len(accepted)} sent to LLM "
            f"(semantic_threshold={OUTLOOK_SEMANTIC_THRESHOLD}, "
            f"gate decision: {gate.decision})")

    return accepted, info


# ---------------------------------------------------------------------------
# LLM synthesis prompt
# ---------------------------------------------------------------------------

OUTLOOK_PROMPT = """You are analyzing current coffee market conditions to assess near-term directional pressure on prices. You have two types of evidence: structured quantitative signals and recent news articles that passed a relevance gate.

QUANTITATIVE SIGNALS:
{signals_block}

RECENT NEWS EVIDENCE (already relevance-filtered):
{evidence_block}

TASK — follow these steps in order:

STEP 1: Evaluate each signal INDEPENDENTLY.
For each available signal (seasonal calendar, weather, CFTC positioning, currency, price trend, historical analogs, news), state whether it points toward UPWARD pressure, DOWNWARD pressure, or is NEUTRAL on coffee prices, and explain why in one sentence. Cite the specific data. For news articles, cite by document_id.

STEP 2: Assess signal agreement.
Count how many signals point upward, downward, and neutral. If most signals agree, confidence is HIGH. If signals are mixed but lean one way, confidence is MEDIUM. If signals roughly split, confidence is LOW.

STEP 3: Synthesize a directional lean.
Based on your signal-by-signal analysis, state the overall lean (upward / downward / balanced) with the confidence level from Step 2.

STEP 4: Write a 2-3 sentence summary for a non-expert reader explaining what the current conditions suggest and which factors are driving it. Every factual claim from news must cite its document_id, e.g. "(source: doc_abc123)". Do NOT predict a specific price. Do NOT give financial advice.

RULES:
- Base your assessment ONLY on the signals and documents provided. No outside knowledge.
- If a signal is missing (not provided), skip it — do not guess.
- If the news evidence is too vague or off-topic to inform the outlook, still assess based on the quantitative signals alone and note that news evidence was inconclusive.
- Historical analogs are context ("in similar past periods, X happened"), not predictions.

Respond with ONLY a JSON object in exactly this shape:
{{
  "signal_assessments": [
    {{"factor": "...", "direction": "upward" | "downward" | "neutral", "reasoning": "...", "data_point": "..."}}
  ],
  "n_upward": <int>,
  "n_downward": <int>,
  "n_neutral": <int>,
  "confidence": "high" | "medium" | "low",
  "lean": "upward" | "downward" | "balanced",
  "summary": "...",
  "citations": [{{"document_id": "...", "supports": "..."}}]
}}
"""


def _format_signals_block(signals: dict) -> str:
    """Format the collected signals as structured text for the LLM prompt."""
    lines = []
    if "seasonal" in signals:
        s = signals["seasonal"]
        lines.append(f"SEASONAL CALENDAR: Brazil is in '{s['brazil_phase']}' phase. "
                     f"Vietnam: '{s['vietnam_phase']}'. Key risk: {s['key_risk']}. "
                     f"(Source: {s['source']})")

    if "weather_observed" in signals:
        w = signals["weather_observed"]
        lines.append(f"WEATHER (OBSERVED, as of {w['as_of']}): "
                     f"Dryness z-score: {w['dryness_zscore']:+.2f} (positive = drier than normal). "
                     f"Frost days in past week: {w['frost_7d_count']}. "
                     f"30-day rainfall: {w['prcp_30d_mm']} mm. "
                     f"(Source: {w['source']})")

    if "weather_forecast" in signals:
        wf = signals["weather_forecast"]
        lines.append(f"WEATHER (FORECAST): "
                     f"Expected rainfall next 7 days: {wf['precip_7d_mm']} mm. "
                     f"Minimum temperature next 7 days: {wf['min_temp_7d']}°C. "
                     f"Frost risk days (7d/14d): {wf['frost_days_7d']}/{wf['frost_days_14d']}. "
                     f"Dry days next week: {wf['dry_days_7d']}/7. "
                     f"(Source: {wf['source']})")

    if "cftc" in signals:
        c = signals["cftc"]
        direction = "long (bullish)" if c["net_speculative"] > 0 else "short (bearish)"
        lines.append(f"CFTC POSITIONING (report date {c['report_date']}): "
                     f"Net speculative: {c['net_speculative']:+,.0f} contracts ({direction}). "
                     f"Percentile in 1-year range: {c['net_spec_percentile']}th. "
                     f"Week-over-week change: {c['net_speculative_change']:+,.0f} contracts. "
                     f"(Source: {c['source']})")

    if "currency" in signals:
        fx = signals["currency"]
        lines.append(f"BRL/USD: {fx['brl_usd_rate']} (1-month change: {fx['change_1m_pct']:+.1f}%, "
                     f"1-year percentile: {fx['percentile_1y']}th). "
                     f"A weaker real encourages Brazilian exports -> bearish for prices. "
                     f"(Source: {fx['source']})")

    if "price" in signals:
        p = signals["price"]
        lines.append(f"PRICE CONTEXT (as of {p['as_of']}): "
                     f"{p['current_cents_lb']} cents/lb. "
                     f"Week: {p['week_change_pct']:+.1f}%, Month: {p['month_change_pct']:+.1f}%. "
                     f"1-year percentile: {p['percentile_1y']}th. "
                     f"Recent volatility: {p['recent_vol_annualized']}% (vs {p['hist_vol_annualized']}% historical). "
                     f"(Source: {p['source']})")

    if "historical_analogs" in signals:
        h = signals["historical_analogs"]
        w, m = h["week"], h["month"]
        lines.append(f"HISTORICAL ANALOGS ({w['n_analogs']} most similar past days): "
                     f"Next week: median {w['median_pct']:+.1f}%, range {w['p10_pct']:+.1f}% to {w['p90_pct']:+.1f}%, "
                     f"{w['share_up_pct']:.0f}% rose. "
                     f"Next month: median {m['median_pct']:+.1f}%, range {m['p10_pct']:+.1f}% to {m['p90_pct']:+.1f}%, "
                     f"{m['share_up_pct']:.0f}% rose. "
                     f"(Source: {h['source']})")

    return "\n\n".join(lines) if lines else "(No quantitative signals available)"


def _format_evidence_block(documents: list) -> str:
    """Format gated news articles for the LLM prompt.
    Text capped at 600 chars per article to keep total prompt within free-tier
    token limits (12 articles × ~600 chars ≈ 7200 chars ≈ 2000 tokens)."""
    if not documents:
        return "(No news articles passed the relevance gate for this date)"
    lines = []
    for doc in documents:
        lines.append(
            f"- document_id: {doc.get('document_id')}\n"
            f"  title: {doc.get('title', '')}\n"
            f"  date: {doc.get('publication_date', '')}\n"
            f"  text: {(doc.get('text') or doc.get('title') or '')[:600]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-processing: fix incomplete LLM responses
# ---------------------------------------------------------------------------

def _postprocess_outlook(parsed: dict) -> dict:
    """Derive aggregate fields from signal_assessments when the model leaves
    them empty. Some models (Groq free-tier, smaller open-source models) fill
    the per-signal array correctly but leave summary, counts, lean, and
    confidence at their defaults. This function computes them deterministically
    so the dashboard always gets consistent data.

    Rules:
      - n_upward / n_downward / n_neutral: counted from signal_assessments
      - lean: majority direction wins; tie -> "balanced"
      - confidence: all agree -> "high"; >60% agree -> "medium"; else "low"
      - summary: if empty, built from signal_assessments reasoning
    Only overrides a field when it looks wrong (e.g. counts are 0 but
    assessments exist, or lean says "balanced" when 7/8 signals agree).
    """
    assessments = parsed.get("signal_assessments", [])
    if not assessments:
        return parsed  # nothing to derive from

    # --- Count directions ---
    n_up = sum(1 for s in assessments if s.get("direction") == "upward")
    n_down = sum(1 for s in assessments if s.get("direction") == "downward")
    n_neut = sum(1 for s in assessments if s.get("direction") == "neutral")
    total = n_up + n_down + n_neut

    # Always override counts — the model's counts are unreliable
    parsed["n_upward"] = n_up
    parsed["n_downward"] = n_down
    parsed["n_neutral"] = n_neut

    # --- Derive lean ---
    if total > 0:
        if n_down > n_up and n_down > n_neut:
            derived_lean = "downward"
        elif n_up > n_down and n_up > n_neut:
            derived_lean = "upward"
        else:
            derived_lean = "balanced"

        # Override if model's lean contradicts the actual counts
        model_lean = parsed.get("lean", "balanced")
        if model_lean != derived_lean:
            parsed["lean"] = derived_lean

    # --- Derive confidence from signal agreement ---
    if total > 0:
        majority = max(n_up, n_down, n_neut)
        agreement_ratio = majority / total
        if agreement_ratio >= 0.80:
            derived_conf = "high"
        elif agreement_ratio >= 0.55:
            derived_conf = "medium"
        else:
            derived_conf = "low"

        # Override if model's confidence looks wrong
        model_conf = parsed.get("confidence", "low")
        if model_conf != derived_conf:
            parsed["confidence"] = derived_conf

    # --- Derive summary if empty ---
    summary = (parsed.get("summary") or "").strip()
    if not summary and assessments:
        lean = parsed.get("lean", "balanced")
        conf = parsed.get("confidence", "low")

        # Group factors by direction
        down_factors = [s["factor"] for s in assessments if s.get("direction") == "downward"]
        up_factors = [s["factor"] for s in assessments if s.get("direction") == "upward"]

        parts = []
        if lean == "downward":
            parts.append(
                f"Current market conditions suggest downward pressure on coffee prices, "
                f"driven by {', '.join(down_factors[:3])}."
            )
            if up_factors:
                parts.append(
                    f"Partially offsetting: {', '.join(up_factors[:2])} point upward."
                )
        elif lean == "upward":
            parts.append(
                f"Current market conditions suggest upward pressure on coffee prices, "
                f"driven by {', '.join(up_factors[:3])}."
            )
            if down_factors:
                parts.append(
                    f"Partially offsetting: {', '.join(down_factors[:2])} point downward."
                )
        else:
            parts.append(
                "Market signals are mixed with no clear directional consensus."
            )

        parts.append(f"Signal agreement is {conf} ({n_down} downward, {n_up} upward, {n_neut} neutral).")
        parsed["summary"] = " ".join(parts)

    return parsed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_outlook(allow_live: bool = True) -> dict:
    """Generate a RAG-powered directional outlook.

    Returns a dict with:
      - lean: "upward" / "downward" / "balanced"
      - confidence: "high" / "medium" / "low"
      - summary: plain-English synthesis for the dashboard
      - signal_assessments: per-signal chain-of-thought reasoning
      - citations: cited news documents
      - drivers: formatted list for dashboard rendering
      - signals_collected: raw signal data used
      - retrieval_info: what happened during news retrieval
    """
    # 1. Collect quantitative signals
    signals = _collect_signals()

    # 2. Retrieve and gate news
    accepted_docs = []
    retrieval_info = "News retrieval skipped (allow_live=False)"
    if allow_live:
        try:
            as_of = signals.get("price", {}).get("as_of")
            if not as_of:
                as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            accepted_docs, retrieval_info = _retrieve_outlook_news(as_of)
        except Exception as exc:
            retrieval_info = f"News retrieval failed: {type(exc).__name__}: {exc}"

    # 3. Build prompt and call Gemini
    signals_block = _format_signals_block(signals)
    evidence_block = _format_evidence_block(accepted_docs)

    prompt = OUTLOOK_PROMPT.format(
        signals_block=signals_block,
        evidence_block=evidence_block,
    )

    def without_synthesis(label: str, summary: str, error: str = None) -> dict:
        """The signal data on its own. Used when no model was called and when
        the call failed, so the two cases produce the same shape."""
        return {
            "lean": "balanced",
            "confidence": "low",
            "label": label,
            "summary": summary,
            "signal_assessments": [],
            "citations": [],
            "drivers": _signals_to_fallback_drivers(signals),
            "signals_collected": signals,
            "retrieval_info": retrieval_info,
            "accepted_documents": [],
            **({"error": error} if error else {}),
        }

    # No network and no model when live access is off: see the note in
    # daily_briefing.generate_daily_briefing. Skipping the news retrieval
    # alone still called the model, on an empty evidence block.
    if not allow_live:
        return without_synthesis(
            "Signals only",
            "The signals below are read from stored data. No model was called, "
            "so there is no written assessment.")

    try:
        from src.rag.explainer import _call_model, log_call
        response, model_used = _call_model(prompt)
        parsed = json.loads(response.text)
        log_call("outlook", getattr(response, "usage_metadata", None),
                 "OUTLOOK", model=model_used)
    except Exception as exc:
        # Fallback: return signal data without LLM synthesis
        return {
            "lean": "balanced",
            "confidence": "low",
            "label": "Assessment unavailable",
            "summary": f"Could not generate outlook: {type(exc).__name__}. "
                       f"Signal data is shown below without synthesis.",
            "signal_assessments": [],
            "citations": [],
            "drivers": _signals_to_fallback_drivers(signals),
            "signals_collected": signals,
            "retrieval_info": retrieval_info,
            "accepted_documents": [],
            "error": str(exc),
        }

    # 4. Post-process: some models (notably Groq free-tier) fill
    #    signal_assessments but leave the aggregate fields empty. Derive
    #    them from the assessments so the dashboard always gets valid data.
    parsed = _postprocess_outlook(parsed)

    # 5. Run faithfulness audit on citations
    faithfulness_report = None
    if parsed.get("citations") and accepted_docs:
        try:
            from src.rag.faithfulness import check_citations
            faithfulness_report = check_citations(
                {"explanation": parsed.get("summary", ""), "citations": parsed["citations"]},
                accepted_docs,
            )
        except Exception:
            pass

    # 6. Format output
    lean = parsed["lean"]
    confidence = parsed["confidence"]

    label_map = {
        ("upward", "high"): "Strong upward pressure",
        ("upward", "medium"): "Upward pressure",
        ("upward", "low"): "Slight upward lean",
        ("downward", "high"): "Strong downward pressure",
        ("downward", "medium"): "Downward pressure",
        ("downward", "low"): "Slight downward lean",
        ("balanced", "high"): "Mixed signals, no clear direction",
        ("balanced", "medium"): "Mixed signals",
        ("balanced", "low"): "Mixed signals, uncertain",
    }
    label = label_map.get((lean, confidence), "Mixed signals")

    # Build drivers list from signal assessments
    drivers = []
    arrow_map = {"upward": 1, "downward": -1, "neutral": 0}
    for sa in parsed.get("signal_assessments", []):
        drivers.append({
            "factor": sa.get("factor", ""),
            "direction": sa.get("direction", "neutral"),
            "score": arrow_map.get(sa.get("direction", "neutral"), 0),
            "why": sa.get("reasoning", ""),
            "data_point": sa.get("data_point", ""),
        })

    return {
        "lean": lean,
        "confidence": confidence,
        "label": label,
        "summary": parsed.get("summary", ""),
        "signal_assessments": parsed.get("signal_assessments", []),
        "n_upward": parsed["n_upward"],
        "n_downward": parsed["n_downward"],
        "n_neutral": parsed["n_neutral"],
        "citations": parsed.get("citations", []),
        "drivers": drivers,
        "signals_collected": signals,
        "retrieval_info": retrieval_info,
        "accepted_documents": accepted_docs,
        "faithfulness_report": faithfulness_report,
    }


def _signals_to_fallback_drivers(signals: dict) -> list:
    """When the LLM call fails, produce a basic driver list from raw signals."""
    drivers = []
    if "seasonal" in signals:
        phase = signals["seasonal"]["brazil_phase"].lower()
        if "harvest" in phase:
            drivers.append({"factor": "Brazil harvest", "direction": "downward",
                            "score": -1, "why": f"Brazil is in {phase} phase (supply increase)",
                            "data_point": phase})
        elif "flowering" in phase or "growing" in phase:
            drivers.append({"factor": "Brazil crop phase", "direction": "neutral",
                            "score": 0, "why": f"Brazil is in {phase} phase",
                            "data_point": phase})

    if "weather_observed" in signals:
        w = signals["weather_observed"]
        if w["frost_7d_count"] > 0:
            drivers.append({"factor": "Frost detected", "direction": "upward",
                            "score": 1, "why": f"{w['frost_7d_count']} frost day(s) — supply risk",
                            "data_point": f"{w['frost_7d_count']} days"})
        elif w["dryness_zscore"] > 1.5:
            drivers.append({"factor": "Drought stress", "direction": "upward",
                            "score": 1, "why": f"Dryness z-score {w['dryness_zscore']:+.1f} — supply risk",
                            "data_point": f"z={w['dryness_zscore']:+.1f}"})

    if "currency" in signals:
        fx = signals["currency"]
        if fx["change_1m_pct"] > 2:
            drivers.append({"factor": "Weak BRL", "direction": "downward",
                            "score": -1, "why": "Weaker real encourages Brazilian exports",
                            "data_point": f"{fx['change_1m_pct']:+.1f}% 1m"})
        elif fx["change_1m_pct"] < -2:
            drivers.append({"factor": "Strong BRL", "direction": "upward",
                            "score": 1, "why": "Stronger real discourages Brazilian exports",
                            "data_point": f"{fx['change_1m_pct']:+.1f}% 1m"})

    return drivers


if __name__ == "__main__":
    import sys
    allow = "--no-live" not in sys.argv
    result = generate_outlook(allow_live=allow)
    print(json.dumps(result, indent=2, default=str))
