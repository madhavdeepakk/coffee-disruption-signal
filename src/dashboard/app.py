"""
Coffee market intelligence dashboard.

Opens with the price outlook and the market brief (what matters right now
across supply, demand, events and context). The price chart, anomaly flags, and the RAG
explanation engine are supporting sections below.

Usage:
    streamlit run src/dashboard/app.py
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config.commodities import get_commodity
from src.modeling.anomaly_detector import compute_anomalies, load_contract_switches
from src.rag import outcome as outcome_mod
from src.advisory.buying_brief import generate_brief, _generate_headline
from src.advisory.outlook_rag import generate_outlook
from src.advisory.daily_briefing import generate_daily_briefing
from src.advisory.sentiment_log import log_sentiment, load_sentiment_log, get_sentiment_streak

READ_ONLY = os.environ.get("READ_ONLY", "").strip().lower() in ("1", "true", "yes")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

COMMODITY_KEY = "coffee"
cfg = get_commodity(COMMODITY_KEY)

st.set_page_config(page_title="Coffee Market Intelligence", layout="centered")

# --- Custom styling for a clean, professional look ---
st.markdown("""
<style>
    /* Tighten up the default Streamlit spacing */
    .block-container { padding-top: 2rem; max-width: 800px; }

    /* Section headers */
    h1 { font-size: 1.6rem !important; font-weight: 600 !important;
         letter-spacing: -0.02em; margin-bottom: 0.25rem !important; }
    h2 { font-size: 1.25rem !important; font-weight: 600 !important;
         margin-top: 1.5rem !important; }
    h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

    /* Observation cards */
    .obs-card {
        border-left: 3px solid #4a7c9b;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.75rem;
        background: #fafbfc;
        border-radius: 0 4px 4px 0;
    }
    .obs-card.urgent {
        border-left-color: #c0392b;
        background: #fdf2f2;
    }
    .obs-card .obs-topic {
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.2rem;
        color: #2c3e50;
    }
    .obs-card .obs-text {
        font-size: 0.88rem;
        line-height: 1.5;
        color: #34495e;
    }
    .obs-card .obs-source {
        font-size: 0.75rem;
        color: #95a5a6;
        margin-top: 0.3rem;
    }

    /* Headline block */
    .headline-block {
        border-left: 4px solid #2c3e50;
        padding: 0.8rem 1rem;
        margin: 0.75rem 0 1.25rem 0;
        background: #f8f9fa;
        font-size: 0.95rem;
        line-height: 1.55;
        color: #2c3e50;
    }

    /* Category section title */
    .cat-title {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #7f8c8d;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #ecf0f1;
        margin-bottom: 0.5rem;
    }

    /* Outlook */
    .outlook-box {
        border: 1px solid #e3e6ea;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
        background: #fbfbfc;
    }
    .lean {
        display: inline-block;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        color: #fff;
        margin-bottom: 0.5rem;
    }
    .lean.up { background: #b03a2e; }
    .lean.down { background: #1f6f8b; }
    .lean.balanced { background: #7f8c8d; }
    .outlook-summary { font-size: 0.95rem; line-height: 1.55; color: #2c3e50; }
    .outlook-range { font-size: 0.88rem; color: #34495e; margin-top: 0.5rem; }
    .driver { font-size: 0.86rem; line-height: 1.5; color: #34495e; margin: 0.15rem 0; }
    .driver .arrow { display: inline-block; width: 1.2rem; font-weight: 700; }
    .driver .arrow.up { color: #b03a2e; }
    .driver .arrow.down { color: #1f6f8b; }
    .driver .arrow.flat { color: #95a5a6; }
    .src { font-size: 0.85rem; line-height: 1.5; margin: 0.2rem 0; color: #34495e; }
    .src .meta { color: #95a5a6; font-size: 0.78rem; }

    /* Daily briefing */
    .briefing-headline {
        font-size: 1.05rem;
        font-weight: 600;
        color: #2c3e50;
        border-left: 4px solid #e67e22;
        padding: 0.6rem 0.9rem;
        margin: 0.5rem 0 0.75rem 0;
        background: #fef9f0;
        border-radius: 0 4px 4px 0;
    }
    .briefing-section {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #34495e;
        margin-bottom: 0.6rem;
    }
    .briefing-section .section-title {
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #7f8c8d;
        margin-bottom: 0.2rem;
    }
    .briefing-dev {
        font-size: 0.9rem;
        line-height: 1.5;
        color: #34495e;
        padding-left: 0.8rem;
        border-left: 2px solid #e0e4e8;
        margin: 0.3rem 0;
    }
    .watch-item {
        font-size: 0.88rem;
        color: #34495e;
        padding: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data + pipeline helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_prices(price_file: str) -> pd.DataFrame:
    path = REPO_ROOT / "data" / "raw" / price_file
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


@st.cache_data
def detect_anomalies(price_file: str, fingerprint: float) -> pd.DataFrame:
    prices = load_prices(price_file)
    if prices.empty:
        return pd.DataFrame()
    res = compute_anomalies(prices[["date", "price"]].copy(),
                            reported_returns=load_contract_switches(COMMODITY_KEY))
    res["date"] = pd.to_datetime(res["date"])
    return res.sort_values("date")


def pipeline_output(date_str: str):
    path = RESULTS_DIR / f"pipeline_output_{COMMODITY_KEY}_{date_str}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


@st.cache_data
def cached_decisions(fingerprint: str) -> dict:
    import glob
    out = {}
    for p in glob.glob(str(RESULTS_DIR / f"pipeline_output_{COMMODITY_KEY}_*.json")):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        date = Path(p).stem.replace(f"pipeline_output_{COMMODITY_KEY}_", "")
        dec = (d.get("explanation_result") or {}).get("decision")
        out[date] = ("explained" if dec == "EXPLAINED"
                     else "error" if dec in ("API_ERROR", "PARSE_ERROR")
                     else "refused" if dec else "other")
    return out


def run_pipeline(date_str: str) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run(
            [sys.executable, "-m", "src.pipeline", "--date", date_str,
             "--commodity", COMMODITY_KEY],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace", env=env,
        )
        return r.returncode == 0, (r.stdout or "") + "\n" + (r.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return False, f"Timed out after 300s.\n{exc}"


def refresh_to_today() -> tuple[bool, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    logs = []
    for step in (["-m", "src.modeling.fetch_price_data", "--commodity", COMMODITY_KEY],
                 ["-m", "src.modeling.anomaly_detector", "--commodity", COMMODITY_KEY]):
        try:
            r = subprocess.run([sys.executable, *step], cwd=str(REPO_ROOT),
                               capture_output=True, text=True, timeout=180,
                               encoding="utf-8", errors="replace", env=env)
            logs.append((r.stdout or "") + "\n" + (r.stderr or ""))
            if r.returncode != 0:
                return False, "\n".join(logs)
        except subprocess.TimeoutExpired as exc:
            return False, f"Timed out.\n{exc}"
    return True, "\n".join(logs)


@st.cache_data
def article_index(fingerprint: int) -> dict:
    """Map document_id -> article metadata from the local GDELT cache, so saved
    explanations can show article titles and links instead of raw ids."""
    idx = {}
    cache_dir = REPO_ROOT / "data" / "gdelt_cache"
    for p in cache_dir.glob("*.json"):
        try:
            entry = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        for art in (entry.get("response") or {}).get("articles", []) or []:
            url = art.get("url") or ""
            if not url:
                continue
            doc_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            seen = art.get("seendate") or ""
            date = f"{seen[6:8]}/{seen[4:6]}/{seen[0:4]}" if len(seen) >= 8 else ""
            idx[doc_id] = {"title": art.get("title") or url, "url": url,
                           "domain": art.get("domain") or "", "date": date}
    return idx


_ID_RE = re.compile(r"(?:doc_)?([0-9a-f]{16})")

# --- Evidence posture classification ---
_SUPPLY_TERMS = {
    "frost", "drought", "harvest", "crop", "production", "output", "yield",
    "weather", "rainfall", "rain", "plantation", "farmer", "grower",
    "export", "shipment", "shipping", "freight", "logistics", "port",
    "inventory", "stock", "warehouse", "storage", "supply", "surplus",
    "deficit", "shortage", "arabica", "robusta", "bean",
}
_DEMAND_TERMS = {
    "demand", "consumption", "import", "buyer", "roaster", "retail",
    "consumer", "market", "sales", "growth", "economy", "gdp",
    "inflation", "recession", "trade", "tariff", "sanction",
}
_EVENT_TERMS = {
    "strike", "protest", "war", "conflict", "regulation", "policy",
    "government", "election", "currency", "real", "dollar", "exchange",
    "speculation", "futures", "hedge", "fund", "ipo", "merger",
    "pandemic", "covid", "lockdown", "volcano", "earthquake", "flood",
}


def _classify_evidence(title: str) -> str:
    """Classify a document title as supply, demand, or event."""
    title_lower = title.lower()
    words = set(re.findall(r"[a-z]+", title_lower))
    s = len(words & _SUPPLY_TERMS)
    d = len(words & _DEMAND_TERMS)
    e = len(words & _EVENT_TERMS)
    if s >= d and s >= e and s > 0:
        return "supply"
    if d >= s and d >= e and d > 0:
        return "demand"
    if e > 0:
        return "event"
    return "other"


def render_evidence_posture(output: dict):
    """Show a supply/demand/event signal breakdown of accepted evidence."""
    gate = output.get("gate_result", {}) or {}
    accepted_ids = gate.get("accepted_document_ids", [])
    if not accepted_ids:
        return

    cache = article_index(len(list((REPO_ROOT / "data" / "gdelt_cache").glob("*.json"))))
    saved = {d.get("document_id"): d for d in (output.get("sources") or [])}

    counts = {"supply": 0, "demand": 0, "event": 0, "other": 0}
    for doc_id in accepted_ids:
        meta = saved.get(doc_id) or cache.get(doc_id)
        title = (meta or {}).get("title", "") or ""
        category = _classify_evidence(title)
        counts[category] += 1

    total = sum(counts.values())
    if total == 0:
        return

    # Only show categories with signals
    active = {k: v for k, v in counts.items() if v > 0 and k != "other"}
    if not active:
        return

    direction = (output.get("direction_summary") or {})
    consistent = direction.get("consistent", 0)
    inconsistent = direction.get("inconsistent", 0)

    labels = {"supply": "Supply-side", "demand": "Demand-side", "event": "Event-driven"}
    icons = {"supply": "🌿", "demand": "📊", "event": "⚡"}

    cols = st.columns(len(active) + (1 if consistent + inconsistent > 0 else 0))
    for i, (cat, count) in enumerate(active.items()):
        pct = round(100 * count / total)
        cols[i].metric(f"{icons.get(cat, '')} {labels.get(cat, cat)}", f"{count} signals", f"{pct}%")

    if consistent + inconsistent > 0:
        aligned_pct = round(100 * consistent / (consistent + inconsistent))
        cols[-1].metric("🎯 Direction Match", f"{aligned_pct}%",
                        f"{consistent} of {consistent + inconsistent} articles")


def _number_citations(text: str, order: dict, url_map: dict | None = None) -> str:
    """Replace '(source: doc_abc..., doc_def...)' with clickable numbered refs."""
    url_map = url_map or {}

    def num(doc_id):
        if doc_id not in order:
            order[doc_id] = len(order) + 1
        return order[doc_id]

    def _link(doc_id):
        n = num(doc_id)
        url = url_map.get(doc_id)
        if url:
            return f'<a href="{url}" target="_blank" style="text-decoration:none;color:#2563eb;font-weight:600">[{n}]</a>'
        return f"[{n}]"

    def group(m):
        ids = _ID_RE.findall(m.group(1))
        if not ids:
            return ""
        return " " + "".join(_link(i) for i in dict.fromkeys(ids))

    text = re.sub(r"\s*\((?:sources?|see)\s*:?\s*([^)]*)\)", group, text, flags=re.I)
    text = re.sub(r"\[?\bdoc_([0-9a-f]{16})\b\]?", lambda m: _link(m.group(1)), text)
    return text


def render_explanation(output: dict):
    explanation = output.get("explanation_result", {}) or {}
    outcome = outcome_mod.classify(
        retrieval_meta=output.get("retrieval_meta"),
        gate_result=output.get("gate_result"),
        explanation_result=explanation,
        direction_summary=output.get("direction_summary"),
        faithfulness_report=output.get("faithfulness_report"),
    )
    decision = explanation.get("decision")

    # --- Brief info bar ---
    model_used = explanation.get("model_used", "")
    confidence = explanation.get("confidence", "")
    r_meta = output.get("retrieval_meta")
    info_parts = []
    if model_used:
        info_parts.append(f"Model: **{model_used}**")
    if confidence:
        info_parts.append(f"Confidence: **{confidence}**")
    n_docs = (r_meta or {}).get("documents_returned", 0) or output.get("documents_retrieved", 0)
    if n_docs:
        info_parts.append(f"**{n_docs}** news articles analyzed")
    if info_parts:
        st.caption(" · ".join(info_parts))

    if decision == "EXPLAINED":
        st.success(f"**Explained.** {outcome.detail}")

        citations = explanation.get("citations", []) or []
        supports = {c.get("document_id"): c.get("supports", "") for c in citations}
        # Runs decided by the blind-evidence rule carry the quote each reason
        # rests on; show it under the source so the reader can check it.
        for c in citations:
            if c.get("quote"):
                quote = str(c["quote"]).replace("<", "&lt;").replace(">", "&gt;")
                supports[c.get("document_id")] = (f'{c.get("supports", "")} '
                                                  f'&mdash; &ldquo;{quote}&rdquo;')

        saved = {d.get("document_id"): d for d in (output.get("sources") or [])}
        cache = article_index(len(list((REPO_ROOT / "data" / "gdelt_cache").glob("*.json"))))

        # Build URL lookup so inline citations become clickable links
        all_ids = set()
        all_ids.update(_ID_RE.findall(explanation.get("explanation", "") or ""))
        all_ids.update(c.get("document_id", "") for c in citations)
        url_map = {}
        for doc_id in all_ids:
            meta = saved.get(doc_id) or cache.get(doc_id)
            if meta and meta.get("url"):
                url_map[doc_id] = meta["url"]

        order = {}
        text = explanation.get("explanation", "") or ""
        # From pipeline version 11 the explanation is itself the list of
        # sources: one line each with headline, outlet, date, reported move,
        # reason and quote. The source marker at the end of each line becomes
        # the link, and no second list is drawn under it.
        one_line_per_source = any("reported_move" in c for c in citations)
        if one_line_per_source:
            text = text.replace("<", "&lt;").replace(">", "&gt;")
        body = _number_citations(text, order, url_map)
        st.markdown(body, unsafe_allow_html=True)

        for c in citations:
            order.setdefault(c.get("document_id"), len(order) + 1)

        if order and not one_line_per_source:
            st.markdown("**Sources**")
            for doc_id, n in sorted(order.items(), key=lambda kv: kv[1]):
                meta = saved.get(doc_id) or cache.get(doc_id)
                note = supports.get(doc_id, "")
                if meta and meta.get("url"):
                    title = (meta.get("title") or meta["url"]).replace("[", "(").replace("]", ")")
                    extra = " · ".join(x for x in (meta.get("domain", ""),
                                                   meta.get("date") or meta.get("publication_date", "")) if x)
                    line = (f'<div class="src">[{n}] <a href="{meta["url"]}" target="_blank">{title}</a>'
                            + (f' <span class="meta">{extra}</span>' if extra else "")
                            + (f'<br><span class="meta">{note}</span>' if note else "") + "</div>")
                else:
                    line = f'<div class="src">[{n}] {note or "News article"}</div>'
                st.markdown(line, unsafe_allow_html=True)
    elif decision in ("API_ERROR", "PARSE_ERROR"):
        # Show specific error details so users know what went wrong
        error_reason = explanation.get("reason", "") or ""
        if "RESOURCE_EXHAUSTED" in error_reason or "429" in error_reason:
            st.error("**Rate limited.** The LLM API quota is temporarily exhausted. "
                     "Wait a minute and try again.")
        elif "API_KEY" in error_reason.upper() or "authentication" in error_reason.lower():
            st.error("**API key issue.** The explanation service could not authenticate. "
                     "Check that GEMINI_API_KEY is set correctly in .env.")
        elif "PARSE_ERROR" in decision:
            st.error("**Parse error.** The LLM returned a response that could not be "
                     "parsed. This is usually a transient issue — try again.")
        elif "timeout" in error_reason.lower() or "connect" in error_reason.lower():
            st.error("**Network error.** Could not reach the explanation service. "
                     "Check your internet connection and try again.")
        else:
            st.error(f"**Service error.** The explanation service could not be reached "
                     f"for this date. Try again shortly. "
                     f"({'Detail: ' + error_reason[:120] if error_reason else 'This says nothing about the market itself.'})")
    else:
        st.warning(f"**No explanation available.** {outcome.detail}")
        gate = output.get("gate_result", {}) or {}
        reason = gate.get("reason", "") or explanation.get("reason", "")
        total_ct = gate.get("total_documents_considered", 0)

        with st.expander("Why?", expanded=True):
            if reason == "no_documents_retrieved":
                st.markdown("No news articles could be found for this date.")
            elif reason == "all_documents_below_noise_floor":
                st.markdown("We found articles, but none of them were about coffee markets.")
            elif "direction_guard" in (explanation.get("reason") or ""):
                st.markdown(f"We found {total_ct} articles and the model drafted an "
                            f"explanation, but more of the relevant articles described "
                            f"prices moving the opposite way than described this move. "
                            f"An explanation built on that is withheld.")
            elif "blind_evidence" in (explanation.get("reason") or ""):
                if "other_way" in explanation["reason"]:
                    st.markdown(f"We checked {total_ct} articles. The most recent ones "
                                f"that give a reason for a price move describe prices "
                                f"moving the opposite way to this one, so no explanation "
                                f"is offered.")
                else:
                    st.markdown(f"We checked {total_ct} articles, but none published on "
                                f"the day or in the two trading days before gives a "
                                f"reason, with a quote we could find in the article, for "
                                f"prices moving this way.")
            elif "model_judged_evidence_insufficient" in reason:
                st.markdown(f"We found {total_ct} articles and some looked relevant, "
                            f"but none contained enough detail to confidently explain "
                            f"what caused the price move.")
            else:
                accepted_ct = gate.get("accepted_document_count", 0)
                st.markdown(f"We checked {total_ct} news articles but only {accepted_ct} "
                            f"were relevant enough. The evidence wasn't strong enough "
                            f"to give a reliable explanation.")

        st.caption("The system only explains when the evidence clearly supports it — "
                   "no guessing.")


def render_outlook(outlook: dict, price_now: float):
    if not outlook:
        return
    lean = outlook.get("lean", "balanced")
    confidence = outlook.get("confidence", "low")

    # Confidence badge color
    conf_colors = {"high": "#27ae60", "medium": "#f39c12", "low": "#95a5a6"}
    conf_color = conf_colors.get(confidence, "#95a5a6")

    # Build URL map for inline citation links
    outlook_accepted = outlook.get("accepted_documents", [])
    outlook_url_map = {}
    for d in outlook_accepted:
        did = (d.get("document_id") or "").replace("doc_", "")
        if did and d.get("url"):
            outlook_url_map[did] = d["url"]

    def _clean_outlook(text: str) -> str:
        order = {}
        return _number_citations(text, order, outlook_url_map)

    html = [f'<div class="outlook-box">',
            f'<span class="lean {lean}">{outlook.get("label", "")}</span>',
            f' <span style="font-size:0.75rem; background:{conf_color}; color:#fff; '
            f'padding:0.1rem 0.4rem; border-radius:3px; margin-left:0.4rem;">'
            f'{confidence} confidence</span>',
            f'<div class="outlook-summary">{_clean_outlook(outlook.get("summary", ""))}</div>']

    # Signal agreement counts
    n_up = outlook.get("n_upward", 0)
    n_down = outlook.get("n_downward", 0)
    n_neut = outlook.get("n_neutral", 0)
    if n_up + n_down + n_neut > 0:
        html.append(f'<div style="font-size:0.82rem; color:#7f8c8d; margin-top:0.4rem;">'
                    f'Signal tally: {n_up} upward · {n_down} downward · {n_neut} neutral</div>')

    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    with st.expander("Signal-by-signal reasoning", expanded=False):
        rows = []
        for d in sorted(outlook.get("drivers", []), key=lambda d: -abs(d.get("score", 0))):
            score = d.get("score", 0)
            if score > 0:
                arrow, cls = "▲", "up"
            elif score < 0:
                arrow, cls = "▼", "down"
            else:
                arrow, cls = "●", "flat"
            why = d.get("why", "")
            data_pt = d.get("data_point", "")
            detail = why
            if data_pt and data_pt not in why:
                detail += f" ({data_pt})"
            rows.append(f'<div class="driver"><span class="arrow {cls}">{arrow}</span>'
                        f'<b>{d.get("factor", "")}</b>: {detail}</div>')
        st.markdown("".join(rows), unsafe_allow_html=True)

    # News citations
    citations = outlook.get("citations", [])
    accepted = outlook.get("accepted_documents", [])
    if citations and accepted:
        with st.expander("Cited news sources", expanded=False):
            docs_by_id = {}
            for doc in accepted:
                did = doc.get("document_id", "")
                docs_by_id[did] = doc
                # Also map without doc_ prefix
                if did.startswith("doc_"):
                    docs_by_id[did[4:]] = doc
            for c in citations:
                cid = c.get("document_id", "")
                doc = docs_by_id.get(cid) or docs_by_id.get(cid.replace("doc_", ""))
                supports = c.get("supports", "")
                if doc and doc.get("url"):
                    title = (doc.get("title") or doc["url"]).replace("[", "(").replace("]", ")")
                    st.markdown(f'<div class="src"><a href="{doc["url"]}" target="_blank">{title}</a>'
                                f'<br><span class="meta">{supports}</span></div>',
                                unsafe_allow_html=True)
                elif supports:
                    st.markdown(f'<div class="src">{supports}</div>', unsafe_allow_html=True)

    # Faithfulness audit
    fr = outlook.get("faithfulness_report")
    if fr:
        with st.expander("Citation audit", expanded=False):
            ok = fr.get("n_ok", 0)
            total = fr.get("n_citations", 0)
            missing = fr.get("n_missing_document", 0)
            weak = fr.get("n_weak_support", 0)
            st.markdown(f"**{ok}/{total}** citations verified · "
                        f"{weak} weak support · {missing} missing document"
                        + ("" if fr.get("citations_resolve", True)
                           else " · ⚠️ unresolved citations detected"))



# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

prices = load_prices(cfg.price_file)
price_path = REPO_ROOT / "data" / "raw" / cfg.price_file
fp = price_path.stat().st_mtime if price_path.exists() else 0.0
anomalies = detect_anomalies(cfg.price_file, fp)

if prices.empty or anomalies.empty:
    st.title("Coffee Market Intelligence")
    st.error("No price data found. Run: `python -m src.modeling.fetch_price_data --commodity coffee`")
    st.stop()

today = datetime.now().date()
latest_row = anomalies.iloc[-1]
latest_dt = latest_row["date"].date()
latest_ds = latest_dt.strftime("%Y-%m-%d")
stale_days = (today - latest_dt).days
flagged = anomalies[anomalies["anomaly_flag"] == True]  # noqa: E712


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: INTELLIGENCE BRIEF
# ═══════════════════════════════════════════════════════════════════════════

st.title("Coffee Market Intelligence")

stale_note = f"  ·  {stale_days} days old, use Refresh below" if stale_days > 3 else ""
st.caption(f"Arabica coffee futures (ICE, KC=F)  ·  Data as of {latest_dt:%d %b %Y}{stale_note}")

# --- Key metrics ---
_p = prices["price"].to_numpy()
_now = float(_p[-1])
_mo = (_now / _p[-22] - 1) * 100 if len(_p) > 22 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Price (US¢/lb)", f"{_now:.2f}")
c2.metric("1-Month Change", f"{_mo:+.1f}%")
c3.metric("Anomalies Detected", f"{len(flagged)}")

@st.cache_data(ttl=3600, show_spinner="Loading market brief...")
def cached_brief(key):
    return generate_brief(key)


def render_daily_briefing(briefing: dict):
    """Render the daily market intelligence briefing."""
    if not briefing:
        return

    # Build URL map from accepted documents so inline citations become links
    accepted = briefing.get("accepted_documents", [])
    brief_url_map = {}
    for d in accepted:
        did = (d.get("document_id") or "").replace("doc_", "")
        if did and d.get("url"):
            brief_url_map[did] = d["url"]

    def _clean(text: str) -> str:
        """Replace (source: ...) with clickable numbered refs."""
        order = {}
        return _number_citations(text, order, brief_url_map)

    headline = briefing.get("headline", "")
    if headline:
        st.markdown(f'<div class="briefing-headline">{headline}</div>',
                    unsafe_allow_html=True)

    # Price action
    pa = briefing.get("price_action", "")
    if pa:
        st.markdown(f'<div class="briefing-section">'
                    f'<div class="section-title">Price Action</div>{_clean(pa)}</div>',
                    unsafe_allow_html=True)

    # Key developments
    devs = briefing.get("key_developments", [])
    if devs:
        with st.expander("Key Developments", expanded=True):
            for dev in devs:
                st.markdown(f'<div class="briefing-dev">{_clean(dev)}</div>',
                            unsafe_allow_html=True)

    # Weather & crop
    wc = briefing.get("weather_crop", "")
    if wc:
        with st.expander("Weather & Crop Conditions", expanded=False):
            st.markdown(f'<div class="briefing-section">{_clean(wc)}</div>',
                        unsafe_allow_html=True)

    # Positioning
    pos = briefing.get("positioning", "")
    if pos:
        with st.expander("Speculative Positioning", expanded=False):
            st.markdown(f'<div class="briefing-section">{_clean(pos)}</div>',
                        unsafe_allow_html=True)

    # Watch list
    watch = briefing.get("watch_list", [])
    if watch:
        with st.expander("Watch List", expanded=False):
            for item in watch:
                st.markdown(f'<div class="watch-item">• {_clean(item)}</div>',
                            unsafe_allow_html=True)

    # Sources
    citations = briefing.get("citations", [])
    accepted = briefing.get("accepted_documents", [])
    if citations and accepted:
        with st.expander("News Sources", expanded=False):
            docs_by_id = {d.get("document_id", ""): d for d in accepted}
            for c in citations:
                cid = c.get("document_id", "")
                doc = docs_by_id.get(cid) or docs_by_id.get(cid.replace("doc_", ""))
                supports = c.get("supports", "")
                if doc and doc.get("url"):
                    title = (doc.get("title") or doc["url"]).replace("[", "(").replace("]", ")")
                    source_type = doc.get("_source", "")
                    st.markdown(f'<div class="src"><a href="{doc["url"]}" target="_blank">{title}</a>'
                                f'<br><span class="meta">{supports}</span>'
                                f'{f" · <span class=meta>{source_type}</span>" if source_type else ""}'
                                f'</div>', unsafe_allow_html=True)
                elif supports:
                    st.markdown(f'<div class="src">{supports}</div>', unsafe_allow_html=True)

    sources_used = briefing.get("sources_used", 0)
    model_gen = briefing.get("model_generated", False)
    st.caption(f"Briefing from {sources_used} news sources · "
               f"{'LLM-synthesized' if model_gen else 'Data-only (LLM unavailable)'} · "
               f"Refreshes hourly")


try:
    brief = cached_brief(COMMODITY_KEY)

    # --- Daily market briefing (RAG-powered) ---
    st.subheader("Daily Market Briefing")

    @st.cache_data(ttl=3600, show_spinner="Generating daily briefing...")
    def cached_daily_briefing():
        return generate_daily_briefing(allow_live=not READ_ONLY)

    try:
        briefing_data = cached_daily_briefing()
        render_daily_briefing(briefing_data)
    except Exception as briefing_exc:
        st.warning(f"Daily briefing unavailable: {briefing_exc}")

    st.divider()

    # --- Price outlook (RAG-powered) ---
    st.subheader("Price Outlook")

    @st.cache_data(ttl=3600, show_spinner="Generating outlook...")
    def cached_outlook():
        return generate_outlook(allow_live=not READ_ONLY)

    try:
        outlook_data = cached_outlook()
        render_outlook(outlook_data, brief.current_price)
    except Exception as outlook_exc:
        outlook_data = None
        st.warning(f"Outlook unavailable: {outlook_exc}")

    # --- Log sentiment & show timeline ---
    try:
        _briefing_for_log = briefing_data if "briefing_data" in dir() else None
        _outlook_for_log = outlook_data if outlook_data else None
        if _outlook_for_log or _briefing_for_log:
            log_sentiment(outlook=_outlook_for_log, briefing=_briefing_for_log)

        sentiment_entries = load_sentiment_log()
        if len(sentiment_entries) >= 2:
            streak = get_sentiment_streak(sentiment_entries)
            if streak["days"] > 1:
                streak_label = (f"{'🔴' if streak['direction'] == 'bearish' else '🟢' if streak['direction'] == 'bullish' else '⚪'} "
                                f"Outlook has been {streak['direction']} for {streak['days']} consecutive days")
                st.caption(streak_label)

            _sent_dates = [e["date"] for e in sentiment_entries]
            _lean_map = {"upward": 1, "balanced": 0, "downward": -1}
            _conf_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
            _lean_vals = [_lean_map.get(e.get("lean", "balanced"), 0) for e in sentiment_entries]
            _conf_vals = [_conf_map.get(e.get("confidence", "low"), 0.3) for e in sentiment_entries]
            _colors = ["#b03a2e" if v > 0 else "#1f6f8b" if v < 0 else "#95a5a6" for v in _lean_vals]

            with st.expander("Sentiment Timeline", expanded=False):
                fig_s = go.Figure()
                fig_s.add_trace(go.Bar(
                    x=_sent_dates, y=_lean_vals,
                    marker_color=_colors,
                    marker_opacity=[c for c in _conf_vals],
                    name="Lean",
                    hovertemplate="%{x}<br>Lean: %{customdata[0]}<br>Confidence: %{customdata[1]}<extra></extra>",
                    customdata=[[e.get("lean", "?"), e.get("confidence", "?")] for e in sentiment_entries],
                ))
                fig_s.update_layout(
                    template="plotly_white", height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(tickvals=[-1, 0, 1], ticktext=["Bearish", "Neutral", "Bullish"],
                               range=[-1.3, 1.3]),
                    xaxis_title="",
                )
                st.plotly_chart(fig_s, use_container_width=True)
                st.caption("Bar height = direction, opacity = confidence. "
                           f"{len(sentiment_entries)} data points logged.")
    except Exception:
        pass  # sentiment logging/display is non-critical

    st.subheader("Market Brief")

    # --- Headline ---
    headline = _generate_headline(brief.observations)
    st.markdown(f'<div class="headline-block">{headline}</div>',
                unsafe_allow_html=True)

    # --- Category sections ---
    _section_titles = {
        "supply": "Supply",
        "demand": "Demand & Market",
        "events": "Events",
        "context": "Context",
    }

    for cat_key in ["supply", "demand", "events", "context"]:
        obs_list = [o for o in brief.observations if o.category == cat_key]
        if not obs_list:
            continue

        title = _section_titles.get(cat_key, cat_key.title())
        has_important = any(o.importance <= 2 for o in obs_list)

        with st.expander(f"**{title}**", expanded=has_important):
            for idx, o in enumerate(sorted(obs_list, key=lambda x: x.importance)):
                card_class = "obs-card urgent" if o.importance == 1 else "obs-card"
                st.markdown(
                    f'<div class="{card_class}">'
                    f'<div class="obs-topic">{o.topic}</div>'
                    f'<div class="obs-text">{o.text}</div>'
                    f'<div class="obs-source">{o.source}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if o.detail:
                    with st.expander("Detailed analysis",
                                     expanded=False):
                        st.write(o.detail)

except Exception as e:
    st.error(f"Could not load the market brief: {e}")


st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: PRICE CHART
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Price History")
st.caption("Daily closing price since 2018. Triangles mark days when the price "
           "moved far outside its recent normal range.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=prices["date"], y=prices["price"], mode="lines",
                         name="Price", line=dict(color="#5B7C99", width=1.3)))
up = flagged[flagged["z_score"] > 0]
down = flagged[flagged["z_score"] < 0]
fig.add_trace(go.Scatter(x=up["date"], y=up["price"], mode="markers",
                         name="Unusual jump up",
                         marker=dict(color="#C0392B", size=7, symbol="triangle-up")))
fig.add_trace(go.Scatter(x=down["date"], y=down["price"], mode="markers",
                         name="Unusual drop",
                         marker=dict(color="#2471A3", size=7, symbol="triangle-down")))
fig.update_layout(template="plotly_white", height=380,
                  margin=dict(l=10, r=10, t=10, b=10),
                  hovermode="x unified", yaxis_title="Price (US cents/lb)",
                  legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)
st.caption(f"{len(flagged)} unusual days out of {len(anomalies)} trading days.")


st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: LATEST DAY + REFRESH
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Latest Trading Day")

latest_flagged = bool(latest_row["anomaly_flag"])
latest_sigma = latest_row["z_score"]
latest_dir = "up" if latest_sigma > 0 else "down"

if latest_flagged:
    st.write(f"**{latest_dt:%d %b %Y}: unusual {latest_dir} move.** "
             f"The price moved well outside its recent range.")
else:
    st.write(f"**{latest_dt:%d %b %Y}: normal day.** No unusual price movement.")

if not READ_ONLY:
    if st.button("Refresh to today (fetch latest prices)"):
        with st.spinner("Fetching latest prices and re-running detection..."):
            ok, log = refresh_to_today()
        if ok:
            st.cache_data.clear()
            st.success("Refreshed.")
            st.rerun()
        else:
            with st.expander("Refresh log"):
                st.code(log)
            st.error("Refresh failed (Yahoo Finance may be unreachable).")

latest_cached = pipeline_output(latest_ds)
if latest_flagged and latest_cached is not None:
    render_explanation(latest_cached)
    render_evidence_posture(latest_cached)
elif latest_flagged and not READ_ONLY:
    if st.button(f"Explain {latest_dt:%d %b %Y}", type="primary"):
        with st.spinner("Retrieving news and generating an explanation (1-2 min)..."):
            ok, log = run_pipeline(latest_ds)
        if ok:
            st.cache_data.clear()
            st.rerun()
        else:
            with st.expander("Log"):
                st.code(log)


st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: EXPLAIN ANY DAY
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Explain a Day")

st.caption("Pick a flagged day to see what happened, or switch to 'any date' to try "
           "any day in the history. For a normal day the system will correctly say "
           "there is nothing unusual to explain.")

mode = st.radio("Which day?", ["Flagged days", "Any date"], horizontal=True,
                label_visibility="collapsed")

selected = None
if mode == "Flagged days":
    decisions = cached_decisions(f"{len(list(RESULTS_DIR.glob('pipeline_output_coffee_*.json')))}")
    all_dates = flagged.sort_values("date", ascending=False)["date"].dt.strftime("%Y-%m-%d").tolist()

    only_explained = st.checkbox(
        "Only show days that already have an explanation", value=True)

    if only_explained:
        opts = [d for d in all_dates if decisions.get(d) == "explained"]
        if not opts:
            st.info("No days with a saved explanation yet. Untick the box to see all flagged days.")
            opts = all_dates
    else:
        opts = all_dates

    def _label(d: str) -> str:
        status = decisions.get(d)
        tag = {"explained": "  ·  explained", "refused": "  ·  no explanation (weak evidence)",
               "error": "  ·  service error", "other": ""}.get(status, "  ·  not run yet")
        return f"{d}{tag}"

    selected = st.selectbox("Flagged day", opts, format_func=_label)
else:
    picked = st.date_input("Any date", value=latest_dt,
                           min_value=prices["date"].min().date(),
                           max_value=prices["date"].max().date())
    selected = picked.strftime("%Y-%m-%d")

if selected:
    match = anomalies[anomalies["date"] == pd.to_datetime(selected)]
    if match.empty:
        st.info(f"{selected} is not a trading day (weekend or holiday).")
    else:
        row = match.iloc[0]
        is_flagged = bool(row["anomaly_flag"])
        st.write(f"**{pd.to_datetime(selected):%d %b %Y}:** price {row['price']:.2f} cents/lb, "
                 + ("flagged as an unusual move." if is_flagged else "a normal day."))

        cached = pipeline_output(selected)
        if cached is not None:
            render_explanation(cached)
            render_evidence_posture(cached)
        elif READ_ONLY:
            st.info("No saved explanation for this day. Live explanation is off on the public site.")
        else:
            if st.button(f"Explain {pd.to_datetime(selected):%d %b %Y}", key="explain_any"):
                with st.spinner("Retrieving news and generating an explanation (1-2 min)..."):
                    ok, log = run_pipeline(selected)
                if ok:
                    st.cache_data.clear()
                    st.rerun()
                else:
                    with st.expander("Log"):
                        st.code(log)


# ===================================================================
# SYSTEM EVALUATION
# ===================================================================
# Hidden for now. The stored evaluation_report.json was produced by runs under
# an earlier pipeline version and scored against the original answer key,
# whose four should-refuse labels were later found to be wrong
# (data/labeling/LABELLING_RULE.md). Showing those figures next to the
# current ones invites a comparison between two different tests. Rebuild the
# reports after the next run (python -m src.evaluation.metrics) and set this
# back to True.
SHOW_SYSTEM_EVALUATION = False

if SHOW_SYSTEM_EVALUATION:
  st.divider()
  st.subheader("System Evaluation")

  eval_report_path = RESULTS_DIR / "evaluation_report.json"
  ablation_path = RESULTS_DIR / "ablation_study.json"

  _has_eval = eval_report_path.exists()
  _has_ablation = False
  if ablation_path.exists():
      # Only show an ablation produced by the like-for-like version of the
      # study (every configuration gating the same scored documents). Results
      # from the earlier version compared different inputs and are not shown.
      try:
          with open(ablation_path, encoding="utf-8") as _af:
              _has_ablation = bool(json.load(_af).get("like_for_like"))
      except (OSError, ValueError):
          _has_ablation = False

  if _has_eval or _has_ablation:
      eval_tab_names = []
      if _has_eval:
          eval_tab_names.append("Pipeline Metrics")
      if _has_ablation:
          eval_tab_names.append("Ablation Study")

      eval_tabs = st.tabs(eval_tab_names)
      tab_idx = 0

      if _has_eval:
          with eval_tabs[tab_idx]:
              with open(eval_report_path, encoding="utf-8") as _ef:
                  metrics = json.load(_ef)

              ret = metrics.get("retrieval", {})
              cit = metrics.get("citation_accuracy", {})
              dir_ = metrics.get("direction_alignment", {})
              div = metrics.get("source_diversity", {})
              dist = metrics.get("outcome_distribution", {})

              st.caption(f"Evaluated {metrics.get('dates_evaluated', 0)} anomaly dates, "
                         f"{metrics.get('dates_explained', 0)} with LLM explanations. "
                         f"Small sample: see results/evaluation_report.md for the intervals.")

              def _ci(pair):
                  return (f"95% CI {pair[0]}-{pair[1]}%"
                          if pair and pair[0] is not None else None)

              acc = (metrics.get("decision_accuracy") or {}).get("as_run") or {}
              overall = acc.get("overall") or {}

              # Key metrics row
              m1, m2, m3, m4 = st.columns(4)
              m1.metric("Explained", f"{ret.get('llm_explain_rate', 0)}%",
                        help=_ci(ret.get("llm_explain_ci95")))
              if overall.get("n"):
                  m2.metric("Correct decision", f"{overall['pct']}%",
                            help=f"{overall['k']} of {overall['n']} labelled dates: explained "
                                 f"when it should, refused when it should. If the answer key "
                                 f"has no should-refuse date, explaining everything would "
                                 f"score 100% here. {_ci(overall.get('ci95'))}")
              else:
                  m2.metric("Gate pass rate", f"{ret.get('gate_explain_rate', 0)}%")
              m3.metric("Citations resolve", f"{cit.get('citations_resolve_rate', 'N/A')}%"
                         if cit.get("citations_resolve_rate") is not None else "N/A",
                         help="The cited document was one the model was shown. This rules out "
                              "invented sources; it does not show the document supports the claim.")
              m4.metric("Evidence direction", f"{dir_.get('direction_consistent_rate', 0)}%",
                        help="Explained runs whose accepted evidence leaned the same way as the "
                             "move (word-list heuristic). A property of the evidence, not a check "
                             "that the explanation is right.")

              refuse = acc.get("refuse_labelled") or {}
              if refuse.get("n"):
                  st.caption(f"Refused when it should have: {refuse['k']} of {refuse['n']} "
                             f"dates labelled as having no clean cause.")

              # Detail expanders
              with st.expander("Retrieval Performance"):
                  st.markdown(
                      f"- **{ret.get('avg_documents_retrieved', 0):.1f}** articles retrieved per date on average\n"
                      f"- **{ret.get('zero_retrieval_rate', 0)}%** of dates had zero articles retrieved\n"
                      f"- Gate passes **{ret.get('gate_explain_rate', 0)}%** of anomalies, "
                      f"LLM ultimately explains **{ret.get('llm_explain_rate', 0)}%** — "
                      f"the gap reflects the LLM's own evidence quality judgment"
                  )

              with st.expander("Source Diversity"):
                  st.markdown(
                      f"- **{div.get('avg_unique_domains', 0):.1f}** unique source domains per explanation\n"
                      f"- **{div.get('avg_accepted_documents', 0):.1f}** accepted documents per explanation\n"
                      f"- **{div.get('single_source_rate', 0)}%** of explanations rely on a single source"
                  )

              if dist:
                  with st.expander("Outcome Distribution"):
                      dist_df = pd.DataFrame(
                          [{"Outcome": k.replace("_", " ").title(), "Count": v}
                           for k, v in dist.items()]
                      )
                      st.dataframe(dist_df, use_container_width=True, hide_index=True)

          tab_idx += 1

      if _has_ablation:
          with eval_tabs[tab_idx]:
              with open(ablation_path, encoding="utf-8") as _af:
                  ablation = json.load(_af)

              summary = ablation.get("summary", {})
              n_dates = ablation.get("n_dates", 0)

              st.caption(f"Tested {len(summary)} configurations across {n_dates} anomaly dates"
                         + ("" if ablation.get("semantic_scores_available", True)
                            else " - semantic scores were unavailable for this run"))

              # Build comparison table
              rows = []
              for config, stats in summary.items():
                  rows.append({
                      "Configuration": config.replace("_", " ").title(),
                      "EXPLAIN Rate": f"{stats['explain_rate']}%",
                      "Avg Retrieved": stats["avg_retrieved"],
                      "Avg Accepted": stats["avg_accepted"],
                      "Avg Best Score": f"{stats['avg_best_score']:.3f}",
                  })

              st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

              # Visual comparison
              configs = list(summary.keys())
              explain_rates = [summary[c]["explain_rate"] for c in configs]

              fig = go.Figure()
              fig.add_trace(go.Bar(
                  x=[c.replace("_", " ") for c in configs],
                  y=explain_rates,
                  marker_color=["#27ae60" if r > 50 else "#e67e22" if r > 0 else "#e74c3c"
                                 for r in explain_rates],
                  text=[f"{r}%" for r in explain_rates],
                  textposition="outside",
              ))
              fig.update_layout(
                  title="Gate EXPLAIN Rate by Configuration",
                  yaxis_title="EXPLAIN Rate (%)",
                  yaxis_range=[0, 110],
                  height=350,
                  margin=dict(t=40, b=40),
                  showlegend=False,
              )
              st.plotly_chart(fig, use_container_width=True)

              # What the table shows - generated by the study from its own numbers
              findings = ablation.get("findings") or []
              if findings:
                  with st.expander("What the table shows", expanded=True):
                      for line in findings:
                          st.markdown(f"- {line}")
              st.caption("Every configuration gates the same scored documents for each date. "
                         "No model is called, so this measures the gate, not the final decision.")
  else:
      st.info("Run the evaluation framework to see system metrics here: "
              "`python -m src.evaluation.metrics`")
