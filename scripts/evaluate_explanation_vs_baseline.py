"""
Does the RAG layer (relevance gate + LLM synthesis + honest refusal) actually
beat the obvious baseline of "just show the retrieved headlines"?

This is the missing product-level evaluation. Every other eval measures a
component; this one asks whether the expensive parts (the gate and the LLM)
earn their place against a trivial alternative.

Two baselines are compared against the full system, scored on the hand-labelled
set (data/labeling/labels_by_written_rule.csv), using only CACHED pipeline
outputs (no live API calls):

  1. DECISION VALUE  (always computable)
     - System:   the gated decision (EXPLAIN vs REFUSE) the pipeline actually made.
     - Baseline: "never refuse" - explain whenever at least one article was
                 retrieved (i.e. a headline dump with no gate and no refusal).
     The baseline gets every REFUSE-labelled date wrong by construction. The
     delta on the refuse-cases is exactly the value the gate + refusal add.

  2. CAUSE COVERAGE  (best-effort; needs article titles)
     - For EXPLAIN-labelled dates the system explained, does the explanation
       actually surface the documented cause (term overlap with the label's
       `known_cause`), and does it do so better than the raw top-3 headlines?
     This is a lexical proxy for "conveys the true cause", not human judgement,
     and is reported as such. Titles are recovered from the run's `sources`
     field, or reconstructed from the GDELT disk cache; dates with neither are
     skipped for this metric only.

Usage:
    python -m scripts.evaluate_explanation_vs_baseline
    python -m scripts.evaluate_explanation_vs_baseline --min-confidence high
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = REPO_ROOT / "data" / "labeling" / "labels_by_written_rule.csv"
RESULTS_DIR = REPO_ROOT / "results"
GDELT_CACHE_DIR = REPO_ROOT / "data" / "gdelt_cache"
REPORT_PATH = RESULTS_DIR / "explanation_vs_baseline.md"

_CONF_RANK = {"low": 0, "medium": 1, "high": 2}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "as", "at", "is", "was", "were", "are", "be", "been", "that", "this",
    "it", "its", "up", "down", "over", "after", "before", "into", "amid", "due",
    "s", "t", "not", "no", "than", "then", "but", "out", "off", "per", "plus",
    "cause", "causes", "caused", "causing", "drove", "driven", "drives", "spike",
    "spiked", "surge", "surged", "rally", "rose", "fell", "move", "day", "price",
    "prices", "coffee", "arabica", "market", "markets",
}


def load_labels(min_confidence: str) -> list:
    floor = _CONF_RANK.get(min_confidence, 0)
    rows = []
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # EXCLUDE rows are not real moves (the price series switched contract).
            if (row.get("expected_outcome") or "").strip().upper() not in ("EXPLAIN", "REFUSE"):
                continue
            if _CONF_RANK.get(row.get("label_confidence", "low"), 0) >= floor:
                rows.append(row)
    return rows


def load_run(date: str) -> dict | None:
    p = RESULTS_DIR / f"pipeline_output_coffee_{date}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def system_decision(run: dict) -> str | None:
    """EXPLAINED / REFUSED, or None for a fault (model unavailable etc.)."""
    dec = (run.get("explanation_result") or {}).get("decision")
    if dec in ("API_ERROR", "PARSE_ERROR"):
        return None
    return "EXPLAINED" if dec == "EXPLAINED" else "REFUSED"


def baseline_decision(run: dict) -> str:
    """'Never refuse' baseline: explain whenever anything was retrieved."""
    return "EXPLAINED" if int(run.get("documents_retrieved", 0) or 0) > 0 else "REFUSED"


def _content_terms(text: str) -> set:
    toks = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return {t for t in toks if t not in _STOPWORDS}


def _recover_titles(date: str, run: dict) -> list:
    """Best-effort list of retrieved article titles for a date.

    Prefers the run's own `sources` (accepted docs, newer runs). Falls back to
    replaying the coffee GDELT queries against the on-disk cache. Returns [] if
    nothing is recoverable (older run + cold cache)."""
    titles = [s.get("title", "") for s in (run.get("sources") or []) if s.get("title")]
    if titles:
        return titles

    # Reconstruct from the GDELT cache by rebuilding the query params.
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from src.config.commodities import get_commodity
        from src.rag.gdelt_client import build_query_params, DEFAULT_SORT, cache_get
        from datetime import datetime, timedelta
    except Exception:
        return []

    cfg = get_commodity("coffee")
    end_dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
    start_dt = end_dt - timedelta(days=11)
    sdt, edt = start_dt.strftime("%Y%m%d%H%M%S"), end_dt.strftime("%Y%m%d%H%M%S")
    seen, out = set(), []
    for q in cfg.gdelt_queries:
        params = build_query_params(q, 20, DEFAULT_SORT, sdt, edt)
        raw = cache_get(GDELT_CACHE_DIR, params, verbose=False)
        if not raw:
            continue
        for art in raw.get("articles", []) or []:
            u = art.get("url", "")
            if u and u not in seen:
                seen.add(u)
                out.append(art.get("title", "") or "")
    return out


def _coverage(text: str, cause_terms: set) -> float:
    if not cause_terms:
        return float("nan")
    present = cause_terms & _content_terms(text)
    return len(present) / len(cause_terms)


def evaluate(labels: list) -> dict:
    decision_rows, coverage_rows = [], []

    for lab in labels:
        date = lab["date"].strip()
        expected = lab["expected_outcome"].strip().upper()  # EXPLAIN / REFUSE
        run = load_run(date)
        if run is None:
            continue
        sysd = system_decision(run)
        if sysd is None:
            continue  # fault, exclude

        base = baseline_decision(run)
        exp_dec = "EXPLAINED" if expected == "EXPLAIN" else "REFUSED"
        decision_rows.append({
            "date": date, "expected": exp_dec,
            "system": sysd, "system_ok": sysd == exp_dec,
            "baseline": base, "baseline_ok": base == exp_dec,
        })

        # Cause coverage only on EXPLAIN dates the system explained.
        if expected == "EXPLAIN" and sysd == "EXPLAINED":
            cause_terms = _content_terms(lab.get("known_cause", ""))
            expl = (run.get("explanation_result") or {}).get("explanation", "") or ""
            titles = _recover_titles(date, run)
            headline_blob = " ".join(titles[:3])
            cov_rag = _coverage(expl, cause_terms)
            cov_base = _coverage(headline_blob, cause_terms) if titles else float("nan")
            coverage_rows.append({
                "date": date, "n_cause_terms": len(cause_terms),
                "cov_rag": cov_rag, "cov_base": cov_base,
                "have_titles": bool(titles),
            })

    return {"decision": decision_rows, "coverage": coverage_rows}


def _acc(rows, key):
    n = len(rows)
    return (sum(1 for r in rows if r[key]) / n) if n else None


def _pct(x):
    return "n/a" if x is None else f"{x*100:.0f}%"


def write_report(res: dict, mode: str) -> str:
    d = res["decision"]
    refuse = [r for r in d if r["expected"] == "REFUSED"]
    explain = [r for r in d if r["expected"] == "EXPLAINED"]

    cov = [r for r in res["coverage"] if r["have_titles"]
           and r["cov_rag"] == r["cov_rag"] and r["cov_base"] == r["cov_base"]]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None

    L = []
    L.append("# Explanation value vs. a headline-dump baseline\n")
    L.append("Does the gate + LLM + refusal layer beat the trivial alternative of "
             "showing whatever was retrieved? Scored on cached outputs only "
             f"(mode: `{mode}`). No live API calls.\n")

    L.append("## 1. Decision value (gate + refusal)\n")
    L.append("The baseline never refuses: it 'explains' whenever any article was "
             "retrieved. The system uses the relevance gate and can refuse.\n")
    L.append(f"- Graded dates: **{len(d)}**  ({len(explain)} EXPLAIN, {len(refuse)} REFUSE)")
    L.append(f"- **System accuracy: {_pct(_acc(d,'system_ok'))}**  "
             f"(baseline: {_pct(_acc(d,'baseline_ok'))})")
    L.append(f"- On REFUSE-labelled dates: **system {_pct(_acc(refuse,'system_ok'))}** "
             f"vs baseline {_pct(_acc(refuse,'baseline_ok'))} "
             f"({len(refuse)} dates) &mdash; this gap is the value the gate + refusal add.")
    L.append(f"- On EXPLAIN-labelled dates: system {_pct(_acc(explain,'system_ok'))} "
             f"vs baseline {_pct(_acc(explain,'baseline_ok'))}.\n")

    L.append("| Date | Expected | System | ok | Baseline (never-refuse) | ok |")
    L.append("|------|----------|--------|----|------------------------|----|")
    for r in d:
        L.append(f"| {r['date']} | {r['expected']} | {r['system']} | "
                 f"{'y' if r['system_ok'] else 'n'} | {r['baseline']} | "
                 f"{'y' if r['baseline_ok'] else 'n'} |")
    L.append("")

    L.append("## 2. Cause coverage (lexical proxy)\n")
    L.append("For EXPLAIN dates the system explained: fraction of the label's "
             "documented-cause terms that appear in the RAG explanation vs. in the "
             "raw top-3 headlines. A proxy for 'conveys the true cause', not human "
             "judgement.\n")
    if cov:
        L.append(f"- Dates scored (titles available): **{len(cov)}**")
        L.append(f"- Mean cause-term coverage &mdash; **RAG explanation: "
                 f"{_pct(mean([r['cov_rag'] for r in cov]))}**, "
                 f"top-3 headlines: {_pct(mean([r['cov_base'] for r in cov]))}\n")
        L.append("| Date | Cause terms | RAG coverage | Headline coverage |")
        L.append("|------|-------------|--------------|-------------------|")
        for r in cov:
            L.append(f"| {r['date']} | {r['n_cause_terms']} | "
                     f"{_pct(r['cov_rag'])} | {_pct(r['cov_base'])} |")
    else:
        L.append("- No dates had recoverable article titles (older runs + cold "
                 "GDELT cache). Re-run the pipeline so outputs carry `sources`, then "
                 "this metric populates.")
    L.append("")

    L.append("## Reading this\n")
    L.append("- The decision-value result is the headline: a headline dump cannot "
             "refuse, so it fails every genuinely-unexplainable day. That is the "
             "product thesis, quantified.")
    L.append("- Cause coverage is a weak lexical proxy. Treat a higher RAG number as "
             "'synthesis did not lose the cause', not as proof of superiority.\n")

    report = "\n".join(L) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-confidence", default="low",
                    choices=["low", "medium", "high"])
    args = ap.parse_args()
    labels = load_labels(args.min_confidence)
    res = evaluate(labels)
    report = write_report(res, args.min_confidence)
    print(report)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
