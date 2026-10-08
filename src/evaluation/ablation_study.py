"""
Ablation study: what does each retrieval/gating component change?

Every configuration is applied to the SAME scored documents for a date. The
documents are rebuilt once per date from the on-disk caches (GDELT responses
and extracted article text), scored with both channels (keyword and
semantic), and then gated under each configuration. Only the thing being
ablated differs between rows.

That like-for-like property is the whole point. An earlier version of this
study took the full-pipeline row from stored outputs (semantic scores, full
text) and every other row from title-only keyword scores, so the gap it
reported between "full pipeline" and "keyword only" compared two different
scorings of two different inputs, and its gate-threshold rows never saw a
semantic score at all. Its conclusions did not follow from its table.

Gate configurations (same documents, different acceptance rule)
  full_pipeline   keyword >= 0.4 OR semantic >= 0.85   (production)
  semantic_only   semantic >= 0.85
  keyword_only    keyword >= 0.4
  no_gate         accept everything above the noise floor
  semantic_0.80   keyword >= 0.4 OR semantic >= 0.80
  semantic_0.90   keyword >= 0.4 OR semantic >= 0.90

Preparation configurations (production gate, one clean-up step removed)
  no_window_filter      keep articles dated outside the retrieval window
  no_duplicate_collapse keep syndicated copies as separate documents

What is measured
  - gate pass rate (does the gate let the date through to the model)
  - accepted documents per date, and how many are distinct stories
  - accepted documents dated after the anomaly date
  - on hand-labelled dates: how often the gate passes an EXPLAIN-labelled
    date and blocks a REFUSE-labelled one

No model is called. A date that passes the gate would go on to the LLM in
production, so this measures the gate, not the final decision.

Requires the embedding model (src/rag/vector_store.py). Without it every
semantic score is missing and the semantic rows are meaningless, so the
script stops unless --allow-keyword-only is passed.

Usage:
    python -m src.evaluation.ablation_study
    python -m src.evaluation.ablation_study --dates 10
"""

import argparse
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import load_labels, rate, fmt_rate   # noqa: E402
from src.rag.evidence import normalize_title                      # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
GDELT_CACHE_DIR = REPO_ROOT / "data" / "gdelt_cache"

# name -> gate_documents keyword arguments
GATE_CONFIGS = {
    "full_pipeline": {},
    "semantic_only": {"scoring_mode": "semantic_only"},
    "keyword_only": {"scoring_mode": "keyword_only"},
    "no_gate": {"keyword_threshold": 0.0, "semantic_threshold": 0.0},
    "semantic_0.80": {"semantic_threshold": 0.80},
    "semantic_0.90": {"semantic_threshold": 0.90},
}

# name -> retrieve_evidence keyword arguments (gated with the production gate)
PREP_CONFIGS = {
    "no_window_filter": {"enforce_window": False},
    "no_duplicate_collapse": {"collapse_duplicates": False},
}

CONFIGS = list(GATE_CONFIGS) + list(PREP_CONFIGS)

CONFIG_NOTES = {
    "full_pipeline": "Production: keyword >= 0.4 OR semantic >= 0.85.",
    "semantic_only": "Keyword channel ignored.",
    "keyword_only": "Semantic channel ignored.",
    "no_gate": "Everything above the noise floor is accepted.",
    "semantic_0.80": "Semantic threshold lowered to 0.80.",
    "semantic_0.90": "Semantic threshold raised to 0.90.",
    "no_window_filter": "Production gate; out-of-window articles not removed.",
    "no_duplicate_collapse": "Production gate; syndicated copies not collapsed.",
}


def cached_dates() -> list:
    """Coffee dates that have a stored pipeline output and at least one GDELT
    cache entry for their window, oldest first. Only these can be rebuilt
    offline and identically on every run."""
    from datetime import datetime, timedelta
    from src.config.commodities import get_commodity
    from src.rag.gdelt_client import build_query_params, cache_get, DEFAULT_SORT
    from src.pipeline import RETRIEVAL_WINDOW_DAYS_BEFORE, MAX_ARTICLES_PER_QUERY

    cfg = get_commodity("coffee")
    dates = []
    for path in sorted(RESULTS_DIR.glob("pipeline_output_coffee_*.json")):
        date = path.stem.replace("pipeline_output_coffee_", "")
        end_dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
        start_dt = end_dt - timedelta(days=RETRIEVAL_WINDOW_DAYS_BEFORE + 1)
        start, end = start_dt.strftime("%Y%m%d%H%M%S"), end_dt.strftime("%Y%m%d%H%M%S")
        if any(cache_get(GDELT_CACHE_DIR,
                         build_query_params(q, MAX_ARTICLES_PER_QUERY, DEFAULT_SORT, start, end))
               is not None for q in cfg.gdelt_queries):
            dates.append(date)
    return dates


def load_scored_documents(date: str, **retrieve_kwargs) -> list:
    """Rebuild and score the documents for a date from the caches only: no
    GDELT call, no article fetch, no RSS. Both score channels are computed by
    the same code the live pipeline uses."""
    from src.config.commodities import get_commodity
    from src.pipeline import retrieve_evidence, search_arguments

    cfg = get_commodity("coffee")
    with contextlib.redirect_stdout(io.StringIO()):
        return retrieve_evidence(
            date, cfg.gdelt_queries, cfg.semantic_reference_query,
            cache_dir=GDELT_CACHE_DIR,
            commodity_key=None,      # no live news aggregator: not reproducible
            fallback_dir=None,
            gdelt_live=False,
            fetch_text="cache",
            **search_arguments(cfg),
            **retrieve_kwargs,
        )


def gate_once(date: str, config: str, documents: list, gate_kwargs: dict) -> dict:
    from src.rag.relevance_gate import gate_documents

    result = gate_documents(copy.deepcopy(documents), **gate_kwargs)
    accepted = result.accepted_documents
    stories = {normalize_title(d.get("title", "")) or d.get("url", "") for d in accepted}
    return {
        "date": date,
        "config": config,
        "n_retrieved": len(documents),
        "n_accepted": len(accepted),
        "n_distinct_stories": len(stories),
        "n_accepted_after_date": sum(1 for d in accepted
                                     if (d.get("publication_date") or "") > date),
        "gate_decision": result.decision,
        "best_score": round(result.best_score, 4),
        "gate_reason": result.reason,
        "error": None,
    }


def summarize(results: list, labels: dict) -> dict:
    summary = {}
    for config in CONFIGS:
        rows = [r for r in results if r["config"] == config]
        n = len(rows)
        if not n:
            continue
        passed = [r for r in rows if r["gate_decision"] == "EXPLAIN"]
        accepted = sum(r["n_accepted"] for r in rows)

        exp = [r for r in rows
               if (labels.get(r["date"], {}).get("expected_outcome") or "").upper() == "EXPLAIN"]
        ref = [r for r in rows
               if (labels.get(r["date"], {}).get("expected_outcome") or "").upper() == "REFUSE"]

        summary[config] = {
            "n_dates": n,
            "n_explain": len(passed),
            "explain_rate": round(100 * len(passed) / n, 1),
            "explain": rate(len(passed), n),
            "avg_retrieved": round(sum(r["n_retrieved"] for r in rows) / n, 1),
            "avg_accepted": round(accepted / n, 1),
            "avg_distinct_stories": round(sum(r["n_distinct_stories"] for r in rows) / n, 1),
            "avg_best_score": round(sum(r["best_score"] for r in rows) / n, 3),
            "accepted_after_date": rate(sum(r["n_accepted_after_date"] for r in rows), accepted),
            "passes_explain_labelled": rate(
                sum(1 for r in exp if r["gate_decision"] == "EXPLAIN"), len(exp)),
            "blocks_refuse_labelled": rate(
                sum(1 for r in ref if r["gate_decision"] != "EXPLAIN"), len(ref)),
            "n_errors": 0,
        }
    return summary


def _decisions(results: list, config: str) -> dict:
    return {r["date"]: r["gate_decision"] for r in results if r["config"] == config}


def findings(summary: dict, results: list) -> list:
    """Statements about what the table shows. Each is conditional on the
    numbers; none is written in advance."""
    out = []
    base = _decisions(results, "full_pipeline")
    full = summary.get("full_pipeline")
    if not full:
        return out
    n = full["n_dates"]

    def differing(config):
        other = _decisions(results, config)
        return sorted(d for d in base if other.get(d) != base[d])

    for config, label in (("semantic_only", "Removing the keyword channel"),
                          ("keyword_only", "Removing the semantic channel"),
                          ("no_gate", "Removing the gate entirely")):
        s = summary.get(config)
        if not s:
            continue
        diff = differing(config)
        if not diff:
            out.append(f"{label} changes the gate decision on none of the {n} dates "
                       f"(accepted documents per date: {full['avg_accepted']} -> {s['avg_accepted']}).")
        else:
            out.append(f"{label} changes the gate decision on {len(diff)} of {n} dates "
                       f"({', '.join(diff[:8])}{', ...' if len(diff) > 8 else ''}); pass rate "
                       f"{full['explain_rate']}% -> {s['explain_rate']}%, accepted documents per "
                       f"date {full['avg_accepted']} -> {s['avg_accepted']}.")

    for config in ("semantic_0.80", "semantic_0.90"):
        s = summary.get(config)
        if s:
            diff = differing(config)
            out.append(f"Semantic threshold {config.split('_')[1]}: pass rate "
                       f"{s['explain_rate']}% ({len(diff)} date(s) change), "
                       f"{s['avg_accepted']} accepted per date vs {full['avg_accepted']} at 0.85.")

    nw = summary.get("no_window_filter")
    if nw:
        out.append(f"Without the local window filter, {fmt_rate(nw['accepted_after_date'])} of "
                   f"accepted documents are dated after the anomaly; with it, "
                   f"{fmt_rate(full['accepted_after_date'])}.")
    nd = summary.get("no_duplicate_collapse")
    if nd:
        out.append(f"Without duplicate collapsing the gate accepts {nd['avg_accepted']} documents "
                   f"per date, of which {nd['avg_distinct_stories']} are distinct stories; with it, "
                   f"{full['avg_accepted']} accepted and {full['avg_distinct_stories']} distinct.")

    if full["explain_rate"] >= 90:
        out.append(f"The production gate passes {full['explain_rate']}% of these dates. At the "
                   f"level of whole dates it filters almost nothing: its effect is on which "
                   f"documents reach the model, not on whether the model is called.")
    return out


def generate_report(summary: dict, results: list, dates: list, semantic_available: bool) -> str:
    lines = [
        "# Ablation Study",
        "",
        f"**Dates**: {len(dates)} coffee anomaly dates with cached retrieval, "
        f"{min(dates)} to {max(dates)}.",
        "",
        "Every row gates the same scored documents for each date; only the named "
        "component differs. Documents are rebuilt from the on-disk caches and scored "
        "with both channels. No model is called, so this measures the gate, not the "
        "final explain/refuse decision.",
        "",
    ]
    if not semantic_available:
        lines += ["> **Semantic scores were not available for this run.** Every row that "
                  "depends on the semantic channel is not meaningful. Re-run on a machine "
                  "where the embedding model loads.", ""]

    lines += [
        "| Configuration | Gate pass rate | Accepted / date | Distinct stories / date | "
        "Accepted dated after anomaly |",
        "|---|---|---|---|---|",
    ]
    for config in CONFIGS:
        s = summary.get(config)
        if not s:
            continue
        lines.append(f"| {config} | {fmt_rate(s['explain'])} | {s['avg_accepted']} | "
                     f"{s['avg_distinct_stories']} | {fmt_rate(s['accepted_after_date'])} |")

    full = summary.get("full_pipeline", {})
    if full.get("passes_explain_labelled", {}).get("n") or full.get("blocks_refuse_labelled", {}).get("n"):
        lines += ["", "## Against the hand labels", "",
                  "A gate that helped would pass the EXPLAIN-labelled dates and block the "
                  "REFUSE-labelled ones.", "",
                  "| Configuration | Passes EXPLAIN-labelled | Blocks REFUSE-labelled |",
                  "|---|---|---|"]
        for config in GATE_CONFIGS:
            s = summary.get(config)
            if s:
                lines.append(f"| {config} | {fmt_rate(s['passes_explain_labelled'])} | "
                             f"{fmt_rate(s['blocks_refuse_labelled'])} |")

    lines += ["", "## What each configuration is", ""]
    lines += [f"- **{c}**: {CONFIG_NOTES[c]}" for c in CONFIGS if c in summary]

    lines += ["", "## What the table shows", ""]
    notes = findings(summary, results)
    lines += [f"{i}. {text}" for i, text in enumerate(notes, 1)] or ["No findings computed."]

    lines += [
        "", "## Limits", "",
        "- Body text comes from the on-disk text cache, so a document whose text was "
        "never fetched successfully is scored on its title alone - the same as in a "
        "live run where that fetch failed.",
        "- The live news aggregator (Google News, industry feeds) is not used here "
        "because its results are not reproducible; these rows cover GDELT-retrieved "
        "documents only.",
        f"- {len(dates)} dates. Intervals are 95% Wilson intervals and are wide.",
        "",
    ]
    return "\n".join(lines)


def run_ablation(n_dates: int = None, allow_keyword_only: bool = False,
                 verbose: bool = True) -> dict:
    """Run the ablation across dates and configurations."""
    dates = cached_dates()
    if n_dates:
        dates = dates[-n_dates:]
    if not dates:
        raise SystemExit("No dates with both a stored pipeline output and cached retrieval.")
    labels = load_labels()

    if verbose:
        print(f"=== Ablation study: {len(CONFIGS)} configurations x {len(dates)} dates ===\n")

    results = []
    semantic_available = True
    for i, date in enumerate(dates, 1):
        documents = load_scored_documents(date)
        has_semantic = any(d.get("semantic_score") is not None for d in documents)
        if documents and not has_semantic:
            semantic_available = False
            if not allow_keyword_only:
                raise SystemExit(
                    "Semantic scores are missing (the embedding model did not load), so "
                    "the semantic rows of this study would be meaningless. Fix the model "
                    "load (pip install onnxruntime tokenizers huggingface_hub, with "
                    "network access on first use) or pass --allow-keyword-only to run "
                    "the keyword rows anyway.")
        for config, kwargs in GATE_CONFIGS.items():
            results.append(gate_once(date, config, documents, kwargs))
        for config, kwargs in PREP_CONFIGS.items():
            results.append(gate_once(date, config, load_scored_documents(date, **kwargs), {}))
        if verbose:
            row = next(r for r in results if r["date"] == date and r["config"] == "full_pipeline")
            print(f"  [{i}/{len(dates)}] {date}: {row['n_retrieved']} documents, "
                  f"{row['n_accepted']} accepted, gate {row['gate_decision']}")

    summary = summarize(results, labels)
    output = {
        "n_dates": len(dates),
        "dates_tested": dates,
        "configurations": CONFIGS,
        "semantic_scores_available": semantic_available,
        "like_for_like": True,
        "summary": summary,
        "findings": findings(summary, results),
        "results": results,
    }

    output_path = RESULTS_DIR / "ablation_study.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    report = generate_report(summary, results, dates, semantic_available)
    report_path = RESULTS_DIR / "ablation_study_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    if verbose:
        print(f"\n{report}")
        print(f"\nSaved to {output_path}")
        print(f"Report at {report_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Pipeline component ablation study")
    parser.add_argument("--dates", type=int, default=None,
                        help="Use only the N most recent cached dates (default: all)")
    parser.add_argument("--allow-keyword-only", action="store_true",
                        help="Run even if the embedding model is unavailable "
                             "(semantic rows will not be meaningful)")
    args = parser.parse_args()
    run_ablation(n_dates=args.dates, allow_keyword_only=args.allow_keyword_only)


if __name__ == "__main__":
    main()
