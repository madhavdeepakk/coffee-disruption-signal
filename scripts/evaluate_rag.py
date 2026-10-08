"""
Quantitative summary of the RAG explanation layer across all pipeline runs.

Reads every results/pipeline_output_*.json (both the older
pipeline_output_<date>.json and the newer pipeline_output_<commodity>_<date>.json
naming) and tabulates what actually happened, so the RAG track's behaviour is
reported as numbers rather than anecdotes:
  - decision mix: EXPLAINED vs INSUFFICIENT_EVIDENCE vs PARSE_ERROR, and WHY
    a refusal happened (no documents retrieved, gate rejected, or the LLM
    itself judged the retrieved evidence insufficient - three very different
    things this project deliberately keeps distinct).
  - retrieval: documents retrieved and accepted per run, best gate score.
  - direction: how many accepted docs were direction-consistent vs
    inconsistent (present only for runs produced after the direction feature).
  - faithfulness: whether every citation resolved to a retrieved document,
    and any weak-support flags (present only for runs produced after the
    faithfulness feature).

Runs entirely offline on the saved JSON outputs - no network.

Usage:
    python -m scripts.evaluate_rag
    python -m scripts.evaluate_rag --results-dir results
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MD_PATH = REPO_ROOT / "results" / "rag_evaluation.md"


def _refusal_reason(rec: dict) -> str:
    er = rec.get("explanation_result") or {}
    gate = rec.get("gate_result") or {}
    if er.get("decision") == "EXPLAINED":
        return "-"
    reason = er.get("reason") or ""
    if "model_judged" in reason:
        return "LLM judged evidence insufficient"
    if gate.get("reason") == "no_documents_retrieved":
        return "no documents retrieved"
    if gate.get("decision") != "EXPLAIN":
        return f"gate: {gate.get('reason', 'rejected')}"
    return reason or "unknown"


def load_runs(results_dir: Path) -> list:
    runs = []
    for path in sorted(results_dir.glob("pipeline_output_*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        anomaly = rec.get("anomaly") or {}
        gate = rec.get("gate_result") or {}
        er = rec.get("explanation_result") or {}
        fr = rec.get("faithfulness_report")
        ds = rec.get("direction_summary") or {}
        runs.append({
            "file": path.name,
            "date": anomaly.get("date", path.stem.split("_")[-1]),
            "decision": er.get("decision", "?"),
            "refusal_reason": _refusal_reason(rec),
            "docs_retrieved": rec.get("documents_retrieved", gate.get("total_documents_considered")),
            "accepted": gate.get("accepted_document_count"),
            "best_score": gate.get("best_score"),
            "n_citations": len(er.get("citations") or []),
            "direction": ds,
            "faithfulness": fr,
        })
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No pipeline_output_*.json files found in {args.results_dir}.")

    n = len(runs)
    explained = sum(1 for r in runs if r["decision"] == "EXPLAINED")
    insufficient = sum(1 for r in runs if r["decision"] == "INSUFFICIENT_EVIDENCE")
    parse_err = sum(1 for r in runs if r["decision"] == "PARSE_ERROR")

    # Faithfulness aggregate (only over runs that produced citations AND have a report)
    with_fr = [r for r in runs if r["faithfulness"]]
    fr_all_resolve = sum(1 for r in with_fr if r["faithfulness"].get("citations_resolve"))
    fr_weak = sum((r["faithfulness"].get("n_weak_support") or 0) for r in with_fr)
    fr_missing = sum((r["faithfulness"].get("n_missing_document") or 0) for r in with_fr)

    md = [
        "# RAG Explanation Layer — Quantitative Evaluation", "",
        f"Across **{n}** pipeline runs found in `{args.results_dir.name}/`:", "",
        f"- **Explained: {explained}** · **Insufficient-evidence (refused): {insufficient}** · "
        f"Parse-error: {parse_err}",
        f"- Refusals broken down by cause below — the point of the RAG safety design is that a "
        f"refusal for \"no documents\" (a retrieval problem) and a refusal because the LLM "
        f"judged real evidence didn't fit (a reasoning safeguard) are different and stay labeled.",
    ]
    if with_fr:
        md.append(f"- **Citation faithfulness** (over {len(with_fr)} run(s) with citations + a "
                  f"faithfulness report): {fr_all_resolve}/{len(with_fr)} had every citation "
                  f"resolve to a retrieved document; {fr_missing} fabricated-id citation(s), "
                  f"{fr_weak} weak-support flag(s) in total.")
    else:
        md.append("- Citation faithfulness: no runs carry a faithfulness report yet "
                  "(re-run the pipeline to populate it).")
    md += ["", "| date | decision | refusal cause | retrieved | accepted | best | cites | dir(c/n/i) | cites resolve |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        ds = r["direction"]
        dstr = (f"{ds.get('consistent',0)}/{ds.get('neutral',0)}/{ds.get('inconsistent',0)}"
                if ds else "-")
        fr = r["faithfulness"]
        fstr = ("yes" if fr and fr.get("citations_resolve") else ("NO" if fr else "-"))
        best = f"{r['best_score']:.3f}" if isinstance(r["best_score"], (int, float)) else "-"
        md.append(f"| {r['date']} | {r['decision']} | {r['refusal_reason']} | "
                  f"{r['docs_retrieved']} | {r['accepted']} | {best} | {r['n_citations']} | "
                  f"{dstr} | {fstr} |")
    md.append("")

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text("\n".join(md), encoding="utf-8")

    print(f"{n} runs: {explained} explained, {insufficient} refused, {parse_err} parse-error")
    for r in runs:
        print(f"  {r['date']}: {r['decision']:22s} ({r['refusal_reason']}) "
              f"docs={r['docs_retrieved']} accepted={r['accepted']}")
    print(f"\nSaved report to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
