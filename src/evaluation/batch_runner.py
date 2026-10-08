"""
Batch evaluation runner: run the pipeline on the top anomalies by z-score
and collect structured results for the evaluation framework.

This does NOT make live GDELT API calls (cache-only mode) and does NOT
call the LLM for fresh explanations. It uses whatever cached GDELT
responses and news aggregator results are available. For dates that
already have a pipeline_output JSON, it reads the existing result
instead of re-running.

The output is a single JSON file (results/evaluation_results.json)
consumed by the evaluation metrics framework and the dashboard.

Usage:
    python -m src.evaluation.batch_runner
    python -m src.evaluation.batch_runner --top 40    # more dates
    python -m src.evaluation.batch_runner --rerun      # ignore existing outputs
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def select_top_anomalies(n: int = 30) -> pd.DataFrame:
    """Pick the top-N anomalies by absolute z-score."""
    path = RESULTS_DIR / "anomaly_detections.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No anomaly detections at {path}. "
            "Run: python -m src.modeling.anomaly_detector --commodity coffee"
        )
    df = pd.read_csv(path)
    flagged = df[df["anomaly_flag"] == True].copy()  # noqa: E712
    flagged["abs_z"] = flagged["z_score"].abs()
    return flagged.nlargest(n, "abs_z")


def load_existing_output(date_str: str, commodity: str = "coffee") -> dict | None:
    """Load an existing pipeline output JSON if available."""
    path = RESULTS_DIR / f"pipeline_output_{commodity}_{date_str}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def run_pipeline_for_date(date_str: str, commodity: str = "coffee") -> dict:
    """Run the retrieval + gate pipeline for one date (no LLM call).

    Uses cache-only GDELT (no live API), skips text fetching to be fast.
    Returns the gate result and document counts.
    """
    from src.config.commodities import get_commodity
    from src.pipeline import retrieve_evidence, get_anomaly, search_arguments, GDELT_CACHE_DIR
    from src.rag.relevance_gate import gate_documents
    from src.rag import direction as direction_mod

    cfg = get_commodity(commodity)

    try:
        anomaly = get_anomaly(date_str, cfg.anomalies_file)
    except SystemExit:
        return {"date": date_str, "error": "not_in_anomaly_file"}

    try:
        documents, retrieval_meta = retrieve_evidence(
            date_str,
            cfg.gdelt_queries,
            cfg.semantic_reference_query,
            cache_dir=GDELT_CACHE_DIR,
            commodity_key=cfg.key,
            return_meta=True,
            **search_arguments(cfg),
            gdelt_live=False,
            fetch_text=False,
        )
    except Exception as exc:
        return {
            "date": date_str,
            "anomaly": anomaly,
            "error": str(exc),
            "documents_retrieved": 0,
        }

    gate_result = gate_documents(documents)

    # Direction analysis
    accepted = gate_result.accepted_documents
    direction_summary = {}
    if accepted:
        accepted = direction_mod.rerank(
            accepted, anomaly.get("direction"), strict=False
        )
        direction_summary = direction_mod.summarize(accepted)

    return {
        "date": date_str,
        "anomaly": anomaly,
        "retrieval_meta": retrieval_meta.to_dict(),
        "gate_result": gate_result.to_dict(),
        "documents_retrieved": len(documents),
        "direction_summary": direction_summary,
        "error": None,
    }


def collect_existing_outputs(commodity: str = "coffee",
                             verbose: bool = True) -> list[dict]:
    """Collect all existing pipeline output files."""
    import glob as glob_mod
    outputs = sorted(glob_mod.glob(
        str(RESULTS_DIR / f"pipeline_output_{commodity}_*.json")
    ))
    results = []
    for p in outputs:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        date_str = Path(p).stem.replace(f"pipeline_output_{commodity}_", "")
        dec = (data.get("explanation_result") or {}).get("decision", "?")
        if verbose:
            print(f"  {date_str}  (existing: {dec})")
        data["_source"] = "existing_output"
        data["_has_llm_explanation"] = True
        results.append(data)
    return results


def run_batch(top_n: int = 30, rerun: bool = False,
              existing_only: bool = False,
              verbose: bool = True) -> dict:
    """Run the batch evaluation across top anomalies.

    If existing_only=True, skip z-score selection and just collect all
    existing pipeline output files. This avoids diluting metrics with
    dates that have no GDELT cache.
    """
    if existing_only:
        if verbose:
            print("=== Batch Evaluation Runner (existing outputs only) ===\n")
        results = collect_existing_outputs(verbose=verbose)
        output = {
            "run_info": {
                "mode": "existing_only",
                "dates_evaluated": len(results),
                "dates_with_llm_explanation": len(results),
                "dates_gate_only": 0,
            },
            "results": results,
        }
        output_path = RESULTS_DIR / "evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        if verbose:
            print(f"\nDone. {len(results)} existing pipeline outputs collected.")
            print(f"Saved to {output_path}")
        return output

    top = select_top_anomalies(top_n)

    if verbose:
        print(f"=== Batch Evaluation Runner ===")
        print(f"Evaluating top {len(top)} anomalies by z-score\n")

    results = []
    for i, (_, row) in enumerate(top.iterrows()):
        date_str = row["date"]
        z = row["z_score"]

        # Check for existing full pipeline output first
        existing = load_existing_output(date_str)
        if existing and not rerun:
            if verbose:
                dec = (existing.get("explanation_result") or {}).get(
                    "decision", "?"
                )
                print(
                    f"  [{i+1}/{len(top)}] {date_str}  z={z:+.2f}  "
                    f"(existing: {dec})"
                )
            existing["_source"] = "existing_output"
            existing["_has_llm_explanation"] = True
            results.append(existing)
            continue

        # Run retrieval + gate only (no LLM)
        if verbose:
            print(
                f"  [{i+1}/{len(top)}] {date_str}  z={z:+.2f}  "
                f"(running retrieval+gate)...",
                end=" ",
                flush=True,
            )
        result = run_pipeline_for_date(date_str)
        result["_source"] = "batch_run"
        result["_has_llm_explanation"] = False
        results.append(result)

        if verbose:
            gate = result.get("gate_result", {})
            if isinstance(gate, dict):
                dec = gate.get("decision", "?")
                n_acc = gate.get("accepted_document_count", 0)
            else:
                dec = "ERROR"
                n_acc = 0
            print(f"gate={dec}  docs={result.get('documents_retrieved', 0)}  "
                  f"accepted={n_acc}")

    output = {
        "run_info": {
            "top_n": top_n,
            "dates_evaluated": len(results),
            "dates_with_llm_explanation": sum(
                1 for r in results if r.get("_has_llm_explanation")
            ),
            "dates_gate_only": sum(
                1 for r in results if not r.get("_has_llm_explanation")
            ),
        },
        "results": results,
    }

    output_path = RESULTS_DIR / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    if verbose:
        info = output["run_info"]
        print(f"\nDone. {info['dates_evaluated']} dates evaluated.")
        print(f"  {info['dates_with_llm_explanation']} with full LLM explanation")
        print(f"  {info['dates_gate_only']} with gate-only results")
        print(f"Saved to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation runner for the pipeline"
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of top anomalies to evaluate (default: 30)"
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run even if existing pipeline output exists"
    )
    parser.add_argument(
        "--existing-only", action="store_true",
        help="Only collect existing pipeline outputs (no new retrieval)"
    )
    args = parser.parse_args()
    run_batch(top_n=args.top, rerun=args.rerun,
              existing_only=args.existing_only)


if __name__ == "__main__":
    main()
