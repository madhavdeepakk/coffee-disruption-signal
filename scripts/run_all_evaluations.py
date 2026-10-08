"""
Reproducibility script: run every evaluation in the project end-to-end.

This script runs all evaluations in sequence and reports which passed, which
failed, and where the outputs are. It does NOT require a GEMINI_API_KEY
(evaluations that need live API calls are skipped unless --live is passed).

Outputs:
  - results/detector_evaluation.md       (anomaly detector recall/specificity)
  - results/rag_evaluation.md            (RAG explain/refuse profile)
  - results/explanation_backtest.md       (decision-calibration backtest)
  - results/forecast_evaluation.md        (directional forecaster, walk-forward)
  - results/hf_forecast_evaluation.md     (ARIMA + Chronos comparison)
  - results/evaluation_report.md          (pipeline metrics, decision accuracy on labels)
  - results/baseline_comparison_report.md (decisions vs. trivial baselines)
  - results/relevance_gate_calibration.md (gate thresholds, in-sample and held-out)
  - results/retrieval_evaluation.md       (ranking quality against labelled documents)
  - results/ops_report.md                 (tokens, fallback rate, failure rate)
  - results/ablation_study_report.md      (like-for-like ablation; --live)
  - results/robustness_report.md          (placebo / flipped-direction tests; --live)
  - data/processed/unified_daily.csv      (multi-source daily table)

Usage:
    python -m scripts.run_all_evaluations           # offline evaluations only
    python -m scripts.run_all_evaluations --live     # includes live API calls
    python -m scripts.run_all_evaluations --verbose  # show full output
"""

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, cmd: list[str], live_only: bool = False,
             live: bool = False, verbose: bool = False) -> dict:
    """Run one evaluation step."""
    if live_only and not live:
        return {"name": name, "status": "skipped", "reason": "needs --live flag",
                "duration": 0}

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        # Live steps hit rate-limited APIs and can legitimately take hours.
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=not verbose,
            text=True, timeout=None if live_only else 600,
            encoding="utf-8", errors="replace",
        )
        duration = time.time() - start
        if result.returncode == 0:
            print(f"  PASSED ({duration:.1f}s)")
            return {"name": name, "status": "passed", "duration": duration}
        else:
            stderr = result.stderr if not verbose else ""
            print(f"  FAILED (exit {result.returncode}, {duration:.1f}s)")
            if stderr and not verbose:
                # Print last 5 lines of error
                for line in stderr.strip().split("\n")[-5:]:
                    print(f"    {line}")
            return {"name": name, "status": "failed", "duration": duration,
                    "exit_code": result.returncode, "stderr": stderr[-500:] if stderr else ""}
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        print(f"  TIMEOUT ({duration:.1f}s)")
        return {"name": name, "status": "timeout", "duration": duration}
    except Exception as e:
        duration = time.time() - start
        print(f"  ERROR: {e}")
        return {"name": name, "status": "error", "duration": duration, "error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Include evaluations that need live API calls")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full command output")
    args = parser.parse_args()

    py = sys.executable

    steps = [
        # 1. Build unified multi-source table
        ("Build unified daily table",
         [py, "-m", "scripts.build_unified_table"], False),

        # 2. Anomaly detector evaluation
        ("Anomaly detector evaluation (all commodities)",
         [py, "-m", "scripts.evaluate_detector", "--all"], False),

        # 3. RAG evaluation (from cached outputs)
        ("RAG evaluation (cached outputs)",
         [py, "-m", "scripts.evaluate_rag"], False),

        # 4. Explanation backtest (cached)
        ("Explanation backtest (cached)",
         [py, "-m", "scripts.explanation_backtest"], False),

        # 5. Explanation backtest (live — re-runs pipeline)
        ("Explanation backtest (live)",
         [py, "-m", "scripts.explanation_backtest", "--live"], True),

        # 6. Directional forecaster (price-only + multivariate)
        ("Directional forecaster (price-only + weather)",
         [py, "-m", "src.modeling.forecast", "--multivariate"], False),

        # 7. Alternative model evaluation (ARIMA)
        ("Alternative model evaluation (ARIMA)",
         [py, "-m", "src.modeling.hf_forecast", "--arima-only"], False),

        # 8. Alternative model evaluation (ARIMA + Chronos)
        ("Alternative model evaluation (ARIMA + Chronos)",
         [py, "-m", "src.modeling.hf_forecast"], True),

        # 9. Pipeline metrics over the stored runs (with confidence intervals)
        ("Collect stored pipeline runs",
         [py, "-m", "src.evaluation.batch_runner", "--existing-only"], False),
        ("Pipeline metrics + decision accuracy on labelled dates",
         [py, "-m", "src.evaluation.metrics"], False),

        # 10. Decisions vs. trivial baselines (always-explain, gate-only)
        ("Baseline comparison (decision level)",
         [py, "-m", "src.evaluation.baseline_comparison"], False),

        # 11. Gate thresholds: in-sample and held-out, and ranking quality
        ("Relevance gate calibration (with held-out check)",
         [py, "-m", "scripts.calibrate_thresholds"], False),
        ("Retrieval quality against labelled documents",
         [py, "-m", "scripts.evaluate_retrieval"], False),

        # 12. Token / fallback / failure statistics from the call log
        ("Operations report",
         [py, "-m", "scripts.ops_report"], False),

        # 13. Citation audit (only meaningful once the sheet has been labelled)
        ("Citation audit sheet",
         [py, "-m", "scripts.audit_citations", "--export"], False),

        # 14. Like-for-like ablation - needs the embedding model, so --live only
        ("Ablation study (same documents, one component changed)",
         [py, "-m", "src.evaluation.ablation_study"], True),

        # 15. Robustness: placebo days, flipped direction, no-documents control.
        #     Needs retrieval and a model key; slow. --live only.
        ("Robustness evaluation",
         [py, "-m", "src.evaluation.robustness"], True),

        # 16. Unit tests
        ("Unit tests",
         [py, "-m", "pytest", "-q", "--tb=short"], False),
    ]

    print(f"Running {'all' if args.live else 'offline'} evaluations...")
    print(f"Repository: {REPO_ROOT}")
    total_start = time.time()

    results = []
    for name, cmd, live_only in steps:
        r = run_step(name, cmd, live_only=live_only, live=args.live,
                     verbose=args.verbose)
        results.append(r)

    total_duration = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    passed = [r for r in results if r["status"] == "passed"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    other = [r for r in results if r["status"] not in ("passed", "failed", "skipped")]

    for r in results:
        icon = {"passed": "OK", "failed": "FAIL", "skipped": "SKIP",
                "timeout": "TIME", "error": "ERR"}.get(r["status"], "?")
        dur = f"({r['duration']:.1f}s)" if r["duration"] > 0 else ""
        reason = f" — {r.get('reason', '')}" if r.get("reason") else ""
        print(f"  [{icon}] {r['name']} {dur}{reason}")

    print(f"\n  {len(passed)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped, {len(other)} other")
    print(f"  Total time: {total_duration:.0f}s")

    # Output file locations
    print("\n  Output files:")
    outputs = [
        "data/processed/unified_daily.csv",
        "results/detector_evaluation.md",
        "results/rag_evaluation.md",
        "results/explanation_backtest.md",
        "results/forecast_evaluation.md",
        "results/hf_forecast_evaluation.md",
        "results/evaluation_report.md",
        "results/baseline_comparison_report.md",
        "results/relevance_gate_calibration.md",
        "results/retrieval_evaluation.md",
        "results/ops_report.md",
        "results/ablation_study_report.md",
        "results/robustness_report.md",
    ]
    for o in outputs:
        p = REPO_ROOT / o
        exists = "exists" if p.exists() else "MISSING"
        print(f"    {o} [{exists}]")

    # Exit code: 0 if all non-skipped passed
    if failed or other:
        sys.exit(1)


if __name__ == "__main__":
    main()
