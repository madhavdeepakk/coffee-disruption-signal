"""
Builds a small, human-labelable set of retrieved documents so the relevance
gate's two thresholds (src/rag/relevance_gate.py's KEYWORD_THRESHOLD,
SEMANTIC_THRESHOLD) can be calibrated against human judgments instead of
staying reasoned placeholders. This is the Week 4 calibration task.

Must be run on a machine that can reach GDELT and has the ONNX semantic
scorer working (a real environment, not an offline one), since it calls the
same src.pipeline.retrieve_evidence() the pipeline uses. The documents and
scores you label are therefore the same kind the gate sees in production,
not synthetic examples.

What it does:
  1. Retrieves documents for a handful of already-detected coffee anomaly
     dates (the same ones from results/anomaly_detector_validation.md - a
     mix of events already run through the pipeline and ones not yet run,
     for variety).
  2. Pools all retrieved documents together, then takes a stratified sample
     across the score range (not just the top-scoring ones). Calibrating a
     threshold needs clear examples on both ends and the ambiguous middle,
     not just a pile of obviously-good matches.
  3. Writes data/labeling/relevance_labeling_set.csv with a blank
     human_label column to fill in by hand.

After collecting, open the CSV, read each title/text_snippet/url, and fill
in human_label as exactly "relevant" or "not_relevant" for each row you can
judge (leave genuinely-unsure rows blank; calibrate_thresholds.py ignores
blanks). Aim for the full ~15-20 rows; judge "relevant" as whether the
document plausibly helps explain this specific anomaly, not just whether it
is about coffee in general.

Re-running is safe: labels already entered are kept. A row that is sampled
again keeps its label, and a labelled row that is not sampled again stays in
the file.

The sample is spread evenly over anomaly dates (stratified by score within
each date), because the per-date ranking metrics and the leave-one-date-out
threshold check both need several labelled documents for every date.

Usage:
    python -m scripts.collect_labeling_set
    python -m scripts.collect_labeling_set --n 200       # the size worth labelling
    python -m scripts.collect_labeling_set --offline     # on-disk caches only
"""

import argparse
import csv
import math
from pathlib import Path

from src.config.commodities import get_commodity
from src.pipeline import retrieve_evidence, get_anomaly, GDELT_CACHE_DIR
from src.rag.rerank import anomaly_query

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "labeling" / "relevance_labeling_set.csv"
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CACHE_DIR = GDELT_CACHE_DIR  # reuse the same on-disk GDELT cache the
# pipeline uses, so re-collecting the labeling set doesn't re-trigger the
# 429 storm for windows already fetched once (see src/pipeline.py).

# The original seven dates the first 20-document sheet was drawn from. Kept
# as the fallback when no stored pipeline outputs exist yet.
LABELING_DATES = [
    "2021-07-19",  # 2021 Brazil drought+frost, first flagged day
    "2024-09-23",  # 2024 Brazil drought+Typhoon Yagi
    "2024-10-07",  # 2024 Brazil drought+Typhoon Yagi, second flagged day
    "2025-08-15",  # 2025 sharp spike+reversal, initial spike
    "2025-09-15",  # 2025 sharp spike+reversal, the reversal side
    "2026-07-07",  # already explained with high confidence - good "easy relevant" anchor
    "2026-08-28",  # already correctly refused - good "topically related but not relevant" anchor
]

COLUMNS = [
    "document_id", "anomaly_date", "anomaly_move", "title", "url", "publication_date",
    "days_before_anomaly", "n_copies", "text_snippet",
    "keyword_score", "semantic_score", "anomaly_query_score", "human_label",
]


def default_dates(commodity_key: str) -> list:
    """Every date with a stored pipeline output for this commodity, so the
    sheet covers the same dates the evaluation does. Falls back to the
    original seven if there are none."""
    prefix = f"pipeline_output_{commodity_key}_"
    dates = sorted(p.stem.replace(prefix, "") for p in RESULTS_DIR.glob(f"{prefix}*.json"))
    return dates or LABELING_DATES


def collect_all_documents(commodity_key: str, dates: list, cache_dir=DEFAULT_CACHE_DIR,
                          offline: bool = False) -> list:
    cfg = get_commodity(commodity_key)
    kwargs = dict(cache_dir=cache_dir, commodity_key=cfg.key)
    if offline:
        # caches only: no GDELT call, no RSS, no article fetch
        kwargs.update(commodity_key=None, gdelt_live=False, fetch_text="cache", fallback_dir=None)

    # The anomaly-specific query score is recorded alongside the two gate
    # scores so scripts/evaluate_retrieval.py can compare all three rankings
    # against the labels. It needs the embedding model; without it the column
    # is left blank.
    try:
        from src.rag.vector_store import score_documents_against_query
    except ImportError:
        score_documents_against_query = None

    all_docs = []
    for date_str in dates:
        print(f"\n=== Retrieving for {date_str} ===")
        try:
            anomaly = get_anomaly(date_str, cfg.anomalies_file)
        except SystemExit:
            print("  not in the detections file, skipped")
            continue
        docs = retrieve_evidence(date_str, cfg.gdelt_queries, cfg.semantic_reference_query, **kwargs)
        if docs and score_documents_against_query is not None:
            try:
                scores = score_documents_against_query(docs, anomaly_query(anomaly))
                for d, s in zip(docs, scores):
                    d["anomaly_query_score"] = s
            except Exception as exc:  # noqa: BLE001 - model unavailable: leave the column blank
                print(f"  anomaly-query scoring unavailable: {type(exc).__name__}")
                score_documents_against_query = None
        pct = anomaly.get("pct_move")
        move = f"{pct:+.1f}% ({anomaly['direction']})" if pct is not None else anomaly["direction"]
        for d in docs:
            d["anomaly_date"] = date_str
            d["anomaly_move"] = move
        all_docs.extend(docs)
    return all_docs


def stratified_sample(documents: list, n: int) -> list:
    """
    Sort by best score (keyword or semantic, whichever is higher) and take
    an evenly-spaced sample across the full sorted list - this deliberately
    includes clear top matches, clear non-matches, AND the ambiguous middle,
    which is exactly what you need to see to pick a sane threshold. Taking
    only the top-N would give you a labeling set with no negative examples
    at all, which can't calibrate anything.
    """
    if len(documents) <= n:
        return documents
    ranked = sorted(
        documents,
        key=lambda d: max(d.get("retrieval_score", 0.0) or 0.0, d.get("semantic_score", 0.0) or 0.0),
        reverse=True,
    )
    step = len(ranked) / n
    return [ranked[int(i * step)] for i in range(n)]


def sample_per_date(documents: list, n: int) -> list:
    """Spread the sample evenly over anomaly dates, stratified by score
    within each date. Ranking metrics (precision@k per date) and the
    leave-one-date-out threshold check both need several labelled documents
    for EVERY date, which a single pooled sample does not give."""
    by_date = {}
    for d in documents:
        by_date.setdefault(d["anomaly_date"], []).append(d)
    if not by_date:
        return []
    per_date = max(1, math.ceil(n / len(by_date)))
    sample = []
    for date in sorted(by_date):
        sample.extend(stratified_sample(by_date[date], per_date))
    return sample


def _row(d: dict) -> dict:
    sem = d.get("semantic_score")
    aq = d.get("anomaly_query_score")
    return {
        "document_id": d["document_id"],
        "anomaly_date": d["anomaly_date"],
        "anomaly_move": d.get("anomaly_move", ""),
        "title": d.get("title", ""),
        "url": d.get("url", ""),
        "publication_date": d.get("publication_date", ""),
        "days_before_anomaly": "" if d.get("days_before_anomaly") is None else d["days_before_anomaly"],
        "n_copies": d.get("n_copies", 1),
        "text_snippet": (d.get("text") or "")[:300].replace("\n", " "),
        "keyword_score": round(d.get("retrieval_score", 0.0) or 0.0, 4),
        "semantic_score": "" if sem is None else round(sem, 4),
        "anomaly_query_score": "" if aq is None else round(aq, 4),
        "human_label": "",   # fill this in by hand
    }


def read_existing(path: Path = OUTPUT_PATH) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def merge_with_existing(new_rows: list, existing: list) -> list:
    """Never lose a label. A re-collected row keeps the label it already had,
    and a labelled row that was not re-sampled is kept as it was."""
    def key(r):
        return (r["document_id"], r["anomaly_date"])
    labelled = {key(r): r for r in existing if (r.get("human_label") or "").strip()}
    seen = set()
    out = []
    for row in new_rows:
        prior = labelled.get(key(row))
        if prior:
            row["human_label"] = prior["human_label"]
        seen.add(key(row))
        out.append(row)
    for k, prior in labelled.items():
        if k not in seen:
            out.append({col: prior.get(col, "") for col in COLUMNS})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Target labeling-set size (default 20)")
    parser.add_argument("--commodity", default="coffee", help="Commodity key (default coffee)")
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Anomaly dates to draw from (default: every date with a stored "
                             "pipeline output)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the GDELT response cache and force fresh retrieval")
    parser.add_argument("--offline", action="store_true",
                        help="Use only the on-disk GDELT and text caches (no network)")
    args = parser.parse_args()

    dates = args.dates or default_dates(args.commodity)
    all_docs = collect_all_documents(
        args.commodity, dates,
        cache_dir=None if args.no_cache else DEFAULT_CACHE_DIR, offline=args.offline,
    )
    print(f"\nRetrieved {len(all_docs)} total documents across {len(dates)} dates.")

    # De-dupe by (document, date): the same article can legitimately surface
    # for more than one anomaly date, and whether it is relevant depends on
    # the date, so it is a separate judgement each time.
    seen, deduped = set(), []
    for d in all_docs:
        k = (d["document_id"], d["anomaly_date"])
        if k not in seen:
            seen.add(k)
            deduped.append(d)

    sample = sample_per_date(deduped, args.n)
    existing = read_existing()
    rows = merge_with_existing([_row(d) for d in sample], existing)
    kept = sum(1 for r in rows if (r.get("human_label") or "").strip())
    print(f"Sampled {len(sample)} documents across {len({d['anomaly_date'] for d in sample})} "
          f"dates; {kept} existing label(s) kept.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_PATH}")
    print(
        "\nNext step: open the CSV, read each title/text_snippet/url against the anomaly_move "
        "for its date, and fill in the human_label column with exactly 'relevant' or "
        "'not_relevant' for each row you can confidently judge (leave genuinely-unsure rows "
        "blank). 'relevant' means the document plausibly helps explain THAT move on THAT "
        "date, not merely that it is about coffee. Then run:\n"
        "    python -m scripts.calibrate_thresholds\n"
        "    python -m scripts.evaluate_retrieval"
    )


if __name__ == "__main__":
    main()
