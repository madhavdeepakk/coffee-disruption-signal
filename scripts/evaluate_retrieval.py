"""
Retrieval quality against the hand-labelled documents.

The relevance gate decides which documents the model sees, using two scores
per document. This script asks how well each score separates documents a
person judged relevant from ones judged not relevant, using the labelled
sheet (data/labeling/relevance_labeling_set.csv).

For every score column present in the sheet:

  AUC          Probability that a randomly chosen relevant document scores
               higher than a randomly chosen irrelevant one. 0.5 is chance.
               Threshold-free, so it compares scores on different scales.
               Shown with a bootstrap 95% interval (resampling whole anomaly
               dates, since documents for one date are not independent).
  P@k, R@k     Per anomaly date: of the top k documents by this score, how
               many are relevant; and how many of the date's relevant
               documents made the top k. Averaged over dates that have at
               least k labelled documents and at least one relevant one.
  MRR          Mean reciprocal rank of the first relevant document.

Score columns it understands:
  keyword_score        English keyword overlap (src/rag/retriever.py)
  semantic_score       similarity to the fixed commodity query
  anomaly_query_score  similarity to a query built from the move's direction
                       (src/rag/rerank.py) - present only in sheets collected
                       after that was added
  fused                reciprocal-rank fusion of semantic and anomaly-query
                       scores within each date - what --rerank would use

Runs offline on the labelled CSV.

Usage:
    python -m scripts.evaluate_retrieval
    python -m scripts.evaluate_retrieval --k 3 5
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_thresholds import load_labeled, DEFAULT_INPUT_PATH   # noqa: E402
from src.rag.rerank import reciprocal_rank_fusion                            # noqa: E402

OUTPUT_MD_PATH = REPO_ROOT / "results" / "retrieval_evaluation.md"
SCORE_COLUMNS = ["keyword_score", "semantic_score", "anomaly_query_score"]
BOOTSTRAP_SAMPLES = 2000


def auc(scores: list, relevant: list):
    """Rank-based AUC (Mann-Whitney). Ties count half. None if only one class."""
    pos = [s for s, r in zip(scores, relevant) if r]
    neg = [s for s, r in zip(scores, relevant) if not r]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_auc(df: pd.DataFrame, column: str, seed: int = 7,
                  samples: int = BOOTSTRAP_SAMPLES) -> tuple:
    """95% interval for AUC, resampling anomaly dates with replacement."""
    groups = [g for _, g in df.groupby("anomaly_date")]
    if len(groups) < 3:
        return (None, None)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        picked = pd.concat([rng.choice(groups) for _ in groups])
        value = auc(picked[column].tolist(), picked["relevant"].tolist())
        if value is not None:
            values.append(value)
    if len(values) < samples // 2:
        return (None, None)
    values.sort()
    return (round(values[int(0.025 * len(values))], 3),
            round(values[int(0.975 * len(values)) - 1], 3))


def ranking_metrics(df: pd.DataFrame, column: str, k: int) -> dict:
    """Mean P@k, R@k and MRR over dates with >= k labelled documents and at
    least one relevant one."""
    p_at_k, r_at_k, rr = [], [], []
    for _, group in df.groupby("anomaly_date"):
        if len(group) < k or not group["relevant"].any():
            continue
        ranked = group.sort_values(column, ascending=False)
        top = ranked.head(k)
        p_at_k.append(top["relevant"].mean())
        r_at_k.append(top["relevant"].sum() / group["relevant"].sum())
        first = next(i for i, rel in enumerate(ranked["relevant"], 1) if rel)
        rr.append(1.0 / first)
    n = len(p_at_k)
    if not n:
        return {"n_dates": 0}
    return {"n_dates": n, "p_at_k": round(sum(p_at_k) / n, 3),
            "r_at_k": round(sum(r_at_k) / n, 3), "mrr": round(sum(rr) / n, 3)}


def add_fused(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date reciprocal-rank fusion of the semantic and anomaly-query scores."""
    df = df.copy()
    df["fused"] = 0.0
    for _, group in df.groupby("anomaly_date"):
        fused = reciprocal_rank_fusion(group["semantic_score"].tolist(),
                                       group["anomaly_query_score"].tolist())
        df.loc[group.index, "fused"] = fused
    return df


def evaluate(df: pd.DataFrame, ks: list) -> dict:
    df = df.copy()
    df["relevant"] = df["human_label"] == "relevant"

    # Each score is evaluated on the rows that have it. A sheet built up over
    # time has older rows without the newer score columns; dropping a column
    # because some rows lack it would hide exactly the comparison the newer
    # column was added for.
    subsets = {}
    for column in SCORE_COLUMNS:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().any():
                sub = df[values.notna()].copy()
                sub[column] = values[values.notna()]
                subsets[column] = sub
    if "semantic_score" in subsets and "anomaly_query_score" in subsets:
        both = subsets["anomaly_query_score"]
        both = both[pd.to_numeric(both["semantic_score"], errors="coerce").notna()].copy()
        both["semantic_score"] = pd.to_numeric(both["semantic_score"])
        if len(both):
            subsets["fused"] = add_fused(both)
            # same rows as "fused", so the two can be compared directly
            subsets["semantic_score (same rows as fused)"] = both.rename(
                columns={"semantic_score": "semantic_score (same rows as fused)"})

    out = {"n_documents": len(df), "n_relevant": int(df["relevant"].sum()),
           "n_dates": int(df["anomaly_date"].nunique()), "scores": {}}
    for column, sub in subsets.items():
        value = auc(sub[column].tolist(), sub["relevant"].tolist())
        out["scores"][column] = {
            "n_rows": len(sub),
            "auc": None if value is None else round(value, 3),
            "auc_ci95": bootstrap_auc(sub, column),
            "at_k": {k: ranking_metrics(sub, column, k) for k in ks},
        }
    return out


def report(result: dict, ks: list, source: str) -> str:
    L = ["# Retrieval Evaluation", "",
         f"{result['n_documents']} hand-labelled documents ({result['n_relevant']} relevant) "
         f"from {result['n_dates']} anomaly dates (`{source}`).", "",
         "| Score | Rows | AUC | 95% interval |"
         + "".join(f" P@{k} | R@{k} |" for k in ks) + " MRR | Dates |",
         "|---|---|---|---|" + "---|---|" * len(ks) + "---|---|"]
    for name, s in result["scores"].items():
        lo, hi = s["auc_ci95"]
        interval = "n/a" if lo is None else f"{lo}-{hi}"
        cells = ""
        for k in ks:
            at = s["at_k"][k]
            cells += (f" {at['p_at_k']} | {at['r_at_k']} |" if at["n_dates"] else " n/a | n/a |")
        last = s["at_k"][ks[-1]]
        first = s["at_k"][ks[0]]
        L.append(f"| {name} | {s['n_rows']} | {s['auc']} | {interval} |{cells} "
                 f"{first.get('mrr', 'n/a')} | {first['n_dates']}"
                 + (f" / {last['n_dates']}" if len(ks) > 1 else "") + " |")
    L += ["",
          "AUC 0.5 is chance; 1.0 is perfect separation of relevant from irrelevant. The "
          "interval is a bootstrap over anomaly dates. P@k, R@k and MRR are averaged over "
          "dates with at least k labelled documents and at least one relevant one (the "
          "Dates column gives how many, for the smallest and largest k).", ""]

    scores = result["scores"]
    notes = []
    if "keyword_score" in scores and "semantic_score" in scores:
        kw, sem = scores["keyword_score"]["auc"], scores["semantic_score"]["auc"]
        if kw is not None and sem is not None:
            notes.append(f"Semantic score AUC {sem} against keyword score AUC {kw}.")
    if "fused" in scores and scores["fused"]["auc"] is not None:
        sem = scores["semantic_score (same rows as fused)"]["auc"]
        fused = scores["fused"]["auc"]
        verb = "improves on" if fused > sem else ("matches" if fused == sem else "is worse than")
        notes.append(f"On the {scores['fused']['n_rows']} rows that have both scores, fusing in "
                     f"the anomaly-specific query ({fused}) {verb} the semantic score alone "
                     f"({sem}). The intervals decide whether that difference means anything.")
    elif "anomaly_query_score" not in scores:
        notes.append("The sheet has no anomaly_query_score column, so the optional second-stage "
                     "ranking (src/rag/rerank.py) cannot be assessed here. Re-collect the sheet "
                     "with `python -m scripts.collect_labeling_set` to add it.")
    if result["n_dates"] < 10:
        notes.append(f"Only {result['n_dates']} dates: the intervals are wide and per-date "
                     f"ranking figures rest on very few dates.")
    L += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5])
    args = parser.parse_args()

    labelled = load_labeled(args.input)
    result = evaluate(labelled, args.k)
    source = (args.input.relative_to(REPO_ROOT).as_posix()
              if args.input.is_relative_to(REPO_ROOT) else str(args.input))
    text = report(result, args.k, source)
    OUTPUT_MD_PATH.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Saved {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
