"""Labelling-sheet collection, threshold cross-validation, retrieval metrics."""
import pandas as pd

from scripts import calibrate_thresholds as ct
from scripts import collect_labeling_set as cl
from scripts import evaluate_retrieval as er
from src.evaluation.metrics import rate


def test_recollecting_never_loses_a_label():
    existing = [
        {"document_id": "a", "anomaly_date": "2024-09-23", "title": "kept and resampled",
         "human_label": "relevant"},
        {"document_id": "b", "anomaly_date": "2024-09-23", "title": "labelled, not resampled",
         "human_label": "not_relevant"},
        {"document_id": "c", "anomaly_date": "2024-09-23", "title": "never labelled",
         "human_label": ""},
    ]
    new = [{"document_id": "a", "anomaly_date": "2024-09-23", "title": "kept and resampled",
            "human_label": ""},
           {"document_id": "z", "anomaly_date": "2025-08-15", "title": "new", "human_label": ""}]
    rows = cl.merge_with_existing(new, existing)
    by_id = {r["document_id"]: r for r in rows}
    assert by_id["a"]["human_label"] == "relevant"
    assert by_id["b"]["human_label"] == "not_relevant"        # still in the file
    assert by_id["z"]["human_label"] == "" and "c" not in by_id


def test_sample_is_spread_over_dates():
    docs = [{"document_id": f"{date}-{i}", "anomaly_date": date, "semantic_score": 0.7 + i / 100}
            for date in ("d1", "d2", "d3", "d4") for i in range(20)]
    sample = cl.sample_per_date(docs, 20)
    counts = pd.Series([d["anomaly_date"] for d in sample]).value_counts()
    assert len(sample) == 20 and set(counts) == {5}


def _labelled():
    rows = []
    # Relevant documents score higher on the semantic channel, on every date.
    for date in ("d1", "d2", "d3", "d4"):
        for score, label in ((0.91, "relevant"), (0.88, "relevant"), (0.86, "relevant"),
                             (0.82, "not_relevant"), (0.78, "not_relevant")):
            rows.append({"anomaly_date": date, "semantic_score": score,
                         "keyword_score": 0.0, "human_label": label})
    return pd.DataFrame(rows)


def test_held_out_threshold_performance():
    df = _labelled()
    cv = ct.leave_one_date_out(df, "semantic_score", "high_precision")
    assert cv["n_folds"] == 4 and cv["folds_skipped"] == 0
    assert cv["precision"] == rate(12, 12) and cv["recall"] == rate(12, 12)
    prod = ct.performance_at(df, "semantic_score", 0.90)
    assert prod["precision"] == rate(4, 4) and prod["recall"] == rate(4, 12)


def test_held_out_check_needs_several_dates():
    df = _labelled()
    df["anomaly_date"] = "only-one"
    assert "error" in ct.leave_one_date_out(df, "semantic_score", "best_f1")


def test_auc():
    assert er.auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert er.auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert er.auc([0.5, 0.5], [True, False]) == 0.5              # a tie is a coin flip
    assert er.auc([0.5, 0.6], [True, True]) is None              # one class only


def test_retrieval_metrics_per_date():
    result = er.evaluate(_labelled(), [3])
    sem = result["scores"]["semantic_score"]
    assert sem["auc"] == 1.0
    assert sem["at_k"][3] == {"n_dates": 4, "p_at_k": 1.0, "r_at_k": 1.0, "mrr": 1.0}
    assert result["scores"]["keyword_score"]["auc"] == 0.5       # all zeros: no information
    assert "fused" not in result["scores"]                        # no anomaly-query column


def test_fused_ranking_is_evaluated_when_the_column_is_present():
    df = _labelled()
    df["anomaly_query_score"] = df["semantic_score"]
    result = er.evaluate(df, [3])
    assert result["scores"]["fused"]["auc"] == 1.0
    assert "improves on" in er.report(result, [3], "x") or "matches" in er.report(result, [3], "x")
