"""Evaluation scripts: RAG summary loader + detector metrics (synthetic data)."""
import json
import tempfile
import types
from pathlib import Path

import pandas as pd

from scripts import evaluate_rag, evaluate_detector


def test_rag_refusal_reason_classification():
    assert evaluate_rag._refusal_reason(
        {"explanation_result": {"decision": "EXPLAINED"}}) == "-"
    assert evaluate_rag._refusal_reason(
        {"explanation_result": {"decision": "INSUFFICIENT_EVIDENCE", "reason": "model_judged_evidence_insufficient"}}
    ) == "LLM judged evidence insufficient"
    assert evaluate_rag._refusal_reason(
        {"explanation_result": {"decision": "INSUFFICIENT_EVIDENCE"},
         "gate_result": {"decision": "INSUFFICIENT_EVIDENCE", "reason": "no_documents_retrieved"}}
    ) == "no documents retrieved"


def test_rag_load_runs_reads_dir():
    with tempfile.TemporaryDirectory() as dd:
        rec = {
            "anomaly": {"date": "2025-08-15"},
            "gate_result": {"decision": "EXPLAIN", "best_score": 0.9,
                            "accepted_document_count": 5, "total_documents_considered": 25},
            "documents_retrieved": 25,
            "explanation_result": {"decision": "EXPLAINED", "citations": [{"document_id": "x", "supports": "y"}]},
            "faithfulness_report": {"citations_resolve": True, "n_weak_support": 0, "n_missing_document": 0},
            "direction_summary": {"consistent": 3, "neutral": 1, "inconsistent": 1},
        }
        (Path(dd) / "pipeline_output_coffee_2025-08-15.json").write_text(json.dumps(rec), encoding="utf-8")
        runs = evaluate_rag.load_runs(Path(dd))
        assert len(runs) == 1
        assert runs[0]["decision"] == "EXPLAINED"
        assert runs[0]["direction"]["consistent"] == 3
        assert runs[0]["faithfulness"]["citations_resolve"] is True


def test_detector_eval_recall_and_specificity(monkeypatch):
    with tempfile.TemporaryDirectory() as dd:
        results = Path(dd) / "results"
        results.mkdir()
        # 2 event days flagged in-window, control window fully quiet.
        rows = [
            {"date": "2021-06-05", "anomaly_flag": True, "anomaly_type": "shock"},   # in event
            {"date": "2021-06-10", "anomaly_flag": False, "anomaly_type": ""},
            {"date": "2019-02-01", "anomaly_flag": False, "anomaly_type": ""},        # in control
            {"date": "2019-02-02", "anomaly_flag": False, "anomaly_type": ""},
            {"date": "2023-01-01", "anomaly_flag": True, "anomaly_type": "trend"},    # elsewhere
        ]
        pd.DataFrame(rows).to_csv(results / "fake.csv", index=False)

        fake_cfg = types.SimpleNamespace(
            display_name="Fake", anomalies_file="fake.csv",
            known_events=[("evt", "2021-06-01", "2021-06-30", "src")],
            control_period=("2019-01-01", "2019-03-31", "quiet"),
        )
        monkeypatch.setattr(evaluate_detector, "REPO_ROOT", Path(dd))
        monkeypatch.setattr(evaluate_detector, "get_commodity", lambda k: fake_cfg)

        r = evaluate_detector.evaluate_commodity("coffee")
        assert r["events_detected"] == 1 and r["n_events"] == 1
        assert r["event_recall"] == 1.0
        assert r["control_false_positives"] == 0
        assert r["control_specificity"] == 1.0
        assert r["flagged_in_event"] == 1
        assert r["flagged_elsewhere"] == 1
