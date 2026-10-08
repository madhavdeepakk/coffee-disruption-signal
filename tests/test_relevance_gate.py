"""Relevance gate accept/reject logic, including the split thresholds."""
from src.rag import relevance_gate as rg


def test_keyword_strong_accepts():
    r = rg.gate_documents([{"document_id": "a", "retrieval_score": 0.85}])
    assert r.decision == "EXPLAIN"


def test_weak_keyword_rejected():
    r = rg.gate_documents([{"document_id": "c", "retrieval_score": 0.30}])
    assert r.decision == "INSUFFICIENT_EVIDENCE"
    assert r.reason == "documents_retrieved_but_below_relevance_threshold"


def test_no_documents():
    r = rg.gate_documents([])
    assert r.decision == "INSUFFICIENT_EVIDENCE"
    assert r.reason == "no_documents_retrieved"


def test_semantic_only_above_bar_accepts():
    r = rg.gate_documents([{"document_id": "f", "retrieval_score": 0.0, "semantic_score": 0.88}])
    assert r.decision == "EXPLAIN"


def test_semantic_between_old_and_new_threshold_now_rejected():
    # 0.80 cleared the old 0.75 bar but not the calibrated 0.85 bar.
    assert rg.SEMANTIC_THRESHOLD == 0.85
    r = rg.gate_documents([{"document_id": "g", "retrieval_score": 0.0, "semantic_score": 0.80}])
    assert r.decision == "INSUFFICIENT_EVIDENCE"


def test_noise_floor_reason():
    r = rg.gate_documents([{"document_id": "d", "retrieval_score": 0.0, "semantic_score": 0.01}])
    assert r.reason == "all_documents_below_noise_floor"
