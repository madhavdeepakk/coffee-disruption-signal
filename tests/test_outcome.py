"""Outcome classification: the labelled decision + confidence tier.

These lock in the distinction the module exists to make - RETRIEVAL_FAILED
(an outage) must never be reported as a refusal, and a direction conflict
must surface as REFUSED_CONFLICTING, not a flat 'insufficient evidence'.
"""
from src.rag import outcome as oc
from src.rag.outcome import RetrievalMeta
from src.rag.relevance_gate import GateResult


def _gate(decision, reason, accepted=0, considered=0):
    return GateResult(
        decision=decision, reason=reason, best_score=0.9,
        accepted_documents=[{"document_id": f"d{i}"} for i in range(accepted)],
        all_scored_documents=[{"document_id": f"s{i}"} for i in range(considered)],
    )


def test_retrieval_outage_is_not_a_refusal():
    # Every live query failed, nothing cached, nothing fell back, 0 docs.
    meta = RetrievalMeta(queries_attempted=3, network_attempts=3, network_failures=3,
                         network_successes=0, cache_hits=0, fallback_added=0,
                         documents_returned=0)
    gate = _gate("INSUFFICIENT_EVIDENCE", "no_documents_retrieved", 0, 0)
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=None)
    assert out.tier == oc.RETRIEVAL_FAILED
    assert out.is_fault is True
    assert out.is_refusal is False


def test_retrieval_ran_but_empty_is_a_refusal_not_an_outage():
    # Network SUCCEEDED (returned a response) but the window was genuinely quiet.
    meta = RetrievalMeta(queries_attempted=3, network_attempts=3, network_failures=0,
                         network_successes=3, cache_hits=0, fallback_added=0,
                         documents_returned=0)
    gate = _gate("INSUFFICIENT_EVIDENCE", "no_documents_retrieved", 0, 0)
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=None)
    assert out.tier == oc.REFUSED_NO_EVIDENCE
    assert out.is_refusal is True
    assert out.is_fault is False


def test_cache_hit_counts_as_retrieval_running():
    meta = RetrievalMeta(queries_attempted=2, network_attempts=0, network_failures=0,
                         network_successes=0, cache_hits=2, fallback_added=0,
                         documents_returned=0)
    gate = _gate("INSUFFICIENT_EVIDENCE", "no_documents_retrieved", 0, 0)
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=None)
    assert out.tier == oc.REFUSED_NO_EVIDENCE  # not RETRIEVAL_FAILED


def test_weak_evidence_below_threshold():
    meta = RetrievalMeta(network_successes=3, documents_returned=5)
    gate = _gate("INSUFFICIENT_EVIDENCE",
                 "documents_retrieved_but_below_relevance_threshold", 0, 5)
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=None)
    assert out.tier == oc.REFUSED_WEAK_EVIDENCE
    assert out.is_refusal is True


def test_direction_conflict_surfaces_as_conflicting():
    meta = RetrievalMeta(network_successes=3, documents_returned=4)
    gate = _gate("EXPLAIN", "2_document(s)_above_threshold", 2, 4)
    result = {"decision": "INSUFFICIENT_EVIDENCE"}
    summary = {"consistent": 0, "neutral": 0, "inconsistent": 2}
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=result,
                      direction_summary=summary)
    assert out.tier == oc.REFUSED_CONFLICTING
    assert out.is_refusal is True


def test_model_refusal_without_conflict_is_weak():
    meta = RetrievalMeta(network_successes=3, documents_returned=4)
    gate = _gate("EXPLAIN", "2_document(s)_above_threshold", 2, 4)
    result = {"decision": "INSUFFICIENT_EVIDENCE"}
    summary = {"consistent": 1, "neutral": 1, "inconsistent": 0}
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=result,
                      direction_summary=summary)
    assert out.tier == oc.REFUSED_WEAK_EVIDENCE


def test_api_error_is_system_error_not_refusal():
    meta = RetrievalMeta(network_successes=3, documents_returned=4)
    gate = _gate("EXPLAIN", "2_document(s)_above_threshold", 2, 4)
    result = {"decision": "API_ERROR"}
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=result)
    assert out.tier == oc.SYSTEM_ERROR
    assert out.is_fault is True
    assert out.is_refusal is False


def test_strong_explanation():
    meta = RetrievalMeta(network_successes=3, documents_returned=4)
    gate = _gate("EXPLAIN", "2_document(s)_above_threshold", 2, 4)
    result = {"decision": "EXPLAINED", "confidence": "high",
              "citations": [{"document_id": "d0"}]}
    summary = {"consistent": 2, "neutral": 0, "inconsistent": 0}
    faith = {"citations_resolve": True, "n_missing_document": 0}
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=result,
                      direction_summary=summary, faithfulness_report=faith)
    assert out.tier == oc.EXPLAINED_STRONG
    assert out.is_refusal is False and out.is_fault is False


def test_explained_is_tentative_when_single_source():
    meta = RetrievalMeta(network_successes=3, documents_returned=2)
    gate = _gate("EXPLAIN", "1_document(s)_above_threshold", 1, 2)
    result = {"decision": "EXPLAINED", "confidence": "high", "citations": []}
    summary = {"consistent": 1, "neutral": 0, "inconsistent": 0}
    faith = {"citations_resolve": True, "n_missing_document": 0}
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=result,
                      direction_summary=summary, faithfulness_report=faith)
    assert out.tier == oc.EXPLAINED_TENTATIVE


def test_explained_tentative_when_unfaithful_citation():
    meta = RetrievalMeta(network_successes=3, documents_returned=4)
    gate = _gate("EXPLAIN", "2_document(s)_above_threshold", 2, 4)
    result = {"decision": "EXPLAINED", "confidence": "high", "citations": []}
    summary = {"consistent": 2, "neutral": 0, "inconsistent": 0}
    faith = {"citations_resolve": False, "n_missing_document": 1}
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=result,
                      direction_summary=summary, faithfulness_report=faith)
    assert out.tier == oc.EXPLAINED_TENTATIVE


def test_explained_is_tentative_when_a_citation_is_flagged():
    meta = RetrievalMeta(network_successes=3, documents_returned=4)
    gate = _gate("EXPLAIN", "2_document(s)_above_threshold", 2, 4)
    result = {"decision": "EXPLAINED", "confidence": "high", "citations": [{"document_id": "d0"}]}
    summary = {"consistent": 2, "neutral": 0, "inconsistent": 0}
    faith = {"citations_resolve": True, "n_missing_document": 0, "n_low_rank": 1}
    out = oc.classify(retrieval_meta=meta, gate_result=gate, explanation_result=result,
                      direction_summary=summary, faithfulness_report=faith)
    assert out.tier == oc.EXPLAINED_TENTATIVE
    assert "weakly matched" in out.detail


def test_guard_withholds_on_a_majority_against_and_not_on_a_tie():
    assert oc.guard_fires({"consistent": 2, "inconsistent": 8}) is True
    assert oc.guard_fires({"consistent": 0, "inconsistent": 1}) is True
    # The three ties seen in real runs were all on dates with a documented cause.
    assert oc.guard_fires({"consistent": 1, "inconsistent": 1}) is False
    assert oc.guard_fires({"consistent": 3, "inconsistent": 3}) is False
    assert oc.guard_fires({"consistent": 0, "inconsistent": 0}) is False
    assert oc.guard_fires({}) is False and oc.guard_fires(None) is False


def test_the_first_guard_rule_can_still_be_scored():
    assert oc.guard_fires({"consistent": 3, "inconsistent": 3}, "ties") is True
    assert oc.guard_fires({"consistent": 4, "inconsistent": 3}, "ties") is False
    assert oc.guard_fires({"consistent": 0, "inconsistent": 0}, "ties") is False
    try:
        oc.guard_fires({"consistent": 1, "inconsistent": 1}, "sometimes")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown rule must not silently mean 'never fires'")


def test_a_tied_explanation_is_kept_but_not_called_strong():
    result = {"decision": "EXPLAINED", "explanation": "x", "confidence": "high",
              "citations": [{"document_id": "d1", "supports": "y"}]}
    tie = {"consistent": 3, "inconsistent": 3}
    assert oc.apply_direction_guard(result, tie) is result
    out = oc.classify(retrieval_meta=None,
                      gate_result={"decision": "EXPLAIN", "accepted_document_count": 6,
                                   "total_documents_considered": 100},
                      explanation_result=result, direction_summary=tie,
                      faithfulness_report=None)
    assert out.tier == oc.EXPLAINED_TENTATIVE
