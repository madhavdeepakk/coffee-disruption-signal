"""Golden end-to-end safety test: the refuse-don't-hallucinate path.

This is the project's headline behaviour, so it gets a dedicated golden
test. It wires the REAL relevance gate + REAL direction analysis + REAL
outcome classifier together (only the LLM's verdict is supplied, since that
is the external call) and asserts:

  * evidence that conflicts on price direction, where the model declines,
    surfaces as a REFUSED_CONFLICTING outcome - NOT EXPLAINED, and flagged
    as a refusal, not a fault; and
  * clean, direction-consistent evidence the model explains surfaces as an
    EXPLAINED outcome.

If a future change lets conflicting/weak evidence produce an explanation,
this test fails loudly.
"""
from src.rag import outcome as oc
from src.rag.relevance_gate import gate_documents
from src.rag import direction as direction_mod
from src.rag.outcome import RetrievalMeta


def _healthy_meta(n):
    return RetrievalMeta(queries_attempted=3, network_attempts=3, network_successes=3,
                         documents_returned=n)


def test_conflicting_evidence_refuses_not_explains():
    # Two on-topic, gate-passing articles that point in OPPOSITE directions.
    # The anomaly is a DROP ("down"); one article says prices fell (consistent),
    # the other says prices rallied (inconsistent).
    docs = [
        {"document_id": "d_down", "title": "Coffee prices fall as ample rain eases supply fears",
         "text": "Arabica coffee futures dropped as rain returned to Brazil, easing "
                 "supply concerns and pushing prices lower.", "semantic_score": 0.90},
        {"document_id": "d_up", "title": "Coffee rallies on frost threat to Brazilian crop",
         "text": "Coffee futures surged and rose sharply on fears a frost would damage "
                 "the harvest, lifting prices higher.", "semantic_score": 0.89},
    ]
    gate = gate_documents(docs)
    assert gate.decision == "EXPLAIN"  # both are on-topic; the gate passes them

    accepted = direction_mod.rerank(gate.accepted_documents, "down", strict=False)
    summary = direction_mod.summarize(accepted)
    assert summary["inconsistent"] >= 1  # the conflict is real and detected

    # The model, reading contradictory evidence, declines (defence in depth).
    model_result = {"decision": "INSUFFICIENT_EVIDENCE", "explanation": None,
                    "citations": [], "confidence": None}

    out = oc.classify(retrieval_meta=_healthy_meta(len(docs)), gate_result=gate,
                      explanation_result=model_result, direction_summary=summary)

    assert out.tier == oc.REFUSED_CONFLICTING
    assert out.is_refusal is True
    assert out.is_fault is False
    # And it must NOT read as an explanation of any kind.
    assert out.tier not in oc.EXPLAINED_TIERS


def test_clean_consistent_evidence_explains():
    docs = [
        {"document_id": "d1", "title": "Coffee surges on Brazil frost damage",
         "text": "Arabica coffee futures surged after frost damaged Brazil's crop, "
                 "with prices rising sharply on supply fears.", "semantic_score": 0.91},
        {"document_id": "d2", "title": "Frost losses lift coffee to multi-year high",
         "text": "Prices rose again as traders raised frost damage estimates, pushing "
                 "coffee higher.", "semantic_score": 0.90},
    ]
    gate = gate_documents(docs)
    assert gate.decision == "EXPLAIN"
    accepted = direction_mod.rerank(gate.accepted_documents, "up", strict=False)
    summary = direction_mod.summarize(accepted)

    model_result = {"decision": "EXPLAINED",
                    "explanation": "Frost damage in Brazil drove coffee prices up (source: d1).",
                    "citations": [{"document_id": "d1", "supports": "frost drove prices up"}],
                    "confidence": "high"}
    faith = {"citations_resolve": True, "n_missing_document": 0}

    out = oc.classify(retrieval_meta=_healthy_meta(len(docs)), gate_result=gate,
                      explanation_result=model_result, direction_summary=summary,
                      faithfulness_report=faith)
    assert out.tier in oc.EXPLAINED_TIERS
    assert out.is_refusal is False
