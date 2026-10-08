"""Optional second-stage ranking."""
from src.rag import rerank as rr
from src.rag import direction as direction_mod


def test_query_follows_the_direction_of_the_move():
    assert "fall" in rr.anomaly_query({"direction": "down"})
    assert "rise" in rr.anomaly_query({"direction": "up"})
    assert "wheat" in rr.anomaly_query({"direction": "up"}, "wheat")


def test_rank_fusion_rewards_agreement_between_rankings():
    # item 0 is top in both lists; item 2 is top in one and last in the other
    fused = rr.reciprocal_rank_fusion([0.9, 0.5, 0.8], [0.9, 0.6, 0.1])
    assert fused[0] > fused[2] and fused[0] > fused[1]
    assert rr.reciprocal_rank_fusion() == []


def test_rerank_orders_by_fused_score_and_records_both_scores():
    docs = [{"document_id": "generic", "semantic_score": 0.90},
            {"document_id": "specific", "semantic_score": 0.89},
            {"document_id": "weak", "semantic_score": 0.86}]
    specific_scores = {"generic": 0.70, "specific": 0.95, "weak": 0.60}
    out = rr.rerank_by_anomaly_query(
        docs, {"direction": "down"},
        lambda documents, query: [specific_scores[d["document_id"]] for d in documents])
    assert [d["document_id"] for d in out][-1] == "weak"
    assert all("anomaly_query_score" in d and "rerank_score" in d for d in out)
    assert rr.rerank_by_anomaly_query([], {"direction": "up"}, None) == []


def test_direction_ordering_respects_the_rerank_order_within_a_group():
    docs = [
        {"document_id": "a", "title": "Coffee falls on rain", "text": "", "semantic_score": 0.95,
         "rerank_score": 0.010},
        {"document_id": "b", "title": "Coffee drops as supply recovers", "text": "",
         "semantic_score": 0.86, "rerank_score": 0.030},
        {"document_id": "c", "title": "Coffee soars on frost", "text": "", "semantic_score": 0.99,
         "rerank_score": 0.050},
    ]
    out = direction_mod.rerank(docs, "down")
    # c contradicts the move so it goes last whatever its score; b outranks a on rerank_score
    assert [d["document_id"] for d in out] == ["b", "a", "c"]
