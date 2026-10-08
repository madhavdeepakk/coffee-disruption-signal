"""Direction-aware relevance heuristic."""
from src.rag import direction as d


def test_detects_up_down_neutral():
    assert d.detect_price_direction("Coffee prices surge on Brazil frost", "jumped 5%")["direction"] == "up"
    assert d.detect_price_direction("Preço do café cai em Nova York", "queda de 1,83%")["direction"] == "down"
    assert d.detect_price_direction("Coffee losses on rain forecasts", "prices fell as rain eased")["direction"] == "down"
    assert d.detect_price_direction("A local coffee festival opens", "feature story")["direction"] == "neutral"


def test_whole_word_matching_no_substring_false_positive():
    # "supply" must not trigger the "up" term
    res = d.detect_price_direction("Global coffee supply chains", "supply and demand discussion")
    assert res["direction"] == "neutral"


def test_consistency_rules():
    assert d.direction_consistency("up", "up") == "consistent"
    assert d.direction_consistency("up", "down") == "inconsistent"
    assert d.direction_consistency("neutral", "down") == "neutral"
    assert d.direction_consistency("up", None) == "neutral"


def test_rerank_orders_consistent_first_without_dropping():
    docs = [
        {"document_id": "up", "title": "Coffee soars on frost", "text": "", "semantic_score": 0.80},
        {"document_id": "down", "title": "Coffee falls as rain returns", "text": "", "semantic_score": 0.88},
        {"document_id": "neu", "title": "A coffee festival opens", "text": "", "semantic_score": 0.83},
    ]
    ranked = d.rerank(docs, "down")
    assert ranked[0]["document_id"] == "down"       # consistent leads
    assert ranked[-1]["document_id"] == "up"         # inconsistent last
    assert len(ranked) == 3                          # nothing dropped by default


def test_rerank_strict_drops_inconsistent():
    docs = [
        {"document_id": "up", "title": "Coffee soars on frost", "text": "", "semantic_score": 0.9},
        {"document_id": "down", "title": "Coffee falls hard", "text": "", "semantic_score": 0.5},
    ]
    ranked = d.rerank(docs, "down", strict=True)
    assert [x["document_id"] for x in ranked] == ["down"]
