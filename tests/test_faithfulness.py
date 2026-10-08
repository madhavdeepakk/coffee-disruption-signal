"""Citation-faithfulness audit."""
from src.rag import faithfulness as f


DOCS = [
    {"document_id": "aaa111", "title": "Coffee up on frost",
     "text": "Arabica rose 5% after frost hit Minas Gerais; exports fell 28% this year."},
    {"document_id": "bbb222", "title": "US tariff on Brazil",
     "text": "The US imposed a 50% tariff on Brazilian exports effective August 2025."},
]


def test_flags_fabricated_document_id():
    expl = {
        "explanation": "Prices rose (source: doc_aaa111).",
        "citations": [
            {"document_id": "doc_aaa111", "supports": "Arabica rose after frost damaged Minas Gerais."},
            {"document_id": "zzz999", "supports": "Claim citing a document never retrieved."},
        ],
    }
    rep = f.check_citations(expl, DOCS)
    assert rep["n_missing_document"] == 1
    assert rep["citations_resolve"] is False


def test_shared_numbers_rescue_cross_language_claim():
    # Claim shares "50" with the source even if lexical overlap were low.
    expl = {"explanation": "", "citations": [
        {"document_id": "bbb222", "supports": "A 50% tariff took effect."}]}
    rep = f.check_citations(expl, DOCS)
    c = rep["per_citation"][0]
    assert "50" in c["shared_numbers"]
    assert c["flag"] == "ok"


def test_unresolved_inline_source_detected():
    expl = {"explanation": "Prices moved (source: doc_nope000).", "citations": []}
    rep = f.check_citations(expl, DOCS)
    assert rep["unresolved_inline_source_ids"] == ["nope000"]
    assert rep["citations_resolve"] is False


def test_semantic_fn_used_when_provided():
    calls = {"n": 0}
    def fake_sem(claim, doc_text):
        calls["n"] += 1
        return 0.95
    expl = {"explanation": "", "citations": [
        {"document_id": "aaa111", "supports": "totally unrelated words here"}]}
    rep = f.check_citations(expl, DOCS, semantic_fn=fake_sem)
    assert calls["n"] == 1
    assert rep["per_citation"][0]["support_method"] == "semantic"
    assert rep["per_citation"][0]["flag"] == "ok"  # high semantic score


# --- support rank ------------------------------------------------------------

MANY = [{"document_id": f"m{i}", "title": f"t{i}", "text": f"text {i}"} for i in range(8)]


def _batch_scorer(best_index):
    """Scores document `best_index` highest, the rest in descending order after it."""
    def score(claim, texts):
        return [1.0 if i == best_index else 0.9 - 0.01 * i for i in range(len(texts))]
    return score


def test_citation_ranked_first_among_shown_documents_is_ok():
    expl = {"explanation": "", "citations": [{"document_id": "m5", "supports": "a claim"}]}
    rep = f.check_citations(expl, MANY, semantic_fn=lambda c, t: 0.9,
                            semantic_batch_fn=_batch_scorer(5))
    c = rep["per_citation"][0]
    assert c["support_rank"] == 1 and c["support_rank_of"] == 8
    assert c["flag"] == "ok" and rep["n_low_rank"] == 0


def test_citation_far_down_the_ranking_is_flagged_even_with_a_high_score():
    # Absolute similarity is 0.9 (passes the old check); but six other documents
    # are closer to the claim than the one cited for it.
    expl = {"explanation": "", "citations": [{"document_id": "m7", "supports": "a claim"}]}
    rep = f.check_citations(expl, MANY, semantic_fn=lambda c, t: 0.9,
                            semantic_batch_fn=_batch_scorer(0))
    c = rep["per_citation"][0]
    assert c["support_rank"] == 8
    assert c["flag"] == "low_rank" and rep["n_low_rank"] == 1
    assert c["closest_document_id"] == "m0"
    assert rep["citations_resolve"] is True     # advisory: the id itself is real


def test_rank_is_not_used_when_too_few_documents_were_shown():
    expl = {"explanation": "", "citations": [{"document_id": "bbb222", "supports": "frost in Minas"}]}
    rep = f.check_citations(expl, DOCS, semantic_fn=lambda c, t: 0.9,
                            semantic_batch_fn=lambda c, texts: [1.0, 0.1])
    c = rep["per_citation"][0]
    assert c["support_rank"] == 2 and c["flag"] == "ok"


def test_rank_falls_back_to_lexical_overlap_without_a_model():
    expl = {"explanation": "", "citations": [
        {"document_id": "aaa111", "supports": "Arabica rose after frost hit Minas Gerais"}]}
    rep = f.check_citations(expl, DOCS)
    c = rep["per_citation"][0]
    assert c["support_rank"] == 1 and c["support_rank_method"] == "lexical"
