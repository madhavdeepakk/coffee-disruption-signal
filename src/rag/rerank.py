"""
Optional second-stage ranking against the specific move being explained.

The semantic score every document gets is its similarity to ONE fixed
reference query per commodity ("coffee Brazil frost drought price disruption
supply"). That measures whether a document is about coffee supply shocks in
general. It says nothing about whether the document is about THIS move, and
it leans towards supply-shock stories, which mostly describe prices rising.

This module scores the already-accepted documents a second time against a
query built from the anomaly itself (which way the price moved), then merges
the two rankings with reciprocal rank fusion.

Status: OFF by default (pipeline flag --rerank). It only changes the ORDER
documents are handed to the model, which matters when the evidence has to be
trimmed to fit a prompt budget. It has not been shown to improve anything
yet: scripts/evaluate_retrieval.py compares the rankings against the
hand-labelled documents, and this should stay off until that comparison says
it helps.
"""

RRF_K = 60   # the usual constant; damps the influence of any single ranking

_UP_QUERY = "{name} prices rise surge jump rally higher futures gain"
_DOWN_QUERY = "{name} prices fall drop decline slump lower futures loss"


def anomaly_query(anomaly: dict, commodity_name: str = "coffee arabica") -> str:
    """A short query describing the move: commodity + direction words."""
    template = _UP_QUERY if anomaly.get("direction") == "up" else _DOWN_QUERY
    return template.format(name=commodity_name)


def _ranks(scores: list) -> list:
    """Rank positions (1 = best) for a list of scores, ties broken by order."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for position, index in enumerate(order, 1):
        ranks[index] = position
    return ranks


def reciprocal_rank_fusion(*score_lists: list) -> list:
    """Fuse several score lists (aligned by index) into one fused score per
    item. Uses ranks, not raw scores, so lists on different scales combine
    without any normalisation."""
    if not score_lists:
        return []
    n = len(score_lists[0])
    fused = [0.0] * n
    for scores in score_lists:
        for i, rank in enumerate(_ranks(scores)):
            fused[i] += 1.0 / (RRF_K + rank)
    return fused


def rerank_by_anomaly_query(documents: list, anomaly: dict, score_fn,
                            commodity_name: str = "coffee arabica") -> list:
    """Attach anomaly_query_score and rerank_score to each document and
    return them ordered by rerank_score, best first.

    score_fn(documents, query) -> list of similarities, aligned with
    documents (src.rag.vector_store.score_documents_against_query).
    """
    if not documents:
        return documents
    query = anomaly_query(anomaly, commodity_name)
    specific = [float(s) for s in score_fn(documents, query)]
    topical = [max(d.get("retrieval_score", 0.0) or 0.0, d.get("semantic_score", 0.0) or 0.0)
               for d in documents]
    fused = reciprocal_rank_fusion(topical, specific)
    for doc, s, f in zip(documents, specific, fused):
        doc["anomaly_query_score"] = round(s, 4)
        doc["rerank_score"] = round(f, 6)
    return sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
