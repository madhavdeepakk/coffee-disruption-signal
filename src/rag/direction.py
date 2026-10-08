"""
Direction-aware relevance: does a retrieved document describe the price
moving in the SAME direction as the detected anomaly?

WHY this exists (the gap the threshold calibration proved): the semantic
scorer tells us a document is *about* coffee-price disruption, but not
whether it explains THIS anomaly. The clearest real example is the
2021-07-19 run - a -3.7% price DROP - where retrieval returned real,
on-topic articles about frost/drought pushing coffee prices UP. Those
scored high (topically relevant) and passed the gate, and only the LLM,
at the very last step, caught that price-RISE evidence can't explain a
price DROP and refused. The labeled-set calibration then showed why the
gate couldn't catch it earlier: relevant (0.80-0.91) and not-relevant
(0.75-0.84) semantic scores overlap heavily, because semantic similarity
measures topic, not direction.

This module adds a cheap, transparent direction signal to move some of that
discrimination earlier. It is a lexicon heuristic, not a model - it is
`neutral` (no signal) on many documents, especially short or non-English
ones, and it is used as a soft re-ranking signal plus annotation, not as a
hard filter by default. That keeps it from silently turning a valid
explanation into a refusal (the failure mode of most concern here is the
opposite - forcing an unsupported explanation - but a heuristic that wrongly
drops good evidence is still a regression). The LLM's own direction check
remains the authority; this orders the evidence so direction-consistent
documents lead, and reports the split.

Multilingual note: the retrieved corpus is heavily Portuguese/Spanish/
Italian (Brazil coverage), so the lexicons below include the common
price-movement words in those languages, not just English - otherwise the
signal would be `neutral` on exactly the documents that matter most.
"""

import re
from typing import Optional

# Price-movement vocabulary, by direction. English + the Portuguese/Spanish/
# Italian terms that actually show up in this project's real retrieved
# articles (alta/queda, sube/cae, etc.). Kept as whole-word matches so "up"
# doesn't fire inside "supply" (see _count_terms).
UP_TERMS = [
    # English
    "surge", "surged", "soar", "soared", "soaring", "jump", "jumped", "rally",
    "rallied", "spike", "spiked", "climb", "climbed", "rise", "rises", "rising",
    "rose", "gain", "gains", "gained", "higher", "advance", "advanced",
    "skyrocket", "skyrocketed", "rocket", "boom", "boomed", "record", "records",
    "up", "upward", "uptick", "escalate", "escalated",
    # Portuguese
    "alta", "sobe", "subir", "subiu", "dispara", "disparou", "salto", "avanca",
    "avancam", "avancou", "aumenta", "aumentou", "aumento", "recorde",
    # Spanish
    "sube", "subio", "alza", "dispara", "disparo", "aumenta", "aumento",
    # Italian
    "aumenta", "aumento", "sale", "salgono", "impennata", "rincaro", "rincari",
]

DOWN_TERMS = [
    # English
    "fall", "falls", "fell", "falling", "drop", "drops", "dropped", "decline",
    "declined", "declines", "tumble", "tumbled", "plunge", "plunged", "slump",
    "slumped", "retreat", "retreated", "lower", "ease", "eased", "easing",
    "dip", "dipped", "sink", "sank", "slide", "slid", "crash", "crashed",
    "down", "downward", "cheaper", "loss", "losses", "pullback", "selloff",
    # Portuguese
    "queda", "cai", "cair", "caiu", "recuo", "recua", "baixa", "despenca",
    "despencou", "desaba", "desabou",
    # Spanish
    "cae", "cayo", "baja", "bajo", "desploma", "desplome", "retroceso", "caida",
    # Italian
    "cala", "scende", "scendono", "ribasso", "crollo", "crolla",
]

# Supply-side context terms, used as a WEAK secondary signal only. A supply
# shock (frost, drought, ban) is normally upward pressure on price; supply
# relief (rain, bumper crop) is downward. Weighted far below explicit
# price-movement words, because a document can mention "drought" while
# reporting a price fall for other reasons - the movement words win.
SUPPLY_SHORTAGE_TERMS = [
    "frost", "drought", "shortage", "crop damage", "export ban", "embargo",
    "disease", "deficit", "disruption", "shortfall", "geada", "seca", "escassez",
    "sequia", "helada",
]
SUPPLY_RELIEF_TERMS = [
    "rain", "rains", "rainfall", "bumper", "surplus", "good harvest",
    "improved weather", "recovery", "chuva", "chuvas", "lluvia", "cosecha récord",
]

MOVEMENT_WEIGHT = 1.0
SUPPLY_WEIGHT = 0.35  # secondary signal only - never overrides explicit movement words


def _tokenize(text: str) -> list:
    # Lowercased word tokens; keeps accented Latin letters so PT/ES/IT terms
    # match. Whole-word tokenization is what stops "up" matching in "supply".
    return re.findall(r"[a-záàâãéêíóôõúüçñ]+", (text or "").lower())


def _count_terms(tokens: list, joined: str, terms: list) -> int:
    """Count occurrences of each term. Single-word terms match tokens
    (whole-word); multi-word terms are substring-matched on the joined text."""
    token_set = tokens  # list, so repeated words count repeatedly
    total = 0
    for term in terms:
        if " " in term:
            total += joined.count(term)
        else:
            total += token_set.count(term)
    return total


def detect_price_direction(title: str, text: str) -> dict:
    """
    Infer the price direction a document describes: "up", "down", or
    "neutral" (no usable signal). Title terms count double - a direction word
    in the headline is much stronger evidence than one buried in the body.

    Returns {direction, up_score, down_score, confidence, evidence} where
    confidence is |up-down| / (up+down) in [0,1] (0 when no signal).
    """
    title_tokens = _tokenize(title)
    text_tokens = _tokenize(text)
    title_joined = " ".join(title_tokens)
    text_joined = " ".join(text_tokens)

    up = MOVEMENT_WEIGHT * (2 * _count_terms(title_tokens, title_joined, UP_TERMS)
                            + _count_terms(text_tokens, text_joined, UP_TERMS))
    down = MOVEMENT_WEIGHT * (2 * _count_terms(title_tokens, title_joined, DOWN_TERMS)
                              + _count_terms(text_tokens, text_joined, DOWN_TERMS))
    up += SUPPLY_WEIGHT * (2 * _count_terms(title_tokens, title_joined, SUPPLY_SHORTAGE_TERMS)
                           + _count_terms(text_tokens, text_joined, SUPPLY_SHORTAGE_TERMS))
    down += SUPPLY_WEIGHT * (2 * _count_terms(title_tokens, title_joined, SUPPLY_RELIEF_TERMS)
                             + _count_terms(text_tokens, text_joined, SUPPLY_RELIEF_TERMS))

    total = up + down
    if total == 0:
        return {"direction": "neutral", "up_score": 0.0, "down_score": 0.0,
                "confidence": 0.0, "evidence": "no directional terms found"}
    if up == down:
        return {"direction": "neutral", "up_score": up, "down_score": down,
                "confidence": 0.0, "evidence": "up/down signals balanced"}
    direction = "up" if up > down else "down"
    confidence = abs(up - down) / total
    return {"direction": direction, "up_score": round(up, 2),
            "down_score": round(down, 2), "confidence": round(confidence, 3),
            "evidence": f"up={up:.1f} vs down={down:.1f}"}


def direction_consistency(doc_direction: str, anomaly_direction: Optional[str]) -> str:
    """
    Compare a document's inferred direction to the anomaly's actual direction.
    Returns "consistent", "inconsistent", or "neutral". A neutral document (no
    signal) is never called inconsistent - absence of evidence isn't evidence
    of contradiction.
    """
    if not anomaly_direction or doc_direction == "neutral":
        return "neutral"
    return "consistent" if doc_direction == anomaly_direction else "inconsistent"


def _effective_score(doc: dict) -> float:
    kw = doc.get("retrieval_score", 0.0) or 0.0
    sem = doc.get("semantic_score", 0.0) or 0.0
    return max(kw, sem)


def annotate_documents(documents: list, anomaly_direction: Optional[str]) -> list:
    """Attach price_direction / direction_consistency / direction_confidence
    to each document in place, and return the list."""
    for doc in documents:
        d = detect_price_direction(doc.get("title", ""), doc.get("text", ""))
        doc["price_direction"] = d["direction"]
        doc["direction_confidence"] = d["confidence"]
        doc["direction_consistency"] = direction_consistency(d["direction"], anomaly_direction)
    return documents


def rerank(documents: list, anomaly_direction: Optional[str], strict: bool = False) -> list:
    """
    Re-order documents so direction-consistent evidence leads, then neutral,
    then inconsistent - each group ordered by its existing best (keyword/
    semantic) score. This changes only the ORDER in which evidence is handed
    to the LLM (consistent first), never which documents the gate accepted -
    so no EXPLAIN/refuse decision can flip from re-ranking alone.

    strict=True (opt-in, off by default) additionally DROPS direction-
    inconsistent documents. Use only when you explicitly want the heuristic
    to hard-filter; the default trusts the LLM to make the final direction
    call and just orders the evidence for it.
    """
    documents = annotate_documents(documents, anomaly_direction)
    rank = {"consistent": 0, "neutral": 1, "inconsistent": 2}
    if strict:
        documents = [d for d in documents if d.get("direction_consistency") != "inconsistent"]
    def within_group(d):
        # If the optional second-stage ranking ran (src/rag/rerank.py), keep
        # its order inside each direction group; otherwise the usual score.
        fused = d.get("rerank_score")
        return -(fused if fused is not None else _effective_score(d))

    # rerank_score and the keyword/semantic score are on different scales, so
    # they are never mixed: either every document has a rerank_score or none.
    return sorted(
        documents,
        key=lambda d: (rank.get(d.get("direction_consistency", "neutral"), 1),
                       within_group(d)),
    )


def news_lean(documents: list) -> dict:
    """
    Absolute up-vs-down lean of a set of documents (independent of any
    anomaly): how many describe prices rising vs falling. Used by the
    directional-forecast news signal (src/modeling/forecast.py). Returns
    counts and a lean label; 'mixed' when neither side clearly dominates.
    """
    up = down = neutral = 0
    for doc in documents:
        d = detect_price_direction(doc.get("title", ""), doc.get("text", ""))
        if d["direction"] == "up":
            up += 1
        elif d["direction"] == "down":
            down += 1
        else:
            neutral += 1
    directional = up + down
    if directional == 0:
        lean = "no signal"
    elif up >= 2 * max(down, 1) or (down == 0 and up > 0):
        lean = "bullish (up-leaning)"
    elif down >= 2 * max(up, 1) or (up == 0 and down > 0):
        lean = "bearish (down-leaning)"
    else:
        lean = "mixed"
    return {"up": up, "down": down, "neutral": neutral, "total": len(documents), "lean": lean}


def summarize(documents: list) -> dict:
    """Counts by direction-consistency, for the pipeline printout / output."""
    out = {"consistent": 0, "neutral": 0, "inconsistent": 0}
    for d in documents:
        out[d.get("direction_consistency", "neutral")] = \
            out.get(d.get("direction_consistency", "neutral"), 0) + 1
    return out


def _self_test():
    cases = [
        ("Coffee prices surge on Brazil frost fears", "Arabica futures jumped 5%.", "up"),
        ("Preço do café cai em Nova York após susto com geadas", "queda de 1,83%", "down"),
        ("Coffee prices post moderate losses on rain forecasts", "prices fell as rain eased drought", "down"),
        ("Australian coffee growers brew new opportunities", "a feature about local farms", "neutral"),
    ]
    print("direction.detect_price_direction self-test:")
    for title, text, expected in cases:
        d = detect_price_direction(title, text)
        ok = "OK " if d["direction"] == expected else "XX "
        print(f"  {ok}expected={expected:9s} got={d['direction']:9s} ({d['evidence']})")

    docs = [
        {"document_id": "a", "title": "Coffee soars on frost", "text": "", "semantic_score": 0.80},
        {"document_id": "b", "title": "Coffee falls as rain returns", "text": "", "semantic_score": 0.88},
        {"document_id": "c", "title": "A coffee festival opens", "text": "", "semantic_score": 0.83},
    ]
    print("\n  rerank for a DOWN anomaly (consistent 'falls' should lead despite lower score):")
    for d in rerank(docs, "down"):
        print(f"    {d['document_id']}: dir={d['price_direction']:8s} "
              f"consistency={d['direction_consistency']:12s} sem={d.get('semantic_score')}")


if __name__ == "__main__":
    _self_test()
