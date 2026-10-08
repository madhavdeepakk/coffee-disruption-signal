"""
Citation-faithfulness check: after the explainer produces a grounded
explanation, verify that its citations actually hold up.

WHY this exists: the whole premise of the RAG track is "don't hallucinate an
explanation." The relevance gate stops the LLM from being handed junk, and
the prompt requires a document_id for every claim - but nothing yet checks
that the cited document (a) exists in what was actually retrieved, and (b)
genuinely says what the citation claims it says. A fluent model can cite a
real document_id next to a claim that document doesn't support, which is
exactly the failure this project says it wants to prevent. This module is
the after-the-fact audit of that.

Two checks, strongest first:
  1. ID RESOLUTION (hard, language-independent, reliable): every cited
     document_id - both in the structured `citations` list and in the inline
     "(source: doc_xxx)" markers inside the explanation text - must be one of
     the documents actually passed to the explainer. A citation to an id that
     was never retrieved is a fabrication, full stop.
  2. SUPPORT OVERLAP (advisory): how well the citation's claim matches the
     cited document's own text. Uses the multilingual embedding model when
     available (a claim written in English can be supported by a Portuguese
     source - lexical overlap alone would wrongly flag those, so semantic
     similarity is preferred), falling back to a lexical + shared-numbers
     measure offline. Shared numbers ("28%", "50%") are strong cross-language
     evidence and are reported separately.

  3. SUPPORT RANK (advisory): among all the documents the model was shown,
     is the cited one near the top by similarity to the claim? Added because
     check 2's absolute score is not discriminating with this embedding model
     (see _support_rank). Only applied when there are enough documents for a
     rank to mean something.

None of these establishes that the document ENTAILS the claim - similarity
is not entailment. That is measured separately, against human labels, by
scripts/audit_citations.py.

Report-only: it annotates the pipeline output and prints a summary, it does
not suppress an explanation. Check 1 failing is a red flag worth surfacing
clearly; check 2 is a softer signal (heuristic thresholds), so it is
reported as a score, not a verdict.
"""

import re
from typing import Callable, Optional

# A cited claim scoring below this against its source (semantic cosine, ~0..1)
# is flagged as weakly-supported. Heuristic, tuned to the multilingual-e5
# model's compressed-high band (see relevance_gate.py) - a genuinely
# supported claim typically scores well above this.
SEMANTIC_SUPPORT_THRESHOLD = 0.60
# Lexical fallback (offline / no model): fraction of the claim's content words
# that appear in the source text.
LEXICAL_SUPPORT_THRESHOLD = 0.15

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "by", "at", "from", "that", "this", "these", "those", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "has", "have", "had",
    "which", "due", "amid", "into", "over", "than", "per", "s", "their", "they",
}

_INLINE_SOURCE_RE = re.compile(r"source:\s*(?:doc_)?([A-Za-z0-9]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _normalize_id(raw: str) -> str:
    """Strip a leading 'doc_' marker so 'doc_abc123' and 'abc123' compare equal."""
    if raw is None:
        return ""
    raw = str(raw).strip()
    return raw[4:] if raw.lower().startswith("doc_") else raw


def _content_words(text: str) -> set:
    toks = re.findall(r"[a-záàâãéêíóôõúüçñ]+", (text or "").lower())
    return {t for t in toks if len(t) > 2 and t not in _STOPWORDS}


def _numbers(text: str) -> set:
    return {n.replace(",", ".") for n in _NUMBER_RE.findall(text or "")}


def _lexical_support(claim: str, doc_text: str) -> float:
    claim_words = _content_words(claim)
    if not claim_words:
        return 0.0
    doc_words = _content_words(doc_text)
    return len(claim_words & doc_words) / len(claim_words)


# A citation is flagged low_rank when at least this many documents were shown
# to the model and the cited one is not among the SUPPORT_RANK_TOP most
# similar to the claim. See _support_rank for why rank is used at all.
SUPPORT_RANK_MIN_CANDIDATES = 6
SUPPORT_RANK_TOP = 3


def _support_rank(claim: str, cited_id: str, docs_by_id: dict, score_all) -> Optional[dict]:
    """Where does the cited document rank among everything the model was
    shown, by similarity to the claim?

    The absolute similarity score cannot fail in practice: the embedding
    model puts any two coffee-market texts at 0.75-0.95, and the threshold is
    0.60, so every citation in the stored runs passed (lowest score 0.83).
    Rank does not depend on that compressed scale. If the claim is closer to
    several OTHER documents than to the one cited for it, the citation is
    suspect whatever the absolute number says.

    score_all(claim, [texts]) -> [scores]. Returns
    {"rank", "n_candidates", "best_document_id"} or None if scoring failed.
    """
    ids = list(docs_by_id)
    texts = [(docs_by_id[i].get("text") or docs_by_id[i].get("title") or "") for i in ids]
    try:
        scores = [float(x) for x in score_all(claim, texts)]
    except Exception:  # noqa: BLE001 - never let scoring break the audit
        return None
    if len(scores) != len(ids):
        return None
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    ranked_ids = [ids[i] for i in order]
    if cited_id not in ranked_ids:
        return None
    return {"rank": ranked_ids.index(cited_id) + 1, "n_candidates": len(ids),
            "best_document_id": ranked_ids[0]}


def check_citations(explanation_result: dict, accepted_documents: list,
                    semantic_fn: Optional[Callable[[str, str], float]] = None,
                    semantic_batch_fn: Optional[Callable[[str, list], list]] = None) -> dict:
    """
    Audit an explanation's citations against the documents actually retrieved.

    semantic_fn(claim, doc_text) -> cosine similarity in ~[0,1]; if None, a
    lexical + shared-numbers fallback is used (deterministic, offline-safe).

    semantic_batch_fn(claim, [doc_text, ...]) -> [similarity, ...]; when given
    (or, without it, using the lexical measure), each citation also gets a
    support_rank among all documents shown to the model.

    accepted_documents should be the documents that were in the prompt. A
    citation to a document that was retrieved but cut from the prompt is
    reported as missing: the model could not have read it.

    Returns a report dict (JSON-serializable) suitable for storing in the
    pipeline output and printing.
    """
    docs_by_id = {}
    for d in accepted_documents or []:
        docs_by_id[_normalize_id(d.get("document_id", ""))] = d

    if semantic_batch_fn is not None:
        rank_scorer, rank_method = semantic_batch_fn, "semantic"
    else:
        def rank_scorer(claim, texts):
            return [_lexical_support(claim, t) for t in texts]
        rank_method = "lexical"

    citations = (explanation_result or {}).get("citations") or []
    # Saved outputs and other callers may hold citations in whatever shape the
    # model produced; only well-formed ones can be audited.
    citations = [c for c in citations if isinstance(c, dict)] if isinstance(citations, list) else []
    explanation_text = str((explanation_result or {}).get("explanation") or "")

    per_citation = []
    n_missing = 0
    n_weak = 0
    n_low_rank = 0
    for cite in citations:
        cid = _normalize_id(cite.get("document_id", ""))
        claim = cite.get("supports", "") or ""
        doc = docs_by_id.get(cid)
        if doc is None:
            n_missing += 1
            per_citation.append({
                "document_id": cid, "id_found": False, "support_score": None,
                "shared_numbers": [], "flag": "missing_document",
            })
            continue
        doc_text = (doc.get("text") or doc.get("title") or "")
        shared_numbers = sorted(_numbers(claim) & _numbers(doc_text))
        if semantic_fn is not None:
            try:
                support = float(semantic_fn(claim, doc_text))
            except Exception:  # noqa: BLE001 - never let scoring break the audit
                support = _lexical_support(claim, doc_text)
                method = "lexical(fallback)"
            else:
                method = "semantic"
            threshold = SEMANTIC_SUPPORT_THRESHOLD
        else:
            support = _lexical_support(claim, doc_text)
            method = "lexical"
            threshold = LEXICAL_SUPPORT_THRESHOLD
        # A weak textual match is forgiven if the claim and source share a
        # concrete number (e.g. "28%") - strong, language-independent evidence
        # the claim came from that document.
        weak = support < threshold and not shared_numbers
        if weak:
            n_weak += 1

        rank_info = _support_rank(claim, cid, docs_by_id, rank_scorer) if claim else None
        low_rank = bool(
            rank_info
            and rank_info["n_candidates"] >= SUPPORT_RANK_MIN_CANDIDATES
            and rank_info["rank"] > SUPPORT_RANK_TOP
            and not shared_numbers
        )
        if low_rank:
            n_low_rank += 1

        entry = {
            "document_id": cid, "id_found": True,
            "support_score": round(support, 3), "support_method": method,
            "shared_numbers": shared_numbers,
            "flag": "weak_support" if weak else ("low_rank" if low_rank else "ok"),
        }
        if rank_info:
            entry.update({
                "support_rank": rank_info["rank"],
                "support_rank_of": rank_info["n_candidates"],
                "support_rank_method": rank_method,
                "closest_document_id": rank_info["best_document_id"],
            })
        per_citation.append(entry)

    # Inline "(source: doc_xxx)" markers inside the prose - catch any that
    # don't resolve to a retrieved document, even if they weren't listed in
    # the structured citations array.
    inline_ids = {_normalize_id(m) for m in _INLINE_SOURCE_RE.findall(explanation_text)}
    unresolved_inline = sorted(i for i in inline_ids if i and i not in docs_by_id)

    overall_ok = (n_missing == 0 and not unresolved_inline)
    return {
        "n_citations": len(citations),
        "n_ok": sum(1 for c in per_citation if c["flag"] == "ok"),
        "n_weak_support": n_weak,
        "n_low_rank": n_low_rank,
        "n_missing_document": n_missing,
        "unresolved_inline_source_ids": unresolved_inline,
        "citations_resolve": overall_ok,
        "per_citation": per_citation,
        "note": ("All cited document_ids resolve to retrieved documents."
                 if overall_ok else
                 "WARNING: one or more citations reference documents that were "
                 "not in the retrieved/accepted set - possible fabrication."),
    }


def format_report(report: dict) -> str:
    """One-line-per-citation human summary for the pipeline printout."""
    lines = [
        f"Citation faithfulness: {report['n_ok']}/{report['n_citations']} ok, "
        f"{report['n_weak_support']} weak-support, {report.get('n_low_rank', 0)} low-rank, "
        f"{report['n_missing_document']} missing-doc"
        + ("" if report["citations_resolve"] else "  <-- UNRESOLVED CITATIONS")
    ]
    for c in report["per_citation"]:
        if c["flag"] == "ok":
            continue
        extra = (f" support={c['support_score']}" if c.get("support_score") is not None else "")
        if c.get("support_rank"):
            extra += f" rank={c['support_rank']}/{c['support_rank_of']}"
        lines.append(f"  [{c['flag']}] doc {c['document_id']}{extra}")
    if report["unresolved_inline_source_ids"]:
        lines.append(f"  inline source ids not retrieved: {report['unresolved_inline_source_ids']}")
    return "\n".join(lines)


def _self_test():
    docs = [
        {"document_id": "aaa111", "title": "Coffee up on frost",
         "text": "Arabica rose 5% after frost hit Minas Gerais; exports fell 28% this year."},
        {"document_id": "bbb222", "title": "US tariff on Brazil",
         "text": "The US imposed a 50% tariff on Brazilian exports effective August 2025."},
    ]
    # One good citation, one number-supported cross-topic, one fabricated id.
    expl = {
        "explanation": "Prices rose on frost (source: doc_aaa111) and a 50% tariff (source: doc_bbb222).",
        "citations": [
            {"document_id": "doc_aaa111", "supports": "Arabica rose after frost damaged Minas Gerais and exports fell 28%."},
            {"document_id": "bbb222", "supports": "A 50% US tariff on Brazilian exports took effect in August 2025."},
            {"document_id": "zzz999", "supports": "A cited claim about a document that was never retrieved."},
        ],
    }
    rep = check_citations(expl, docs)  # lexical mode (no model)
    print(format_report(rep))
    assert rep["n_missing_document"] == 1
    assert rep["citations_resolve"] is False
    assert any(c["shared_numbers"] for c in rep["per_citation"])
    print("\nfaithfulness self-test OK")


if __name__ == "__main__":
    _self_test()
