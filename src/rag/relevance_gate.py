"""
Relevance gate: retrieve -> score -> threshold -> accept/reject.

Per docs/rag_design.md and the team proposal (S3.3): this is the most
important safety mechanism in the RAG track. An LLM given weak or off-topic
retrieved documents will still write a fluent, confident-sounding
explanation - the gate exists to catch that before the LLM ever sees the
documents, by refusing to pass anything through when the evidence itself
isn't strong enough.

Notes:
  - The Week 4 calibration was run on a small scale. Both thresholds were
    checked against a 20-document labeled set (scripts/collect_labeling_set.py
    -> hand labels -> scripts/calibrate_thresholds.py, report in
    results/relevance_gate_calibration.md). SEMANTIC_THRESHOLD moved
    0.75 -> 0.85 (0.75 was too permissive - it admitted 7 of 8 not-relevant
    documents); KEYWORD_THRESHOLD stayed 0.4 (the keyword channel is
    degenerate on the mostly-non-English real retrieval, so its sweep gives
    no usable signal - see each constant's comment). Calibration against 20
    one-judge labels is informed, not statistically robust, and one-line
    reversible.
  - Semantic scoring (src/rag/vector_store.py, running on onnxruntime +
    tokenizers) is wired into the live pipeline and confirmed working on real
    documents: German/Portuguese/Spanish articles that scored 0.000 on the
    keyword scorer scored 0.84-0.90 on the semantic scorer and were correctly
    accepted, covering the non-English language gap the keyword scorer has.
  - The gate uses two separate thresholds because a single shared cutoff was
    ineffective once semantic scores were in the mix: embedding similarity
    for topically related text sits in a compressed, generally-high band
    (~0.7-0.95) even for only loosely relevant documents, a different
    distribution than the keyword score. A document is accepted if either
    score clears its own bar, and SEMANTIC_THRESHOLD is set higher for that
    reason.
  - Not yet done: the offline comparison of semantic vs. keyword score
    against the Week 1 manual-labeled set (vector_store.py's own `main()`)
    has not been run; that is what would let SEMANTIC_THRESHOLD move from a
    reasoned value to a fully validated one.
  - This gate decides accept/reject given a score already attached to each
    document. It does not recompute scores itself.

Usage as a library:
    from src.rag.relevance_gate import gate_documents
    result = gate_documents(documents_for_this_anomaly)
    if result["decision"] == "EXPLAIN":
        ... pass result["accepted_documents"] to the explainer ...
    else:
        ... show "insufficient evidence" ...

Usage for a quick self-test:
    python -m src.rag.relevance_gate
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Scoring mode: configurable via SCORING_MODE env var or gate_documents kwarg.
#
# The ablation study (results/ablation_study.json) showed that the keyword
# scorer contributes 0% explain rate on its own (all non-English articles
# score 0.0), while semantic-only matches the full pipeline at 80%. This
# makes the keyword channel effectively redundant for the current corpus.
# Rather than removing it (it may help on English-heavy retrieval), we make
# the mode configurable so the paper can demonstrate the finding and future
# users can choose.
#
# Modes:
#   "dual"          — accept if EITHER keyword OR semantic clears its bar
#                     (production default, backward-compatible)
#   "semantic_only" — ignore keyword scores entirely
#   "keyword_only"  — ignore semantic scores entirely
# ---------------------------------------------------------------------------
SCORING_MODE = os.environ.get("SCORING_MODE", "dual").strip().lower()

# See the notes above - both are reasoned values, not frozen,
# labeled-set-validated thresholds.
#
# KEYWORD_THRESHOLD: chosen from the Week 2 comparison's illustrative score
# bands (>=0.5 was labeled "R-like" there), erring conservative per the
# project's stated principle: a higher refusal rate is preferable to an
# unsupported explanation.
#
# Left at 0.4 after calibration (see SEMANTIC_THRESHOLD below and
# results/relevance_gate_calibration.md). The keyword channel is not the
# workhorse on real retrieval: the 20-doc labeled set was dominated by
# non-English articles (Portuguese/Spanish/Arabic/Polish/Italian) that score
# ~0 on this English-only keyword scorer regardless of relevance - the
# language-coverage gap the semantic channel covers. The keyword sweep is
# degenerate here (best-F1 lands at 0.0, i.e. "accept everything", which is
# meaningless; precision>=0.9 needs 0.305 but recalls only ~0.17). Neither
# supports moving the bar: lowering it would just accept noise, and the
# semantic channel already picks up the non-English matches. 0.4 stays as a
# conservative bar for the cases where a keyword score is meaningfully
# non-zero.
KEYWORD_THRESHOLD = 0.4

# SEMANTIC_THRESHOLD: set higher than KEYWORD_THRESHOLD. Cosine similarity
# from this embedding model runs in a compressed, generally-high band for
# anything in the same rough topic area (observed 0.84-0.90 even for
# documents only loosely about the specific anomaly), so reusing 0.4 here
# would accept almost everything retrieved, regardless of how central it is
# to the story.
#
# Calibrated (was 0.75) against a 20-document labeled set built by
# scripts/collect_labeling_set.py from 7 coffee anomaly dates and scored by
# scripts/calibrate_thresholds.py - see
# results/relevance_gate_calibration.md. Findings that drove 0.75 -> 0.85:
#   - 0.75 was too permissive: it admitted 7 of the 8 documents hand-labeled
#     not relevant (their semantic scores ran 0.75-0.84), leaving the gate
#     nearly ineffective on the semantic side.
#   - 0.85 is the lowest threshold in the sweep reaching precision 1.0 (with
#     >=2 docs flagged) on that set. It trades recall down to ~0.5, the
#     correct direction under the project's stated principle (below, and
#     docs/final_evaluation_DRAFT.md): a higher refusal rate beats an
#     unsupported explanation. The relevant/not-relevant semantic scores
#     overlap heavily (relevant 0.80-0.91, not-relevant 0.75-0.84), so no
#     threshold separates them cleanly - 0.85 is the precision-first pick,
#     not a clean divider.
#   - Checked against the 5 completed pipeline runs (2021-07-19, 2024-09-23,
#     2024-10-07, 2025-08-15, 2025-09-15): every one has a best document
#     score of 0.888-0.909, so 0.85 keeps >=1 accepted document in each and
#     flips no EXPLAIN/refuse decision - it only trims lower-signal documents
#     out of the evidence set handed to the LLM.
# Caveat: 20 documents labeled by one judge is illustrative calibration, not
# a statistically robust one - checked against real labeled examples, not a
# validated production threshold. One-line reversible; revisit if the
# labeled set grows or is re-judged.
SEMANTIC_THRESHOLD = 0.85

# Kept for backward compatibility with any caller still importing the old
# single-threshold name; not used by gate_documents() below.
DEFAULT_THRESHOLD = KEYWORD_THRESHOLD

# Below this, a document isn't even worth counting toward "we have some
# evidence, just weak" - it's treated as noise. Prevents a pile of
# near-zero-score documents from looking like "several pieces of weak
# evidence" when they're really just retrieval noise. Applied to whichever
# score is higher, same as before.
MINIMUM_SCORE_TO_COUNT = 0.05


def _effective_score(doc: dict) -> float:
    """
    Used only for ranking/reporting (best_score, sort order) - NOT for the
    accept/reject decision itself anymore (see _passes_threshold). A
    document may have a keyword retrieval_score (Week 2, always present)
    and/or a semantic_score (Week 3). Reports whichever is higher.
    """
    kw = doc.get("retrieval_score", 0.0) or 0.0
    sem = doc.get("semantic_score", 0.0) or 0.0
    return max(kw, sem)


def _passes_threshold(doc: dict, keyword_threshold: float = KEYWORD_THRESHOLD,
                       semantic_threshold: float = SEMANTIC_THRESHOLD,
                       scoring_mode: str = None) -> bool:
    """
    Accept/reject decision: a document passes if either score clears its
    own threshold - not if the higher of the two clears a single shared
    bar. See the module notes for why a shared bar was wrong (semantic
    scores run high enough that a keyword-calibrated bar accepted
    everything). A document with no semantic_score (semantic scoring
    unavailable - see pipeline.py) is judged on its keyword score alone.

    scoring_mode overrides the module-level SCORING_MODE:
      "dual"          — accept if EITHER score clears its bar (default)
      "semantic_only" — ignore keyword scores
      "keyword_only"  — ignore semantic scores
    """
    mode = (scoring_mode or SCORING_MODE).strip().lower()

    kw = doc.get("retrieval_score", 0.0) or 0.0
    sem = doc.get("semantic_score")

    if mode == "keyword_only":
        return kw >= keyword_threshold
    elif mode == "semantic_only":
        return sem is not None and sem >= semantic_threshold
    else:  # "dual" (default)
        if kw >= keyword_threshold:
            return True
        if sem is not None and sem >= semantic_threshold:
            return True
        return False


@dataclass
class GateResult:
    decision: str  # "EXPLAIN" or "INSUFFICIENT_EVIDENCE"
    reason: str
    best_score: float
    accepted_documents: list = field(default_factory=list)
    all_scored_documents: list = field(default_factory=list)
    keyword_threshold_used: float = KEYWORD_THRESHOLD
    semantic_threshold_used: float = SEMANTIC_THRESHOLD
    evaluated_at_utc: str = ""

    def to_dict(self):
        return {
            "decision": self.decision,
            "reason": self.reason,
            "best_score": self.best_score,
            "accepted_document_count": len(self.accepted_documents),
            "accepted_document_ids": [d.get("document_id") for d in self.accepted_documents],
            "total_documents_considered": len(self.all_scored_documents),
            "keyword_threshold_used": self.keyword_threshold_used,
            "semantic_threshold_used": self.semantic_threshold_used,
            "evaluated_at_utc": self.evaluated_at_utc,
        }


def gate_documents(documents: list, keyword_threshold: float = KEYWORD_THRESHOLD,
                    semantic_threshold: float = SEMANTIC_THRESHOLD,
                    scoring_mode: str = None) -> GateResult:
    """
    Core gate logic. Three distinct rejection paths, kept distinct in the
    reason string (not just a single "no" flag) because they're different
    failure modes that should be reported differently downstream:
      - "no documents retrieved" = retrieval infrastructure problem
        (see docs/gdelt_failure_policy_DRAFT.md) - GDELT/fetch failed, not
        a relevance judgment at all.
      - "documents retrieved but all below threshold" = the gate doing its
        actual job: real evidence existed and was judged too weak.

    Acceptance is per-document, per-score (see _passes_threshold) - a
    document passes if its keyword score clears keyword_threshold OR its
    semantic score clears semantic_threshold, not by comparing a single
    combined number against one shared bar. best_score (for reporting/
    ranking only) is still the max of the two scores, via _effective_score.
    """
    now = datetime.now(timezone.utc).isoformat()

    if not documents:
        return GateResult(
            decision="INSUFFICIENT_EVIDENCE",
            reason="no_documents_retrieved",
            best_score=0.0,
            accepted_documents=[],
            all_scored_documents=[],
            keyword_threshold_used=keyword_threshold,
            semantic_threshold_used=semantic_threshold,
            evaluated_at_utc=now,
        )

    ranked = [(_effective_score(d), d) for d in documents]
    ranked = [(s, d) for s, d in ranked if s >= MINIMUM_SCORE_TO_COUNT]
    ranked.sort(key=lambda x: x[0], reverse=True)

    if not ranked:
        return GateResult(
            decision="INSUFFICIENT_EVIDENCE",
            reason="all_documents_below_noise_floor",
            best_score=0.0,
            accepted_documents=[],
            all_scored_documents=documents,
            keyword_threshold_used=keyword_threshold,
            semantic_threshold_used=semantic_threshold,
            evaluated_at_utc=now,
        )

    best_score = ranked[0][0]
    accepted = [d for s, d in ranked if _passes_threshold(d, keyword_threshold, semantic_threshold, scoring_mode)]

    if not accepted:
        return GateResult(
            decision="INSUFFICIENT_EVIDENCE",
            reason="documents_retrieved_but_below_relevance_threshold",
            best_score=best_score,
            accepted_documents=[],
            all_scored_documents=documents,
            keyword_threshold_used=keyword_threshold,
            semantic_threshold_used=semantic_threshold,
            evaluated_at_utc=now,
        )

    return GateResult(
        decision="EXPLAIN",
        reason=f"{len(accepted)}_document(s)_above_threshold",
        best_score=best_score,
        accepted_documents=accepted,
        all_scored_documents=documents,
        keyword_threshold_used=keyword_threshold,
        semantic_threshold_used=semantic_threshold,
        evaluated_at_utc=now,
    )


def _self_test():
    """
    The four required test cases from the team plan's Week 6 task (strong
    evidence, weak evidence, irrelevant evidence, no documents), plus two
    that exercise the semantic side of the gate - the original four only set
    retrieval_score, so they pass identically whether or not
    SEMANTIC_THRESHOLD exists.

    The two semantic cases' semantic_score numbers are illustrative, chosen
    to sit on either side of SEMANTIC_THRESHOLD to check the split logic;
    they are not measured from a real embedding run (unlike the 0.84-0.90
    numbers in the module notes, which are).
    """
    cases = {
        "strong evidence (keyword)": [
            {"document_id": "a", "title": "Coffee spikes on Brazil frost", "retrieval_score": 0.85},
            {"document_id": "b", "title": "El Nino threatens coffee harvest", "retrieval_score": 0.6},
        ],
        "weak evidence (keyword)": [
            {"document_id": "c", "title": "Global coffee output hits record highs", "retrieval_score": 0.3},
        ],
        "irrelevant evidence (keyword)": [
            {"document_id": "d", "title": "Ilkley Food and Drink Festival 2026", "retrieval_score": 0.0},
            {"document_id": "e", "title": "Mercadona ice cream makers", "retrieval_score": 0.02},
        ],
        "strong evidence (semantic only, non-English)": [
            {"document_id": "f", "title": "Café dispara com geada no Brasil",
             "retrieval_score": 0.0, "semantic_score": 0.88},
        ],
        "loosely-related evidence (semantic only, below the higher bar)": [
            {"document_id": "g", "title": "Global food prices tick up amid weather worries",
             "retrieval_score": 0.0, "semantic_score": 0.60},
        ],
        "no documents": [],
    }
    print(f"Relevance gate self-test (keyword_threshold={KEYWORD_THRESHOLD}, "
          f"semantic_threshold={SEMANTIC_THRESHOLD}):\n")
    for label, docs in cases.items():
        result = gate_documents(docs)
        print(f"  {label}: decision={result.decision}  reason={result.reason}  "
              f"best_score={result.best_score:.2f}  accepted={len(result.accepted_documents)}")


if __name__ == "__main__":
    _self_test()
