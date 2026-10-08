"""
Outcome classification: turn the raw pipeline signals into one labelled
decision with a confidence tier.

Why this module exists
----------------------
The pipeline produces several separate signals - whether retrieval even
worked, the relevance gate's accept/reject, the model's own
explain/refuse, the direction agreement of the evidence, and the citation
faithfulness audit. Historically these collapsed into just two visible
outcomes: "EXPLAINED" or "INSUFFICIENT_EVIDENCE". That hid two things that
matter a great deal for a system whose whole thesis is *honest refusal*:

  1. "I could not retrieve any evidence" (an infrastructure failure - GDELT
     was down / rate-limited) is NOT the same as "I retrieved evidence and
     judged it too weak" (the safety mechanism doing its job). The first is
     an outage; the second is the product working. Conflating them lets a
     flaky news API masquerade as a principled refusal.

  2. A confident, well-cited, direction-consistent explanation and a
     hedged, single-source one both showed up as a flat "EXPLAINED". The
     reader had no way to see which explanations to trust more.

This module is the single source of truth for that classification, so the
CLI pipeline and the dashboard describe an outcome identically. It reads
the signals; it does not recompute them.

Tiers
-----
  EXPLAINED_STRONG      - grounded explanation, high model confidence, no
                          direction conflict, citations verified.
  EXPLAINED_TENTATIVE   - grounded explanation, but hedged: lower
                          confidence, a single source, some direction
                          disagreement, or an unverified citation.
  REFUSED_CONFLICTING   - evidence existed but contradicted itself on the
                          price direction, so no honest causal story could
                          be told. (The strongest demonstration of the
                          refuse-don't-hallucinate design.)
  REFUSED_WEAK_EVIDENCE - evidence was retrieved but too weak/off-topic to
                          support a causal explanation (gate reject, or the
                          model's own second-look refusal).
  REFUSED_NO_EVIDENCE   - retrieval succeeded but genuinely found nothing
                          in the lookahead-safe window. A real "the news is
                          quiet" answer, not an outage.
  RETRIEVAL_FAILED      - retrieval itself could not run (every news query
                          failed, no cache, no fallback). An outage, NOT a
                          refusal and NOT a judgement about the evidence.
  SYSTEM_ERROR          - the model call or its parsing failed. A fault to
                          fix, not a statement about the market.

is_refusal is True only for the four REFUSED_* / no-evidence tiers - the
cases where the system deliberately declined. RETRIEVAL_FAILED and
SYSTEM_ERROR are explicitly NOT refusals; they are faults.
"""

from dataclasses import dataclass

# Tier constants (stable strings; used as keys in the UI and in tests).
EXPLAINED_STRONG = "EXPLAINED_STRONG"
EXPLAINED_TENTATIVE = "EXPLAINED_TENTATIVE"
REFUSED_CONFLICTING = "REFUSED_CONFLICTING"
REFUSED_WEAK_EVIDENCE = "REFUSED_WEAK_EVIDENCE"
REFUSED_NO_EVIDENCE = "REFUSED_NO_EVIDENCE"
RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
SYSTEM_ERROR = "SYSTEM_ERROR"

REFUSAL_TIERS = frozenset({
    REFUSED_CONFLICTING, REFUSED_WEAK_EVIDENCE, REFUSED_NO_EVIDENCE,
})
EXPLAINED_TIERS = frozenset({EXPLAINED_STRONG, EXPLAINED_TENTATIVE})
FAULT_TIERS = frozenset({RETRIEVAL_FAILED, SYSTEM_ERROR})

# Short human labels for each tier (no emojis - professional UI).
TIER_LABEL = {
    EXPLAINED_STRONG: "Explained (strong)",
    EXPLAINED_TENTATIVE: "Explained (tentative)",
    REFUSED_CONFLICTING: "Refused - evidence conflicts on direction",
    REFUSED_WEAK_EVIDENCE: "Refused - evidence too weak",
    REFUSED_NO_EVIDENCE: "Refused - no evidence found",
    RETRIEVAL_FAILED: "Could not retrieve evidence (retrieval outage)",
    SYSTEM_ERROR: "System error (model unavailable)",
}


@dataclass
class RetrievalMeta:
    """Health of the retrieval step, independent of relevance.

    Populated by src.pipeline.retrieve_evidence. The distinction this
    captures - did retrieval *run*? - is what separates RETRIEVAL_FAILED
    (an outage) from REFUSED_NO_EVIDENCE (retrieval ran and the window was
    genuinely quiet)."""
    queries_attempted: int = 0
    network_attempts: int = 0      # queries that actually hit the network (not served from cache)
    network_failures: int = 0      # of those, how many raised after retries
    network_successes: int = 0     # of those, how many returned a response (even if 0 articles)
    cache_hits: int = 0            # queries served from the on-disk cache
    reused_cache_hits: int = 0     # earlier per-topic answers read from the cache only (never requested)
    fallback_added: int = 0        # articles merged from the local news-fallback file
    documents_returned: int = 0    # unique documents handed to the gate
    dropped_out_of_window: int = 0  # articles a source returned that were dated outside the window
    collapsed_duplicates: int = 0   # syndicated copies folded into another document

    def retrieval_ran(self) -> bool:
        """True if at least one source produced a usable response: a live
        query returned, a cached response was read, or a fallback file
        supplied articles. False means nothing worked - an outage."""
        return (self.network_successes > 0 or self.cache_hits > 0 or self.reused_cache_hits > 0
                or self.fallback_added > 0)

    def to_dict(self) -> dict:
        return {
            "queries_attempted": self.queries_attempted,
            "network_attempts": self.network_attempts,
            "network_failures": self.network_failures,
            "network_successes": self.network_successes,
            "cache_hits": self.cache_hits,
            "reused_cache_hits": self.reused_cache_hits,
            "fallback_added": self.fallback_added,
            "documents_returned": self.documents_returned,
            "dropped_out_of_window": self.dropped_out_of_window,
            "collapsed_duplicates": self.collapsed_duplicates,
            "retrieval_ran": self.retrieval_ran(),
        }


def _as_get(obj, name, default=None):
    """Read `name` from either a dataclass/object or a plain dict. The live
    pipeline passes objects; the dashboard passes the saved to_dict() JSON.
    classify() must handle both."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _gate_considered(gate_result) -> int:
    if gate_result is None:
        return 0
    if isinstance(gate_result, dict):
        return gate_result.get("total_documents_considered", 0) or 0
    return len(getattr(gate_result, "all_scored_documents", []) or [])


def _gate_accepted(gate_result) -> int:
    if gate_result is None:
        return 0
    if isinstance(gate_result, dict):
        return gate_result.get("accepted_document_count", 0) or 0
    return len(getattr(gate_result, "accepted_documents", []) or [])


def _retrieval_ran(retrieval_meta) -> bool:
    if retrieval_meta is None:
        return True  # unknown -> assume it ran, so we never mislabel as an outage
    if isinstance(retrieval_meta, dict):
        if "retrieval_ran" in retrieval_meta:
            return bool(retrieval_meta["retrieval_ran"])
        return bool((retrieval_meta.get("network_successes", 0) or 0)
                    or (retrieval_meta.get("cache_hits", 0) or 0)
                    or (retrieval_meta.get("fallback_added", 0) or 0))
    return retrieval_meta.retrieval_ran()


def _direction_conflict(direction_summary: dict) -> bool:
    """Evidence disagrees with the anomaly's own direction more than it
    agrees - the signature of the REFUSED_CONFLICTING case."""
    if not direction_summary:
        return False
    consistent = direction_summary.get("consistent", 0) or 0
    inconsistent = direction_summary.get("inconsistent", 0) or 0
    # Conflict when contradictory evidence is present and not outweighed by
    # supporting evidence. (>= so a 1-1 split still reads as unresolved.)
    return inconsistent > 0 and inconsistent >= consistent


def direction_conflict(direction_summary: dict) -> bool:
    """Public name for the conflict test the tier classifier uses (a tie
    counts as unresolved: it files a refusal under REFUSED_CONFLICTING and
    keeps an explanation out of EXPLAINED_STRONG)."""
    return _direction_conflict(direction_summary)


# Which rule the direction guard applies. Written into every pipeline output,
# and scripts/run_eval_set.py re-runs an output whose stored decision is not
# the one the current rule gives.
#   "majority": withhold when MORE accepted documents point against the move
#               than with it.
#   "ties":     the first version - withhold on a tie as well.
GUARD_RULE = "majority"
GUARD_RULES = ("majority", "ties")


def guard_fires(direction_summary: dict, rule: str = None) -> bool:
    """Would the direction guard withhold an explanation given this tally of
    accepted documents? `rule` defaults to GUARD_RULE; the evaluation code
    passes "ties" to score the first version next to the current one."""
    rule = rule or GUARD_RULE
    if rule not in GUARD_RULES:
        raise ValueError(f"unknown guard rule {rule!r}; expected one of {GUARD_RULES}")
    if not direction_summary:
        return False
    consistent = direction_summary.get("consistent", 0) or 0
    inconsistent = direction_summary.get("inconsistent", 0) or 0
    if rule == "ties":
        return inconsistent > 0 and inconsistent >= consistent
    return inconsistent > consistent


def apply_direction_guard(explanation_result: dict, direction_summary: dict) -> dict:
    """Withhold an explanation the accepted evidence does not support on
    direction.

    The model is asked to refuse when the evidence points the wrong way, and
    usually does. It does not always: in the stored runs it explained a -3.7%
    day from evidence that was 13-to-2 about prices rising, and a -5.6% day
    from evidence that was 5-to-0 the wrong way. Both dates were labelled
    should-refuse. This check makes that refusal deterministic instead of
    leaving it to whichever model answered.

    The rule (guard_fires): more accepted documents point against the move
    than with it. The first version also withheld on a tie. In the runs made
    after it was written the guard fired on a tie three times (1-1, 1-1 and
    3-3), each time on a date hand-labelled as having a documented cause,
    while every firing that matched a should-refuse label had a wide margin
    (8-2, 11-2, 4-0, 10-2, 26-5). A tie in a word-count over a handful of
    headlines is not evidence against the move, so ties no longer withhold;
    src/evaluation/metrics.py scores both rules on the same stored runs.

    If explanation_result is an explanation AND guard_fires() holds,
    returns a NEW dict whose decision is INSUFFICIENT_EVIDENCE (so the
    classifier files it under REFUSED_CONFLICTING), with the model's original
    text preserved under "withheld_explanation". Otherwise returns the input
    object unchanged - callers can test `result is original` to see whether
    the guard fired.

    Known costs. The direction signal is a word-list heuristic
    (src/rag/direction.py); when it misreads the evidence this turns a good
    explanation into a refusal. And it counts every accepted document in the
    ten-day window equally, so on a reversal day inside a rally most of the
    evidence is the rally's news and the guard withholds whatever the model
    wrote, including an explanation built on same-day reports of the
    reversal. Both are the project's stated preference (refuse rather than
    explain without support) and both are measured by
    src/evaluation/metrics.py rather than assumed to be zero.
    """
    if not explanation_result or explanation_result.get("decision") != "EXPLAINED":
        return explanation_result
    if not guard_fires(direction_summary or {}):
        return explanation_result
    consistent = (direction_summary or {}).get("consistent", 0) or 0
    inconsistent = (direction_summary or {}).get("inconsistent", 0) or 0
    guarded = {k: v for k, v in explanation_result.items()
               if k not in ("explanation", "citations", "confidence", "decision", "reason")}
    guarded.update({
        "decision": "INSUFFICIENT_EVIDENCE",
        "explanation": None,
        "citations": [],
        "confidence": explanation_result.get("confidence"),
        "reason": (f"direction_guard: {inconsistent} accepted document(s) pointed against "
                   f"the move vs {consistent} with it"),
        "withheld_explanation": {
            "explanation": explanation_result.get("explanation"),
            "citations": explanation_result.get("citations", []),
            "confidence": explanation_result.get("confidence"),
        },
    })
    return guarded


def _faithfulness_ok(faithfulness_report) -> bool:
    """True when the citation audit found no unfaithful citation. None =
    not run (no citations to check) = treated as neutral/ok."""
    if not faithfulness_report:
        return True
    # faithfulness.check_citations reports per-citation results; a report is
    # 'ok' if it flags nothing as unfaithful/unresolved.
    if isinstance(faithfulness_report, dict):
        # src.rag.faithfulness.check_citations report shape: an audit is clean
        # when every cited id resolves to a retrieved document and none are
        # flagged missing. n_weak_support is a softer signal and does not by
        # itself sink the audit.
        if "citations_resolve" in faithfulness_report:
            return (bool(faithfulness_report.get("citations_resolve"))
                    and (faithfulness_report.get("n_missing_document", 0) == 0))
    return True


def _citations_flagged(faithfulness_report) -> int:
    """How many citations the audit flagged as weakly supported or as ranked
    far from their claim. These do not sink the audit, but an explanation
    carrying one is not called strong."""
    if not isinstance(faithfulness_report, dict):
        return 0
    return ((faithfulness_report.get("n_weak_support", 0) or 0)
            + (faithfulness_report.get("n_low_rank", 0) or 0))


@dataclass
class Outcome:
    tier: str
    label: str
    headline: str          # one-line, UI-ready
    detail: str            # 1-2 sentences explaining *why* this tier
    is_refusal: bool
    is_fault: bool

    def to_dict(self) -> dict:
        return {
            "tier": self.tier, "label": self.label, "headline": self.headline,
            "detail": self.detail, "is_refusal": self.is_refusal,
            "is_fault": self.is_fault,
        }


def classify(*, retrieval_meta, gate_result, explanation_result,
             direction_summary=None, faithfulness_report=None) -> Outcome:
    """Map the pipeline's signals to one tier. Precedence matters:

    1. System faults (model/parse error) first - they mask everything.
    2. Retrieval outage next - "we couldn't look" outranks any judgement
       about evidence, because there was no evidence to judge.
    3. Gate refusals (evidence retrieved but rejected).
    4. Model's own refusal (gate passed, model declined) - split into a
       direction-conflict case and a generic weak case.
    5. Explanation delivered - strong vs tentative.
    """
    direction_summary = direction_summary or {}
    exp_decision = (explanation_result or {}).get("decision")

    # 1. System fault - not a market statement, a bug/outage to fix.
    if exp_decision in ("API_ERROR", "PARSE_ERROR"):
        return Outcome(
            tier=SYSTEM_ERROR, label=TIER_LABEL[SYSTEM_ERROR],
            headline="The explanation model was unavailable.",
            detail="The retrieval and gating ran, but the language model call "
                   "or its response could not be completed. This is a system "
                   "fault, not a judgement about the market - retry when the "
                   "model is available.",
            is_refusal=False, is_fault=True,
        )

    # 2. Retrieval outage - nothing to judge because nothing could be fetched.
    #    Only when the gate itself saw zero documents AND retrieval never ran.
    docs_considered = _gate_considered(gate_result)
    retrieval_ran = _retrieval_ran(retrieval_meta)
    if docs_considered == 0 and not retrieval_ran:
        na = _as_get(retrieval_meta, "network_attempts", 0) or 0
        nf = _as_get(retrieval_meta, "network_failures", 0) or 0
        return Outcome(
            tier=RETRIEVAL_FAILED, label=TIER_LABEL[RETRIEVAL_FAILED],
            headline="Evidence could not be retrieved - the news source did not respond.",
            detail=f"Every news query failed to return ({nf} of {na} live "
                   "attempts errored, no cached or fallback articles were "
                   "available). This is a retrieval outage, not a refusal: the "
                   "system had nothing to evaluate. It says nothing about "
                   "whether news for this date exists.",
            is_refusal=False, is_fault=True,
        )

    gate_decision = _as_get(gate_result, "decision", None)
    gate_reason = _as_get(gate_result, "reason", "") or ""

    # In blind mode recent documents just under the gate's bar are read too, so
    # a run can have a reading - and a decision by the rule - without the gate
    # having passed anything. The gate's verdict only stands when nothing was
    # read.
    reading_was_made = ("blind_evidence" in (explanation_result or {})
                        or str((explanation_result or {}).get("reason") or "")
                        .startswith("blind_evidence:"))

    # 3. Gate rejected the retrieved evidence.
    if gate_decision != "EXPLAIN" and not reading_was_made:
        if gate_reason in ("no_documents_retrieved", "all_documents_below_noise_floor"):
            return Outcome(
                tier=REFUSED_NO_EVIDENCE, label=TIER_LABEL[REFUSED_NO_EVIDENCE],
                headline="Retrieval ran, but found no on-topic news in the window.",
                detail="News retrieval succeeded and returned nothing relevant "
                       "in the lookahead-safe window before this date. This is a "
                       "genuine 'the news was quiet' result - the system declines "
                       "to invent a cause rather than reach for an unrelated story.",
                is_refusal=True, is_fault=False,
            )
        # documents_retrieved_but_below_relevance_threshold
        return Outcome(
            tier=REFUSED_WEAK_EVIDENCE, label=TIER_LABEL[REFUSED_WEAK_EVIDENCE],
            headline="Evidence was retrieved but judged too weak to explain the move.",
            detail=f"{docs_considered} document(s) were retrieved, but none "
                   "cleared the relevance gate. The system refuses rather than "
                   "build a confident-sounding explanation on weak or off-topic "
                   "material.",
            is_refusal=True, is_fault=False,
        )

    # Gate passed. Look at what the model did with the accepted evidence.
    accepted_n = _gate_accepted(gate_result)

    # 4a. Refusal by the blind-evidence rule (src/rag/blind_evidence.py): the
    #     model read the recent documents without knowing the move, and the
    #     rule found nothing recent that gives a reason for this direction.
    blind_reason = str((explanation_result or {}).get("reason") or "")
    if exp_decision == "INSUFFICIENT_EVIDENCE" and blind_reason.startswith("blind_evidence:"):
        c = direction_summary.get("consistent", 0) or 0
        i = direction_summary.get("inconsistent", 0) or 0
        if i > 0:
            return Outcome(
                tier=REFUSED_CONFLICTING, label=TIER_LABEL[REFUSED_CONFLICTING],
                headline="The most recent evidence points the other way.",
                detail=f"Of the newest documents that give a reason for a price move, "
                       f"{i} point against this move and {c} with it. An explanation "
                       "resting on older or less direct material is not offered.",
                is_refusal=True, is_fault=False,
            )
        if "no_report_gives_an_event_as_the_reason_for_a_move_of_this_size" in blind_reason:
            return Outcome(
                tier=REFUSED_WEAK_EVIDENCE, label=TIER_LABEL[REFUSED_WEAK_EVIDENCE],
                headline="Recent reports point this way, but none fits a move of this size.",
                detail="The newest reports say prices moved in this direction. None of them "
                       "both gives an event as the reason and describes a move this large: "
                       "they report an ordinary day, or give only trading reasons such as "
                       "technical adjustments. That does not explain an unusual move.",
                is_refusal=True, is_fault=False,
            )
        return Outcome(
            tier=REFUSED_WEAK_EVIDENCE, label=TIER_LABEL[REFUSED_WEAK_EVIDENCE],
            headline="No recent document gives a reason for a move in this direction.",
            detail="Documents passed the relevance gate, but none published on the day "
                   "or in the two trading days before it states a reason, backed by a "
                   "quote found in the document, for prices moving this way.",
            is_refusal=True, is_fault=False,
        )

    # 4. Model's own second-look refusal (defence in depth beyond the gate).
    if exp_decision == "INSUFFICIENT_EVIDENCE":
        if _direction_conflict(direction_summary):
            c = direction_summary.get("consistent", 0)
            i = direction_summary.get("inconsistent", 0)
            return Outcome(
                tier=REFUSED_CONFLICTING, label=TIER_LABEL[REFUSED_CONFLICTING],
                headline="Evidence contradicted itself on price direction - no honest cause could be given.",
                detail=f"The accepted articles disagreed with the actual move "
                       f"({i} pointed the opposite way, {c} agreed), so no single "
                       "causal story is supported. The system declines rather than "
                       "cherry-pick the articles that happen to fit - the clearest "
                       "case of refusing to hallucinate a reason.",
                is_refusal=True, is_fault=False,
            )
        return Outcome(
            tier=REFUSED_WEAK_EVIDENCE, label=TIER_LABEL[REFUSED_WEAK_EVIDENCE],
            headline="Evidence passed the gate but read as inconclusive on a close look.",
            detail="The relevance gate accepted the evidence on score, but the "
                   "model judged it too vague or non-causal to support a real "
                   "explanation and declined. This is the second, closer-reading "
                   "safety check doing its job.",
            is_refusal=True, is_fault=False,
        )

    # 5. Explanation delivered. Strong vs tentative.
    confidence = str((explanation_result or {}).get("confidence") or "").lower()
    conflict = _direction_conflict(direction_summary)
    faithful = _faithfulness_ok(faithfulness_report)
    flagged = _citations_flagged(faithfulness_report)
    strong = (confidence == "high" and not conflict and faithful and accepted_n >= 2
              and flagged == 0)

    if "blind_evidence" in (explanation_result or {}):
        # "high" from the rule already means two supporting documents and none
        # against; how many documents the gate itself accepted is beside the point.
        strong = confidence == "high" and not conflict and faithful and flagged == 0
        # Explained by the blind-evidence rule: the tiers mean how much recent,
        # quoted evidence the explanation rests on.
        n_with = direction_summary.get("consistent", 0) or 0
        n_against = direction_summary.get("inconsistent", 0) or 0
        if strong:
            return Outcome(
                tier=EXPLAINED_STRONG, label=TIER_LABEL[EXPLAINED_STRONG],
                headline="Explained from recent, quoted evidence.",
                detail=f"{n_with} recent documents each give a reason for a move in this "
                       "direction, each reason is backed by a quote found in the document, "
                       "and no equally recent document points the other way.",
                is_refusal=False, is_fault=False,
            )
        reasons = []
        if n_with < 2:
            reasons.append("it rests on a single document")
        if n_against:
            reasons.append(f"{n_against} equally recent document(s) point the other way")
        if not faithful:
            reasons.append("a citation could not be fully verified")
        elif flagged:
            reasons.append(f"{flagged} citation(s) were flagged as weakly matched to their source")
        why = "; ".join(reasons) if reasons else "the evidence is thinner than the strong bar requires"
        return Outcome(
            tier=EXPLAINED_TENTATIVE, label=TIER_LABEL[EXPLAINED_TENTATIVE],
            headline="Explained, but treat as tentative.",
            detail=f"A recent document gives a quoted reason for a move in this direction, "
                   f"but {why}.",
            is_refusal=False, is_fault=False,
        )

    if strong:
        # "Verified" here means what the audit can actually establish: each
        # citation points to a document the model was shown and none was
        # flagged. It does not mean a person confirmed the document supports
        # the claim - see scripts/audit_citations.py for that.
        return Outcome(
            tier=EXPLAINED_STRONG, label=TIER_LABEL[EXPLAINED_STRONG],
            headline="Explained, with strong, direction-consistent evidence.",
            detail=f"{accepted_n} sources cleared the gate, the evidence leaned the "
                   "same way as the move, the model reported high confidence, and "
                   "every citation points to a document the model was shown.",
            is_refusal=False, is_fault=False,
        )

    reasons = []
    if confidence and confidence != "high":
        reasons.append(f"model confidence was {confidence}")
    if accepted_n < 2:
        reasons.append("only a single source cleared the gate")
    if conflict:
        reasons.append("some evidence disagreed on direction")
    if not faithful:
        reasons.append("a citation could not be fully verified")
    elif flagged:
        reasons.append(f"{flagged} citation(s) were flagged as weakly matched to their source")
    why = "; ".join(reasons) if reasons else "the evidence was thinner than the strong bar requires"
    return Outcome(
        tier=EXPLAINED_TENTATIVE, label=TIER_LABEL[EXPLAINED_TENTATIVE],
        headline="Explained, but treat as tentative.",
        detail=f"An explanation is supported by the cited evidence, but "
               f"{why}. Weigh it accordingly.",
        is_refusal=False, is_fault=False,
    )
