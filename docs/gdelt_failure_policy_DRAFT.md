# GDELT Live-Failure Policy (DRAFT — needs Madhav's sign-off, then goes in rag_design.md)

Problem: DOC 2.0 has failed 1 of 3 queries even after 3 retries in 2
separate live runs (Week 1 and Week 2). At daily-automation scale (Yashika's
Week 8 scheduled job), some days will have zero usable GDELT response for a
flagged anomaly. We need a decided, written behavior for that rather than an
implicit assumption.

Proposed policy:

1. If all retry attempts fail for every query constructed for a given
   anomaly, the retriever returns zero documents for that anomaly — same
   as a real "no evidence found" result — and this is logged distinctly
   from "we found documents but they were irrelevant" (different failure
   mode, should not be conflated in reporting/evaluation).
2. Per the relevance-gating principle already in the plan, zero documents
   retrieved routes straight to INSUFFICIENT EVIDENCE, same downstream
   path as weak/irrelevant evidence. No special-casing needed at the
   explanation-generation stage — the gate already handles "not enough to
   work with."
3. The daily scheduled job (Yashika, Week 8) should NOT hard-fail or block
   on a GDELT outage for one day's anomaly. It logs the failure, the
   anomaly still gets flagged on the dashboard, evidence panel shows
   "retrieval failed" (distinct from "insufficient evidence found" — one
   is an infrastructure problem, the other is the safety mechanism working
   as designed) and moves on.
4. This failure rate and its distinction from genuine insufficient-evidence
   cases should be reported as a named number in Week 8's evaluation
   ("insufficient-evidence rate") — a retrieval infrastructure failure
   inflating that rate is a different finding than the relevance gate
   correctly rejecting weak evidence, and conflating them would misstate
   what the gate is actually catching.

Open decision for Madhav: does "retrieval failed" need its own visible
dashboard state distinct from "insufficient evidence," or is folding it into
the same UI state acceptable with the distinction only tracked in
logs/evaluation data? This affects Week 7's dashboard work, not urgent now.
