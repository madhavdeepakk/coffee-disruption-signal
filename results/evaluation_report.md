# Pipeline Evaluation Report

**30** stored pipeline runs. Rates are shown with 95% Wilson intervals; at this sample size the intervals are wide and the point estimates should not be quoted without them.

> **Mixed pipeline versions**: {'2': 6, '1': 24}. Runs from different versions were retrieved, gated or prompted differently and are not directly comparable. Re-run the older dates before quoting a combined figure.

Models that answered: not recorded (18), openai/gpt-oss-120b (7), gemini-3.6-flash (5).

2 run(s) are on dates the current detections file does not flag as anomalies (2026-08-28, 2026-09-11). They were run on request and are included, but they are not detected events.

In 66.7% (20/30; 95% CI 48.8-80.8%) of runs at least one news-API query failed after retries (in 2, none was answered). Those runs searched with fewer queries than intended: 55.0% (11/20; 95% CI 34.2-74.2%) of them ended in a refusal, against 50.0% (5/10; 95% CI 23.7-76.3%) of the rest. A refusal on such a run may reflect the API rather than the news; `python -m scripts.run_eval_set --run --retry-incomplete` re-asks only the failed queries.

## 1. Outcomes

Every run falls in exactly one tier.

| Outcome tier | Runs |
|---|---|
| REFUSED_WEAK_EVIDENCE | 10 |
| EXPLAINED_STRONG | 8 |
| REFUSED_CONFLICTING | 6 |
| EXPLAINED_TENTATIVE | 5 |
| SYSTEM_ERROR | 1 |
| **Total** | **30** |

5 run(s) were saved before the outcome tier existed and are classified here from their stored signals.

- Explained: 43.3% (13/30; 95% CI 27.4-60.8%)
- Refused: 53.3% (16/30; 95% CI 36.1-69.8%)
- Faults (model or retrieval unavailable): 3.3% (1/30; 95% CI 0.6-16.7%)
- Explained, faults excluded: 44.8% (13/29; 95% CI 28.4-62.5%)

An explanation rate is not a quality measure on its own: a system that explained every date would score 100% here. Section 2 is the one that checks decisions against ground truth.

## 2. Decision accuracy on the labelled set

18 dates carry a hand label (EXPLAIN: a documented cause exists; REFUSE: no clean cause, the system should decline). Graded: 16. Ended in a fault: 1. No run yet: 1.

| | As run | With direction guard applied |
|---|---|---|
| Correct decision | 62.5% (10/16; 95% CI 38.6-81.5%) | 75.0% (12/16; 95% CI 50.5-89.8%) |
| Explained when it should | 69.2% (9/13; 95% CI 42.4-87.3%) | 69.2% (9/13; 95% CI 42.4-87.3%) |
| Refused when it should | 33.3% (1/3; 95% CI 6.1-79.2%) | 100.0% (3/3; 95% CI 43.8-100.0%) |

Decisions that disagree with the label:

| Date | Label | System | Evidence direction (with / against) | Model |
|---|---|---|---|---|
| 2020-03-19 | EXPLAIN | REFUSED | 1 / 0 | not recorded |
| 2021-07-19 | REFUSE | EXPLAINED | 2 / 13 | openai/gpt-oss-120b |
| 2023-09-20 | REFUSE | EXPLAINED | 0 / 5 | openai/gpt-oss-120b |
| 2023-11-30 | EXPLAIN | REFUSED | 1 / 1 | openai/gpt-oss-120b |
| 2024-04-30 | EXPLAIN | REFUSED | 1 / 13 | openai/gpt-oss-120b |
| 2024-09-23 | EXPLAIN | REFUSED | 9 / 2 | gemini-3.6-flash |

**2 date(s) labelled REFUSE were explained.** For a system whose claim is that it declines rather than invent a cause, this is the number that matters most, and the sample behind it (3 graded REFUSE labels) is too small to bound the true rate.

The "with direction guard" column re-scores the same stored runs as if the guard had been on. The guard was designed after looking at these runs, so its improvement here is in-sample and is not evidence it generalises - that needs dates it was not designed on (see results/robustness_report.md).

Caveats: labels are single-annotator; a correct EXPLAIN decision means the system chose to explain, not that its text names the documented cause.

## 3. Citations

- Citations checked: 33 across 10 explained run(s) (3 explained run(s) have no citation report).
- Cited document was in the evidence shown to the model: 100.0% (33/33; 95% CI 89.6-100.0%)
- Claim-to-document similarity: minimum 0.795, median 0.859; flagged weak: 0.

What this establishes: the model did not invent source ids. What it does not: that each cited document supports the claim attached to it. The similarity score cannot fail in practice (any two coffee-market texts score 0.75-0.95 with this embedding model, against a 0.60 threshold), so a 0% weak-support rate carries no information.

No human citation audit has been recorded yet. Run `python -m scripts.audit_citations --export`, label the sheet, then `--score` to add a measured support rate here.

## 4. Evidence

- Relevance gate passed: 96.7% (29/30; 95% CI 83.3-99.4%)
- Documents retrieved per run: 61.0 on average; runs with none: 0
- Accepted documents, over the 18 run(s) that saved their sources: 188, of which 159 are distinct stories. Copies of another accepted document: 15.4% (29/188; 95% CI 11.0-21.3%)
- Accepted documents dated after the anomaly date: 14.9% (28/188; 95% CI 10.5-20.7%)
- Per explained run: 13.8 accepted documents, 9.5 distinct stories, 9.6 domains (over 8 explained run(s) that saved their sources)
- Explained from a single distinct story: 0.0% (0/8; 95% CI 0.0-32.4%)

The gate passes 96.7% of dates, so in these runs it does almost no filtering at the date level; the refusals come from the model's own judgement, not from the gate.

Documents dated after the anomaly were used as evidence in these runs even though the design excludes them. Pipeline version 2 filters them locally; these runs predate that.

Domain counts overstate independence when the same wire story appears on several sites; distinct stories is the better measure of how many sources an explanation rests on.

## 5. Evidence direction (heuristic)

Of 13 explained runs, 10 have a direction tally. Accepted evidence leaned the same way as the move in 80.0% (8/10; 95% CI 49.0-94.3%).

Explained despite evidence leaning against the move: 2021-07-19 (2 with / 13 against), 2023-09-20 (0 with / 5 against).

This measures the retrieved documents with a word list (src/rag/direction.py). It is not a check that the explanation is correct.

## 6. By year

| Year | Explained | Refused | Fault |
|---|---|---|---|
| 2018 | 0 | 5 | 0 |
| 2020 | 0 | 1 | 0 |
| 2021 | 5 | 1 | 0 |
| 2022 | 1 | 0 | 0 |
| 2023 | 2 | 1 | 0 |
| 2024 | 1 | 2 | 1 |
| 2025 | 3 | 0 | 0 |
| 2026 | 1 | 6 | 0 |

Dates the answering model could have seen in training are a weaker test of retrieval than dates after its knowledge cutoff. That split needs the cutoff of each model used and is reported by src/evaluation/robustness.py.
