# Pipeline Evaluation Report

**75** stored pipeline runs. Rates are shown with 95% Wilson intervals; at this sample size the intervals are wide and the point estimates should not be quoted without them.

> **Mixed pipeline versions**: {'3': 51, '1': 8, '11': 16}. Runs from different versions were retrieved, gated or prompted differently and are not directly comparable. Re-run the older dates before quoting a combined figure.

Models that answered: openai/gpt-oss-120b (47), qwen/qwen3.8-27b (16), not recorded (12).

More than one model answered, and they are not interchangeable. The model's own decision (before the direction guard), by model: openai/gpt-oss-120b explained 40.4% (19/47; 95% CI 27.6-54.7%); qwen/qwen3.8-27b explained 50.0% (8/16; 95% CI 28.0-72.0%). The models did not see the same dates, so this is not a like-for-like comparison, but a pooled rate over a mixed set partly measures which model happened to answer. Set EXPLAINER_PROVIDERS to one provider for a run that is meant to be compared with another.

5 stored run(s) are on dates whose flagged move was not a real market move of that size - the price series switched futures contract that day (data/labeling/contract_switch_dates.csv) - and are left out of everything below: 2018-05-21, 2020-03-19, 2021-03-22, 2023-03-22, 2023-09-20.

2 run(s) are on dates the current detections file does not flag as anomalies (2026-08-28, 2026-09-11). They were run on request and are included, but they are not detected events.

In 70.7% (53/75; 95% CI 59.6-79.8%) of runs at least one news-API query failed after retries (in 1, none was answered). Those runs searched with fewer queries than intended: 81.1% (43/53; 95% CI 68.6-89.4%) of them ended in a refusal, against 50.0% (11/22; 95% CI 30.7-69.3%) of the rest. A refusal on such a run may reflect the API rather than the news; `python -m scripts.run_eval_set --run --retry-incomplete` re-asks only the failed queries.

## 1. Outcomes

Every run falls in exactly one tier.

| Outcome tier | Runs |
|---|---|
| REFUSED_WEAK_EVIDENCE | 29 |
| REFUSED_CONFLICTING | 25 |
| EXPLAINED_TENTATIVE | 11 |
| EXPLAINED_STRONG | 10 |
| **Total** | **75** |

2 run(s) were saved before the outcome tier existed and are classified here from their stored signals.

- Explained: 28.0% (21/75; 95% CI 19.1-39.0%)
- Refused: 72.0% (54/75; 95% CI 61.0-80.9%)
- Faults (model or retrieval unavailable): 0.0% (0/75; 95% CI 0.0-4.9%)
- Explained, faults excluded: 28.0% (21/75; 95% CI 19.1-39.0%)

An explanation rate is not a quality measure on its own: a system that explained every date would score 100% here. Section 2 is the one that checks decisions against ground truth.

## 2. Decision accuracy on the labelled set

16 dates carry a label (EXPLAIN: a documented cause exists; REFUSE: no clean cause, the system should decline). Graded: 16. Ended in a fault: 0. No run yet: 0.

**Explaining every date would score 100.0% (16/16; 95% CI 80.6-100.0%) here.** The "correct decision" row below is worth something only where it beats that.

**No graded date in this answer key is one where refusing is right.** So this section measures one thing only: how often the system fails to explain a move that has a documented cause. Whether it refuses when it should is not measured here at all; that is what the invented-move test is for (results/robustness_report.md).

| | As run | Model alone | Guard: more against than with | Guard: a tie also withholds |
|---|---|---|---|---|
| Correct decision | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) |
| Explained when it should | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) |
| Refused when it should | n/a (0 cases) | n/a (0 cases) | n/a (0 cases) | n/a (0 cases) |

16 of these runs were decided by the blind-evidence rule, which has no direction guard: for them all four columns show the decision as run. The guard columns only differ for runs made by the earlier path.

"As run" is the decision each stored run ended with, under whatever version produced it. The other three columns are computed from the same stored runs: the model's own answer, and that answer after each guard rule, so they are comparable across versions. The pipeline uses the "more against than with" rule.

Decisions that disagree with the label:

| Date | Label | System | Evidence direction (with / against) | Model |
|---|---|---|---|---|
| 2023-01-31 | EXPLAIN | REFUSED | 0 / 0 | openai/gpt-oss-120b |
| 2024-04-30 | EXPLAIN | REFUSED | 0 / 0 | openai/gpt-oss-120b |
| 2024-09-23 | EXPLAIN | REFUSED | 0 / 0 | openai/gpt-oss-120b |
| 2024-12-02 | EXPLAIN | REFUSED | 0 / 0 | openai/gpt-oss-120b |
| 2025-04-07 | EXPLAIN | REFUSED | 0 / 0 | openai/gpt-oss-120b |
| 2025-08-15 | EXPLAIN | REFUSED | 0 / 0 | openai/gpt-oss-120b |

The guard and both of its rules were chosen after looking at runs on these labelled dates, so the guard columns are in-sample and are not evidence the guard generalises - that needs inputs it was not designed on (see results/robustness_report.md).

Caveats: each label rests on one round of research that nobody has checked independently, and rows whose notes start with CHECK are borderline; the labels were made after the system had been run on these dates, from market reports and without sight of its answers; a correct EXPLAIN decision means the system chose to explain, not that its text names the documented cause.

## 3. Citations

- Citations checked: 52 across 21 explained run(s) (0 explained run(s) have no citation report).
- Cited document was in the evidence shown to the model: 100.0% (52/52; 95% CI 93.1-100.0%)
- Claim-to-document similarity: minimum 0.78, median 0.852; flagged weak: 0.
- Cited document not among the closest to its claim: 9.5% (4/42; 95% CI 3.8-22.1%)

What this establishes: the model did not invent source ids. What it does not: that each cited document supports the claim attached to it. The similarity score cannot fail in practice (any two coffee-market texts score 0.75-0.95 with this embedding model, against a 0.60 threshold), so a 0% weak-support rate carries no information.

No human citation audit has been recorded yet. Run `python -m scripts.audit_citations --export`, label the sheet, then `--score` to add a measured support rate here.

## 4. Evidence

- Relevance gate passed: 94.7% (71/75; 95% CI 87.1-97.9%)
- Documents retrieved per run: 199.0 on average; runs with none: 0
- Accepted documents, over the 63 run(s) that saved their sources: 2838, of which 2832 are distinct stories. Copies of another accepted document: 0.2% (6/2838; 95% CI 0.1-0.5%)
- Accepted documents dated after the anomaly date: 0.0% (0/2838; 95% CI 0.0-0.1%)
- Per explained run: 15.4 accepted documents, 86.8 distinct stories, 60.3 domains (over 19 explained run(s) that saved their sources)
- Explained from a single distinct story: 5.3% (1/19; 95% CI 0.9-24.6%)

The gate passes 94.7% of dates, so in these runs it does almost no filtering at the date level; the refusals come from the model's own judgement, not from the gate.

Domain counts overstate independence when the same wire story appears on several sites; distinct stories is the better measure of how many sources an explanation rests on.

## 5. Evidence direction (heuristic)

Of 21 explained runs, 21 have a direction tally. Accepted evidence leaned the same way as the move in 90.5% (19/21; 95% CI 71.1-97.3%).

Explained despite evidence leaning against the move: 2022-11-15 (1 with / 1 against), 2023-06-08 (2 with / 2 against).

This measures the retrieved documents with a word list (src/rag/direction.py). It is not a check that the explanation is correct.

### Runs decided by the blind-evidence rule

16 run(s). The model read the recent documents without being told the move: it returned a reading for 100.0% (161/161; 95% CI 97.7-100.0%) of the documents it was shown, and of the reasons it reported, the quote was found in the document for 97.1% (68/70; 95% CI 90.2-99.2%). A reason whose quote is not found is discarded before the rule runs.

| Move | Runs | Explained |
|---|---|---|
| down | 6 | 50.0% (3/6; 95% CI 18.8-81.2%) |
| up | 10 | 70.0% (7/10; 95% CI 39.7-89.2%) |

Direction test on every explained date: had prices moved the opposite way, the same readings would still have produced an explanation for 0.0% (0/10; 95% CI 0.0-27.8%). That can only happen when the deciding evidence is split evenly, and such explanations are marked tentative. Under the earlier design, where the model was told the move, this was the test it failed.

Where the refused dates stopped:

| Stopped at | Refused dates |
|---|---|
| Nothing recent was found | 0.0% (0/6; 95% CI 0.0-39.0%) |
| Recent articles found, none relevant enough to read | 0.0% (0/6; 95% CI 0.0-39.0%) |
| Recent articles read, none gave a reason with a quote | 33.3% (2/6; 95% CI 9.7-70.0%) |
| A reason was found, for the opposite move | 66.7% (4/6; 95% CI 30.0-90.3%) |

The first two rows are search and filtering: no change to the reading or the rule can explain a date for which nothing recent and relevant was retrieved.

Dates labelled as explainable that were refused:

| Date | Stopped at | Retrieved | Of those recent | Relevant enough to read |
|---|---|---|---|---|
| 2023-01-31 | Recent articles read, none gave a reason with a quote | 595 | 397 | 138 |
| 2024-04-30 | A reason was found, for the opposite move | 611 | 432 | 163 |
| 2024-09-23 | Recent articles read, none gave a reason with a quote | 491 | 348 | 158 |
| 2024-12-02 | A reason was found, for the opposite move | 585 | 434 | 273 |
| 2025-04-07 | A reason was found, for the opposite move | 591 | 418 | 184 |
| 2025-08-15 | A reason was found, for the opposite move | 634 | 331 | 142 |

The same stored readings under other settings of the rule (no model is called again). The first row is the pipeline's setting. If the rows differ a lot, the setting is doing the work and the result should not be trusted; "pooled" counts every recent reading equally, the shape of rule that made the earlier guard refuse most down days.

| Rule | Recent means | Explained, all runs | Explained when it should | Refused when it should |
|---|---|---|---|---|
| freshest | same day or up to 2 trading days before | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | n/a (0 cases) |
| freshest | same day or up to 1 trading day before | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | n/a (0 cases) |
| freshest | same day or up to 3 trading days before | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | n/a (0 cases) |
| pooled | same day or up to 2 trading days before | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | n/a (0 cases) |

Other versions of what counts as evidence, scored the same way from the same readings. A version is only worth adopting if it raises the last column without lowering the one before it. With this few labelled dates a difference of one date is not evidence.

| What counts as evidence | Explained, all runs | Explained when it should | Refused when it should |
|---|---|---|---|
| As the pipeline runs | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | n/a (0 cases) |
| Only documents that cleared the relevance filter itself | 43.8% (7/16; 95% CI 23.1-66.8%) | 43.8% (7/16; 95% CI 23.1-66.8%) | n/a (0 cases) |
| Ignore reports of a move under a quarter the size of the actual one | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | n/a (0 cases) |
| Both of the last two | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | n/a (0 cases) |
| Only articles that report a price move, not background events (added after seeing these dates) | 62.5% (10/16; 95% CI 38.6-81.5%) | 62.5% (10/16; 95% CI 38.6-81.5%) | n/a (0 cases) |
| Counting reports of older and forecast moves too (the rule before version 9) | 75.0% (12/16; 95% CI 50.5-89.8%) | 75.0% (12/16; 95% CI 50.5-89.8%) | n/a (0 cases) |
| Without the size check (a report need not describe a move this large) | 75.0% (12/16; 95% CI 50.5-89.8%) | 75.0% (12/16; 95% CI 50.5-89.8%) | n/a (0 cases) |
| Counting trading reasons too ("technical adjustments") | 75.0% (12/16; 95% CI 50.5-89.8%) | 75.0% (12/16; 95% CI 50.5-89.8%) | n/a (0 cases) |
| Counting reports about another market too (robusta, local prices) | 68.8% (11/16; 95% CI 44.4-85.8%) | 68.8% (11/16; 95% CI 44.4-85.8%) | n/a (0 cases) |
| Without all three (the rule before version 11) | 93.8% (15/16; 95% CI 71.7-98.9%) | 93.8% (15/16; 95% CI 71.7-98.9%) | n/a (0 cases) |

Recent documents read although they scored just under the relevance filter's bar: 125, of which 22 gave a reason with a quote that was found. The "only documents that cleared the filter" row above is the result without them.

Which search found the documents. "Read candidates" are documents relevant enough to be read; "gave a reason" are those whose reading counted as evidence; the last two columns are documents an explanation rests on and the number of dates they decided. A document found by two searches is credited to the one that returned it first.

| Search | Read candidates | Gave a reason | In an explanation | Dates |
|---|---|---|---|---|
| combined search, recent days | 1030 | 15 | 9 | 8 |
| combined search, full window | 870 | 12 | 8 | 6 |
| combined search, day of the move | 481 | 7 | 4 | 4 |
| news feed | 203 | 1 | 1 | 1 |
| coffee Brazil drought | 16 | 0 | 0 | 0 |
| coffee Brazil frost | 3 | 0 | 0 | 0 |
| coffee Minas Gerais | 1 | 0 | 0 | 0 |
| coffee export Vietnam | 12 | 0 | 0 | 0 |
| coffee futures | 12 | 0 | 0 | 0 |
| coffee harvest rain | 10 | 0 | 0 | 0 |
| coffee prices arabica | 1 | 0 | 0 | 0 |
| coffee prices settle | 6 | 0 | 0 | 0 |
| coffee supply shipping | 16 | 0 | 0 | 0 |
| coffee tariff | 4 | 0 | 0 | 0 |

Across every run, the direction guard withholds 42.1% (8/19; 95% CI 23.1-63.7%) of the explanations the model wrote. By direction of the move:

| Move | Runs | Model explained | Guard withholds | Explained after the guard |
|---|---|---|---|---|
| down | 29 | 12 | 66.7% (8/12; 95% CI 39.1-86.2%) | 13.8% (4/29; 95% CI 5.5-30.6%) |
| up | 30 | 7 | 0.0% (0/7; 95% CI 0.0-35.4%) | 23.3% (7/30; 95% CI 11.8-40.9%) |

If the guard withholds far more on one direction than the other, it is not only catching unsupported explanations: it is also reporting which way the retrieved news leans. The GDELT queries are about supply shocks (frost, drought, tariffs, shipping), which are stories about prices rising, so on a day prices fell the tally starts against the move. A should-refuse label set made only of days in the direction the guard already refuses cannot show that the guard works.

## 6. By year

| Year | Explained | Refused | Fault |
|---|---|---|---|
| 2018 | 2 | 7 | 0 |
| 2019 | 1 | 11 | 0 |
| 2020 | 1 | 10 | 0 |
| 2021 | 8 | 4 | 0 |
| 2022 | 3 | 8 | 0 |
| 2023 | 2 | 3 | 0 |
| 2024 | 1 | 3 | 0 |
| 2025 | 2 | 2 | 0 |
| 2026 | 1 | 6 | 0 |

Dates the answering model could have seen in training are a weaker test of retrieval than dates after its knowledge cutoff. That split needs the cutoff of each model used and is reported by src/evaluation/robustness.py.
