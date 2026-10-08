# Evaluation

One place for what has been measured, what it shows, and what has not been measured yet. Every figure comes from a script named in the last section and a report in this folder. Rates carry 95% Wilson intervals.

## The claim under test

The system flags unusual coffee-price moves, explains them from cited news, and declines when the evidence is weak, missing or contradictory rather than produce a plausible-sounding cause. So two things have to hold: it explains moves that have a documented cause, and it refuses moves that do not.

## Status of these numbers

The 26 stored pipeline runs were made before pipeline version 2 (local window filter, duplicate collapsing, token-budgeted prompts, prompt v2, temperature 0, direction guard). Sections 1 to 5 describe the system as it was when those runs were made. They are the baseline the changes have to beat, and they are not flattering. Sections 9 and 10 are the evaluations that test the changed system. Section 10 has been run; section 9 only in part.

Pipeline version 3 exists because the first live run of version 2 went wrong. On 2024-09-23, a date labelled EXPLAIN, the model refused on the grounds that the accepted documents were "published in the days prior". The move was on a Monday and the documents were from the Friday before: one trading day, which the prompt presented as "3 days before" next to a rule telling the model to distrust documents from "several days before". Version 3 states each document's age in trading days and treats the same or previous trading day as timely. That is one date and one observation, and the fix has not been seen to work on the model that refused: the re-run of that date was explained, but by the fallback model, because the primary's free quota had run out. Section 9 compares v1 and v2 on the same evidence.

## 1. Decision accuracy on the hand-labelled dates

`evaluation_report.md`, `explanation_backtest.md`. 18 dates are labelled EXPLAIN (documented cause) or REFUSE (no clean cause). 15 are graded, 2 ended in a fault, 1 has no run.

| | As run | Same runs with the direction guard |
|---|---|---|
| Correct decision | 66.7% (10/15; CI 41.7-84.8%) | 80.0% (12/15; CI 54.8-93.0%) |
| Explained when it should | 75.0% (9/12; CI 46.8-91.1%) | 75.0% (9/12) |
| Refused when it should | 33.3% (1/3; CI 6.1-79.2%) | 100.0% (3/3; CI 43.8-100%) |

The misses:

| Date | Label | System | Evidence with / against the move | What happened |
|---|---|---|---|---|
| 2021-07-19 | REFUSE | explained | 2 / 13 | Called the drop "a pull-back after the earlier rally"; no source says so |
| 2023-09-20 | REFUSE | explained | 0 / 5 | Explained from one article published six days earlier |
| 2020-03-19 | EXPLAIN | refused | 1 / 0 | One accepted document of 53 retrieved |
| 2023-11-30 | EXPLAIN | refused | 1 / 1 | Retrieved evidence was about prices falling |
| 2024-04-30 | EXPLAIN | refused | 1 / 13 | Five copies of one article about a record high the week before |

Two kinds of failure. The first two are the system doing the thing it claims not to do. The last three are retrieval not finding the cause, after which refusing was the right response to what was retrieved.

The guard column is in-sample: the guard was written after reading these runs. It shows the rule is consistent with these 15 dates, not that it generalises. With three should-refuse dates, no figure for refusal accuracy is bounded tightly enough to quote without its interval.

A correct EXPLAIN decision means the system chose to explain. Whether the text names the documented cause is read by a person and is not scored.

## 2. Against trivial baselines

`baseline_comparison_report.md`, same 15 dates.

| System | Explain recall | Refusal specificity | Balanced accuracy | Raw accuracy |
|---|---|---|---|---|
| Always explain | 12/12 | 0/3 | 50.0% | 80.0% |
| Gate only | 12/12 | 0/3 | 50.0% | 80.0% |
| Pipeline as run | 9/12 | 1/3 | 54.2% | 66.7% |
| Pipeline with guard (in-sample) | 9/12 | 3/3 | 87.5% | 80.0% |

On raw accuracy the pipeline as run loses to explaining everything, because 12 of the 15 labels are EXPLAIN and it missed three of those. On balanced accuracy it is barely ahead. The relevance gate passes every one of these dates, including all three should-refuse ones: at the level of whole dates the gate filters nothing, and every correct refusal came from the model.

## 3. Citations

`evaluation_report.md`. 33 of 33 citations (CI 89.6-100%) refer to a document the model was shown, so it is not inventing sources.

That is all this establishes. The pipeline's support score is an embedding similarity with a 0.60 threshold, and any two coffee-market texts score 0.75 to 0.95, so it has never flagged anything and cannot in practice. Whether cited documents support their claims is unmeasured. `scripts/audit_citations.py` exports the claims next to the text the model saw, for labelling by hand; once labelled it reports the support rate and how well the automated checks agree with the labels.

## 4. Retrieval and the gate

`retrieval_evaluation.md`, `relevance_gate_calibration.md`. 20 hand-labelled documents from 5 anomaly dates.

- Semantic score AUC 0.81 (CI 0.53-0.91). Keyword score AUC 0.42 (CI 0.24-0.70), no better than chance: most retrieved articles are not in English and the keyword scorer is.
- The semantic threshold of 0.85 has precision 100% (6/6) and recall 50% (6/12) on these documents, but it was chosen on them. Choosing the threshold without one anomaly date and testing on that date gives precision 71.4% (5/7; CI 35.9-91.8%) and recall 41.7% (5/12). The held-out figure is the one to quote.

## 5. Evidence quality in the stored runs

`evaluation_report.md` section 4, over the 14 runs that saved their sources.

- 29 of 182 accepted documents (15.9%; CI 11.3-21.9%) are dated after the anomaly they were used to explain. GDELT returned the day after the requested end of the window (204 of 1,867 cached articles), and nothing checked.
- 36 of 182 (19.8%; CI 14.6-26.2%) are copies of another accepted document: the same wire story on a different site. Explained runs averaged 13.8 accepted documents but 9.5 distinct stories.
- The two fault outcomes (2024-09-23, 2024-12-02) were the fallback provider rejecting prompts of 8,400 and 11,000 tokens built from 24-25 accepted documents, many of them copies.
- For 41 of the 151 trend-type anomalies in the detections file, the direction the pipeline assigned (the sign of that day's move) is the opposite of the multi-day move that was flagged.

All four are fixed in pipeline version 2 and covered by tests.

One thing is not fixed, because it is not the pipeline's to fix. In 18 of the 26 stored runs (69.2%; CI 50.0-83.5%) at least one GDELT query failed after retries, and in one run none was answered. The pipeline carries on with what the other queries and sources returned, so those runs searched with fewer queries than intended. They were refused 8 times in 18 against 3 in 8 for the rest, which at this size shows nothing either way, but it means a refusal can be the API's doing rather than the news being thin. Each output records its failed queries, the evaluation report counts them, and `run_eval_set --run --retry-incomplete` sends only the failed queries again.

Two of the 26 stored runs are on days the current detections file does not flag. 2026-08-28 was run on request on an ordinary day (+1.2%). 2026-09-11 was run when the price file showed 284.25 for that day, a -9.8% move; the re-fetched data shows 313.65 and a -0.5% move. Both were refused, which is the right outcome for a day on which nothing happened, but the second is a reminder that a freshly quoted price can be wrong and an anomaly flagged on it can disappear.

## 6. Anomaly detector

`detector_evaluation.md`. Unchanged by this round of work.

| Commodity | Known events flagged | False flags in quiet control period | Days flagged overall |
|---|---|---|---|
| Coffee | 4/4 | 0 of 61 days | 12.7% (after section 9d) |
| Crude oil | 5/5 | 0 of 63 days | 12.2% |
| Wheat | 4/4 | 0 of 62 days | 11.7% |

Event recall on a handful of events the windows were chosen around is a sanity check, not a precision figure: most flagged days fall outside any labelled window and are unlabelled.

## 7. Prediction experiments: null results

`forecast_evaluation.md`, `hf_forecast_evaluation.md`, `backtest_outlook_report.md`. Walk-forward, against the naive baseline.

| Model | Next day | Next 5 days |
|---|---|---|
| Logistic regression, price features | 50.2% vs 50.7% | 47.0% vs 50.8% |
| Logistic regression, price + weather | 49.4% vs 50.6% | 46.9% vs 51.3% |
| ARIMA(5,1,0) | 46.4% vs 51.5% | 48.7% vs 52.9% |
| Analog outlook (50 dates) | - | 34.0% vs 44.0% random |

Nothing beats its baseline. The dashboard's outlook section rests on the analog model and states this beneath it.

## 8. Operations

`ops_report.md`. 99 logged model calls. Of the 54 explanation attempts, 13 failed (24.1%; CI 14.6-36.9%); other call types log successes only. 52.3% of successful calls were answered by the fallback provider rather than the primary. Explanation prompts had a median of about 3,000 tokens and a maximum of 17,656. Latency was not logged before pipeline version 2.

Half the calls being answered by a different model is an evaluation problem, not only an operations one: 2021-07-19 was refused by the primary and explained by the fallback.

## 9. Robustness evaluation: the control condition

`src/evaluation/robustness.py` measures how often the pipeline explains when refusing is correct: the same evidence with the move's direction reversed, and quiet days with an invented move. It also runs the same model with no documents, to separate what retrieval contributes from what the model recalls from training. It reports the model's own decision and the decision after the direction guard, and can run prompt v1 and v2 on the same evidence.

Only the control condition (real anomaly, its own evidence, cached retrieval) has been run: 29 dates under both prompts, 24 graded each, 5 with nothing in the cache. `robustness_report.md` has the tables. What it shows:

- Prompt v2 changed the model's decision on 3 of 24 dates, each time from explain to refuse. One of the three is a should-refuse date (2023-09-20). On the labelled dates it cost no explanation: 7 of 11 explained under either prompt, and 1 of 4 should-refuse dates refused under v1 against 2 of 4 under v2. That is a difference of one date.
- The guard, under its current rule, leaves the 7 of 11 alone and brings refusals to 4 of 4. Under its first rule, which also withheld on a tie, it cost one of the seven (2023-01-31, a 1-1 tally). The live runs added two more ties withheld on explain-labelled dates (2020-03-19 at 1-1, 2021-05-05 at 3-3). The tie rule was dropped; `evaluation_report.md` scores both rules side by side.
- The guard has a cost the cached control runs did not show and the live runs did. It counts every accepted document in the ten-day window equally, so on a day that moves against the preceding trend most of the evidence is the trend's news. In the live runs it withheld the model's explanation on 2024-04-30 (1 with, 7 against) and 2024-10-07 (2 with, 11 against), both labelled as explainable, exactly as it did on the should-refuse reversal days 2021-07-19 and 2021-07-30. It cannot tell the two kinds apart; counting only same-day or previous-day documents does not separate them either (2 with / 3 against on 2021-07-19, 2 with / 4 against on 2024-10-07).
- 38 of the 48 graded trials were answered by the fallback model, because the primary's free quota ran out part-way. The primary refused all six trials it answered. This run is mostly a measurement of the fallback model.

Of the conditions that test the refusal claim on inputs nobody tuned on, the placebo condition has now been run under the blind design: section 9f. Version 10 explained 5 of 12 invented moves. For the blind design the flip condition is covered from stored readings (section 9e: 1 of 12 explained dates would also have been explained had prices gone the other way). The closed-book condition has not been run. The direction guard of the earlier design was only ever supported by the labelled dates it was shaped on, and every should-refuse label among those was a reversal day, the one case a direction count is built to catch.

## 9a. The 100-date run under version 3, and what it changed

Stopped after 73 of the 100 dates; the figures here are from the first 71 (54 answered by `openai/gpt-oss-120b`, 13 by `qwen/qwen3.8-27b`, 4 refused at the gate). The outputs are kept under `results/archive/pipeline_v3/`; `python -m src.evaluation.metrics --outputs results/archive/pipeline_v3 --name v3` rebuilds their report.

- The model explained 33 of the 67 dates it answered. The guard withheld 13 of those, all on days prices fell: 13 of 17 down-day explanations, 0 of 16 up-day ones. After the guard, 4 of 34 down days and 16 of 37 up days were explained.
- On the 18 labelled dates: the model alone was right on 13, the guarded pipeline on 14 (10 of 14 explainable, 4 of 4 should-refuse), always explaining on 14, and a rule that explains every up day and refuses every down day, reading no news, on 15. All four should-refuse labels are down days, so those labels cannot tell a guard that detects unsupported explanations from one that detects down days.
- 172 of 340 live GDELT queries failed, and 60 of 71 dates lost at least one. The model explained 10 of the 34 dates that lost three or more queries and 23 of the 37 that lost fewer.

Versions 4 to 7 are the response: direction-neutral queries sent over the recent days only, a decision made by a rule over a direction-blind reading instead of by the model and a guard (`src/rag/blind_evidence.py`), and from version 6 a reading that also covers recent documents just under the relevance filter's bar and records the size and timing of the move each document reports. Nothing in this document measures them yet; two dates were run under version 4 as a check that the path works (2020-03-19 and 2021-05-05, both explained from a same-day report whose reason matches the label). The comparisons that will: the flip and placebo conditions of section 9 with `--prompt-versions blind,v2`, the labelled dates, the explained rate by direction of the move, and the same stored readings re-scored at other settings of the rule (all in `evaluation_report.md` once the set has been re-run). The same report scores three stricter versions of what counts as evidence from the stored readings (filter-cleared documents only; no reports of a move older than a day; no reports of a move far smaller than the actual one) and counts, per search, the documents that reached an explanation. None of the three is switched on, and with 18 labelled dates a difference of one date between them is not a reason to switch one on. Two things the labelled dates still cannot show are missing by construction: a should-refuse day on which prices rose, and reversal days labelled by one consistent rule.

## 9b. Why the news searches fail, measured from the run logs

The run logs of 61 dates record every GDELT request and its HTTP status: 764 requests, 149 answered, 615 rejected with 429 (Too Many Requests). No other failure appears. The body of the rejection reads "Please limit requests to one every 5 seconds".

| Situation | Answered |
|---|---|
| 3 seconds after an answered request | 25 of 125 (20%) |
| 5 seconds after a rejection | 53 of 253 (21%) |
| 10 seconds after a rejection | 38 of 200 (19%) |
| First request of a date, a minute or more after the last | 10 of 57 (18%) |

By query the rate runs from 17% to 23% (six queries with 123 to 128 requests each), by year of the anomaly from 15% to 25%. An answered request is followed by another answer 20% of the time, the same as any other, so answers do not come in runs. Nothing the client did moved the rate: not the wait, not the query, not the date range.

Two consequences. The 3-second pause after an answered request was under GDELT's stated limit and was a fault of ours, but it was not what caused the rejections. And with ten requests per date at one answer in five, a 100-date set needs about 5,000 requests; version 7 asks for two per date and keeps asking, no sooner than every 6 seconds, until they are answered or forty in a row are rejected. The cache-filling pass prints its own answer rate by the wait before each request, so whether a longer wait helps on a given network is measured each time and not assumed.

What these numbers do not show is why GDELT rejects at this rate. A limit counted per address that is shared with other users would look like this; so would a limit applied under load whoever is asking.

## 9c. Version 8 on the 18 labelled dates, and what it showed about the labels

Run on 2026-10-04: 18 dates, no faults, 12 explained and 6 refused. Three models answered (11, 6 and 1 dates) because the free daily quota of each ran out in turn, so this is not a one-model result.

Against the original answer key (`explanation_backtest_labels.csv`):

| | Version 3 with guard | Version 8 | Explain up days, refuse down days |
|---|---|---|---|
| Correct decision | 14/18 | 14/18 (77.8%; CI 54.8-91.0%) | 15/18 |
| Explained when it should | 10/14 | 11/14 | 11/14 |
| Refused when it should | 4/4 | 3/4 | 4/4 |

Other figures from the same 18 runs, all from stored readings:

- 57 of 60 reasons the model reported carried a quote found in the document; 28 of 28 citations refer to a document that was shown.
- For 10 of the 12 explained dates the same readings would have refused the opposite move; the other two were even splits (1-1 and 3-3).
- None of the 6 refusals stopped at search or filtering: 5 found a reason for the opposite move, 1 found no quoted reason.
- The four settings of the rule (freshest at 1, 2 or 3 trading days, and pooled) gave the same decisions on every date.
- Without the recent documents read from just under the relevance bar: 9/14 explained, 4/4 refused.
- Of the 28 documents an explanation rests on, 26 came from the two combined GDELT searches, 2 from the news feed, none from the ten separate queries read from the cache.

**The original labels do not survive the written rule.** The 18 dates were re-researched under `LABELLING_RULE.md`, each from dated market reports and without sight of the system's output, and the result is kept in `labels_by_written_rule.csv` beside the original label:

- All four REFUSE labels change. Three have a documented cause in the reports of the day: 2021-07-19 (a sell-off across markets on Delta-variant cases, the dollar up 2.6% against the real), 2021-07-30 (the frost turned out less damaging than feared), 2024-12-02 (the exchange raised margins and the real fell). The fourth, 2023-09-20, is not a real move (below).
- So "4 of 4 refused when it should" under version 3 was four failures to explain, and the set has no date on which refusing is right. It cannot test the refusal claim at all.
- Three rows are marked for a person to check (2020-03-19, 2023-01-31, 2025-08-15).

Against that key, 16 dates are scored: version 8 is right on 10, a rule that explains every date on 16, and "explain up days, refuse down days" on 10. Every miss is a refusal of a date that has a documented cause. This relabelling was done after the run, by a process that did not see the run; both scores are reported and neither replaces the other until a second person has labelled independently.

**Why it refused them.** The reading step is shown twelve recent documents, chosen by relevance to a fixed phrase about supply shocks. Of the 211 documents read in this run, 133 gave no direction at all, and 73 were a headline with no text; among them were a city guide to coffee shops and a product launch. On 2023-11-30 the Reuters report "Coffee prices jump nearly 7% in New York after stocks drawn" was retrieved, was a candidate, and was not among the twelve. On 2021-07-30 a report headed "Coffee tumbles most since 2008 as Brazil frost concerns ease" was read and returned no direction. Reports that led with another commodity were read from their first 600 characters, before coffee was mentioned, and yielded a direction but no reason. On at least five of the six dates an article giving the cause had been retrieved (on 2024-12-02 that is not confirmed); the selection of what to read, and how much of it, is what failed.

**Flagged moves that are not moves.** The price series follows the nearest futures contract. `contract_switch_dates.csv` lists six of the hundred dates where reports of the day show the active contract moving far less than the series: 2018-05-21 (+1.9% against +5.48%), 2021-03-22 (+0.85% against +3.71%), 2023-03-22 (-1.3% against -4.66%), 2023-09-19 (+0.8% against +5.55%), 2023-09-20 (-1.7% against -5.55%), and 2020-03-19 (+4.1% against +7.96%, a real move overstated). The detector treated these as anomalies when version 8 ran. Section 9d is what was done about it.

**One rule added after seeing these dates.** `reports_only` counts only documents that report a price move, not ones that report an event that would bear on prices. It would have refused 2023-09-20 (explained from one article on planted area) and changed nothing else: 15/18 on the original key. It was chosen with that date in view, the date is one of the six above, and it is off.

**Version 9** is the response to "why it refused them": the twelve documents are chosen market reports first, round-ups are shown their sentences about coffee, a document seen before the exchange opened is counted with the session before, and reports of older or forecast moves no longer count as reports of the day (`src/rag/market_report.py`; the README has the detail). It was designed on these 18 dates and checked by replaying its selection on what version 8 retrieved for them, so its score on them will be in-sample. The 24 dates in `dates_to_label.csv`, whose runs nobody has looked at, are the test.

## 9d. The answer key and the detector, corrected

Section 9c found two faults in the test itself. Both are now fixed in the code, and neither needed a new run.

**The answer key.** `labels_by_written_rule.csv` is the key every report scores against by default: 16 EXPLAIN, 2 EXCLUDE, no REFUSE. Version 8 on that key:

| | Version 8 | Explain every date | Explain up days, refuse down days |
|---|---|---|---|
| Explained a move that has a documented cause | 10/16 (62.5%; CI 38.6-81.5%) | 16/16 | 10/16 |

That is the whole table, because the key has no date where refusing is right. Two things follow and neither is comfortable. On these dates the system does worse than a rule that explains everything and no better than one that reads no news. And the score cannot move in the system's favour for refusing: every refusal on these dates is an error. Whether it refuses when it should is measured only by the invented-move test of section 9, which has not been run under the blind design. Until it is, the project's central claim has no evidence either way.

The original key remains available: `python -m src.evaluation.metrics --labels data/labeling/explanation_backtest_labels.csv --name original_key`.

What the new key rests on: one round of research per date, from market reports of the day, made after the system had been run on these dates but without sight of its answers. Nobody has checked it independently. Three rows are marked CHECK (2020-03-19, 2023-01-31, 2025-08-15).

**The detector.** On the six days in `contract_switch_dates.csv` the detector now uses the move the active contract was reported to have made instead of the jump in the series, for the day itself and inside the rolling baselines later days are compared against. At their real size none of the six is an anomaly (z between 0.5 and 1.8 in absolute value, against a threshold of 2).

| | Before | After |
|---|---|---|
| Days flagged, of 2,197 trading days | 283 (12.9%) | 279 (12.7%) |
| Single-day anomalies | 132 | 130 |
| Known events flagged / false flags in the control period | 4 of 4 / 0 of 61 | 4 of 4 / 0 of 61 |

Sixteen days changed status. The six lost their flag, and so did three slow-build flags in the days after 2018-05-21 that existed only because the jump sat inside their ten-day window. Seven borderline days gained a flag or a second reason for one, because a baseline that no longer contains a jump is narrower: four of them are new single-day anomalies (2021-04-26, 2023-10-19, 2023-11-20, 2023-12-11). None of the other 94 sampled dates lost its flag. The evaluation set is those 94; stored runs on the six are left out of every report and counted there.

**What this does not fix.** The list holds days that were checked against a report, and only the hundred sampled dates were checked. Twelve single-day anomalies outside the sample fall between the 15th and 24th of a delivery month (March, May, July, September, December), where a switch can happen: 2018-07-20, 2018-09-19, 2018-09-20, 2018-12-19, 2019-05-21, 2019-07-16, 2019-12-23, 2020-07-22, 2021-05-18, 2022-05-16, 2023-12-20, 2024-05-17. Some of them are probably artefacts. A date window cannot be made the rule: 2021-07-20 and 2021-07-22, the frost, fall inside one and were real moves of the size shown. The proper fix is a series built from the contract in active trading, which the price source does not provide. Two of the six entries are of medium confidence: 2020-03-19 was a real move of about half the size shown, and for 2023-09-19 the active contract's move is inferred from the next day's report.

## 9e. Version 9 on the labelled dates

Run on 2026-10-04, in about 37 minutes: 18 dates, no faults, every date answered by `openai/gpt-oss-120b`. Two of the 18 are the contract-switch days of section 9d and are not graded.

| On the 16 graded dates | Version 8 | Version 9 | Explain every date |
|---|---|---|---|
| Explained a move that has a documented cause | 10/16 (62.5%; CI 38.6-81.5%) | 12/16 (75.0%; CI 50.5-89.8%) | 16/16 |
| Models that answered | three, by quota | one | |
| Documents read that gave no price direction (all 18 dates) | 128 of 206 (62%) | 52 of 167 (31%) | |
| Reported reasons whose quote was found in the document | 57 of 60 | 66 of 66 | |
| Explained dates that would also have been explained had prices gone the other way | 2 of 12 | 1 of 12 | |
| Citations that refer to a document that was shown | 28 of 28 | 24 of 24 | |

Version 8's counts in the last three rows cover all 18 of its runs, including the two that are no longer graded; they were not recomputed. Against the original answer key the 18 version 9 runs score 13 of 18 (version 8: 14 of 18).

**What changed, date by date.** Three dates went from refused to explained, each from the report that version 8 had retrieved and not read: 2021-07-19 ("coffee and sugar retreat amid a general sell-off in oil and shares", the cause in the answer key), 2021-07-30 (both Bloomberg reports, "frost concerns ease") and 2023-11-30 (the Reuters report, "stocks drawn"). One date went the other way: 2024-09-23. Twelve others kept their decision.

**The four still refused.**

| Date | Where it stopped | What was read |
|---|---|---|
| 2023-01-31 | No reason with a quote | Nothing read reports the day's rise with a reason. Two reports of the day before give a direction ("arabica coffee at 1-month high") and no reason |
| 2024-04-30 | No reason with a quote | The day's report ("agricultural commodities fall sharply in New York") was read first. As shown to the model it gives the size of coffee's fall that morning (2.11%) and no reason for it, so the model was right to return none. The reason in the answer key (a stronger dollar, rain forecasts) is in nothing that was retrieved |
| 2024-09-23 | No reason with a quote | Four articles give the drought as the reason, and all four are forecasts of dearer coffee, which version 9 no longer counts as reports of the day's move. With the rule as it was before version 9 the same readings explain the date: 13/16. A report of the day's rise had also been retrieved ("coffee futures continue to rise amid violent weather swings") and was not read: its headline is in Arabic, which the headline scorer did not cover |
| 2024-12-02 | A reason was found, for the opposite move | No report of the fall was retrieved. What was read is coverage of prices at a 50-year high |

So one is the cost of a rule added in version 9, together with a headline in a language the scorer could not read, and three are searches that did not bring back a report of the day with its reason. Each of the two search requests returns at most 250 results and both reach that limit on nearly every date (about 500 documents per date), so the day's own reports compete for places with five or ten days of coverage.

**What 12 against 10 is worth.** Three dates gained and one lost, on 16 dates, is within what chance produces (exact McNemar test, p = 0.63). Version 9 was designed on these dates and checked by replaying its selection on them, so the score is in-sample. Version 8's figure came from three models and version 9's from one, so the two differ in more than the change being tested. The result that does not depend on the count is the mechanism: the specific reports that had been found and left unread are the ones now cited.

**Whether the explanations name the documented cause.** Read informally against the answer key, not scored: nine of the twelve give the key's cause (drought, the frost and its aftermath, the broad sell-off, the cold-front forecast, the stock draw, the rain forecast, low stocks and weather). 2025-02-10 gives "panic buying" and "crop concerns", which is the key's cause in other words, plus "record high price", which is not a cause. 2025-04-07 gives a general market decline, which matches, and rain in Brazil, which the key does not mention. 2025-08-15 gives tariff pressure and a lower harvest where the key has frost, exports and stocks; that label is marked CHECK. Several reasons are in the language of their article and not in English, which the prompt asks for.

**The two days that are not real moves.** 2023-09-20 was refused: the freshest reports that give a reason are two days old and describe prices edging up 0.3-0.4%. Version 8 had explained that day from a background article. 2020-03-19 was explained from "Coffee jumps 4% on supply chain worries", tied one to one with "Coffee prices fall 8pc on coronavirus pandemic"; the 4% in the first headline is the size the contract-switch list gives for the real move.

**A fault this run exposed.** The reading's answer is limited to 3,000 output tokens, and the model's reasoning counts against the limit. On 7 of the 18 dates the answer stopped at the limit, and 29 of the 196 documents shown got no reading (version 8: 5 of 211). The unread ones are always the last in reading order, and 19 of the 29 are documents from the day of the move. None of the four refused dates had an unread document, so the fault did not cause a refusal here. It may have flattered two explanations: 2021-07-19 rests on one report against one and 2021-07-30 on two against one, and each has an unread same-day document. Pipeline version 10 sends the documents an answer left out again, on their own, to the same model, and stores the count per run (`reading_calls`).

**Version 10, not yet run.** Three changes: the re-reading above; a third search request per date, over the day of the move alone, so that the day's reports get 250 places of their own; and Arabic, Korean and Turkish in the headline scorer. Replaying the new scorer on what version 9 retrieved puts the Arabic report second of the twelve on 2024-09-23 and changes the selection on two other dates (2021-07-20, 2025-08-15) by one document each. The last two changes were made with the four refused dates in view. If the score on these 16 dates rises because of them, that shows the changes do what they were built for and nothing about other dates. Whether the day search finds the missing reports cannot be known before it is run: if GDELT never indexed a report of 2023-01-31, 2024-04-30 or 2024-12-02 that gives the reason, no query will return one. All of it has been tested only with stand-ins for the news source and the model.

**What these 18 runs do not show.** Whether the system refuses when it should: section 9f, where the answer for version 10 is that it does not. How version 9 does on dates it was not designed on: the 24 dates of `dates_to_label.csv` and the rest of the 94 have not been run. Whether a second model reads the same documents the same way: `scripts/compare_models.py` has not been run.

## 9f. The refusal test on invented moves (version 10), and the rule that came out of it

`python -m src.evaluation.robustness --providers groq --experiments placebo --n-placebo 12` was run on 4-5 October 2026 under pipeline version 10. Twelve quiet days were drawn with seed 7, spread from 2018 to 2026: days whose own move and ten-day move are both small and that have no flagged day within five trading days. Each was given the size and direction of a real flagged move. On such a day refusing is the only right answer.

All 36 news requests for the twelve days were answered (121 attempts), so no day ran on part of its news; each had between 434 and 645 documents. One model read every day (`openai/gpt-oss-120b`). No answer was cut short and no follow-up read was needed.

**Result: 5 of 12 invented moves were explained (42%, CI 19-68%).** The trial files are in `data/eval_cache/trials/` (`placebo__<date>__blind__groq__live__s7__p10.json`). The report of that run is `results/robustness_report.md` until a later run replaces it, and is then kept under `results/archive/robustness/groq__live__s7__p10/`.

| Quiet day | Invented move | What the explanation rested on |
|---|---|---|
| 2018-11-19 | -4.8% | One report of the day: the market making technical adjustments after the previous session's rise, and swings in the dollar. No size stated |
| 2019-08-27 | +4.6% | One report of the day, about robusta gaining on demand and the dollar. The price series is arabica |
| 2020-08-24 | -5.8% | One report of the day: technical factors predominated |
| 2021-08-20 | +4.3% | One report of the day (arabica with technical adjustments) against one the other way (a broad sell-off in commodities). A tie, explained as tentative |
| 2022-09-06 | +3.9% | Two reports of the day. One states the day's move: up 0.6% |

The refusals are no better evidence than the explanations. Six of the seven were refused because the day's reports pointed the other way from the invented move (2018-08-10, 2020-06-16, 2023-05-09, 2024-10-23, 2025-01-23, 2026-04-14), and one because no recent document gave a reason with a quote that was found (2019-03-22). The direction of an invented move is arbitrary, so whether it agrees with the day's reports is a coin toss. The version 10 rule checks that a recent report gives a reason for a move in this direction. It does not check that the report describes a move like this one, and a market report gives a reason for prices rising or falling on every trading day.

**The stricter rules already in the code do not help.** Scored on the same stored readings, with no new model call:

| Rule | Invented moves explained | Cost on the 16 labelled dates (version 9 readings) |
|---|---|---|
| As version 10 ran | 5 of 12 | 12 of 16 explained |
| Ignore a report whose stated move is under a quarter of the claimed one | 5 of 12 | 12 of 16 |
| Only reports of a price move, not background events | 5 of 12 | 12 of 16 |
| Only documents that cleared the relevance filter itself | 0 of 12 | 8 of 16 |

The size rule changes nothing because it only drops a report that states a size, and most reports of a quiet day state none. The filter-only rule stops every false explanation and a third of the true ones.

**Version 11: a report has to fit the move.** Built from these trials (`src/rag/blind_evidence.py`, `decide`). The reading is asked three more things about each document, still without being told the move or its size: how big the document's own words make the move (large, small or unstated), whether the reason is an event or only a description of trading, and whether the move is of the watched market or only of another one (robusta, a local price). A report then supports an explanation only if it:

- is about this market;
- gives an event as the reason. "Technical adjustments" and "profit-taking" do not count, which is what the labelling rule has said since it was written;
- describes a move of this size: it states a size of at least a quarter of the flagged move, or, stating none, its words call the move large. A stated size is believed over the words around it. The quarter is the threshold the unused size rule already had; no new number was chosen with these results in view.

Background coverage of an event that reports no price move describes no size, so on its own it can no longer explain a one-day shock; in the version 9 run no explanation rested on such coverage. Reports that fail are set aside, counted neither for the move nor against it, and the rule does not go on to older evidence because of them. Reports pointing the other way count as before. A trend anomaly, flagged on a ten-day move, is exempt from the size check. Each set-aside report is stored with the reason (`blind_evidence.set_aside`), and the refusal has its own wording ("recent reports point this way, but none fits a move of this size").

**What is known about version 11 before it is run.** Only what the stored readings allow, and they did not record size in words, so on them the rule can count a size only where it is stated as a number. That makes it stricter than it will be when run:

| | Version 10 rule | Version 11 rule, sizes as numbers only |
|---|---|---|
| Invented moves explained | 5 of 12 | 0 of 12 |
| Labelled dates explained (version 9 readings) | 12 of 16 | 7 of 16 |

The five labelled dates lost in that table are 2021-05-05, 2021-07-30, 2025-04-07, 2025-08-15 and 2025-09-15. On four of them the supporting report describes a large move in words or in a form the old prompt did not ask for ("tumbles most since 2008", "rises more than 3%", "falling 3-5%"), so the new reading may recover them. 2025-08-15 will probably stay lost: its only supporting report gives a move of 1.1% on a day the series shows +4.6%. So the honest expectation is somewhere between 7 and 11 of 16 explained, against 12 now, in exchange for most false explanations. Neither number is a measurement.

**What would confirm it.** The rule was written with these 12 quiet days in view, so refusing them again proves nothing. Two runs do: the 16 labelled dates under version 11 (`python -m scripts.run_eval_set --fill-cache --run --max 16 --labelled-first --providers groq`), and quiet days the rule was not built on (`python -m src.evaluation.robustness --providers groq --experiments placebo --n-placebo 12 --seed 11`). The report of the second prints the rule with and without the three checks on the same readings.

**Limits of the test itself.** 12 days and one model: the interval on 5 of 12 runs from about a fifth to about two thirds. A quiet day is the easy case for a size check, because its reports describe an ordinary day. The harder case is a real large move that the news of the day does not explain; the two should-refuse dates among the unchecked labels (2018-11-30, 2019-11-11) are of that kind and have not been run.

## 10. Ablation

`ablation_study_report.md`, 26 dates with cached retrieval. Every configuration gates the same scored documents, so only the named component differs. No model is called; this measures the gate.

| Configuration | Dates passing the gate | Accepted per date |
|---|---|---|
| Production (keyword >= 0.4 or semantic >= 0.85) | 23/26 | 4.9 |
| Semantic channel only | 23/26 | 4.7 |
| Keyword channel only | 7/26 | 0.4 |
| No gate | 26/26 | 42.7 |
| Semantic threshold 0.80 | 26/26 | 21.8 |
| Semantic threshold 0.90 | 9/26 | 0.6 |

- The keyword channel does nothing. Removing it changes the gate decision on none of the 26 dates. The "dual-threshold" gate is a semantic threshold with a keyword score attached.
- The gate does not decide which dates get explained. It stops 3 of 26, and none of the 4 should-refuse dates. What it does is cut the evidence handed to the model from 43 documents to 5.
- The 0.85 threshold sits on a steep part of the curve: 0.80 lets through four times as many documents, 0.90 almost none. It was chosen on 20 labelled documents (section 4).
- Without the local window filter, 22 of 149 accepted documents (14.8%; CI 10.0-21.3%) are dated after the anomaly; with it, none of 127.
- Without duplicate collapsing, 6.5 documents are accepted per date, of which 5.2 are distinct stories.

These rows cover GDELT documents only, because the live aggregator's results are not reproducible. Re-run after the 100-date set is in the cache.

## What is not measured

- Version 11, on anything. Its rule has only been applied to readings stored by versions 9 and 10 (section 9f).
- Whether the system refuses a real large move that the news does not explain. The invented-move test uses quiet days.
- Whether explanations name the right cause.
- Whether cited documents support their claims (audit sheet ready, not labelled).
- Anything on more than 26 dates. `data/eval/eval_dates.csv` defines a 100-date sample stratified by year and direction.
- Inter-annotator agreement. All labels are one person's.
- Whether the decision depends on which model does the reading. `scripts/compare_models.py` measures it once a second model has read the set; until `model_comparison.md` exists, every version 4 figure is one model's reading.
- The optional second-stage ranking (`--rerank`), which stays off until `evaluate_retrieval.py` can compare it on a labelled sheet that includes its score.

## Reproducing

```
python -m pytest -q

# offline
python -m src.evaluation.batch_runner --existing-only
python -m src.evaluation.metrics
python -m src.evaluation.metrics --labels data/labeling/explanation_backtest_labels.csv --name original_key   # the labels made before the written rule
python -m src.evaluation.baseline_comparison
python -m scripts.calibrate_thresholds
python -m scripts.evaluate_retrieval
python -m scripts.ops_report
python -m scripts.explanation_backtest
python -m scripts.evaluate_rag
python -m scripts.evaluate_detector --all
python -m src.modeling.anomaly_detector          # rewrites results/anomaly_detections.csv

# needs the embedding model
python -m src.evaluation.ablation_study

# needs retrieval and a model key
python -m scripts.run_eval_set --fill-cache                              # news queries only, no model
python -m scripts.run_eval_set --preview-reading --max 18 --labelled-first            # what would be read; no model
python -m scripts.run_eval_set --fill-cache --run --max 18 --labelled-first --providers groq   # labelled dates only
python -m scripts.run_eval_set --run --labelled-first --providers groq
python -m scripts.run_eval_set --run --retry-incomplete --providers groq   # dates that lost news queries
python -m scripts.compare_models --read groq:qwen/qwen3.8-27b              # a second model reads the same evidence
python -m scripts.compare_models --report                                 # results/model_comparison.md
python -m src.evaluation.robustness --providers groq --cutoff 2024-06-30   # cutoff of the answering model

# needs a person
python -m scripts.audit_citations --export      # label, then --score
python -m scripts.collect_labeling_set --n 200  # label, then calibrate_thresholds
```
