# Coffee Disruption Signal & Explanation System

Detects unusual moves in coffee futures, retrieves the news from the days before each move, and either explains the move with cited evidence or declines to.

The interesting part is the declining. A language model handed a pile of coffee news will write a fluent cause for any price move, real or not. This project is about measuring how often a retrieval pipeline does that, and what it takes to stop it.

## How it works

```
Price data (Yahoo Finance)
        |
        v
Anomaly detection      rolling z-score on daily returns, plus a 10-day
                       cumulative check for moves that build slowly
        |  flagged dates
        v
Evidence retrieval     GDELT + Google News + industry feeds, limited to the
                       10 days up to the move; out-of-window articles and
                       syndicated copies removed
        |  scored documents
        v
Relevance gate         keyword score >= 0.4 OR multilingual semantic score >= 0.85
        |  accepted documents
        v
Blind reading          the model reads the recent documents WITHOUT being told
                       which way prices moved, and reports for each one the
                       direction it describes, the reason it gives, and a quote
        |  readings; a reason counts only if its quote is found in the document
        v
Decision rule          explain only if the newest, most direct evidence gives a
                       reason for a move in the direction prices actually went,
                       in a report that fits the move (version 11)
        |
        v
Explanation and        one line per source: headline, outlet, date, the move
checks                 it reports, the reason, the quote; citation audit,
                       outcome tier
```

An explanation reads like this (pipeline version 11; the headline and quote are the article's own words, the reason is the reading's summary of it):

```
+7.1% day-over-day move (z-score 3.50) on 2023-11-30. Reason given by 1 report from that day:
- "<headline>" (<outlet>, 2023-11-30). Reported move: up 7%. Reason: stocks drawn. Quote: "<the words that state the reason>" (source: <document id>)
```

The model never decides whether to explain. It reads; a fixed rule decides. The earlier design, in which the model was told the move and chose to explain or decline, is kept as `--decision legacy` so the two can be run on the same evidence.

Every run ends in exactly one outcome: explained (strong or tentative), refused (evidence conflicting, weak, or absent), or a fault (retrieval or model unavailable). A fault is never reported as a refusal.

## What the evidence says

All figures are from `results/`, each produced by a script listed under [Reproducing](#reproducing). Intervals are 95% Wilson intervals. The samples are small and the intervals say so.

| Question | Result | Source |
|---|---|---|
| Does the detector flag known events and stay quiet otherwise? | Coffee 4/4 events, crude oil 5/5, wheat 4/4; no flags in each commodity's ~60-day quiet control period | `detector_evaluation.md` |
| Does it explain a move that has a documented cause? | Version 9: 12 of 16 labelled dates (75%, CI 51-90%), one model throughout. Version 8: 10 of 16. In-sample: version 9 was designed on these dates | `EVALUATION.md` sections 9c, 9e |
| Does it refuse when it should? | **No.** Version 10, given an invented move on a quiet day, explained 5 of 12 (42%, CI 19-68%). Version 11 adds a rule built from those failures and has not been run | `EVALUATION.md` section 9f |
| Is it better than always explaining? | No, on these dates: 12 of 16 against 16 of 16. With no should-refuse date in the key, always explaining cannot lose | `EVALUATION.md` sections 9d, 9e |
| Are cited sources real? | 24 of 24 citations in the version 9 runs point to a document the model was shown (CI 86-100%); every one of the 66 reasons the model reported carried a quote found in its document | `EVALUATION.md` section 9e |
| Do cited sources support the claims? | Not yet measured. The automated score cannot fail; a human audit sheet is in place | `scripts/audit_citations.py` |
| Does the semantic score separate relevant from irrelevant documents? | AUC 0.81 (CI 0.53-0.91) on 20 labelled documents; the keyword score is no better than chance (0.42, CI 0.24-0.70) | `retrieval_evaluation.md` |
| Does the gate threshold hold up out of sample? | Precision 100% in-sample, 71% held-out (5/7, CI 36-92%) | `relevance_gate_calibration.md` |
| Can price direction be predicted? | No. Logistic regression 50.2% vs 50.7% baseline; ARIMA 46.4% vs 51.5%; analog outlook 34% vs 44% random | `forecast_evaluation.md`, `backtest_outlook_report.md` |

The headline is rows two to four. On the answer key as re-checked, version 9 failed to explain 4 of 16 moves that market reports of the day explain. And when version 10 was given a move that never happened, it explained 5 of 12: it refused when the invented direction happened to disagree with that day's market reports and explained when it agreed, which is a coin toss and not the claim the project makes. Version 11 requires a report that fits the move (this market, an event as the reason, a move of this size); applied to the stored readings it refuses all 12 and keeps at least 7 of the 12 real explanations, and it has not been run. The step from 10 to 12 is three dates gained and one lost, on the dates the change was designed on; it is not evidence of an improvement that would hold elsewhere. Earlier versions of this table reported 1 of 3 should-refuse dates refused; those labels did not survive the written labelling rule (`results/EVALUATION.md`, section 9c). The sections below are the history of how the system got here, in order.

## What auditing the stored runs turned up

These were found by reading the pipeline's own outputs, not by adding features.

**Evidence from after the move.** The retrieval window is sent to GDELT as a start and end time. GDELT appears to apply the end bound at day granularity, so asking for "up to midnight after the move" returned the whole following day: 204 of 1,867 cached articles (11%) are dated after the requested end, and 29 of 182 accepted documents (16%) in the stored runs postdate the anomaly they were used to explain. The window is now enforced locally.

**The same story counted many times.** De-duplication was by URL, so a wire story on ten sites was ten documents. 36 of 182 accepted documents (20%) were copies of another accepted document. That inflated source counts, skewed the direction tally, and pushed two dates over the fallback model's request limit, which is what their "system error" outcomes were. Copies are now collapsed before text is fetched, and evidence is packed to each provider's limit.

**Wrong direction for slow-building anomalies.** A "trend" anomaly is flagged on its 10-day cumulative move, but the pipeline described it by that day's move and took its direction from that day's sign. For 41 of the 151 trend anomalies the two disagree, so the model was asked to explain a move in the wrong direction.

**Explanations built on the opposite evidence.** On 2021-07-19 (-3.7%) the accepted evidence was 13 documents about prices rising to 2 about them falling; on 2023-09-20 (-5.6%) it was 5 to 0. The fallback model explained both. One explanation's only causal claim was that the drop was "a short-term pull-back after the earlier rally", which appears in no source. The primary model had refused 2021-07-19 in an earlier run, so the outcome depended on which provider happened to answer.

Three changes came out of that: explanations are generated at temperature 0 on the models that support it (Google advises leaving Gemini 3 at its default, so those runs are not exactly repeatable), the prompt gained two rules (kept alongside the original as `v1`/`v2` so both can be run on the same evidence), and a deterministic **direction guard** withholds any explanation when more accepted documents point against the move than with it. Applied to the stored runs the guard takes refusals on should-refuse dates from 1 of 3 to 3 of 3 without changing any other decision. It was designed after looking at those runs, so that is an in-sample result and is reported as one.

**What the first runs of the changed pipeline turned up.** Three more things, none of them flattering.

- The guard's first rule also withheld on a tie. It did that three times (tallies of 1-1, 1-1 and 3-3), each time on a date labelled as having a documented cause, while every firing that matched a should-refuse label had a wide margin. A tie in a word count over a few headlines is not evidence against a move, so ties no longer withhold. The evaluation report scores the model alone and both guard rules on the same stored runs.
- The answering model decides more than the prompt does. Run on the same evidence, the two prompt versions disagreed on 3 of 24 dates. The two models differ far more: in these runs the primary refused every date it answered, and the fallback explained most of what it was shown. In the pipeline runs on labelled dates the fallback explained every should-refuse date it answered, so each correct refusal there came from the guard and not from the model. When the primary's free quota runs out mid-run, a different system is being evaluated, and the report now splits outcomes by model.
- The guard counts every accepted document in the ten-day window equally. On a reversal day inside a rally most of the evidence is the rally's news, so the guard withholds whatever the model wrote, including an explanation that cites same-day reports of the reversal. Every should-refuse label in the set is a day of that kind, which is why the guard scores well on them and why that score says little about any other kind of day.

**Version 4: taking the decision away from the model.** The 100-date run under version 3 showed the guard for what it was. It withheld 13 of the 17 explanations the model wrote for days prices fell, and none of the 16 for days prices rose. The six news queries were all supply shocks (frost, drought, tariffs, shipping), which are stories about prices rising, so on a down day the tally started against the move. Every should-refuse label was a down day, which is why the guard scored 4 of 4 on them, and a rule that reads no news at all, "explain up days, refuse down days", scored 15 of 18 on the labelled dates against the guarded pipeline's 14.

Two changes follow. Four queries were added that are not tied to a direction (market coverage either way, the daily market report, rain and harvest), and they search only the day of the move and the five days before it: a query over the full ten-day window spends most of its twenty results on days too old for the decision to use. And the decision moved to `src/rag/blind_evidence.py`: the model is shown the recent documents without the move and reports what each one says; a reason counts only if the quote the model gives for it is found in the document; then the newest, most direct group of evidence decides (newest day first, and within a day a document that reports a price move before one that only reports events). Since the reading is made without the direction, the same documents cannot be read as supporting both a rise and a fall. That is the property the flip test checks, and it holds by construction here instead of depending on the model's judgement.

What this does not settle: whether the model reads documents accurately, and whether a quiet day's market report, which always offers some reason, is enough to "explain" a move that did not really happen. The placebo test measures the second. Neither has been run yet under version 4.

**Version 6: reading more of what matters, and recording enough to judge the rule.** Not yet run on the evaluation set; nothing below is a result.

- The four recent-window searches ask for 50 results each instead of 20. The decision can only use what was retrieved, and a date with nothing recent retrieved is refused whatever the reading does.
- A recent document is read if it clears the relevance filter or scores just under it (0.80 on the semantic score against the filter's 0.85). The filter's bar was set to keep off-topic text out of an explanation a model wrote freely. The blind reading is a filter of its own: a document that says nothing about coffee prices gets no direction and no reason and counts for nothing. Every reading records which kind of document it came from, so the result without the wider net is one row in the report, not a guess.
- The reading separates the market price from the price in a shop, and records two more things per document: the size of the price move it reports, and whether that move happened on the day, the day before, earlier, or is a forecast.
- Three stricter rules use those fields: count only documents that cleared the filter itself; ignore a report of a move from more than a day before the article; ignore a report of a move under a quarter the size of the actual one. None is switched on. `src/evaluation/metrics.py` scores each from the stored readings, against the same labels, and one is worth adopting only if it refuses more of what should be refused without explaining less of what should be explained.
- Every document records which search returned it, and the report counts, per search, the documents that ended up in an explanation. A search that never gets there is a request per date for nothing.
- Article text is downloaded only for documents recent enough to be read. Older ones keep their headline; the decision never looked at them.

**Version 7: two news requests per date instead of ten.** Not yet run on the evaluation set either. The first attempt to run version 6 stopped in the news search: GDELT answered 15 of 91 requests. The logs of the earlier runs say why. Of 764 requests, 149 were answered, and the rate was the same one in five whether the request came 3, 5 or 10 seconds after the last one or minutes later, for every query and every year (`results/EVALUATION.md`, section 9b). Pacing was not the problem, so the number of requests was the thing to cut. GDELT accepts alternatives in one search, `coffee (frost OR drought OR ...)`, and returns up to 250 results for it, so the six full-window queries became one request and the four recent ones another. The separate queries are still read from the cache where an earlier run has their answers, and are never sent again. The pause between requests went from 3 seconds to 6, because GDELT asks for at most one every 5 and the old pause was under that after every answered request.

Two things about this are untested: whether GDELT accepts the combined searches as written (five test requests were all rejected by the rate limit before they could be judged), and whether one broad search finds what six narrow ones did. The cache-filling pass reports a search GDELT refuses to run separately from one it rate-limits, and the report's table of which search found each deciding document shows what the earlier cached answers add over the combined ones.

**Version 8: the first run of version 7 failed, and why.** The combined searches were accepted and 33 of the 36 requests for the labelled dates were answered. The first four dates then all came back refused, among them 2021-07-20, the day of the Brazil frost. The log showed the cause: with 424 documents for that date instead of about a hundred, the relevance model was given every document in one batch, asked for 5.3 GB of memory and failed; the gate fell back to keyword scores alone, nothing passed it, and the run was saved as a refusal without one document being read. The unit tests had not caught it because they replace the model with a stand-in. Two changes: the model now scores sixteen documents at a time (about 200 MB however many there are, and the same vectors, since padding is masked out), and a run in which the model is installed but scores nothing ends in an error and saves nothing. Before this, a failure of the model looked exactly like weak evidence.

**Version 9: reading the right documents.** Version 8 ran on the 18 labelled dates (`results/EVALUATION.md`, section 9c) and refused six that reports of the day explain. On at least five of them an article giving the cause had been retrieved. The reading step is shown twelve documents, and it was shown the twelve most similar to a phrase about supply shocks: 133 of the 211 it read said nothing about a price move. Four changes, all blind to the direction of the move (`src/rag/market_report.py`):

- **Which twelve.** Headlines are scored for how much they look like a report on coffee's market price (a price, a verb of movement, a figure, a word from the exchange), in the dozen languages the documents come in, and market reports are read first, newest session first. Three places are kept for the day's most relevant other documents: the frost and the hurricane are often reported without a word about prices.
- **Which part.** A round-up of several commodities opens with sugar or cocoa. For those the reading is shown the opening sentence and then the sentences about coffee, copied exactly, so the quote check still holds.
- **Which session.** The exchange opens at 08:15 or 09:15 UTC. A document first seen before 08:00 UTC is counted with the session before: the overnight report of yesterday's settlement was being read as evidence about today.
- **Which move.** A report of a move from an earlier day, or a forecast, no longer counts as a report of the day's move, and a report of the previous day's move counts a session older. "Coffee is up 70% this year" was being counted against a one-day fall.

The selection was checked by replaying it on the documents version 8 had retrieved for these dates: on 2023-11-30 the Reuters report that had gone unread is first of the twelve. That is a check on the 18 dates the change was designed on, and no more than that. `python -m scripts.run_eval_set --preview-reading --max 18 --labelled-first` prints what would be read for each date without calling a model.

**Version 9, measured; version 10.** Version 9 ran on the 18 dates with one model answering all of them (`results/EVALUATION.md`, section 9e). Of the 16 graded dates it explained 12, against version 8's 10: the three days whose report had been retrieved and not read (2021-07-19, 2021-07-30, 2023-11-30) are now explained from that report, and 2024-09-23 is lost. That loss is the "which move" rule: the articles giving the drought as the reason there are forecasts of dearer coffee, not reports of the day's rise, and they no longer count. Of the documents the model read, 31% gave no price direction, down from 62%.

The run also showed a fault. The reading's answer is limited to 3,000 output tokens, the model's reasoning counts against that, and on 7 of the 18 dates the answer stopped at the limit: 29 of the 196 documents shown got no reading, 19 of them from the day of the move. None of the four refused dates was affected, but two of the explained ones rest on a one-to-one or two-to-one count that an unread same-day document could change. Version 10 sends the documents an answer left out again, on their own, to the model that answered, and records how many it took (`reading_calls` in each output).

Version 10 also goes after the four dates still refused. On three of them nothing retrieved reports the day's move with a reason: each search request returns at most 250 results, both reach that limit on nearly every date, and the day's reports compete for places with a week of other coverage. A third request now covers the day of the move alone. On the fourth, a report of the day's rise had been retrieved and went unread because its headline is in Arabic; the headline scorer now reads Arabic, Korean and Turkish. Both changes were made with those four dates in view, so a better score on the 16 dates would not be evidence for them, and the day search may find nothing: if GDELT never indexed a report that gives the reason, no query returns one. Version 10 has been run only in the refusal test below, not on the labelled dates.

**Version 11: a report has to fit the move.** The refusal test was run for the first time under the blind design, on version 10: twelve quiet days, each given the size and direction of a real flagged move. It explained 5 of 12. Each false explanation rested on a routine report of that day: three gave technical adjustments or technical factors as the reason, one was about robusta, and one stated the day's move as 0.6% against a claimed 3.9%. The rule asked only whether a recent report gives a reason for a move in this direction, and a market report does that every day. The reading now also records, still without being told the move or its size, how big each document's words make the move, whether the reason is an event or only a description of trading, and whether the move is of this market. A report supports an explanation only if it is about this market, gives an event as the reason, and describes a move of this size: a stated size of at least a quarter of the flagged move, or, with none stated, words that call it large. Applied to the stored readings, where size was recorded only as a number, that refuses all 12 invented moves and keeps 7 of version 9's 12 real explanations; four of the five lost rest on reports that describe a large move in words, so the run may recover them. It was built with those quiet days in view and has not been run (`results/EVALUATION.md`, section 9f).

**The answer key and the detector, corrected.** Two things found while reading version 8's refusals were faults in the test, not in the system.

- All four should-refuse labels failed the written rule (`data/labeling/LABELLING_RULE.md`) when each date was researched again from market reports of the day: three of the moves have a documented cause and one is not a real move. `labels_by_written_rule.csv` is now the answer key every report scores against: 16 EXPLAIN, 2 EXCLUDE, no REFUSE. The original key is kept (`python -m src.evaluation.metrics --labels data/labeling/explanation_backtest_labels.csv --name original_key`).
- Six of the hundred sampled anomalies are days on which the price series switched from an expired futures contract to the next and shows a move no contract made (`data/labeling/contract_switch_dates.csv`, each with the report it rests on). The detector now measures those days by the move the active contract was reported to have made, which puts all six under its threshold. Single-day anomalies go from 132 to 130: the six go, and four borderline days nearby cross the threshold now that the jumps no longer inflate the baseline they are compared with. The evaluation set is the remaining 94 dates, and stored runs on the six are left out of every report.

The list is of days that were checked against a report, not a rule. Twelve single-day anomalies outside the sample fall between the 15th and 24th of a delivery month, where a switch can happen, and have not been checked; a date window cannot be the rule, because the July 2021 frost days fall inside one and were real. A proper fix needs prices per contract, which the price source does not give.

**Several models, one rule.** The free model tiers run out mid-run, and a run answered by whichever model has quota left measures the mix as much as the method. `scripts/compare_models.py` goes the other way: every model reads every date, on exactly the text the pipeline's model was shown, with no model standing in for another, and the same rule decides from each reading. It reports how often the models reach the same decision (with Cohen's kappa, since two models that both refuse most dates agree by default), lists the dates they split on, and adds a consensus reading in which a document's direction counts only if most of the models report it with a quote found in the document.

```bash
python -m scripts.compare_models --list
python -m scripts.compare_models --read groq:qwen/qwen3.8-27b    # resumable; stops when quota runs out
python -m scripts.compare_models --report                       # results/model_comparison.md
```

## Testing the refusal claim properly

An explanation rate cannot show that a system refuses when it should, because a system that explains everything scores 100%. `src/evaluation/robustness.py` builds inputs where refusing is the right answer and counts how often the pipeline explains anyway. Each condition changes one thing and uses the same code path as the CLI.

| Condition | What changes | Right answer |
|---|---|---|
| true | Nothing: real anomaly, its own evidence | explain where supported |
| flip | The model is told the move went the other way | refuse |
| placebo | A quiet day, its real news, an invented move sized like a real anomaly | refuse |
| transplant | Another anomaly's evidence with dates shifted to fit | marks a limit of the design |
| closed_book | The same model with no documents | separates retrieval from recall of training data |

The placebo condition gives a false-explanation rate. The closed-book condition is the memorization control: a model can "explain" the 2021 Brazil frost without retrieving anything, so retrieval only demonstrably matters on dates after the model's knowledge cutoff.

```bash
python -m src.evaluation.robustness --providers groq --experiments placebo --n-placebo 12   # the refusal test alone
python -m src.evaluation.robustness --providers groq --cutoff 2024-06-30
python -m src.evaluation.robustness --providers groq --experiments true,flip --prompt-versions blind,v2 --retrieval cache
```

The first line is the one that tests the project's claim. Run under version 10 it explained 5 of 12 invented moves (`results/EVALUATION.md`, section 9f); under version 11 it has not been run, and has to be run with another `--seed`, because the version 11 rule was built on the days seed 7 draws. It picks quiet days spread across the years (no flagged day within five trading days), fetches their news with the same patient pass the evaluation set uses, and runs a day only once every one of its news requests has been answered: a day run on part of its news has less to build a false explanation from, and a refusal there would flatter the result. Days still waiting are listed in the report and picked up when the command is run again; a larger `--n-placebo` adds days to the ones already done.

Every trial keeps its readings, so the report also shows what stricter versions of the rule would have decided on the same readings, on the invented moves and on the real ones, without calling a model again. What to expect is not a clean pass. A market report is written every day and gives a reason for that day's small move, so when the invented move points the same way, the rule as it stands has a quoted reason for it. The stricter rule that drops a report whose stated move is under a quarter of the claimed one is aimed at exactly that and cost nothing on the labelled dates; whether it is enough is what this run measures.

The second line runs the blind rule next to the earlier model-decides path on the same cached evidence, for the real dates and their flipped versions.

`--providers` keeps one run to one model family. `--cutoff` is that model's knowledge cutoff, from its model card (June 2024 for `openai/gpt-oss-120b`); it splits the closed-book results into dates the model could and could not have seen in training.

This needs live retrieval and a model key, and is rate-limited. Trials are cached, so it resumes if interrupted.

## Quick start

```bash
pip install -r requirements-pipeline.txt
```

Create `.env` in the project root (it is git-ignored):

```
GEMINI_API_KEY=...
GROQ_API_KEY=...        # optional fallback
OPENAI_API_KEY=...      # optional fallback
```

```bash
python -m src.modeling.fetch_price_data --commodity coffee
python -m src.modeling.anomaly_detector --commodity coffee
python -m src.pipeline --date 2024-09-23 --commodity coffee
python -m streamlit run src/dashboard/app.py
```

The dashboard alone needs only `requirements.txt` and reads precomputed results; see `DEPLOY.md` for hosting it read-only.

Useful pipeline flags: `--decision legacy` (the earlier path: the model is told the move and decides, then the direction guard applies; `--prompt-version v1|v2` and `--no-direction-guard` apply to that path), `--rerank` (experimental second-stage ranking, off by default), `--no-cache`.

Set `EXPLAINER_PROVIDERS=groq` (or `gemini`) to pin a run to one model family instead of whichever provider is up.

## Reproducing

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                  # unit tests; no network, no key

# Offline: rebuilt from files in the repository
python -m src.evaluation.batch_runner --existing-only
python -m src.evaluation.metrics                     # results/evaluation_report.md
python -m src.evaluation.baseline_comparison         # results/baseline_comparison_report.md
python -m scripts.calibrate_thresholds               # results/relevance_gate_calibration.md
python -m scripts.evaluate_retrieval                 # results/retrieval_evaluation.md
python -m scripts.ops_report                         # results/ops_report.md
python -m scripts.evaluate_detector --all            # results/detector_evaluation.md

# Needs the embedding model
python -m src.evaluation.ablation_study              # results/ablation_study_report.md

# Needs retrieval and a model key
python -m scripts.run_eval_set --preview-reading --max 18 --labelled-first            # what would be read; no model
python -m scripts.run_eval_set --fill-cache --run --max 18 --labelled-first --providers groq   # the labelled dates only
python -m scripts.run_eval_set --run                 # pipeline over data/eval/eval_dates.csv
python -m scripts.run_eval_set --run --retry-incomplete   # redo dates whose news queries failed
python -m src.evaluation.robustness                  # results/robustness_report.md

# Needs a person
python -m scripts.audit_citations --export           # then label, then --score
python -m scripts.collect_labeling_set --n 200       # then label, then calibrate_thresholds
```

`python -m scripts.run_all_evaluations` runs the offline set in order; add `--live` for the rest.

## Project layout

```
src/
  pipeline.py                 retrieval, then gate -> explain -> guard -> audit -> outcome
  config/commodities.py       per-commodity queries, known events, control periods
  modeling/
    anomaly_detector.py       daily and cumulative z-score detector
    forecast.py, hf_forecast.py, outlook.py   forecasting experiments (null results)
  rag/
    gdelt_client.py, news_aggregator.py, text_fetch.py   retrieval and caching
    evidence.py               window filter, duplicate collapsing, token-budgeted packing
    vector_store.py           multilingual-e5-small scoring on ONNX Runtime
    relevance_gate.py         dual-threshold gate
    direction.py              does a document describe a rise or a fall
    explainer.py              prompts, provider fallback, call logging
    faithfulness.py           citation checks
    outcome.py                outcome tiers and the direction guard
    rerank.py                 optional second-stage ranking
    blind_evidence.py         direction-blind reading, quote check, decision rule
  evaluation/
    metrics.py                rates with intervals, decision accuracy on labels
    baseline_comparison.py    decisions vs always-explain and gate-only
    ablation_study.py         one component changed, same documents
    robustness.py             placebo, flipped-direction and no-documents tests
  advisory/                   market brief and outlook (descriptive, experimental)
  dashboard/app.py            Streamlit viewer
scripts/                      evaluation and labelling tools
tests/                        unit tests
data/labeling/                hand labels: 18 dates, 20 documents
data/eval/eval_dates.csv      100-date stratified evaluation set
results/                      pipeline outputs and evaluation reports
```

## Limits

- **Sample size.** 18 dates have been run under version 9; 16 are graded and none of them is a should-refuse date. The refusal test is a dozen quiet days. The 94-date set in `data/eval/eval_dates.csv` has been run in full only under version 3. No rate here can be quoted without its interval.
- **The refusal claim failed its first test, and the fix is unmeasured.** Version 10 explained 5 of 12 invented moves. Version 11's rule was built on those days and has not been run on the labelled dates or on other quiet days; on stored readings it costs up to five of the twelve real explanations.
- **Version 9's score is in-sample.** Version 9 was designed on the 16 dates it scores 12 of 16 on. The 24 dates in `data/labeling/dates_to_label.csv` are the held-out test, and their proposed labels have not been checked. Versions 10 and 11 have not been run on the labelled dates. Outputs carry a `pipeline_version`; the metrics report warns when versions are mixed.
- **The labels rest on one round of research.** Nobody has checked them independently, three of the 18 are marked CHECK as borderline, and they were made after the system had been run on these dates, though without sight of its answers. The document labels are single-annotator too.
- **A correct decision is not a correct explanation.** Whether the text names the documented cause is read by a person, not scored.
- **Citation support is unmeasured** until the audit sheet is labelled.
- **The decision is only as good as the reading.** Under version 4 the model's job is to say what each document reports and to quote it. The quote is checked by string matching; whether the direction and the reason were read correctly is not checked by anything but a person.
- **A reason in the news is not a cause.** Market reports attach a reason to every day's move. The rule explains when a recent document gives a reason for a move in the right direction; it cannot tell a real cause from a journalist's habit. The placebo test measures how often that produces an explanation for a move that did not happen.
- **The earlier direction signal is a word list.** It misreads some documents ("rain" is listed as price-negative, but rain delaying a harvest is price-positive). It is still computed and saved for comparison, and it drives the guard on the `--decision legacy` path.
- **Retrieval still leans towards supply-shock stories.** Four direction-neutral queries over the recent days were added, but six of the ten GDELT queries and the semantic reference query the gate scores against are about frost, drought and supply disruption. Days prices fell remain harder to find evidence for. The evaluation report shows, for every refused date, whether it stopped at the search, the filter, the reading or the rule.
- **Retrieval is the bottleneck.** GDELT is slow and rate-limited; about half of article URLs fail to fetch. On the two largest recent moves (2026-07-06, +15.3%; 2026-07-09, +10.1%) the system refused: one document of 27 passed the gate for the first, and for the second the newest accepted article was three days older than the move.
- **Price data gets revised.** One stored run (2026-09-11) recorded a -9.8% move at a price the current data no longer shows; the day is now a -0.5% move and is not flagged. Anomalies detected on fresh quotes should be re-checked against re-fetched prices before anything is concluded from them. The metrics report lists runs on dates the current detections file does not flag.
- **Contract switches are handled by a list.** The price series follows the nearest futures contract and jumps when it changes. Only days in the evaluation sample were checked; other flagged days near a contract's expiry may be artefacts too.
- **The advisory pages are descriptive.** The outlook's direction call was right on 17 of 50 backtest dates against 22 for a random call, and the dashboard says so beneath it.
- **Not financial advice.**

## Design notes

**Two thresholds, not one.** Cosine similarity from the embedding model sits in a narrow high band (roughly 0.75 to 0.95) for anything about coffee markets, so it cannot share a threshold with the keyword score. A document passes if either score clears its own bar. On the labelled set the keyword score carries no signal, because most retrieved articles are not in English.

**The model reads; a rule decides.** A model told that prices rose 5% will find a reason in almost anything it is handed, and a count of which way all the retrieved news leans mostly measures what was searched for. So the model is not told the move, its reading of each document is stored, and the decision is a function of those stored readings, which means any other setting of the rule can be scored afterwards without calling a model again.

**Refusal is a first-class outcome.** Retrieval failure, a gate rejection, a refusal by the rule and (on the earlier path) a model or guard refusal are recorded as different things, because "the news API was down" and "there was no cause to report" should never look the same.

**Caches are part of the method.** Anomaly dates are historical, so their retrieval windows never change. GDELT responses and extracted article text are cached on disk; offline evaluations rebuild their inputs from those caches and call nothing.

**ONNX Runtime instead of sentence-transformers** for the embedding model, to avoid a scipy dependency that some machine policies block at import.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | for explanations | primary model |
| `GROQ_API_KEY`, `OPENAI_API_KEY` | no | fallback providers |
| `EXPLAINER_PROVIDERS` | no | provider order, e.g. `groq` or `gemini,groq` |
| `SCORING_MODE` | no | `dual` (default), `semantic_only`, `keyword_only` |
| `READ_ONLY` | no | `1` disables live calls in the dashboard |
