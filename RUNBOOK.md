# Runbook: what is finished, and the order to do the rest in

Written 11 November 2026. Every figure quoted here has a source file in
`results/`; nothing in this file is an estimate unless it says so.

## Where the three objectives stand

| Objective | State | Evidence |
|---|---|---|
| **1. Detect disruptions** | Done, five commodities | 23 of 23 cited events flagged (coffee 4/4, crude oil 5/5, wheat 4/4, gold 6/6, silver 6/6); 0 false positives in any verified quiet control period. `results/detector_evaluation.md` |
| **2. Explain with cited evidence** | Measured on coffee; built but not run on the other four | Version 11: 10 of 16 labelled dates (CI 39-82%); 2 of 12 invented moves explained on held-out quiet days, against 8 of 12 for the previous rule on the same readings (McNemar p = 0.031). `results/EVALUATION.md` section 9g |
| **1b. Was it the commodity or the market?** | Cross-commodity check run; factor model and numeraire test built, not run | 48 of coffee's 279 flagged days (17.2%) are days the complex moved together; 3 of 16 labelled dates are such days. `EVALUATION.md` section 11 |
| **3. Inflation tracking and pass-through** | Done: United States, India, G20 reference | 25 commodity-and-index pairs, Holm-corrected. Oil into US petrol: +6.0% per 10% rise, p < 0.001, and 31% lower error on an estimate made before publication. India: no pair distinguishable from chance. `results/inflation_report.md` |

What is genuinely not done: an answer key containing should-refuse dates
(objective 2 cannot beat the always-explain baseline without one), the
pipeline run on the four non-coffee commodities, and any measurement of the
reading's `when` field.

## 0. Before anything else: move off the free model tier

This is the cheapest decision available and it unblocks every run below.

`openai/gpt-oss-120b` costs **$0.15 per million input tokens and $0.60 per
million output**. One 16-date evaluation run is about **$0.06**. Every run in
this runbook, several times over, is **under $10**. The Batch API is half
that again and does not consume the request-rate limits at all.

The free tier is not cheaper, it is slower and smaller: 200,000 tokens a day
and 8,000 tokens a minute, against 250,000 a minute on the paid tier. Most of
the three to four minutes each date took was that throttle, not the news API.

Add a card at `console.groq.com`. There is no monthly fee and no minimum.
India's progressive billing thresholds differ from the published ladder, so
read the billing page first.

## 1. Run the three "was it the commodity?" layers

These answer a question none of the evaluation so far could: whether a flagged
move was about the commodity at all. The cross-commodity check has already been
run and found that **48 of coffee's 279 flagged days (17.2%) are days at least
two other watched markets flagged too**, and that 3 of the 16 labelled dates are
such days (`EVALUATION.md` section 11). The other two layers measure the same
thing directly and need series the project had not fetched before.

```
cd /d D:\CAPSTONE
py -m src.data.market_factors
py -m src.modeling.factor_model --all
py -m src.analysis.numeraire --commodity gold --flagged --save
py -m src.analysis.numeraire --commodity coffee --flagged --save
py -m src.evaluation.anomaly_card --commodity coffee --labelled --save
```

All keyless, all offline-capable once the series are down, no model budget at
all. Minutes, not hours.

The one to look at first is gold around **2026-01-30**, the 11% fall. If the
numeraire test shows it survived translation into euro, yen, yuan and rupees,
it was a gold event; if it did not, the dollar moved and gold did not. Nothing
in the project could previously tell those apart.

Two things to check in the output, because they are the known weak spots:

- **Crude oil's window R-squared.** The broad-commodity factor (DBC) is heavily
  weighted to energy, so for crude the factor partly contains the commodity
  itself. A high R-squared there is partly mechanical. The report prints it so
  this is visible rather than hidden.
- **The yuan.** It is managed against the dollar, so it should be the currency
  that moves least in the numeraire tables. If it is moving freely, something
  is wrong with the exchange-rate normalisation.

## 2. The pipeline on the other four commodities (objective 2)

`scripts/run_commodity_eval.py` exists for exactly this and needs **no answer
key**, which is what makes it runnable now. It measures two things per
commodity: the invented-move test on that commodity's own quiet days (on a
quiet day nothing happened, so no labels are needed), and the pipeline on the
one-day shocks the detector flags inside each cited event window.

```
cd /d D:\CAPSTONE
py -m src.modeling.fetch_price_data --commodity gold
py -m src.modeling.fetch_price_data --commodity silver
py -m scripts.run_commodity_eval --commodity crude_oil
py -m scripts.run_commodity_eval --commodity wheat
py -m scripts.run_commodity_eval --commodity gold
py -m scripts.run_commodity_eval --commodity silver
```

Only events from 2017 on are run: the article search does not reach further
back, so an older date would refuse for lack of news, which is a fact about
the archive and not about the system. Everything is resumable - a finished
date is not run again and the report is rebuilt from what is on disk.

Expect the news fetch to dominate. The article search answers about one
request in five, so budget hours per commodity, not minutes, and run them one
at a time so they do not compete for the same requests.

**Read the printed explanations.** The report prints each one on purpose. A
date counted as explained means the system chose to explain, not that its
text names the cited cause; only a person reading them establishes that.

## 3. Close the retrieval confound on the coffee dates

Across the 75 stored runs, 70.7% had at least one news request fail after
retries, and 81.1% of those ended in a refusal against 50.0% of the rest. The
six dates version 11 refused each retrieved 491 to 634 documents, so they are
not starved - but the correlation is in the report and will be asked about.

```
py -m scripts.run_eval_set --run --retry-incomplete
py -m src.evaluation.batch_runner --existing-only
py -m src.evaluation.metrics
```

If a refusal flips to explained, that date was an artefact of the news API.
Run `batch_runner --existing-only` **before** `metrics` every time: metrics
reads `results/evaluation_results.json` when it exists and will otherwise
serve figures from a superseded pipeline version without warning.

## 4. Measure the `when` field (the defect worth presenting)

The reading tells the rule whether a document's price move is the one being
explained: `same_day`, `previous_day`, `earlier`, `forecast`, `unclear`. It is
a hard filter upstream of the whole rule and **its accuracy has never been
measured**. Three cases are already documented in section 9g:

- 2025-08-15 lost a real explanation. A same-day arabica report past the
  relevance filter, stating a 4.86% move against an actual 4.64%, quote
  verified, every version 11 check passed - excluded because the reading
  answered `earlier` on the headline *"Tarifaço ainda impacta mercado do
  café, que sobe 4 % em Nova York"*.
- 2023-04-04 gained a false one, the same error in reverse.
- Across the six refused dates the field reads `unclear` 28 times out of 59.

The measurement is a person's work and needs no quota: take about 50 of the
documents already read in `results/pipeline_output_coffee_*.json`, fill in the
field by hand without looking at the model's answer, and compare. That number
belongs in `EVALUATION.md` beside the rule it gates.

## 5. A larger invented-move test

The held-out result rests on 12 days, so a four-fold drop in false
explanations is still only p = 0.031 paired, and p = 0.37 against the earlier
run. Thirty days would settle it, and costs only model tokens:

```
py -m src.evaluation.robustness --providers groq --experiments placebo --n-placebo 30 --seed 12
```

Hypothesis to test on that seed, **not** before: `MIN_SIZE_RATIO = 0.25` lets
a 1.24% reported move support a 3.7% claimed one, which is how 2025-06-11
survived. Raising it to 0.5 would refuse that day - but it was chosen by
looking at the day it fixes, so it must be scored on days it was not chosen
on or it is worthless.

## 6. The answer key (objective 2's ceiling)

Until a key contains dates where refusing is right, no accuracy figure can
beat always-explain, which scores 16 of 16. This is the one task no code can
do and the one with a lead time, because it needs two people.

1. The 3 rows marked CHECK in `data/labeling/labels_by_written_rule.csv`.
2. The 24 dates in `data/labeling/dates_to_label.csv` (7 marked CHECK), which
   include the first should-refuse candidates.
3. A second labeller on the same dates, independently, so an agreement rate
   can be reported.

There is now a third decision to make, which section 11 of `EVALUATION.md`
sets out: two labelled dates (2024-04-30 and 2025-04-07) are days the whole
complex moved, and the cause the key itself records on both is a macro one -
"a stronger dollar weighed on commodities", "a sell-off across most
commodities". Version 11 refused both and was scored wrong for it. The key
needs a category for a move that is real, documented, and not about this
commodity. **Do not simply relabel those two dates**: changing a label after
seeing which ones a rule failed makes the result circular. Decide the category
first, in writing, then apply it to every date.

Follow `data/labeling/LABELLING_RULE.md` exactly, and label from market
reports of the day without looking at the system's answers. Until this is
done, no document may say the labels have been checked - they rest on one
round of research that nobody has verified.

## 7. Refreshing the inflation data (optional)

`results/inflation_report.md` rebuilds offline from
`data/raw/inflation_series.csv` in about a minute, no key and no network:

```
py -m src.modeling.inflation
```

To pull newer figures, `py -m src.data.cpi` downloads every series from FRED's
public CSV endpoint, which needs no key. Two limits are structural and already
written into the report: India's monthly index and the G20 series come from an
OECD database that stopped updating and end in March 2025, and India has only
the all-items index here - no food or fuel sub-index, which is where
pass-through would show first. A registered key at MOSPI or data.gov.in would
lift both, and registration is free.

## 8. Deployment and the repository

```
py -m pytest -q
```
367 tests, no network and no key. Run before every commit.

The hosted dashboard reads `data/raw/` and `results/`, so both are committed on
purpose. `READ_ONLY=1` disables every model and network call and is the mode
the public deployment runs in; two tests enforce it.

`.github/workflows/` does not exist, so there is no CI. `daily_check.yml` is
sitting in the git-ignored `project-notes/` folder; move it and a `tests.yml`
into `.github/workflows/` with the cron schedule commented out until secrets
are set.

## Standing rules

- `.env` is never committed, and neither is `project-notes/`, `data/text_cache/`
  or `data/eval_cache/` - the last two hold other publishers' article text.
- Every result file carries its `pipeline_version`. Never put figures from two
  versions in one table.
- Always print the baseline next to an accuracy figure. Always carry the
  interval.
- A fault is never reported as a refusal.
- In-sample results say so. Versions 9, 10 and 11 were all designed by
  studying the dates they are scored on; only the seed-11 quiet days and the
  22 unlabelled dates are held out.
