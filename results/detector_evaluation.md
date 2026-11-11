# Anomaly Detector — Quantitative Evaluation

Evaluated against the labeled regions available: each commodity's known historical events (should flag) and its verified-quiet control period (should stay silent). See this script's docstring for why event-level recall + control specificity are the well-defined metrics here, rather than a per-day precision/recall over an unlabeled series.

**Read the recall figures with the chance column beside them.** The detector flags about one trading day in eight and flagged days cluster, so a multi-day window often holds a flag by chance. What separates the detector from chance is how densely it flags inside event windows, and how that changes at stricter thresholds (the second table for each commodity).

## Coffee (Arabica) (`coffee`)

- **Event-level recall: 4/4 (100%)** — known events flagged at least once in-window.
- **Control specificity: 100.0%** — 0 false positive(s) across 61 control days (2019-01-01..2019-03-31).
- Overall flag rate: 279/2204 days (12.66%).
- Flagged-day distribution: 28 inside event windows, 0 in control, 251 elsewhere (unlabeled — may include real undocumented moves).
- **Against chance:** a window placed at random would hold a flag often enough that a detector with no skill would be expected to 'detect' 3.1 of these 4 events. Recall alone therefore says little; density says more: **21% of the days inside event windows are flagged, against 13% of all days** (1.6 times the base rate).

| event | window | trading days | flagged days | detected | chance of a flag in a random window this long | types |
|---|---|---|---|---|---|---|
| 2021 Brazil drought+frost | 2021-06-01..2021-08-15 | 53 | 11 | yes | 95% | shock, shock+trend, trend |
| 2024 Brazil drought+Typhoon Yagi | 2024-08-15..2024-10-15 | 43 | 2 | yes | 89% | shock |
| 2025 sharp spike+reversal | 2025-08-15..2025-09-20 | 25 | 11 | yes | 74% | shock, shock+trend, trend |
| 2026 frost concerns (16%+ single-day surge) | 2026-07-01..2026-07-15 | 10 | 4 | yes | 50% | shock+trend, trend |

Stricter thresholds, recomputed from the price series (the detector runs at 2.0, fixed before validation; the other rows are what a different choice would flag). Under a normal distribution 4.6% of days would lie beyond 2 standard deviations, 1.2% beyond 2.5 and 0.27% beyond 3; the 'daily check alone' column is the measured share.

| threshold | days flagged (either check) | daily check alone | events detected | expected by chance | control false positives | density in event windows vs base |
|---|---|---|---|---|---|---|
| 2.0 | 279/2128 (13.1%) | 6.1% | 4/4 | 3.1 | 0 | 1.6x |
| 2.5 | 133/2128 (6.2%) | 2.5% | 3/4 | 2.3 | 0 | 2.3x |
| 3.0 | 56/2128 (2.6%) | 1.2% | 3/4 | 1.4 | 0 | 4.4x |

## Crude Oil (WTI) (`crude_oil`)

- **Event-level recall: 5/5 (100%)** — known events flagged at least once in-window.
- **Control specificity: 100.0%** — 0 false positive(s) across 63 control days (2023-07-01..2023-09-30).
- Overall flag rate: 420/3444 days (12.20%).
- Flagged-day distribution: 21 inside event windows, 0 in control, 399 elsewhere (unlabeled — may include real undocumented moves).
- **Against chance:** a window placed at random would hold a flag often enough that a detector with no skill would be expected to 'detect' 2.6 of these 5 events. Recall alone therefore says little; density says more: **29% of the days inside event windows are flagged, against 12% of all days** (2.3 times the base rate).

| event | window | trading days | flagged days | detected | chance of a flag in a random window this long | types |
|---|---|---|---|---|---|---|
| 2014 OPEC no-cut decision | 2014-11-24..2014-12-05 | 9 | 5 | yes | 46% | shock, shock+trend, trend |
| 2020 Saudi-Russia price war | 2020-03-06..2020-03-13 | 6 | 6 | yes | 37% | shock+trend, trend |
| 2020 negative WTI price | 2020-04-15..2020-04-22 | 6 | 2 | yes | 37% | shock, shock+trend |
| 2022 Russia invades Ukraine | 2022-02-23..2022-03-09 | 11 | 6 | yes | 51% | shock, shock+trend, trend |
| 2024 Red Sea shipping attacks | 2024-01-01..2024-02-29 | 41 | 2 | yes | 86% | trend |

Stricter thresholds, recomputed from the price series (the detector runs at 2.0, fixed before validation; the other rows are what a different choice would flag). Under a normal distribution 4.6% of days would lie beyond 2 standard deviations, 1.2% beyond 2.5 and 0.27% beyond 3; the 'daily check alone' column is the measured share.

| threshold | days flagged (either check) | daily check alone | events detected | expected by chance | control false positives | density in event windows vs base |
|---|---|---|---|---|---|---|
| 2.0 | 420/3368 (12.5%) | 6.1% | 5/5 | 2.6 | 0 | 2.3x |
| 2.5 | 198/3368 (5.9%) | 3.1% | 4/5 | 1.6 | 0 | 4.0x |
| 3.0 | 99/3368 (2.9%) | 1.7% | 4/5 | 1.0 | 0 | 5.6x |

## Wheat (Chicago SRW) (`wheat`)

- **Event-level recall: 4/4 (100%)** — known events flagged at least once in-window.
- **Control specificity: 100.0%** — 0 false positive(s) across 62 control days (2022-04-01..2022-06-30).
- Overall flag rate: 519/4449 days (11.67%).
- Flagged-day distribution: 27 inside event windows, 0 in control, 492 elsewhere (unlabeled — may include real undocumented moves).
- **Against chance:** a window placed at random would hold a flag often enough that a detector with no skill would be expected to 'detect' 2.4 of these 4 events. Recall alone therefore says little; density says more: **40% of the days inside event windows are flagged, against 12% of all days** (3.3 times the base rate).

| event | window | trading days | flagged days | detected | chance of a flag in a random window this long | types |
|---|---|---|---|---|---|---|
| 2010 Russia export ban | 2010-08-03..2010-08-16 | 10 | 4 | yes | 50% | shock, shock+trend |
| 2012 US drought | 2012-06-20..2012-08-10 | 37 | 10 | yes | 89% | shock+trend, trend |
| 2022 Russia invades Ukraine | 2022-02-23..2022-03-09 | 11 | 10 | yes | 52% | shock, shock+trend, trend |
| 2023 Black Sea grain deal collapse | 2023-07-17..2023-07-28 | 10 | 3 | yes | 50% | shock, shock+trend, trend |

Stricter thresholds, recomputed from the price series (the detector runs at 2.0, fixed before validation; the other rows are what a different choice would flag). Under a normal distribution 4.6% of days would lie beyond 2 standard deviations, 1.2% beyond 2.5 and 0.27% beyond 3; the 'daily check alone' column is the measured share.

| threshold | days flagged (either check) | daily check alone | events detected | expected by chance | control false positives | density in event windows vs base |
|---|---|---|---|---|---|---|
| 2.0 | 519/4373 (11.9%) | 5.7% | 4/4 | 2.4 | 0 | 3.3x |
| 2.5 | 228/4373 (5.2%) | 2.6% | 4/4 | 1.5 | 0 | 5.6x |
| 3.0 | 95/4373 (2.2%) | 1.1% | 4/4 | 0.8 | 0 | 9.5x |

## Gold (COMEX) (`gold`)

- **Event-level recall: 6/6 (100%)** — known events flagged at least once in-window.
- **Control specificity: 100.0%** — 0 false positive(s) across 62 control days (2026-04-01..2026-06-30).
- Overall flag rate: 491/3735 days (13.15%).
- Flagged-day distribution: 22 inside event windows, 0 in control, 469 elsewhere (unlabeled — may include real undocumented moves).
- **Against chance:** a window placed at random would hold a flag often enough that a detector with no skill would be expected to 'detect' 2.4 of these 6 events. Recall alone therefore says little; density says more: **58% of the days inside event windows are flagged, against 13% of all days** (4.3 times the base rate).

| event | window | trading days | flagged days | detected | chance of a flag in a random window this long | types |
|---|---|---|---|---|---|---|
| 2013 April gold crash | 2013-04-12..2013-04-17 | 4 | 4 | yes | 31% | shock+trend, trend |
| 2016 Brexit referendum | 2016-06-23..2016-06-28 | 4 | 1 | yes | 31% | shock |
| 2020 pandemic dash for cash | 2020-03-09..2020-03-27 | 15 | 9 | yes | 68% | shock, shock+trend, trend |
| 2020 August reversal from the record | 2020-08-07..2020-08-13 | 5 | 2 | yes | 36% | shock |
| 2025 October reversal from the record | 2025-10-20..2025-10-23 | 4 | 2 | yes | 31% | shock, shock+trend |
| 2026 January metals crash | 2026-01-28..2026-02-04 | 6 | 4 | yes | 40% | shock, shock+trend, trend |

Stricter thresholds, recomputed from the price series (the detector runs at 2.0, fixed before validation; the other rows are what a different choice would flag). Under a normal distribution 4.6% of days would lie beyond 2 standard deviations, 1.2% beyond 2.5 and 0.27% beyond 3; the 'daily check alone' column is the measured share.

| threshold | days flagged (either check) | daily check alone | events detected | expected by chance | control false positives | density in event windows vs base |
|---|---|---|---|---|---|---|
| 2.0 | 491/3659 (13.4%) | 6.2% | 6/6 | 2.4 | 0 | 4.3x |
| 2.5 | 227/3659 (6.2%) | 3.1% | 6/6 | 1.3 | 0 | 8.9x |
| 3.0 | 96/3659 (2.6%) | 1.5% | 6/6 | 0.6 | 0 | 17.1x |

## Silver (COMEX) (`silver`)

- **Event-level recall: 6/6 (100%)** — known events flagged at least once in-window.
- **Control specificity: 100.0%** — 0 false positive(s) across 62 control days (2026-04-01..2026-06-30).
- Overall flag rate: 515/4238 days (12.15%).
- Flagged-day distribution: 30 inside event windows, 0 in control, 485 elsewhere (unlabeled — may include real undocumented moves).
- **Against chance:** a window placed at random would hold a flag often enough that a detector with no skill would be expected to 'detect' 2.4 of these 6 events. Recall alone therefore says little; density says more: **75% of the days inside event windows are flagged, against 12% of all days** (6.1 times the base rate).

| event | window | trading days | flagged days | detected | chance of a flag in a random window this long | types |
|---|---|---|---|---|---|---|
| 2011 May margin-hike collapse | 2011-05-02..2011-05-06 | 5 | 5 | yes | 35% | shock, shock+trend, trend |
| 2011 September plunge | 2011-09-22..2011-09-26 | 3 | 3 | yes | 25% | shock+trend, trend |
| 2013 April metals crash | 2013-04-12..2013-04-17 | 4 | 4 | yes | 31% | shock, shock+trend, trend |
| 2020 pandemic dash for cash | 2020-03-09..2020-03-27 | 15 | 11 | yes | 67% | shock, shock+trend, trend |
| 2021 retail silver squeeze | 2021-01-28..2021-02-03 | 5 | 2 | yes | 35% | shock, shock+trend |
| 2026 January metals crash | 2026-01-26..2026-02-04 | 8 | 5 | yes | 47% | shock, shock+trend, trend |

Stricter thresholds, recomputed from the price series (the detector runs at 2.0, fixed before validation; the other rows are what a different choice would flag). Under a normal distribution 4.6% of days would lie beyond 2 standard deviations, 1.2% beyond 2.5 and 0.27% beyond 3; the 'daily check alone' column is the measured share.

| threshold | days flagged (either check) | daily check alone | events detected | expected by chance | control false positives | density in event windows vs base |
|---|---|---|---|---|---|---|
| 2.0 | 515/4162 (12.4%) | 6.2% | 6/6 | 2.4 | 0 | 6.1x |
| 2.5 | 249/4162 (6.0%) | 3.5% | 6/6 | 1.4 | 0 | 10.0x |
| 3.0 | 111/4162 (2.7%) | 1.9% | 6/6 | 0.7 | 0 | 21.6x |
