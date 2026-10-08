# Anomaly Detector — Quantitative Evaluation

Evaluated against the labeled regions available: each commodity's known historical events (should flag) and its verified-quiet control period (should stay silent). See this script's docstring for why event-level recall + control specificity are the well-defined metrics here, rather than a per-day precision/recall over an unlabeled series.

## Coffee (Arabica) (`coffee`)

- **Event-level recall: 4/4 (100%)** — known events flagged at least once in-window.
- **Control specificity: 100.0%** — 0 false positive(s) across 61 control days (2019-01-01..2019-03-31).
- Overall flag rate: 279/2197 days (12.70%).
- Flagged-day distribution: 28 inside event windows, 0 in control, 251 elsewhere (unlabeled — may include real undocumented moves).

| event | window | trading days | flagged days | detected | types |
|---|---|---|---|---|---|
| 2021 Brazil drought+frost | 2021-06-01..2021-08-15 | 53 | 11 | yes | shock, shock+trend, trend |
| 2024 Brazil drought+Typhoon Yagi | 2024-08-15..2024-10-15 | 43 | 2 | yes | shock |
| 2025 sharp spike+reversal | 2025-08-15..2025-09-20 | 25 | 11 | yes | shock, shock+trend, trend |
| 2026 frost concerns (16%+ single-day surge) | 2026-07-01..2026-07-15 | 10 | 4 | yes | shock+trend, trend |

## Crude Oil (WTI) (`crude_oil`)

- **Event-level recall: 5/5 (100%)** — known events flagged at least once in-window.
- **Control specificity: 100.0%** — 0 false positive(s) across 63 control days (2023-07-01..2023-09-30).
- Overall flag rate: 420/3444 days (12.20%).
- Flagged-day distribution: 21 inside event windows, 0 in control, 399 elsewhere (unlabeled — may include real undocumented moves).

| event | window | trading days | flagged days | detected | types |
|---|---|---|---|---|---|
| 2014 OPEC no-cut decision | 2014-11-24..2014-12-05 | 9 | 5 | yes | shock, shock+trend, trend |
| 2020 Saudi-Russia price war | 2020-03-06..2020-03-13 | 6 | 6 | yes | shock+trend, trend |
| 2020 negative WTI price | 2020-04-15..2020-04-22 | 6 | 2 | yes | shock, shock+trend |
| 2022 Russia invades Ukraine | 2022-02-23..2022-03-09 | 11 | 6 | yes | shock, shock+trend, trend |
| 2024 Red Sea shipping attacks | 2024-01-01..2024-02-29 | 41 | 2 | yes | trend |

## Wheat (Chicago SRW) (`wheat`)

- **Event-level recall: 4/4 (100%)** — known events flagged at least once in-window.
- **Control specificity: 100.0%** — 0 false positive(s) across 62 control days (2022-04-01..2022-06-30).
- Overall flag rate: 519/4449 days (11.67%).
- Flagged-day distribution: 27 inside event windows, 0 in control, 492 elsewhere (unlabeled — may include real undocumented moves).

| event | window | trading days | flagged days | detected | types |
|---|---|---|---|---|---|
| 2010 Russia export ban | 2010-08-03..2010-08-16 | 10 | 4 | yes | shock, shock+trend |
| 2012 US drought | 2012-06-20..2012-08-10 | 37 | 10 | yes | shock+trend, trend |
| 2022 Russia invades Ukraine | 2022-02-23..2022-03-09 | 11 | 10 | yes | shock, shock+trend, trend |
| 2023 Black Sea grain deal collapse | 2023-07-17..2023-07-28 | 10 | 3 | yes | shock, shock+trend, trend |
