# Explanation backtest

Primary question - **decision calibration**: does the system EXPLAIN the anomalies that have a documented cause, and REFUSE the ones that do not? This is the direct test of the refuse-don't-hallucinate thesis. Scored against hand-labelled historical coffee anomalies (`data/labeling/explanation_backtest_labels.csv`).

Mode: `use-cached`. A row is *correct* when, for an EXPLAIN label, the system chose to explain (and its evidence did not conflict on direction, where that was recorded); for a REFUSE label, when the system refused. System faults (model unavailable) are excluded, not counted as misses. Evidence direction-consistency is a secondary signal and is only present on runs made after that field was added (older cached runs show n/a).

## Summary

- Graded rows: **15** of 18 labelled (1 had no pipeline output, 2 excluded as system faults)
- **Coverage**: 83% of labelled dates have pipeline outputs to score
- **WARNING**: 1 labelled dates have no pipeline output. Run with `--live` to generate them. Accuracy below is computed on the graded subset only and SHOULD NOT be quoted as the system's overall accuracy.
- Overall accuracy on graded rows: **67%** (10/15)
- Correct explanations: **9/12** (75%)
- Correct refusals: **1/3** (33%)

## Per-date

| Date | Expected | Label conf | System | Evidence dir | Correct | Known cause |
|------|----------|-----------|--------|--------------|---------|-------------|
| 2020-03-19 | EXPLAIN | high | REFUSED | consistent | NO | COVID-19 pandemic panic buying and supply chain fears lifted |
| 2021-05-05 | EXPLAIN | high | EXPLAINED | consistent | yes | Sustained drought in Minas Gerais drove arabica higher on cr |
| 2021-07-20 | EXPLAIN | high | EXPLAINED | consistent | yes | Severe frost hit Brazil's coffee belt (Minas Gerais Parana)  |
| 2021-07-22 | EXPLAIN | high | EXPLAINED | consistent | yes | Continued frost aftermath; second frost wave confirmed; arab |
| 2021-07-19 | REFUSE | high | EXPLAINED | conflicted | NO | Down move immediately before the July 2021 Brazil frost spik |
| 2021-07-30 | REFUSE | medium | REFUSED | conflicted | yes | Post-frost profit-taking and technical correction after 30%  |
| 2022-05-11 | EXPLAIN | medium | EXPLAINED | consistent | yes | Tight global supplies from 2021 frost aftermath plus surging |
| 2023-01-31 | EXPLAIN | medium | EXPLAINED | consistent | yes | Rally from multi-month low as funds covered short positions  |
| 2023-09-20 | REFUSE | medium | EXPLAINED | conflicted | NO | Sharp reversal day (-2.74 sigma) amid choppy Sept 2023 tradi |
| 2023-11-30 | EXPLAIN | medium | REFUSED | conflicted | NO | Year-end rally driven by concerns about 2024 Brazil crop aft |
| 2024-04-30 | EXPLAIN | medium | REFUSED | conflicted | NO | Profit-taking after arabica rallied to 248 cents/lb on tight |
| 2024-09-23 | EXPLAIN | high | FAULT | consistent | - | Brazil drought plus Vietnam typhoon drove arabica to record  |
| 2024-12-02 | REFUSE | medium | FAULT | conflicted | - | Sharp pullback (-3.45 sigma) during the late-2024 surge to 4 |
| 2025-02-10 | EXPLAIN | high | EXPLAINED | consistent | yes | Arabica surged toward $4.29/lb peak on worsening Brazil drou |
| 2025-04-07 | EXPLAIN | high | - | - | - | US tariff announcements (50% on Brazilian coffee) triggered  |
| 2025-08-15 | EXPLAIN | high | EXPLAINED | n/a | yes | US 50% tariff on Brazilian coffee imports disrupted exports  |
| 2025-09-15 | EXPLAIN | medium | EXPLAINED | n/a | yes | Continued US tariff disruption and dry Brazil conditions kep |
| 2024-10-07 | EXPLAIN | low | EXPLAINED | n/a | yes | Pullback/profit-taking within the 2024 rally; not a single d |

## Caveats

- Small, coffee-only labelled set: an honest, extensible baseline, not a large-sample accuracy claim. Add rows to the CSV and re-run to grow it.
- 'Correct explanation' checks direction, not that the prose names the exact documented cause - read the printed explanation text to confirm that.
- Low-confidence labels (e.g. mean-reversion down days) are the hardest to ground-truth; use `--min-confidence high` to score only the firm ones.
