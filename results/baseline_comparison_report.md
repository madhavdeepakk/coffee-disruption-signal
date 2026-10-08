# Baseline Comparison

Decisions on 15 hand-labelled dates (12 labelled EXPLAIN, 3 labelled REFUSE). Dates where the pipeline run ended in a fault are left out for every system.

| System | Explain recall | Refusal specificity | Balanced accuracy | Raw accuracy |
|---|---|---|---|---|
| always_explain | 100.0% (12/12; 95% CI 75.7-100.0%) | 0.0% (0/3; 95% CI 0.0-56.2%) | 50.0% | 80.0% (12/15; 95% CI 54.8-93.0%) |
| gate_only | 100.0% (12/12; 95% CI 75.7-100.0%) | 0.0% (0/3; 95% CI 0.0-56.2%) | 50.0% | 80.0% (12/15; 95% CI 54.8-93.0%) |
| pipeline | 75.0% (9/12; 95% CI 46.8-91.1%) | 33.3% (1/3; 95% CI 6.1-79.2%) | 54.2% | 66.7% (10/15; 95% CI 41.7-84.8%) |
| pipeline_guarded | 75.0% (9/12; 95% CI 46.8-91.1%) | 100.0% (3/3; 95% CI 43.8-100.0%) | 87.5% | 80.0% (12/15; 95% CI 54.8-93.0%) |

## Systems

- **always_explain**: Explains whenever anything was retrieved. No gate, no refusal.
- **gate_only**: Explains exactly when the relevance gate passes.
- **pipeline**: Stored runs as made.
- **pipeline_guarded**: Stored runs with the direction guard applied.

## Reading it

On raw accuracy the pipeline as run (66.7%) does not beat explaining everything (80.0%). That is partly the label mix - 12 of 15 dates are labelled EXPLAIN, so never refusing is right most of the time - and partly real: the pipeline refused dates it should have explained and explained dates it should have refused.

Balanced accuracy, which a never-refuse system cannot game: pipeline 54.2%, always-explain 50.0%.

The gate alone blocks 0 of 3 REFUSE-labelled dates. Any refusals the pipeline gets right beyond that come from the model's reading of the evidence, not from the gate.

With the direction guard applied to the same runs, refusal specificity goes from 1/3 to 3/3. The guard was written after looking at these runs, so this is an in-sample result, not a validated one.

The closed-book baseline (same model, no documents) is missing because that experiment has not been run: `python -m src.evaluation.robustness --experiments closed_book`.

## Per date

| Date | Label | always_explain | gate_only | pipeline | pipeline_guarded |
|---|---|---|---|---|---|
| 2020-03-19 | EXPLAIN | EXPLAINED | EXPLAINED | REFUSED | REFUSED |
| 2021-05-05 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |
| 2021-07-19 | REFUSE | EXPLAINED | EXPLAINED | EXPLAINED | REFUSED |
| 2021-07-20 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |
| 2021-07-22 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |
| 2021-07-30 | REFUSE | EXPLAINED | EXPLAINED | REFUSED | REFUSED |
| 2022-05-11 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |
| 2023-01-31 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |
| 2023-09-20 | REFUSE | EXPLAINED | EXPLAINED | EXPLAINED | REFUSED |
| 2023-11-30 | EXPLAIN | EXPLAINED | EXPLAINED | REFUSED | REFUSED |
| 2024-04-30 | EXPLAIN | EXPLAINED | EXPLAINED | REFUSED | REFUSED |
| 2024-10-07 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |
| 2025-02-10 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |
| 2025-08-15 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |
| 2025-09-15 | EXPLAIN | EXPLAINED | EXPLAINED | EXPLAINED | EXPLAINED |

## Limits

- 3 REFUSE-labelled dates. Every specificity figure here rests on that many cases; the intervals say how little that pins down.
- Labels are single-annotator.
- A correct EXPLAIN decision means the system chose to explain. Whether the text names the documented cause is not scored here.
