# Ablation Study

**Dates**: 26 coffee anomaly dates with cached retrieval, 2018-04-27 to 2026-09-11.

Every row gates the same scored documents for each date; only the named component differs. Documents are rebuilt from the on-disk caches and scored with both channels. No model is called, so this measures the gate, not the final explain/refuse decision.

| Configuration | Gate pass rate | Accepted / date | Distinct stories / date | Accepted dated after anomaly |
|---|---|---|---|---|
| full_pipeline | 88.5% (23/26; 95% CI 71.0-96.0%) | 4.9 | 4.9 | 0.0% (0/127; 95% CI 0.0-2.9%) |
| semantic_only | 88.5% (23/26; 95% CI 71.0-96.0%) | 4.7 | 4.7 | 0.0% (0/123; 95% CI 0.0-3.0%) |
| keyword_only | 26.9% (7/26; 95% CI 13.7-46.1%) | 0.4 | 0.4 | 0.0% (0/10; 95% CI 0.0-27.8%) |
| no_gate | 100.0% (26/26; 95% CI 87.1-100.0%) | 42.7 | 42.5 | 0.0% (0/1109; 95% CI 0.0-0.3%) |
| semantic_0.80 | 100.0% (26/26; 95% CI 87.1-100.0%) | 21.8 | 21.8 | 0.0% (0/567; 95% CI 0.0-0.7%) |
| semantic_0.90 | 34.6% (9/26; 95% CI 19.4-53.8%) | 0.6 | 0.6 | 0.0% (0/15; 95% CI 0.0-20.4%) |
| no_window_filter | 88.5% (23/26; 95% CI 71.0-96.0%) | 5.7 | 5.7 | 14.8% (22/149; 95% CI 10.0-21.3%) |
| no_duplicate_collapse | 88.5% (23/26; 95% CI 71.0-96.0%) | 6.5 | 5.2 | 0.0% (0/169; 95% CI 0.0-2.2%) |

## Against the hand labels

A gate that helped would pass the EXPLAIN-labelled dates and block the REFUSE-labelled ones.

| Configuration | Passes EXPLAIN-labelled | Blocks REFUSE-labelled |
|---|---|---|
| full_pipeline | 90.9% (10/11; 95% CI 62.3-98.4%) | 0.0% (0/4; 95% CI 0.0-49.0%) |
| semantic_only | 90.9% (10/11; 95% CI 62.3-98.4%) | 0.0% (0/4; 95% CI 0.0-49.0%) |
| keyword_only | 27.3% (3/11; 95% CI 9.7-56.6%) | 25.0% (1/4; 95% CI 4.6-69.9%) |
| no_gate | 100.0% (11/11; 95% CI 74.1-100.0%) | 0.0% (0/4; 95% CI 0.0-49.0%) |
| semantic_0.80 | 100.0% (11/11; 95% CI 74.1-100.0%) | 0.0% (0/4; 95% CI 0.0-49.0%) |
| semantic_0.90 | 36.4% (4/11; 95% CI 15.2-64.6%) | 0.0% (0/4; 95% CI 0.0-49.0%) |

## What each configuration is

- **full_pipeline**: Production: keyword >= 0.4 OR semantic >= 0.85.
- **semantic_only**: Keyword channel ignored.
- **keyword_only**: Semantic channel ignored.
- **no_gate**: Everything above the noise floor is accepted.
- **semantic_0.80**: Semantic threshold lowered to 0.80.
- **semantic_0.90**: Semantic threshold raised to 0.90.
- **no_window_filter**: Production gate; out-of-window articles not removed.
- **no_duplicate_collapse**: Production gate; syndicated copies not collapsed.

## What the table shows

1. Removing the keyword channel changes the gate decision on none of the 26 dates (accepted documents per date: 4.9 -> 4.7).
2. Removing the semantic channel changes the gate decision on 16 of 26 dates (2018-04-27, 2018-05-21, 2018-08-20, 2021-04-06, 2021-05-05, 2021-07-22, 2021-07-30, 2022-05-11, ...); pass rate 88.5% -> 26.9%, accepted documents per date 4.9 -> 0.4.
3. Removing the gate entirely changes the gate decision on 3 of 26 dates (2020-03-19, 2026-06-16, 2026-09-11); pass rate 88.5% -> 100.0%, accepted documents per date 4.9 -> 42.7.
4. Semantic threshold 0.80: pass rate 100.0% (3 date(s) change), 21.8 accepted per date vs 4.9 at 0.85.
5. Semantic threshold 0.90: pass rate 34.6% (14 date(s) change), 0.6 accepted per date vs 4.9 at 0.85.
6. Without the local window filter, 14.8% (22/149; 95% CI 10.0-21.3%) of accepted documents are dated after the anomaly; with it, 0.0% (0/127; 95% CI 0.0-2.9%).
7. Without duplicate collapsing the gate accepts 6.5 documents per date, of which 5.2 are distinct stories; with it, 4.9 accepted and 4.9 distinct.

## Limits

- Body text comes from the on-disk text cache, so a document whose text was never fetched successfully is scored on its title alone - the same as in a live run where that fetch failed.
- The live news aggregator (Google News, industry feeds) is not used here because its results are not reproducible; these rows cover GDELT-retrieved documents only.
- 26 dates. Intervals are 95% Wilson intervals and are wide.
