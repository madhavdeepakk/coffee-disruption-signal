# Retrieval Evaluation

20 hand-labelled documents (12 relevant) from 5 anomaly dates (`data/labeling/relevance_labeling_set.csv`).

| Score | Rows | AUC | 95% interval | P@3 | R@3 | P@5 | R@5 | MRR | Dates |
|---|---|---|---|---|---|---|---|---|---|
| keyword_score | 20 | 0.417 | 0.24-0.696 | 0.667 | 0.762 | 0.7 | 0.786 | 0.833 | 3 / 2 |
| semantic_score | 20 | 0.812 | 0.531-0.911 | 0.667 | 0.698 | 0.8 | 0.857 | 0.833 | 3 / 2 |

AUC 0.5 is chance; 1.0 is perfect separation of relevant from irrelevant. The interval is a bootstrap over anomaly dates. P@k, R@k and MRR are averaged over dates with at least k labelled documents and at least one relevant one (the Dates column gives how many, for the smallest and largest k).

## Notes

- Semantic score AUC 0.812 against keyword score AUC 0.417.
- The sheet has no anomaly_query_score column, so the optional second-stage ranking (src/rag/rerank.py) cannot be assessed here. Re-collect the sheet with `python -m scripts.collect_labeling_set` to add it.
- Only 5 dates: the intervals are wide and per-date ranking figures rest on very few dates.
