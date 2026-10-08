# Relevance Gate Threshold Calibration

Based on 20 hand-labeled documents (source: `data/labeling/relevance_labeling_set.csv`). Production thresholds (src/rag/relevance_gate.py): KEYWORD_THRESHOLD=0.4, SEMANTIC_THRESHOLD=0.85.

Rates are shown with 95% Wilson intervals.

### Keyword score

- **Best-F1 threshold**: 0.000 (precision=0.60, recall=1.00, f1=0.75, n_flagged=20)
- **Precision>=0.9 threshold** (biases toward refusing over hallucinating, per this project's stated principle): 0.305 (precision=1.00, recall=0.17, n_flagged=2)

- **At the production threshold (0.4)**, on all labelled documents (in-sample if the threshold was chosen from them): precision n/a (0 cases), recall 0.0% (0/12; 95% CI 0.0-24.3%)
- **Held-out, precision>=0.9 rule** (leave-one-date-out, 3 folds, 2 skipped; chosen thresholds ranged 0.305-0.305): precision n/a (0 cases), recall 0.0% (0/4; 95% CI 0.0-49.0%)
- **Held-out, best-F1 rule** (leave-one-date-out, 5 folds; chosen thresholds ranged 0.000-0.000): precision 60.0% (12/20; 95% CI 38.7-78.1%), recall 100.0% (12/12; 95% CI 75.7-100.0%)

### Semantic score

- **Best-F1 threshold**: 0.796 (precision=0.71, recall=1.00, f1=0.83, n_flagged=17)
- **Precision>=0.9 threshold** (biases toward refusing over hallucinating, per this project's stated principle): 0.851 (precision=1.00, recall=0.50, n_flagged=6)

- **At the production threshold (0.85)**, on all labelled documents (in-sample if the threshold was chosen from them): precision 100.0% (6/6; 95% CI 61.0-100.0%), recall 50.0% (6/12; 95% CI 25.4-74.6%)
- **Held-out, precision>=0.9 rule** (leave-one-date-out, 5 folds; chosen thresholds ranged 0.825-0.852): precision 71.4% (5/7; 95% CI 35.9-91.8%), recall 41.7% (5/12; 95% CI 19.3-68.0%)
- **Held-out, best-F1 rule** (leave-one-date-out, 5 folds; chosen thresholds ranged 0.796-0.810): precision 73.3% (11/15; 95% CI 48.0-89.1%), recall 91.7% (11/12; 95% CI 64.6-98.5%)

## How to read this

The best-F1 and precision>=0.9 thresholds above are chosen and scored on the same documents, so their precision and recall are in-sample and optimistic. The held-out lines choose the threshold without one anomaly date and score it on that date; those are the figures to quote.

## Caveat

20 labelled documents from 5 anomaly dates, one annotator. The intervals show how little that pins down. `python -m scripts.collect_labeling_set --n 200` builds a larger sheet (existing labels are kept).