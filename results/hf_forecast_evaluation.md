# Alternative Time-Series Model Evaluation

Walk-forward comparison of alternative forecasters against the project's logistic regression baseline on directional coffee-price forecasting. All models use the same protocol: only past data when predicting, accuracy scored against the naive majority-class baseline.

## Coffee (Arabica)

### ARIMA(5, 1, 0)

- **Next trading day**: accuracy **46.4%** vs baseline 51.5% (edge -5.0 pts, n=379)
- **Next 5 trading days (~1 week)**: accuracy **48.7%** vs baseline 52.9% (edge -4.2 pts, n=378)

### Chronos-T5 (`amazon/chronos-t5-small`, zero-shot)

- **Next trading day**: accuracy **50.0%** vs baseline 55.3% (edge -5.3 pts, n=38)
- **Next 5 trading days (~1 week)**: accuracy **57.9%** vs baseline 57.9% (edge +0.0 pts, n=38)

### Comparison table

| Horizon | Logistic Regression | ARIMA | Chronos-T5 | Baseline |
|---------|-------------------|-------|------------|----------|
| 1 day | 50.2% | 46.4% | 50.0% | 51.5% |
| 5 day | 47.0% | 48.7% | 57.9% | 52.9% |

## Interpretation

If no model beats the baseline by more than ~1-2 percentage points, the result confirms that short-horizon commodity-price direction is close to a coin flip — consistent with weak-form market efficiency. An ARIMA model that sees the full price history, a pre-trained foundation model (Chronos) that has seen diverse time-series patterns, and a logistic regression on hand-crafted features all converge on the same null result. This strengthens the project's thesis that the valuable work is *explanation* (why did the price move?) rather than *prediction* (which way will it move?).
