# Directional Forecaster — Out-of-Sample Evaluation

Experimental, and not financial advice. Short-horizon commodity-price direction is near-random; these numbers are walk-forward (the model is only tested on days after the days it trained on) and shown next to the naive majority-class baseline so any edge over it is visible. See src/modeling/forecast.py.

## Coffee (Arabica) — Price-only features

- **Next trading day**: out-of-sample accuracy **50.2%** vs naive baseline 50.7% (edge -0.5 pts over 1847 walk-forward test days; 49% of those days were up).
- **Next 5 trading days (~1 week)**: out-of-sample accuracy **47.0%** vs naive baseline 50.8% (edge -3.8 pts over 1843 walk-forward test days; 51% of those days were up).

Features used: ret_1, ret_5, ret_10, mom_10, vol_20, zscore_75, dist_ma20

## Coffee (Arabica) — Price + weather features

- **Next trading day**: out-of-sample accuracy **49.4%** vs naive baseline 50.6% (edge -1.3 pts over 1785 walk-forward test days; 49% of those days were up).
- **Next 5 trading days (~1 week)**: out-of-sample accuracy **46.9%** vs naive baseline 51.3% (edge -4.4 pts over 1781 walk-forward test days; 51% of those days were up).

Features used: ret_1, ret_5, ret_10, mom_10, vol_20, zscore_75, dist_ma20, dryness_zscore, frost_7d_count, prcp_30d_sum
