# Outlook Backtest Results (Analog Scenarios)

**Period**: 2018-05-25 to 2026-05-20 (50 dates tested)
**Method**: Historical analog matching (25 nearest neighbors by price features)
**Features**: ret_1, ret_5, ret_10, mom_10, vol_20, zscore_75, dist_ma20, weather_dryness
**Weather included**: yes
**Direction rule**: share_up > 55% = upward, < 45% = downward, else balanced

## Direction Accuracy

| Model | 5-day accuracy | 20-day accuracy |
|-------|---------------|-----------------|
| Analog Scenarios | 34.0% (17/50) | 38.0% (19/50) |
| Momentum (5d trailing) | 42.0% (21/50) | 36.0% (18/50) |
| Random (50/50) | 44.0% (22/50) | 50.0% (25/50) |

## Lean Distribution (5-day horizon)

- balanced: 15 (30%)
- downward: 17 (34%)
- upward: 18 (36%)

## Lean Distribution (20-day horizon)

- balanced: 15 (30%)
- downward: 14 (28%)
- upward: 21 (42%)

## Signal Statistics

- Average share_up (5d analogs): 49.7%
- Average share_up (20d analogs): 50.9%

## Interpretation

This backtest evaluates the analog scenario model — the quantitative backbone of the outlook module. It finds the 25 most similar past days (by price features) and checks whether their forward returns predict direction.

The full RAG outlook adds LLM synthesis of live signals (weather forecasts, CFTC positioning, news) on top of these analogs. That layer cannot be backtested historically because those signals are not available for past dates. What this measures is whether the pattern-matching foundation has any edge.

If analog accuracy is near 50%, that confirms the base finding: short-horizon commodity direction is close to random, and the system's value is in explanation (why did the price move?) not prediction (which way next?). If it's meaningfully above 50%, the feature set captures real regime information.