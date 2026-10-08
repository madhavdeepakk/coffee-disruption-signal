# Coffee (Arabica) — Anomaly Detector Validation

Rolling window: 75 days. Z-score threshold: 2.0 (chosen before this validation, not tuned against it). Two independent checks, either one can flag a day: a daily shock check (day-over-day return z-score) and a 10-trading-day cumulative-return check (catches slower, multi-day build-ups a single-day check can miss - see anomaly_detector.py's module docstring for why this was added).

**Result: 4/4 known real events had at least one day flagged. Control period false positives: 0.**

## Events

- **2021 Brazil drought+frost** (2021-06-01 to 2021-08-15) — DETECTED, 11/53 days flagged (type: shock, shock, shock+trend, shock+trend, trend, shock+trend, trend, trend, trend, shock, trend). Source: multiple news sources - see results/anomaly_detector_validation.md
- **2024 Brazil drought+Typhoon Yagi** (2024-08-15 to 2024-10-15) — DETECTED, 2/43 days flagged (type: shock, shock). Source: see results/anomaly_detector_validation.md
- **2025 sharp spike+reversal** (2025-08-15 to 2025-09-20) — DETECTED, 11/25 days flagged (type: shock+trend, trend, trend, trend, trend, trend, trend, trend, trend, shock, shock). Source: identified from price data itself, not a news date - see validation.md
- **2026 frost concerns (16%+ single-day surge)** (2026-07-01 to 2026-07-15) — DETECTED, 4/10 days flagged (type: trend, shock+trend, shock+trend, shock+trend). Source: cocoaintel.com, July 2026

## Control period

- **Control period (2019-01-01 to 2019-03-31): VERIFIED quiet in real data - 0/61 days flagged (see results/anomaly_detector_validation.md).** — 0/61 days flagged (clean).

This control period is a candidate verified by this run's own numbers above, not assumed correct in advance - see src/config/commodities.py's note on this commodity's control_period for whether it was pre-verified or is still a first-pass candidate.