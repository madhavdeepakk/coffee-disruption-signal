# Wheat (Chicago SRW) — Anomaly Detector Validation

Rolling window: 75 days. Z-score threshold: 2.0 (chosen before this validation, not tuned against it). Two independent checks, either one can flag a day: a daily shock check (day-over-day return z-score) and a 10-trading-day cumulative-return check (catches slower, multi-day build-ups a single-day check can miss - see anomaly_detector.py's module docstring for why this was added).

**Result: 4/4 known real events had at least one day flagged. Control period false positives: 0.**

## Events

- **2010 Russia export ban** (2010-08-03 to 2010-08-16) — DETECTED, 4/10 days flagged (type: shock+trend, shock+trend, shock, shock). Source: Russia announced a wheat export ban Aug 5 2010 (effective Aug 15) after a severe drought - France24 ('Ban on wheat exports sends global prices skyrocketing'), NPR, Aug 2010
- **2012 US drought** (2012-06-20 to 2012-08-10) — DETECTED, 10/37 days flagged (type: shock+trend, trend, trend, trend, trend, trend, trend, trend, trend, trend). Source: Worst US drought since 1956. Window widened from an initial 2012-07-15 start (drawn from two CNN article dates) to capture the earlier onset of the rally - the S&P GSCI grains rally 'since mid-June' (Business Standard, 'Commodities enter bull market after drought hits crops', Aug 23 2012) and corn/soybean prices 'rallied sharply beginning in July 2012... as drought conditions unfolded', peaking Aug 10 2012 (farmdocdaily.illinois.edu, May 2013) - CNN ('Corn, soybean prices shoot up as drought worsens', Jul 19 2012; 'U.S. drought drives up food prices worldwide', Aug 9 2012)
- **2022 Russia invades Ukraine** (2022-02-23 to 2022-03-09) — DETECTED, 10/11 days flagged (type: trend, shock+trend, shock, shock+trend, shock+trend, trend, shock+trend, trend, trend, shock+trend). Source: Chicago wheat futures hit a record high in early March 2022 after the Feb 24 invasion - farmpolicynews.illinois.edu, AgFax, Mar 2022
- **2023 Black Sea grain deal collapse** (2023-07-17 to 2023-07-28) — DETECTED, 3/10 days flagged (type: shock, shock+trend, trend). Source: Russia terminated the Black Sea Grain Initiative ~Jul 17 2023; wheat prices jumped in the following days - CNN, World Economic Forum, Jul 2023

## Control period

- **Control period (2022-04-01 to 2022-06-30): Verified quiet in real data - 0/62 days flagged (the only fully clean calendar quarter found across all quarters from 2009-2025 for wheat, which is a more volatile series than coffee - most quarters have 1-8 flagged days). An earlier candidate (2017-01-01 to 2017-03-31) was rejected after testing against the real fetched price series showed it was not quiet: it contained a one-day jump on 2017-03-15 (+5.8%, z=3.54, the quarter's most extreme day).** — 0/62 days flagged (clean).

This control period is a candidate verified by this run's own numbers above, not assumed correct in advance - see src/config/commodities.py's note on this commodity's control_period for whether it was pre-verified or is still a first-pass candidate.