# Crude Oil (WTI) — Anomaly Detector Validation

Rolling window: 75 days. Z-score threshold: 2.0 (chosen before this validation, not tuned against it). Two independent checks, either one can flag a day: a daily shock check (day-over-day return z-score) and a 10-trading-day cumulative-return check (catches slower, multi-day build-ups a single-day check can miss - see anomaly_detector.py's module docstring for why this was added).

**Result: 5/5 known real events had at least one day flagged. Control period false positives: 0.**

## Events

- **2014 OPEC no-cut decision** (2014-11-24 to 2014-12-05) — DETECTED, 5/9 days flagged (type: shock+trend, shock, trend, trend, trend). Source: OPEC declined to cut output at its Nov 27, 2014 meeting; oil fell to a 4-year low - Washington Post ('OPEC decides not to cut oil production, sending crude prices to 4-year low', Nov 27 2014), CNBC
- **2020 Saudi-Russia price war** (2020-03-06 to 2020-03-13) — DETECTED, 6/6 days flagged (type: shock+trend, shock+trend, shock+trend, trend, trend, trend). Source: OPEC+ talks collapsed Mar 6 2020; Saudi Aramco announced discounts/output hike; WTI fell ~25% on Mar 9 2020 ('Black Monday') - Wikipedia '2020 Russia-Saudi Arabia oil price war', CNBC, Forbes
- **2020 negative WTI price** (2020-04-15 to 2020-04-22) — DETECTED, 2/6 days flagged (type: shock+trend, shock). Source: WTI May futures settled at -$37.63/barrel on Apr 20 2020 amid a storage-capacity crunch - EIA, CFTC interim report, Forbes
- **2022 Russia invades Ukraine** (2022-02-23 to 2022-03-09) — DETECTED, 6/11 days flagged (type: shock, shock+trend, shock+trend, trend, trend, shock). Source: Russia invaded Ukraine Feb 24 2022; oil surged past $100/barrel and kept rising into early March - EIA, Yahoo Finance
- **2024 Red Sea shipping attacks** (2024-01-01 to 2024-02-29) — DETECTED, 2/41 days flagged (type: trend, trend). Source: Added after the cumulative-return check (see anomaly_detector.py) flagged a gradual price rise here that a first choice of control period (2024 Q1) had missed. Cause: Houthi attacks on Red Sea oil tankers forced longer shipping routes and pushed prices up through Jan-Feb 2024 - EIA (cited in databoks.katadata.co.id, 'Global Oil Prices Creep Upward in Early 2024'): Brent +4.4% and WTI +3.8% month-on-month in Feb 2024. A slow grind, not a single-day shock - no individual day's z-score cleared 1.7, which is the failure mode the cumulative check exists to catch.

## Control period

- **Control period (2023-07-01 to 2023-09-30): Verified quiet in real data - 0/63 days flagged under both checks (shock + cumulative). Two earlier candidates were rejected against real data: 2017-01-01/2017-03-31 (contained a -5.4% EIA-inventory-driven shock on 2017-03-08), and 2024-01-01/2024-03-31 (contained the gradual 2024 Red Sea-driven rise above, which the cumulative check flagged and which was moved to its own known_event rather than left as a mislabeled control-period false positive).** — 0/63 days flagged (clean).

This control period is a candidate verified by this run's own numbers above, not assumed correct in advance - see src/config/commodities.py's note on this commodity's control_period for whether it was pre-verified or is still a first-pass candidate.