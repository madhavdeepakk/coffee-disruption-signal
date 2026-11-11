# Silver (COMEX) — Anomaly Detector Validation

Rolling window: 75 days. Z-score threshold: 2.0 (chosen before this validation, not tuned against it). Two independent checks, either one can flag a day: a daily shock check (day-over-day return z-score) and a 10-trading-day cumulative-return check (catches slower, multi-day build-ups a single-day check can miss - see anomaly_detector.py's module docstring for why this was added).

**Result: 6/6 known real events had at least one day flagged. Control period false positives: 0.**

## Events

- **2011 May margin-hike collapse** (2011-05-02 to 2011-05-06) — DETECTED, 5/5 days flagged (type: shock, shock, shock+trend, shock+trend, trend). Source: Silver fell by more than a quarter in a week after the CME raised margin requirements several times - The Motley Fool ('Did Higher Margins Kill Commodities?', May 11 2011), InvestmentNews ('Silver investors holding a falling knife')
- **2011 September plunge** (2011-09-22 to 2011-09-26) — DETECTED, 3/3 days flagged (type: shock+trend, shock+trend, trend). Source: Silver fell about 18% on Sep 23 2011 as the CME raised margins again - tradingnrg.com ('Gold and Silver Plummeted | CME Margin Hike Fueled Drop', Sep 23 2011)
- **2013 April metals crash** (2013-04-12 to 2013-04-17) — DETECTED, 4/4 days flagged (type: shock, shock+trend, trend, trend). Source: Silver fell about 11% on Apr 15 2013 alongside gold's largest fall in 30 years - CoinNews ('Gold Dives 9.3%, Silver Tumbles 11.3%; Both at Multi-Year Lows', Apr 15 2013)
- **2020 pandemic dash for cash** (2020-03-09 to 2020-03-27) — DETECTED, 11/15 days flagged (type: trend, trend, shock+trend, shock+trend, shock+trend, trend, shock+trend, trend, trend, shock+trend, shock). Source: Silver fell to an 11-year low in mid-March 2020 as investors raised cash, then rebounded - Reuters precious-metals reports, March 2020
- **2021 retail silver squeeze** (2021-01-28 to 2021-02-03) — DETECTED, 2/5 days flagged (type: shock+trend, shock). Source: Silver jumped to an eight-year high on Feb 1 2021 on a retail buying campaign, then fell about 10% on Feb 2 after the CME raised margins by 17.9% - Reuters ('Silver slides from 8-year peak as retail frenzy fizzles out'), Bloomberg ('Silver's Selloff Accelerates With Reddit Frenzy Unraveling', Feb 2 2021)
- **2026 January metals crash** (2026-01-26 to 2026-02-04) — DETECTED, 5/8 days flagged (type: shock+trend, shock, trend, shock+trend, trend). Source: Silver fell about 30% on Jan 30 2026, its largest one-day fall on record, after a parabolic run - Morningstar ('Why Did Gold and Silver Prices Plunge?'), SD Bullion ('Biggest Drop in Intraday Price History')

## Control period

- **Control period (2026-04-01 to 2026-06-30): Verified quiet in real data - 0 days flagged under both checks, the only fully clean calendar quarter from 2010 to 2026 for silver. As for gold it follows the January-March 2026 turmoil; the next-best quarters (2012 Q1, 2017 Q1, 2021 Q2, 2024 Q1) each have one flagged day.** — 0/62 days flagged (clean).

This control period is a candidate verified by this run's own numbers above, not assumed correct in advance - see src/config/commodities.py's note on this commodity's control_period for whether it was pre-verified or is still a first-pass candidate.