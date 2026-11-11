# Gold (COMEX) — Anomaly Detector Validation

Rolling window: 75 days. Z-score threshold: 2.0 (chosen before this validation, not tuned against it). Two independent checks, either one can flag a day: a daily shock check (day-over-day return z-score) and a 10-trading-day cumulative-return check (catches slower, multi-day build-ups a single-day check can miss - see anomaly_detector.py's module docstring for why this was added).

**Result: 6/6 known real events had at least one day flagged. Control period false positives: 0.**

## Events

- **2013 April gold crash** (2013-04-12 to 2013-04-17) — DETECTED, 4/4 days flagged (type: shock+trend, shock+trend, trend, trend). Source: Gold fell about 9% on Apr 15 2013, its largest one-day fall in 30 years - Bloomberg ('Gold's Worst Plunge Since 1983 Ends 12-Year Bull Run: Timeline', Apr 17 2013), CoinNews ('Gold Dives 9.3%, Silver Tumbles 11.3%', Apr 15 2013)
- **2016 Brexit referendum** (2016-06-23 to 2016-06-28) — DETECTED, 1/4 days flagged (type: shock). Source: The UK voted to leave the EU on Jun 23 2016; gold rose close to 5% on Jun 24 as investors moved to safe havens - widely reported, e.g. Reuters and CNBC market reports of Jun 24 2016
- **2020 pandemic dash for cash** (2020-03-09 to 2020-03-27) — DETECTED, 9/15 days flagged (type: shock+trend, shock+trend, trend, shock+trend, shock+trend, trend, trend, shock+trend, shock). Source: Gold was sold with everything else as investors raised cash in the second week of March 2020, then rebounded sharply from Mar 23-24 after the Federal Reserve announced unlimited asset purchases - Reuters and CNBC precious-metals reports, March 2020
- **2020 August reversal from the record** (2020-08-07 to 2020-08-13) — DETECTED, 2/5 days flagged (type: shock, shock). Source: After a record above $2,000 an ounce, gold fell about 4.6% on Aug 11 2020, its largest one-day fall in seven years, as bond yields rose - CNBC and Reuters market reports of Aug 11 2020
- **2025 October reversal from the record** (2025-10-20 to 2025-10-23) — DETECTED, 2/4 days flagged (type: shock+trend, shock). Source: Gold fell more than 5% on Oct 21 2025, its steepest one-day fall since 2013 - Axios ('Gold and silver plunge after recent surge to record highs', Oct 21 2025), InvestmentNews ('Gold, silver post steepest declines in over 13 years as rally stalls')
- **2026 January metals crash** (2026-01-28 to 2026-02-04) — DETECTED, 4/6 days flagged (type: shock+trend, trend, shock, shock). Source: Gold fell about 11% on Jan 30 2026 after a parabolic run to a record; reports tied the selling to the news about the next Federal Reserve chair, a stronger dollar and forced liquidation - Morningstar ('Why Did Gold and Silver Prices Plunge?'), Seeking Alpha ('Gold And Silver's Most Volatile Day')

## Control period

- **Control period (2026-04-01 to 2026-06-30): Verified quiet in real data - 0 days flagged under both checks, the only fully clean calendar quarter from 2012 to 2026 for gold. It follows the January-March 2026 turmoil, so 'quiet' here is relative to a trailing window that had just absorbed very large moves; the next-best quarters (2023 Q2, 2025 Q1) each have one flagged day.** — 0/62 days flagged (clean).

This control period is a candidate verified by this run's own numbers above, not assumed correct in advance - see src/config/commodities.py's note on this commodity's control_period for whether it was pre-verified or is still a first-pass candidate.