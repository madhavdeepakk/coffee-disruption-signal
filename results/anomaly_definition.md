# What counts as an anomaly, and what the current definition costs

Produced by `python -m scripts.anomaly_definition`. Nothing here changes the detector: every evaluation figure in `results/` was drawn at the current rule, so these columns sit beside it rather than replacing it.

## 1. Episodes, not days

The 10-day cumulative check tags every day inside a build-up, so one event becomes a run. Both columns describe the same detector; only the second is a fair summary of how often it speaks.

| Commodity | Scorable days | Flagged days | Share | Shock only | Trend only | Both checks | Episodes | Per year |
|---|---|---|---|---|---|---|---|---|
| Coffee (Arabica) | 2122 | 279 | 13.1% | 101 | 149 | 29 | 102 | 11.6 |
| Crude Oil (WTI) | 3366 | 418 | 12.4% | 147 | 214 | 59 | 162 | 11.8 |
| Wheat (Chicago SRW) | 4373 | 519 | 11.9% | 190 | 270 | 59 | 203 | 11.5 |
| Gold (COMEX) | 3659 | 491 | 13.4% | 185 | 264 | 42 | 187 | 12.6 |
| Silver (COMEX) | 4162 | 515 | 12.4% | 209 | 255 | 51 | 222 | 13.2 |

Say "about one episode a month", not "one day in eight". The second is true and misleading: it counts a ten-day build-up as ten anomalies.

## 2. The bar is statistical, not economic

A z-score asks whether a move is unusual for this market lately. It does not ask whether anyone would act on it, and the same bar is a different event in each market.

| Commodity | Median daily move | Median move on a SHOCK day | Smallest shock move | Materiality bar (3x median) |
|---|---|---|---|---|
| Coffee (Arabica) | 1.40% | 4.85% | 2.34% | 4.21% |
| Crude Oil (WTI) | 1.30% | 4.89% | 1.85% | 3.91% |
| Wheat (Chicago SRW) | 1.17% | 4.31% | 1.56% | 3.51% |
| Gold (COMEX) | 0.52% | 2.25% | 1.08% | 1.56% |
| Silver (COMEX) | 0.96% | 4.42% | 1.96% | 2.87% |

The move columns are over days the DAILY check fired. A trend-only day is flagged on its ten-day cumulative move and its own daily move is often near zero, so a move statistic over all flagged days would be meaningless. That is the same conflation the episode table is about.

| Commodity | Daily check | Material | Both | Material, not flagged | Flagged, not material | Overlap (Jaccard) |
|---|---|---|---|---|---|---|
| Coffee (Arabica) | 130 | 143 | 103 | 38 | 174 | 0.61 |
| Crude Oil (WTI) | 204 | 314 | 143 | 146 | 250 | 0.38 |
| Wheat (Chicago SRW) | 249 | 325 | 187 | 121 | 315 | 0.48 |
| Gold (COMEX) | 227 | 436 | 205 | 207 | 262 | 0.45 |
| Silver (COMEX) | 260 | 545 | 227 | 287 | 257 | 0.39 |

These are largely different sets of days. Requiring BOTH clauses - statistically unusual and large enough to act on - is the narrower definition, and the next table says what it costs in cited events.

### Cited events caught, by definition

| Commodity | Events | Current detector | Daily check only | Material only | Both |
|---|---|---|---|---|---|
| Coffee (Arabica) | 4 | 4 | 4 | 4 | 4 |
| Crude Oil (WTI) | 5 | 4 | 3 | 5 | 3 |
| Wheat (Chicago SRW) | 4 | 4 | 4 | 4 | 4 |
| Gold (COMEX) | 6 | 6 | 6 | 6 | 6 |
| Silver (COMEX) | 6 | 6 | 6 | 6 | 6 |

### Thresholds calibrated to this data, not to a normal distribution

Under a normal law |z| >= 2 would fire on 4.55% of days. It does not, because returns are fat-tailed. Quote the measured column, never the 4.55%, and if a stated firing rate is wanted, read the z off this table.

| Commodity | Measured at \|z\|>=2 | Normal law says | z for 5% | z for 2% | z for 1% | Excess kurtosis |
|---|---|---|---|---|---|---|
| Coffee (Arabica) | 6.11% | 4.55% | 2.13 | 2.64 | 3.14 | 1.7 |
| Crude Oil (WTI) | 6.12% | 4.55% | 2.13 | 2.77 | 3.49 | 335.1 |
| Wheat (Chicago SRW) | 5.69% | 4.55% | 2.08 | 2.63 | 3.08 | 1.8 |
| Gold (COMEX) | 6.20% | 4.55% | 2.20 | 2.80 | 3.31 | 5.8 |
| Silver (COMEX) | 6.25% | 4.55% | 2.22 | 2.94 | 3.54 | 4.4 |

## 3. Does a flagged day change what follows?

The label-free test. Median realised volatility of the 20 trading days after a flagged day, against the same after any day, and against the same after a large move the detector did NOT flag. No event list, no news, no answer key, nothing selected by hand.

| Commodity | After a flag | After any day | Lift | After a large unflagged move | Lift over that |
|---|---|---|---|---|---|
| Coffee (Arabica) | 0.02368 | 0.02132 | 1.11x | 0.02184 | 1.08x |
| Crude Oil (WTI) | 0.02308 | 0.01915 | 1.20x | 0.02854 | 0.81x |
| Wheat (Chicago SRW) | 0.01975 | 0.01765 | 1.12x | 0.02376 | 0.83x |
| Gold (COMEX) | 0.00921 | 0.00884 | 1.04x | 0.01137 | 0.81x |
| Silver (COMEX) | 0.01740 | 0.01664 | 1.05x | 0.02371 | 0.73x |

**Read the last column carefully.** A lift above 1 over the baseline says a flagged day is followed by a more turbulent month than an ordinary day. A lift below 1 in the final column says a plain large move predicts that turbulence BETTER than the flag does - which, where it happens, is evidence that the statistical bar is picking days that are unusual without being consequential. This is the strongest argument in the project for adding the materiality clause, and it is also the argument most likely to be made by an examiner, so it belongs in the report.

## What this cannot tell you

**Precision.** Most flagged days sit outside every cited event window and carry no label, so the share of flags that are nothing has no answer here and is not estimated. `--sample` writes a random sample of flagged days to a CSV for a person to judge; labelling it is the only thing that would produce a precision figure. Sampling is random within commodity, not weighted towards large moves, because weighting is how such a figure gets flattered.

**Whether these definitions are better.** The materiality multiple here is a stated choice, not a fitted one. It was not tuned against the cited events, and the event-recall table shows what it costs rather than being adjusted until that cost disappeared.