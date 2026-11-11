# Inflation Tracking and Commodity Pass-Through

United States, India and a global reference (the G20), against the commodities the anomaly detector watches. Produced by `python -m src.modeling.inflation` from `data/raw/inflation_series.csv` (see `src/data/cpi.py` for each source and its limits).

None of this is a forecast, and a regression on monthly figures does not show that a commodity move caused a change in consumer prices. It measures how the two have moved together and whether that helps estimate a figure before it is published.

## 1. Where inflation stands

| | Series | Latest month | Year on year | Last 3 months, annualised | Year on year, 12 months earlier |
|---|---|---|---|---|---|
| United States | all items | 2026-09 | +3.7% | +5.7% | +2.9% |
| United States | core (less food and energy) | 2026-09 | +2.6% | +3.7% | +3.1% |
| United States | food | 2026-09 | +2.5% | +2.6% | +3.2% |
| United States | energy | 2026-09 | +20.4% | +36.8% | +0.4% |
| United States | gasoline | 2026-10 | +37.4% | +72.0% | -1.6% |
| United States | coffee | 2026-09 | +7.1% | -12.4% | +20.8% |
| India | all items | 2025-03 | +3.0% | -1.9% | +4.2% |
| G20 | all items | 2025-03 | +4.2% | - | +7.2% |
| World (annual) | all items, calendar year | 2025 | +3.0% | - | +3.0% |
| United States (annual) | all items, calendar year | 2024 | +2.9% | - | +4.1% |
| India (annual) | all items, calendar year | 2025 | +2.4% | - | +5.0% |

India's monthly index and the G20 series come from an OECD database that stopped updating, so they end where the table says; the annual World Bank rows are the most recent figures available for them without a registered key.

## 2. Pass-through: how much of a commodity price rise reaches consumer prices

25 commodity-and-index pairs were chosen before any result was seen and all are reported. "Pass-through" is the sum of the coefficients on this month's and the previous twelve months' commodity price change: after a year, a 10% rise in the commodity goes with the stated change in the price index. The p-value tests that sum against zero with Newey-West errors; the Holm column corrects it for the 25 pairs tested, and only that column decides the verdict.

| Consumer prices | Commodity | Months | After a 10% rise, index moves | of which within 3 months | p | p (Holm) | R-squared |
|---|---|---|---|---|---|---|---|
| United States: all items | crude oil (WTI) | 306 (2001-02 to 2026-09) | +0.46% | +0.31% | <0.001 | <0.001 | 0.51 |
| United States: all items | food commodities (IMF index) | 306 (2001-02 to 2026-09) | +1.09% | +0.80% | <0.001 | <0.001 | 0.26 |
| United States: all items | wheat | 306 (2001-02 to 2026-09) | +0.43% | +0.21% | <0.001 | 0.009 | 0.16 |
| United States: all items | coffee (arabica) | 306 (2001-02 to 2026-09) | +0.28% | +0.11% | 0.043 | 0.468 | 0.09 |
| United States: all items | all commodities (IMF index) | 306 (2001-02 to 2026-09) | +0.77% | +0.63% | <0.001 | <0.001 | 0.53 |
| United States: all items | gold | 163 (2013-02 to 2026-10) | +0.22% | +0.02% | 0.248 | 1.000 | 0.08 |
| United States: all items | silver | 187 (2011-02 to 2026-10) | +0.27% | +0.11% | 0.005 | 0.078 | 0.09 |
| United States: energy | crude oil (WTI) | 306 (2001-02 to 2026-09) | +3.89% | +3.20% | <0.001 | <0.001 | 0.54 |
| United States: gasoline | crude oil (WTI) | 308 (2001-02 to 2026-09) | +5.98% | +5.66% | <0.001 | <0.001 | 0.56 |
| United States: food | food commodities (IMF index) | 306 (2001-02 to 2026-09) | +1.31% | +0.21% | <0.001 | <0.001 | 0.30 |
| United States: food | wheat | 306 (2001-02 to 2026-09) | +0.49% | +0.08% | <0.001 | 0.003 | 0.24 |
| United States: food | crude oil (WTI) | 306 (2001-02 to 2026-09) | +0.32% | +0.02% | 0.016 | 0.228 | 0.20 |
| United States: coffee | coffee (arabica) | 282 (2003-02 to 2026-09) | +2.27% | +0.46% | <0.001 | <0.001 | 0.26 |
| United States: core (less food and energy) | crude oil (WTI) | 306 (2001-02 to 2026-09) | +0.13% | +0.06% | 0.110 | 1.000 | 0.13 |
| United States: core (less food and energy) | all commodities (IMF index) | 306 (2001-02 to 2026-09) | +0.20% | +0.09% | 0.150 | 1.000 | 0.10 |
| India: all items | crude oil (Brent) in rupees | 290 (2001-02 to 2025-03) | -0.01% | +0.09% | 0.939 | 1.000 | 0.46 |
| India: all items | food commodities (IMF index) in rupees | 290 (2001-02 to 2025-03) | +0.35% | +0.30% | 0.522 | 1.000 | 0.46 |
| India: all items | wheat in rupees | 290 (2001-02 to 2025-03) | +0.25% | +0.07% | 0.274 | 1.000 | 0.48 |
| India: all items | coffee (arabica) in rupees | 290 (2001-02 to 2025-03) | -0.07% | -0.04% | 0.737 | 1.000 | 0.46 |
| India: all items | all commodities (IMF index) in rupees | 290 (2001-02 to 2025-03) | +0.04% | +0.16% | 0.890 | 1.000 | 0.45 |
| India: all items | gold in rupees | 146 (2013-02 to 2025-03) | -0.38% | +0.13% | 0.175 | 1.000 | 0.55 |
| India: all items | silver in rupees | 170 (2011-02 to 2025-03) | -0.11% | +0.12% | 0.603 | 1.000 | 0.53 |
| G20: all items | all commodities (IMF index) (year on year) | 279 (2002-01 to 2025-03) | +0.45% | +0.23% | 0.028 | 0.365 | 0.23 |
| G20: all items | energy commodities (IMF index) (year on year) | 279 (2002-01 to 2025-03) | +0.32% | +0.16% | 0.005 | 0.078 | 0.31 |
| G20: all items | food commodities (IMF index) (year on year) | 279 (2002-01 to 2025-03) | +0.77% | +0.34% | 0.032 | 0.390 | 0.20 |

For the G20 row the target is already a year-on-year rate and the commodity change is year on year too, at lags of 0, 3, 6, 9 and 12 months; the months overlap, so read that row as an association, not as a share passed through.

## 3. Does the commodity price improve the estimate made before publication?

Each month from the first test month on, inflation is estimated twice from the months before it only: from past inflation alone (the baseline), and from past inflation plus the commodity's price changes up to that month, which are known when the month ends and before the CPI is published. A model is better only if its error on those months is lower. Positive improvement means lower error.

| Consumer prices | Commodity | Test months | Baseline error (RMSE, pp) | With the commodity | Improvement | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| United States: all items | crude oil (WTI) | 216 (2008-05 to 2026-09) | 0.279 | 0.226 | +19.0% | 0.033 | 0.667 |
| United States: all items | food commodities (IMF index) | 216 (2008-05 to 2026-09) | 0.279 | 0.261 | +6.4% | 0.068 | 1.000 |
| United States: all items | wheat | 216 (2008-05 to 2026-09) | 0.279 | 0.278 | +0.6% | 0.391 | 1.000 |
| United States: all items | coffee (arabica) | 216 (2008-05 to 2026-09) | 0.279 | 0.285 | -1.9% | 0.889 | 1.000 |
| United States: all items | all commodities (IMF index) | 216 (2008-05 to 2026-09) | 0.279 | 0.223 | +20.3% | 0.066 | 1.000 |
| United States: all items | gold | 73 (2020-05 to 2026-10) | 0.286 | 0.290 | -1.4% | 0.962 | 1.000 |
| United States: all items | silver | 97 (2018-05 to 2026-10) | 0.266 | 0.268 | -0.7% | 0.677 | 1.000 |
| United States: energy | crude oil (WTI) | 216 (2008-05 to 2026-09) | 3.022 | 2.187 | +27.6% | 0.005 | 0.108 |
| United States: gasoline | crude oil (WTI) | 221 (2008-05 to 2026-09) | 5.567 | 3.844 | +31.0% | 0.003 | 0.068 |
| United States: food | food commodities (IMF index) | 216 (2008-05 to 2026-09) | 0.197 | 0.198 | -0.6% | 0.568 | 1.000 |
| United States: food | wheat | 216 (2008-05 to 2026-09) | 0.197 | 0.198 | -0.3% | 0.603 | 1.000 |
| United States: food | crude oil (WTI) | 216 (2008-05 to 2026-09) | 0.197 | 0.202 | -2.7% | 0.861 | 1.000 |
| United States: coffee | coffee (arabica) | 180 (2011-05 to 2026-09) | 0.925 | 0.981 | -6.1% | 0.901 | 1.000 |
| United States: core (less food and energy) | crude oil (WTI) | 216 (2008-05 to 2026-09) | 0.121 | 0.118 | +3.0% | 0.128 | 1.000 |
| United States: core (less food and energy) | all commodities (IMF index) | 216 (2008-05 to 2026-09) | 0.121 | 0.120 | +1.1% | 0.254 | 1.000 |
| India: all items | crude oil (Brent) in rupees | 203 (2008-05 to 2025-03) | 0.615 | 0.621 | -0.9% | 0.936 | 1.000 |
| India: all items | food commodities (IMF index) in rupees | 203 (2008-05 to 2025-03) | 0.615 | 0.620 | -0.7% | 0.808 | 1.000 |
| India: all items | wheat in rupees | 203 (2008-05 to 2025-03) | 0.615 | 0.620 | -0.8% | 0.805 | 1.000 |
| India: all items | coffee (arabica) in rupees | 203 (2008-05 to 2025-03) | 0.615 | 0.626 | -1.7% | 0.999 | 1.000 |
| India: all items | all commodities (IMF index) in rupees | 203 (2008-05 to 2025-03) | 0.615 | 0.623 | -1.2% | 0.970 | 1.000 |
| India: all items | gold in rupees | 59 (2020-05 to 2025-03) | 0.444 | 0.467 | -5.3% | 0.975 | 1.000 |
| India: all items | silver in rupees | 83 (2018-05 to 2025-03) | 0.526 | 0.535 | -1.7% | 0.861 | 1.000 |
| G20: all items | all commodities (IMF index) | 192 (2009-04 to 2025-03) | 0.254 | 0.230 | +9.4% | 0.012 | 0.260 |
| G20: all items | energy commodities (IMF index) | 192 (2009-04 to 2025-03) | 0.254 | 0.234 | +8.0% | 0.010 | 0.228 |
| G20: all items | food commodities (IMF index) | 192 (2009-04 to 2025-03) | 0.254 | 0.241 | +5.0% | 0.006 | 0.134 |

The baseline here is not a strawman: it is the same autoregression with the commodity terms removed, refitted every month. The test is one-sided (is the model's squared error lower?) and is known to be conservative when one model contains the other, as here.

## 4. What the numbers say

- **Pass-through measured, but it does not improve the estimate** (9 of 25): crude oil (WTI) -> United States all items; food commodities (IMF index) -> United States all items; wheat -> United States all items; all commodities (IMF index) -> United States all items; crude oil (WTI) -> United States energy; crude oil (WTI) -> United States gasoline; food commodities (IMF index) -> United States food; wheat -> United States food; coffee (arabica) -> United States coffee.
- **No relationship that can be told from chance** (16 of 25): coffee (arabica) -> United States all items; gold -> United States all items; silver -> United States all items; crude oil (WTI) -> United States food; crude oil (WTI) -> United States core (less food and energy); all commodities (IMF index) -> United States core (less food and energy); crude oil (Brent) (rupees) -> India all items; food commodities (IMF index) (rupees) -> India all items; wheat (rupees) -> India all items; coffee (arabica) (rupees) -> India all items; all commodities (IMF index) (rupees) -> India all items; gold (rupees) -> India all items; silver (rupees) -> India all items; all commodities (IMF index) -> G20 all items; energy commodities (IMF index) -> G20 all items; food commodities (IMF index) -> G20 all items.

Verdicts use the Holm-corrected p-values at 5%, which is strict with this many pairs: several nowcast rows show a large improvement with an uncorrected p-value under 0.05 and are still not counted. A pair listed under "no relationship" is a result, not a gap: for that commodity and that price index the monthly record shows nothing that chance alone would not.

## 5. Limits

- India: the all-items index only, not seasonally adjusted (month-of-year terms are included), ending March 2025. No food or fuel sub-index, which is where pass-through would be expected to show first.
- Fuel prices in India were administered for much of the sample, which weakens the link between the world oil price and what a household paid.
- Monthly averages hide timing within the month; the daily anomaly detector and this monthly analysis are separate instruments.
- Gold and silver start in 2012 and 2010 (monthly averages of the daily futures series), so their rows rest on fewer months.
- The lag lengths, the training window and the pairs were fixed in the code before the results were produced and were not adjusted afterwards.
