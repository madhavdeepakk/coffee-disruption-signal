# Price Outlook — Analog Scenarios + Large-Move Risk

EXPERIMENTAL and NOT financial advice. This does not predict a price; it reports how the price behaved after historically-similar days (a range, not a guess) and whether an unusually large move is more likely than normal right now. Direction over these horizons is near-random; the value here is the honest range and risk level, with accuracy shown. See src/modeling/outlook.py.

## Coffee (Arabica)

Features used: ret_1, ret_5, ret_10, mom_10, vol_20, zscore_75, dist_ma20, weather_dryness  (includes weather dryness)

### Next ~week (5 trading days)
- **Large-move risk model** (walk-forward, 1643 test days): on the days it flagged as elevated, a large move actually happened 25% of the time, vs a 27% base rate — a **lift of 0.92x** (it flagged 47% of days and caught 44% of all large moves). Lift above 1 means real skill.
- **Current large-move risk**: normal (model probability 11% vs a typical 25%; 'large' means a 5-day move bigger than ~5.7%). As of 2026-09-11.
- **Historical analogs** (25 most similar past days): the price then moved a median of +2.7% over the next 5 days, with a typical range of -3.4% to +7.3% (full span -5.2% to +9.2%; 56% rose).

### Next ~month (20 trading days)
- **Large-move risk model** (walk-forward, 1628 test days): on the days it flagged as elevated, a large move actually happened 24% of the time, vs a 25% base rate — a **lift of 0.92x** (it flagged 49% of days and caught 46% of all large moves). Lift above 1 means real skill.
- **Current large-move risk**: normal (model probability 18% vs a typical 25%; 'large' means a 20-day move bigger than ~11.5%). As of 2026-09-11.
- **Historical analogs** (25 most similar past days): the price then moved a median of +3.0% over the next 20 days, with a typical range of -5.9% to +16.5% (full span -12.3% to +20.7%; 56% rose).
