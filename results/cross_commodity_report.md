# Cross-Commodity Check: which flagged days are macro days

The base detector flags a day when one price moved unusually. It does not look at any other market, so it cannot tell a supply disruption from a day the dollar moved and everything repriced. This counts how many of the five watched commodities flag on the same date.

A flagged day is called **macro** when at least 2 of the other commodities flagged it too, and **commodity-specific** otherwise. Only days on which that many other commodities actually have price data are scored, so an early wheat flag is not called specific merely because coffee's series had not started. Days the series switched futures contract are dropped.

| Commodity | Flagged | Scorable | Macro | Commodity-specific | Macro share | Metals counted once |
|---|---|---|---|---|---|---|
| coffee | 279 | 279 | 48 | 231 | 17.2% | 9.7% |
| crude_oil | 420 | 420 | 62 | 358 | 14.8% | 9.0% |
| wheat | 519 | 434 | 58 | 376 | 13.4% | 7.7% |
| gold | 491 | 489 | 106 | 383 | 21.7% | 5.6% |
| silver | 515 | 456 | 93 | 363 | 20.4% | 5.6% |

The last column counts gold and silver as one market. They are substitutes and move together for reasons that are not macro, so the plain macro share overstates agreement for them.

## How the verdict moves with the threshold

Share of flagged days called macro, as the number of other commodities required rises. A share that falls away quickly means the agreement is thin.

| Commodity | 1 other | 2 others | 3 others | 4 others |
|---|---|---|---|---|
| coffee | 48% | 17% | 3% | 0% |
| crude_oil | 37% | 15% | 3% | 0% |
| wheat | 30% | 13% | 2% | 0% |
| gold | 59% | 22% | 3% | 0% |
| silver | 60% | 20% | 4% | 0% |

## Which pairs agree

Jaccard: of the days either flagged, the share on which both did. Two markets with a shared driver score high; two with unrelated physical causes should not.

| Pair | Days both have data | Both flagged | Jaccard |
|---|---|---|---|
| gold / silver | 3734 | 218 | 0.298 |
| coffee / silver | 2196 | 57 | 0.115 |
| coffee / gold | 2197 | 60 | 0.113 |
| crude_oil / silver | 3443 | 74 | 0.095 |
| crude_oil / gold | 3443 | 70 | 0.085 |
| gold / wheat | 3692 | 64 | 0.075 |
| coffee / wheat | 2180 | 37 | 0.073 |
| silver / wheat | 4196 | 68 | 0.072 |
| coffee / crude_oil | 2181 | 37 | 0.072 |
| crude_oil / wheat | 3443 | 49 | 0.063 |

## The answer key's own dates

3 of the 16 scored dates in `labels_by_written_rule.csv` are days the complex moved together.

| Date | Label | Verdict | Also flagged | Cause recorded in the key |
|---|---|---|---|---|
| 2024-04-30 | EXPLAIN | macro | gold, silver | A stronger dollar weighed on commodities, and losses grew after updated forecasts raised the ch… |
| 2025-02-10 | EXPLAIN | macro | wheat, gold | Reports of a hot, dry weather system forming over Brazil's coffee areas added to supply fears, … |
| 2025-04-07 | EXPLAIN | macro | crude_oil, gold, silver | A sell-off across most commodities on trade-war fears after the US tariff announcements; losses… |

On 2 of those 3, the cause the key itself records is a macro one, in its own words. The key marks a date EXPLAIN when a documented cause exists; it never asks whether the cause is specific to this market. A rule built to require a report about this market will refuse such a date and be scored wrong for doing so.

**Nothing here relabels anything.** Changing a label after seeing which dates a rule failed is the move this project exists to refuse. This is evidence that the key needs a third category - a move that is real, documented, and not about this commodity - for the people who own the key to decide on.

## What this does and does not establish

- **Observed.** The counts above are read from the stored detections. No model, no network, no estimate.
- **Not tested.** That a macro day is *caused* by the dollar or by risk appetite is not shown here. Agreement across commodities is consistent with a shared driver; it does not name one. `src/modeling/factor_model.py` measures that directly.
- A commodity-specific verdict is the weaker of the two. It says only that the other four did not flag, which a shared driver too small to flag them would also produce.