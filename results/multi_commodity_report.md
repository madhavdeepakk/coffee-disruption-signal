# Explain or Refuse on Other Commodities

The same pipeline that is measured on coffee, run on crude oil, wheat, gold and silver. Produced by `python -m scripts.run_commodity_eval`. Rates carry 95% Wilson intervals; the samples are small and the intervals say so.

**These commodities have no answer key.** What is measured instead:

- *Explained a documented move*: the days the detector flags as a one-day shock inside each commodity's cited historical events, from 2017 on (the news search reaches no further back). A documented event is weaker than a checked label, and an explanation counted here has not been checked against the cited cause.
- *Explained an invented move*: quiet days given the size and direction of a real shock. Refusing is the only right answer, so every explanation is false. This needs no labels and is the one figure here that is comparable with coffee's.

Always explaining would score 100% on the first column and 100% on the second; always refusing would score 0% on both. A system is doing something only if the first figure is well above the second.

| Commodity | Explained a documented move | Explained an invented move (right answer: none) |
|---|---|---|
| Gold (COMEX) | 100.0% (8/8; 95% CI 67.6-100.0%) | 12.5% (1/8; 95% CI 2.2-47.1%) |

## Gold (COMEX)

Not run, because the news search does not reach back that far: 2013 April gold crash; 2016 Brexit referendum.

| Date | Event it belongs to | Move | Outcome | What was said, or why not |
|---|---|---|---|---|
| 2020-03-13 | 2020 pandemic dash for cash | -4.6% | explained | -4.6% day-over-day move (z-score -4.63), on top of -3.1% over the last 10 trading days on 2020-03-13. Reasons given by 2 reports from that day: - "Gold Falls 4 % As Futures Sold Aggressively While Buying Of Coins And Bars Surges" (gold-eagle.com, 2020-03-13). Reported move: down 4%. Reason: Futures sold aggressively while buying of coins and bars surges. Quote: "Gold Falls 4 % As Futures Sold Aggressively While Buyin |
| 2020-03-23 | 2020 pandemic dash for cash | +5.6% | explained | +5.6% day-over-day move (z-score 4.38), on top of -6.4% over the last 10 trading days on 2020-03-23. Reasons given by 2 reports from that day: - "Gold prices soar after very aggressive Fed stimulus expansion" (kitco.com, 2020-03-23). Reported move: up, size not stated. Reason: aggressive Fed stimulus expansion. Quote: "Gold prices soar after very aggressive Fed stimulus expansion" (source: fc3b27b19798875f) - "Gold p |
| 2020-08-07 | 2020 August reversal from the record | -2.0% | explained | -2.0% day-over-day move (z-score -2.21) on 2020-08-07. Reasons given by 2 reports from that day: - "Gold price retreats on stronger dollar" (mining.com, 2020-08-07). Reported move: down 1.2%. Reason: stronger dollar. Quote: "Gold price retreats on stronger dollar" (source: cdba8522dc8ea981) - "Dollar gains , stock rally ebbs after labor report" (financialpost.com, 2020-08-07). Reported move: down 2%. Reason: Dollar b |
| 2020-08-11 | 2020 August reversal from the record | -4.6% | explained | -4.6% day-over-day move (z-score -4.72) on 2020-08-11. Reasons given by 5 reports from that day: - "Gold Price today August 11 2020 Gold Futures Slump 2 Per Cent As Global Equity Markets , Dollar Rise" (ndtv.com, 2020-08-11). Reported move: down 2%. Reason: Global equity markets rise and dollar rise. Quote: "Gold Price today August 11 2020 Gold Futures Slump 2 Per Cent As Global Equity Markets , Dollar Rise" (source: |
| 2025-10-20 | 2025 October reversal from the record | +3.5% | explained | +3.5% day-over-day move (z-score 2.92), on top of +9.6% over the last 10 trading days on 2025-10-20. Reason given by 1 report from that day: - "Gold prices rise amid expectations of interest rate cut" (thepeninsulaqatar.com, 2025-10-20). Reported move: up 2.8%. Reason: expectations of US interest rate cut and safe‑haven demand. Quote: "Gold prices rise amid expectations of interest rate cut" (source: 68bde55bbb566c7d |
| 2025-10-21 | 2025 October reversal from the record | -5.7% | explained | -5.7% day-over-day move (z-score -5.36) on 2025-10-21. Reasons given by 2 reports from that day: - "Gold Price Today : Spot Gold Slides After Record High ; Futures Volatile as Dollar Firms" (el-balad.com, 2025-10-21). Reported move: down, size not stated. Reason: heavy profit-taking and a firmer U.S. dollar. Quote: "After touching fresh all-time highs in recent sessions, the metal faced heavy profit-taking and a firm |
| 2026-01-30 | 2026 January metals crash | -10.8% | explained | -10.8% day-over-day move (z-score -6.82) on 2026-01-30. Reasons given by 2 reports from that day: - "Gold accelerates losses , silver tumbles 20 % as precious metals rally evaporates" (finance.yahoo.com, 2026-01-30). Reported move: down 11%. Reason: after President Trump selected Kevin Warsh as the next Federal Reserve Chair. Quote: "after President Trump selected Kevin Warsh as the next Federal Reserve Chair" (sourc |
| 2026-02-03 | 2026 January metals crash | +6.1% | explained | +6.1% day-over-day move (z-score 2.85) on 2026-02-03. Reasons given by 3 reports from that day: - "Why gold futures are surging today : Gold price again closer to $5 , 000 – Why gold futures are surging today – How gold has outperformed stocks as gold price forecasts point to $6 , 000 in 2026" (economictimes.indiatimes.com, 2026-02-03). Reported move: up 7%. Reason: renewed demand for safety. Quote: "The immediate dr |

Invented-move test (pipeline version 11, seed 7): explained 12.5% (1/8; 95% CI 2.2-47.1%); 4 quiet day(s) still waiting for news.

- False explanation on 2017-02-07: +2.9% day-over-day move (z-score 2.32) on 2017-02-07. Reason given by 1 report from that day: - "Gold prices climb Rs200 on global cues , jewellers buying" (livemint.com, 2017-02-07). Reported move: up 1.29%. Reason: positive global cues and persistent buying by local jewellers. Quote: "driven by positive global cues and persistent buying by local jewellers." (source: 7214d973ab1dfaf8)

## Limits

- No answer key for these commodities. The documented dates were chosen by a rule (largest shock days inside cited events), not checked one by one, and nobody has read whether each explanation names the cited cause.
- The headline word lists that choose which articles are read were built and tuned on coffee. For the other commodities they were written once and not tuned (src/rag/market_report.py), so their reading is likely to be worse chosen.
- The decision rule and its settings were designed on coffee dates. Every figure here is therefore out of sample for the rule, which is what makes it worth having, and also why it may be lower than coffee's.
- A handful of dates per commodity. One date changes a rate by ten points or more.
