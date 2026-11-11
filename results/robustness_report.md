# Robustness Evaluation

How often the system explains a move, under conditions where explaining is and is not the right answer. Each condition changes one thing about a normal run; everything goes through the same code path as the CLI.

12 trials (0 faults, excluded from rates). Real anomaly dates: 73. Quiet placebo days: 12. Retrieval for real dates: live. Seed 11.
Models that answered: openai/gpt-oss-120b (9), qwen/qwen3.8-27b (3).

![Explained rate by condition](robustness_chart.png)

## Blind-evidence rule

The model reads the recent documents without being told the move, and a fixed rule decides (src/rag/blind_evidence.py). No direction guard is applied on this path, so the two columns are the same.

| Condition | Right answer | Explained: model alone | Explained: with direction guard |
|---|---|---|---|
| Quiet day, real news, invented move | refuse | 16.7% (2/12; 95% CI 4.7-44.8%) | 16.7% (2/12; 95% CI 4.7-44.8%) |

**False-explanation rate.** On quiet days with an invented move the system explained 16.7% (2/12; 95% CI 4.7-44.8%). Nothing happened on those days, so every one of these is an explanation of a move that did not take place. The relevance filter passed 100.0% (12/12; 95% CI 75.7-100.0%) of them.

What it built them from, for a person to read:

| Quiet day | Invented move | What the system said |
|---|---|---|
| 2023-04-04 | +4.7% day-over-day move (z-score 2.26) | +4.7% day-over-day move (z-score 2.26) on 2023-04-04. Reason given by 1 report from that day: - "Arabica Coffee Gains on Strength in the Brazilian Real and Tighter ICE Inventories" (theglobeandmail.com, 2023-04-04). Reported move: up 3.37%. Reason: Strength in the Brazilian real boosted arabica prices. Quote: "Strength in the Brazilian real boosted arabica prices after the real Monday rose to a 2-month high" (source: f4a842ee2307a93e) |
| 2025-06-11 | -3.7% day-over-day move (z-score -2.07) | -3.7% day-over-day move (z-score -2.07) on 2025-06-11. Reason given by 1 report from that day: - "Cacau sobe em Nova York após ajustes técnicos" (globorural.globo.com, 2025-06-11). Reported move: down 1.24%. Reason: progress of the harvest in Brazil, especially arabica. Quote: "diante do andamento da colheita no Brasil, especialmente da variedade arábica" (source: 6d54730a21a2ccbd) |

**The same readings under other versions of the rule.** No model is called again: each row re-applies the rule to the readings these trials stored. A stricter rule is worth adopting only if it explains fewer invented moves without explaining fewer real ones, and it has to be confirmed on days it was not chosen on (a different `--seed` draws other quiet days).

| Rule | Invented moves explained (right answer: none) |
|---|---|
| As the trials ran | 16.7% (2/12; 95% CI 4.7-44.8%) |
| Any report with a reason counts (the rule before version 11) | 66.7% (8/12; 95% CI 39.1-86.2%) |
| Ignore reports of a move under a quarter the size of the claimed one | 58.3% (7/12; 95% CI 32.0-80.7%) |
| Only reports of a price move, not background events | 66.7% (8/12; 95% CI 39.1-86.2%) |
| Only articles that cleared the relevance filter itself | 8.3% (1/12; 95% CI 1.5-35.4%) |

## Limits

- Sample sizes are small; every rate carries its 95% Wilson interval.
- A placebo day's news is whatever was retrievable for that window when this was run. Live retrieval is rate-limited and not perfectly repeatable; the retrieved evidence is cached so a re-report uses the same documents.
- An explanation on a real anomaly is counted as explained, not as correct. Whether it names the documented cause is for the side-by-side table and a human.
- The direction guard and prompt v2 were written after two should-refuse dates were found explained in earlier runs. The placebo days are dates those changes were not designed on. The flip and control conditions reuse the real anomaly dates, including those two, with inputs the changes were not designed on.
- The guard first withheld on a tie as well. That was dropped after it withheld three explanations on dates labelled as having a documented cause (1-1, 1-1 and 3-3 tallies), so the guard column is scored under the current rule: more accepted documents against the move than with it. Cached trials are re-scored, not re-run.
- The blind-evidence rule was written after the earlier runs had been read, and its two settings (recent means the same day or the two trading days before; the newest and most direct group of evidence decides) were fixed before any blind reading was run. The flip condition cannot be passed by tuning those settings: the reading is made without the direction, so the same documents cannot support both a rise and a fall.
- Gemini 3 models are run at their default temperature (Google advises against lowering it), so trials answered by one are not exactly repeatable.
