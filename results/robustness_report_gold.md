# Robustness Evaluation - gold

How often the system explains a move, under conditions where explaining is and is not the right answer. Each condition changes one thing about a normal run; everything goes through the same code path as the CLI.

8 trials (0 faults, excluded from rates). Real anomaly dates: 8. Quiet placebo days: 8. Retrieval for real dates: live. Seed 7.
Models that answered: openai/gpt-oss-120b (8).

![Explained rate by condition](robustness_chart_gold.png)

## Blind-evidence rule

The model reads the recent documents without being told the move, and a fixed rule decides (src/rag/blind_evidence.py). No direction guard is applied on this path, so the two columns are the same.

| Condition | Right answer | Explained: model alone | Explained: with direction guard |
|---|---|---|---|
| Quiet day, real news, invented move | refuse | 12.5% (1/8; 95% CI 2.2-47.1%) | 12.5% (1/8; 95% CI 2.2-47.1%) |

**False-explanation rate.** On quiet days with an invented move the system explained 12.5% (1/8; 95% CI 2.2-47.1%). Nothing happened on those days, so every one of these is an explanation of a move that did not take place. The relevance filter passed 100.0% (8/8; 95% CI 67.6-100.0%) of them.

What it built them from, for a person to read:

| Quiet day | Invented move | What the system said |
|---|---|---|
| 2017-02-07 | +2.9% day-over-day move (z-score 2.32) | +2.9% day-over-day move (z-score 2.32) on 2017-02-07. Reason given by 1 report from that day: - "Gold prices climb Rs200 on global cues , jewellers buying" (livemint.com, 2017-02-07). Reported move: up 1.29%. Reason: positive global cues and persistent buying by local jewellers. Quote: "driven by positive global cues and persistent buying by local jewellers." (source: 7214d973ab1dfaf8) |

**The same readings under other versions of the rule.** No model is called again: each row re-applies the rule to the readings these trials stored. A stricter rule is worth adopting only if it explains fewer invented moves without explaining fewer real ones, and it has to be confirmed on days it was not chosen on (a different `--seed` draws other quiet days).

| Rule | Invented moves explained (right answer: none) |
|---|---|
| As the trials ran | 12.5% (1/8; 95% CI 2.2-47.1%) |
| Any report with a reason counts (the rule before version 11) | 50.0% (4/8; 95% CI 21.5-78.5%) |
| Ignore reports of a move under a quarter the size of the claimed one | 50.0% (4/8; 95% CI 21.5-78.5%) |
| Only reports of a price move, not background events | 50.0% (4/8; 95% CI 21.5-78.5%) |
| Only articles that cleared the relevance filter itself | 0.0% (0/8; 95% CI 0.0-32.4%) |

## Quiet days not run yet

4 quiet day(s) were left out because the news API had not answered all of their requests: 2019-10-18, 2020-04-27, 2022-04-06, 2025-06-11. A day run on part of its news has less to build a false explanation from, so it is not run until the news is complete. Run the same command again to pick them up.

## Limits

- Sample sizes are small; every rate carries its 95% Wilson interval.
- A placebo day's news is whatever was retrievable for that window when this was run. Live retrieval is rate-limited and not perfectly repeatable; the retrieved evidence is cached so a re-report uses the same documents.
- An explanation on a real anomaly is counted as explained, not as correct. Whether it names the documented cause is for the side-by-side table and a human.
- The direction guard and prompt v2 were written after two should-refuse dates were found explained in earlier runs. The placebo days are dates those changes were not designed on. The flip and control conditions reuse the real anomaly dates, including those two, with inputs the changes were not designed on.
- The guard first withheld on a tie as well. That was dropped after it withheld three explanations on dates labelled as having a documented cause (1-1, 1-1 and 3-3 tallies), so the guard column is scored under the current rule: more accepted documents against the move than with it. Cached trials are re-scored, not re-run.
- The blind-evidence rule was written after the earlier runs had been read, and its two settings (recent means the same day or the two trading days before; the newest and most direct group of evidence decides) were fixed before any blind reading was run. The flip condition cannot be passed by tuning those settings: the reading is made without the direction, so the same documents cannot support both a rise and a fall.
- Gemini 3 models are run at their default temperature (Google advises against lowering it), so trials answered by one are not exactly repeatable.
