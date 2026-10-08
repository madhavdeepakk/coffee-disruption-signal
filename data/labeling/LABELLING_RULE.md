# How to label a date

The labels are the answer key the system is graded against. They have to come from the news of the time, with the report relied on recorded beside each label and each row checked by a person. They never come from the system's own output, or the system would be marking its own work.

## For each date

1. Open the search link in the row. It shows coffee market news from three days either side of the date.
2. Look for a report that says why coffee prices moved **on that day, in that direction**. Reports published the day after are the best source: they describe the finished trading day.
3. Fill in the row.

Spend at most ten minutes on a date. If you have not found a reason by then, that is the answer.

## What to write in `expected_outcome`

**EXPLAIN** if a report ties that day's move to a specific event or condition. Examples: frost or drought damage, a rain forecast easing drought fears, a crop or stocks report being released, a currency move, a tariff announcement, a strike or shipping disruption.

**REFUSE** if you find no such report, or the only reasons given are descriptions of trading with no event behind them. These count as no cause: "profit-taking", "technical correction", "fund selling", "consolidation", "choppy trade", "prices eased after recent gains".

**EXCLUDE** if the move is not a real market move of that size. The price series follows the nearest futures contract, and on the day it switches from an expired contract to the next one it can show a jump that no single contract made. If the reports of the day show a much smaller move, write EXCLUDE and say what the reports show. Such a date is left out of the scoring.

The same rule applies to days that reverse a trend. "Fell on profit-taking after the rally" is REFUSE. "Fell after rain was forecast for Minas Gerais" is EXPLAIN.

## The other columns

- `label_confidence`: `high` if the report is specific and from a market source; `medium` if the reason is given but vague or second-hand; `low` if you are unsure the report is about that day.
- `known_cause`: the reason in one sentence, in your own words. For REFUSE, say what you found instead ("only profit-taking mentioned", "no market report found").
- `source_url`: the report you relied on. Leave empty for REFUSE if you found nothing.
- `notes`: where the figures came from and anything odd. A note that starts with `CHECK:` marks a row where the label could reasonably go the other way; read those first.
- `labelled_by`: the initials of the person who checked the row.

## Rules that keep the labels honest

- Do not look at what the system said for the date before labelling it.
- Do not change a label after seeing the system's answer. If you think a label is wrong, note why in `notes` and have the other labeller decide.
- The two labellers work separately and do not compare until both sheets are finished. Disagreements are then discussed and the final label recorded, and the rate of agreement before discussion is reported.

## When both sheets are done

`dates_to_label.csv` holds 24 new dates. `dates_to_label_second_person.csv` holds those 24 plus the 18 already labelled, all blank, so the second person labels everything independently, including the dates the first person labelled earlier under no written rule.

## Two answer keys for the first 18 dates

`explanation_backtest_labels.csv` holds the original labels, made before this rule was written. `labels_by_written_rule.csv` holds the same 18 dates researched again under this rule, each with the report it rests on and the move that report gives for the day, and with the original label in its own column. Five differ, including all four original REFUSE labels.

The second file is the answer key the reports use. It was made after the system had been run on these dates, by research that did not see the system's answers, and nobody has checked it independently; rows whose notes start with CHECK are borderline. Scores against the original labels are still available (`python -m src.evaluation.metrics --labels data/labeling/explanation_backtest_labels.csv --name original_key`).

`contract_switch_dates.csv` lists the dates found so far where the price series shows a move the active contract did not make, each with the reported move (`reported_pct`) and the report it comes from. The detector measures those days by the reported move, so they are no longer flagged, and the evaluation set and every report leave them out. To add a date: add a row with `verdict` EXCLUDE and the reported move, then run `python -m src.modeling.anomaly_detector`.
