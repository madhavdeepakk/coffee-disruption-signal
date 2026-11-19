# Anomaly cards

## coffee — 2020-03-19

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 116.7 | observed | `anomaly_detections.csv` |
| z score | 1.527 | tested | `anomaly_detections.csv` |
| flagged | False | tested | `anomaly_detections.csv` |
| anomaly type | - | tested | `anomaly_detections.csv` |
| contract switch | True | observed | `data/labeling/contract_switch_dates.csv` |
| factor class | nan | tested | `coffee_factor_adjusted.csv` |
| residual z | - | tested | `coffee_factor_adjusted.csv` |
| factor explained return | - | tested | `coffee_factor_adjusted.csv` |
| unexplained return | - | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.04313 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | crude_oil, wheat, gold, silver | observed | `cross_commodity.json` |
| cross commodity verdict | - | tested | `cross_commodity.json` |
| pipeline version | 9 | observed | `pipeline_output_coffee_2020-03-19.json` |
| outcome tier | EXPLAINED_TENTATIVE | tested | `pipeline_output_coffee_2020-03-19.json` |
| decision reason | - | tested | `pipeline_output_coffee_2020-03-19.json` |
| explanation | The +8.0% day-over-day move (z-score 2.94) on 2020-03-19 is matched by 1 report published that day that gives a reason for prices rising: supply chain worrie… | observed | `pipeline_output_coffee_2020-03-19.json` |
| n citations | 1 | observed | `pipeline_output_coffee_2020-03-19.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2020-03-19.json` |

Not run for this day: numeraire. A missing layer is a gap in what is known, not a negative result.

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2021-05-05

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 148.5 | observed | `anomaly_detections.csv` |
| z score | 4.162 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock+trend | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 4.17 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.002079 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.06475 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.1681 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +6.91%, EUR +7.32%, JPY +7.15%, CNY +6.91%, INR +6.83% | observed | `coffee_numeraire.json` |
| survival after translation | 1.017 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -1.345 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 6.943 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2021-05-05.json` |
| outcome tier | EXPLAINED_TENTATIVE | tested | `pipeline_output_coffee_2021-05-05.json` |
| decision reason | - | tested | `pipeline_output_coffee_2021-05-05.json` |
| explanation | +6.9% day-over-day move (z-score 4.16), on top of +12.3% over the last 10 trading days on 2021-05-05. Reasons given by 2 reports from that day: - "Peabirus :… | observed | `pipeline_output_coffee_2021-05-05.json` |
| n citations | 2 | observed | `pipeline_output_coffee_2021-05-05.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2021-05-05.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2021-07-19

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 155.2 | observed | `anomaly_detections.csv` |
| z score | -2.073 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | -2.074 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.001537 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | -0.03915 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.07812 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | crude_oil | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD -3.69%, EUR -3.71%, JPY -3.61%, CNY -3.42%, INR -3.57% | observed | `coffee_numeraire.json` |
| survival after translation | 0.9715 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -1.434 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | -3.393 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2021-07-19.json` |
| outcome tier | EXPLAINED_STRONG | tested | `pipeline_output_coffee_2021-07-19.json` |
| decision reason | - | tested | `pipeline_output_coffee_2021-07-19.json` |
| explanation | -3.7% day-over-day move (z-score -2.07) on 2021-07-19. Reasons given by 2 reports from that day: - "Peabirus : Com delta , dólar e petróleo no radar , arábic… | observed | `pipeline_output_coffee_2021-07-19.json` |
| n citations | 2 | observed | `pipeline_output_coffee_2021-07-19.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2021-07-19.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2021-07-20

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 165.6 | observed | `anomaly_detections.csv` |
| z score | 3.154 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 3.271 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.00189 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.06295 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.09039 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | crude_oil | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +6.70%, EUR +6.88%, JPY +6.37%, CNY +6.88%, INR +7.22% | observed | `coffee_numeraire.json` |
| survival after translation | 1.026 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -1.517 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 6.838 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2021-07-20.json` |
| outcome tier | EXPLAINED_STRONG | tested | `pipeline_output_coffee_2021-07-20.json` |
| decision reason | - | tested | `pipeline_output_coffee_2021-07-20.json` |
| explanation | +6.7% day-over-day move (z-score 3.15) on 2021-07-20. Reasons given by 4 reports from that day: - "Arabica coffee gains more than 6 % on frosts in Brazil , s… | observed | `pipeline_output_coffee_2021-07-20.json` |
| n citations | 4 | observed | `pipeline_output_coffee_2021-07-20.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2021-07-20.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2021-07-22

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 193.6 | observed | `anomaly_detections.csv` |
| z score | 4.245 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock+trend | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 4.245 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.007232 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.08834 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.08918 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +10.03%, EUR +9.95%, JPY +10.36%, CNY +9.74%, INR +9.90% | observed | `coffee_numeraire.json` |
| survival after translation | 0.9901 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -1.416 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 10.15 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2021-07-22.json` |
| outcome tier | EXPLAINED_STRONG | tested | `pipeline_output_coffee_2021-07-22.json` |
| decision reason | - | tested | `pipeline_output_coffee_2021-07-22.json` |
| explanation | +10.0% day-over-day move (z-score 4.25), on top of +27.4% over the last 10 trading days on 2021-07-22. Reasons given by 2 reports from that day: - "Café aráb… | observed | `pipeline_output_coffee_2021-07-22.json` |
| n citations | 2 | observed | `pipeline_output_coffee_2021-07-22.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2021-07-22.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2021-07-30

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 179.6 | observed | `anomaly_detections.csv` |
| z score | -3.336 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | -3.48 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | -0.001228 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | -0.08898 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.07786 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD -8.63%, EUR -8.99%, JPY -9.03%, CNY -9.11%, INR -8.80% | observed | `coffee_numeraire.json` |
| survival after translation | 1.046 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -1.322 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | -8.218 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2021-07-30.json` |
| outcome tier | EXPLAINED_TENTATIVE | tested | `pipeline_output_coffee_2021-07-30.json` |
| decision reason | - | tested | `pipeline_output_coffee_2021-07-30.json` |
| explanation | -8.6% day-over-day move (z-score -3.34) on 2021-07-30. Reasons given by 3 reports from that day: - "Preços do café recuam quase 9 % com geadas no Brasil apar… | observed | `pipeline_output_coffee_2021-07-30.json` |
| n citations | 3 | observed | `pipeline_output_coffee_2021-07-30.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2021-07-30.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2022-05-11

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 220.1 | observed | `anomaly_detections.csv` |
| z score | 4.171 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 3.746 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.008863 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.0671 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.1575 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +7.89%, EUR +8.23%, JPY +7.87%, CNY +7.95%, INR +7.76% | observed | `coffee_numeraire.json` |
| survival after translation | 1.002 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -1.166 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 7.807 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2022-05-11.json` |
| outcome tier | EXPLAINED_STRONG | tested | `pipeline_output_coffee_2022-05-11.json` |
| decision reason | - | tested | `pipeline_output_coffee_2022-05-11.json` |
| explanation | +7.9% day-over-day move (z-score 4.17) on 2022-05-11. Reasons given by 2 reports from that day: - "Café arábica salta 8 % na ICE com clima do Brasil ; robust… | observed | `pipeline_output_coffee_2022-05-11.json` |
| n citations | 2 | observed | `pipeline_output_coffee_2022-05-11.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2022-05-11.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2023-01-31

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 181.8 | observed | `anomaly_detections.csv` |
| z score | 2.746 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock+trend | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 2.569 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.0005829 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.0639 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.03096 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +6.66%, EUR +6.86%, JPY +7.17%, CNY +6.12%, INR +6.75% | observed | `coffee_numeraire.json` |
| survival after translation | 1.022 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -0.1778 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 6.627 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2023-01-31.json` |
| outcome tier | REFUSED_WEAK_EVIDENCE | tested | `pipeline_output_coffee_2023-01-31.json` |
| decision reason | blind_evidence: no_recent_document_gives_a_reason_for_a_price_move | tested | `pipeline_output_coffee_2023-01-31.json` |
| explanation | - | observed | `pipeline_output_coffee_2023-01-31.json` |
| n citations | 0 | observed | `pipeline_output_coffee_2023-01-31.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2023-01-31.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2023-09-20

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 158.2 | observed | `anomaly_detections.csv` |
| z score | -0.8176 | tested | `anomaly_detections.csv` |
| flagged | False | tested | `anomaly_detections.csv` |
| anomaly type | - | tested | `anomaly_detections.csv` |
| contract switch | True | observed | `data/labeling/contract_switch_dates.csv` |
| factor class | nan | tested | `coffee_factor_adjusted.csv` |
| residual z | - | tested | `coffee_factor_adjusted.csv` |
| factor explained return | - | tested | `coffee_factor_adjusted.csv` |
| unexplained return | - | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.04316 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | - | tested | `cross_commodity.json` |
| pipeline version | 9 | observed | `pipeline_output_coffee_2023-09-20.json` |
| outcome tier | REFUSED_CONFLICTING | tested | `pipeline_output_coffee_2023-09-20.json` |
| decision reason | blind_evidence: the_freshest_evidence_points_the_other_way | tested | `pipeline_output_coffee_2023-09-20.json` |
| explanation | - | observed | `pipeline_output_coffee_2023-09-20.json` |
| n citations | 0 | observed | `pipeline_output_coffee_2023-09-20.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2023-09-20.json` |

Not run for this day: numeraire. A missing layer is a gap in what is known, not a negative result.

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2023-11-30

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 195 | observed | `anomaly_detections.csv` |
| z score | 3.499 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 3.71 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | -0.0106 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.07905 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.06759 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +7.08%, EUR +7.40%, JPY +7.07%, CNY +7.31%, INR +6.97% | observed | `coffee_numeraire.json` |
| survival after translation | 1.014 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -0.6294 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 7.562 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2023-11-30.json` |
| outcome tier | EXPLAINED_TENTATIVE | tested | `pipeline_output_coffee_2023-11-30.json` |
| decision reason | - | tested | `pipeline_output_coffee_2023-11-30.json` |
| explanation | +7.1% day-over-day move (z-score 3.50) on 2023-11-30. Reason given by 1 report from that day: - "SOFTS - Coffee prices jump nearly 7 % in New York after stoc… | observed | `pipeline_output_coffee_2023-11-30.json` |
| n citations | 1 | observed | `pipeline_output_coffee_2023-11-30.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2023-11-30.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2024-04-30

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 220.6 | observed | `anomaly_detections.csv` |
| z score | -2.851 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | -2.772 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | -0.0008506 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | -0.05597 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.03446 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | gold, silver | observed | `cross_commodity.json` |
| cross commodity verdict | macro | tested | `cross_commodity.json` |
| move in local currencies pct | USD -5.52%, EUR -5.60%, JPY -6.66%, CNY -5.61%, INR -5.47% | observed | `coffee_numeraire.json` |
| survival after translation | 1.016 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -0.563 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | -5.201 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2024-04-30.json` |
| outcome tier | REFUSED_WEAK_EVIDENCE | tested | `pipeline_output_coffee_2024-04-30.json` |
| decision reason | blind_evidence: no_report_gives_an_event_as_the_reason_for_a_move_of_this_size | tested | `pipeline_output_coffee_2024-04-30.json` |
| explanation | - | observed | `pipeline_output_coffee_2024-04-30.json` |
| n citations | 0 | observed | `pipeline_output_coffee_2024-04-30.json` |
| readings set aside | reason_only_describes_trading | tested | `pipeline_output_coffee_2024-04-30.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2024-09-23

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 263.6 | observed | `anomaly_detections.csv` |
| z score | 2.261 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 2.325 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | -0.001957 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.05212 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.05064 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +5.14%, EUR +5.16%, JPY +5.99%, CNY +4.91%, INR +4.98% | observed | `coffee_numeraire.json` |
| survival after translation | 0.9858 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -1.231 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 5.402 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2024-09-23.json` |
| outcome tier | REFUSED_WEAK_EVIDENCE | tested | `pipeline_output_coffee_2024-09-23.json` |
| decision reason | blind_evidence: no_recent_document_gives_a_reason_for_a_price_move | tested | `pipeline_output_coffee_2024-09-23.json` |
| explanation | - | observed | `pipeline_output_coffee_2024-09-23.json` |
| n citations | 0 | observed | `pipeline_output_coffee_2024-09-23.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2024-09-23.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2024-10-07

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 244.6 | observed | `anomaly_detections.csv` |
| z score | -2.224 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | -2.121 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | -0.003127 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | -0.04748 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.04038 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | crude_oil | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD -4.93%, EUR -4.41%, JPY -3.65%, CNY -5.31%, INR -4.91% | observed | `coffee_numeraire.json` |
| survival after translation | 0.943 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -0.7874 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | -4.92 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2024-10-07.json` |
| outcome tier | EXPLAINED_STRONG | tested | `pipeline_output_coffee_2024-10-07.json` |
| decision reason | - | tested | `pipeline_output_coffee_2024-10-07.json` |
| explanation | -4.9% day-over-day move (z-score -2.22) on 2024-10-07. Reasons given by 2 reports from that day: - "Café : preço em NY cai ao menor patamar em um mês com pre… | observed | `pipeline_output_coffee_2024-10-07.json` |
| n citations | 2 | observed | `pipeline_output_coffee_2024-10-07.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2024-10-07.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2024-12-02

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 299.2 | observed | `anomaly_detections.csv` |
| z score | -3.446 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | -3.489 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.00161 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | -0.07396 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.05449 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD -6.98%, EUR -6.82%, JPY -7.76%, CNY -7.01%, INR -6.89% | observed | `coffee_numeraire.json` |
| survival after translation | 0.9951 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -0.4998 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | -6.668 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2024-12-02.json` |
| outcome tier | REFUSED_WEAK_EVIDENCE | tested | `pipeline_output_coffee_2024-12-02.json` |
| decision reason | blind_evidence: no_report_gives_an_event_as_the_reason_for_a_move_of_this_size | tested | `pipeline_output_coffee_2024-12-02.json` |
| explanation | - | observed | `pipeline_output_coffee_2024-12-02.json` |
| n citations | 0 | observed | `pipeline_output_coffee_2024-12-02.json` |
| readings set aside | reason_only_describes_trading | tested | `pipeline_output_coffee_2024-12-02.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2025-02-10

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 429 | observed | `anomaly_detections.csv` |
| z score | 2.585 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock+trend | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 2.617 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.003848 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.05544 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.01511 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | wheat, gold | observed | `cross_commodity.json` |
| cross commodity verdict | macro | tested | `cross_commodity.json` |
| move in local currencies pct | USD +6.11%, EUR +6.87%, JPY +6.57%, CNY +6.09%, INR +6.02% | observed | `coffee_numeraire.json` |
| survival after translation | 1.035 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -0.5797 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 6.268 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2025-02-10.json` |
| outcome tier | EXPLAINED_STRONG | tested | `pipeline_output_coffee_2025-02-10.json` |
| decision reason | - | tested | `pipeline_output_coffee_2025-02-10.json` |
| explanation | +6.1% day-over-day move (z-score 2.59), on top of +22.9% over the last 10 trading days on 2025-02-10. Reasons given by 2 reports from that day: - "Coffee fut… | observed | `pipeline_output_coffee_2025-02-10.json` |
| n citations | 2 | observed | `pipeline_output_coffee_2025-02-10.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2025-02-10.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2025-04-07

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 344.8 | observed | `anomaly_detections.csv` |
| z score | -2.656 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | -2.757 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.001685 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | -0.06053 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.07454 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | crude_oil, gold, silver | observed | `cross_commodity.json` |
| cross commodity verdict | macro | tested | `cross_commodity.json` |
| move in local currencies pct | USD -5.72%, EUR -5.21%, JPY -6.23%, CNY -5.54%, INR -5.44% | observed | `coffee_numeraire.json` |
| survival after translation | 0.9594 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | 0.8287 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | -5.897 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2025-04-07.json` |
| outcome tier | REFUSED_WEAK_EVIDENCE | tested | `pipeline_output_coffee_2025-04-07.json` |
| decision reason | blind_evidence: no_report_gives_an_event_as_the_reason_for_a_move_of_this_size | tested | `pipeline_output_coffee_2025-04-07.json` |
| explanation | - | observed | `pipeline_output_coffee_2025-04-07.json` |
| n citations | 0 | observed | `pipeline_output_coffee_2025-04-07.json` |
| readings set aside | does_not_describe_a_move_of_this_size | tested | `pipeline_output_coffee_2025-04-07.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2025-08-15

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 341.6 | observed | `anomaly_detections.csv` |
| z score | 2.238 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock+trend | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 2.035 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | -1.234e-05 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.04537 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.008298 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | none | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +4.64%, EUR +5.19%, JPY +5.05%, CNY +4.72%, INR +4.93% | observed | `coffee_numeraire.json` |
| survival after translation | 1.073 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | -0.2691 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 4.525 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2025-08-15.json` |
| outcome tier | REFUSED_WEAK_EVIDENCE | tested | `pipeline_output_coffee_2025-08-15.json` |
| decision reason | blind_evidence: no_report_gives_an_event_as_the_reason_for_a_move_of_this_size | tested | `pipeline_output_coffee_2025-08-15.json` |
| explanation | - | observed | `pipeline_output_coffee_2025-08-15.json` |
| n citations | 0 | observed | `pipeline_output_coffee_2025-08-15.json` |
| readings set aside | does_not_describe_a_move_of_this_size | tested | `pipeline_output_coffee_2025-08-15.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.

## coffee — 2025-09-15

| Field | Value | Evidence | Source |
|---|---|---|---|
| price | 432.4 | observed | `anomaly_detections.csv` |
| z score | 2.144 | tested | `anomaly_detections.csv` |
| flagged | True | tested | `anomaly_detections.csv` |
| anomaly type | shock | tested | `anomaly_detections.csv` |
| factor class | commodity-specific | tested | `coffee_factor_adjusted.csv` |
| residual z | 2.158 | tested | `coffee_factor_adjusted.csv` |
| factor explained return | 0.002534 | tested | `coffee_factor_adjusted.csv` |
| unexplained return | 0.04896 | tested | `coffee_factor_adjusted.csv` |
| window r2 | 0.003248 | tested | `coffee_factor_adjusted.csv` |
| other commodities flagged | wheat | observed | `cross_commodity.json` |
| cross commodity verdict | commodity-specific | tested | `cross_commodity.json` |
| move in local currencies pct | USD +5.28%, EUR +5.36%, JPY +5.60%, CNY +5.37%, INR +5.30% | observed | `coffee_numeraire.json` |
| survival after translation | 1.015 | observed | `coffee_numeraire.json` |
| numeraire reading | specific to the commodity | tested | `coffee_numeraire.json` |
| dollar beta | 0.2743 | tested | `coffee_numeraire.json` |
| unexplained by dollar pct | 5.358 | tested | `coffee_numeraire.json` |
| pipeline version | 11 | observed | `pipeline_output_coffee_2025-09-15.json` |
| outcome tier | EXPLAINED_TENTATIVE | tested | `pipeline_output_coffee_2025-09-15.json` |
| decision reason | - | tested | `pipeline_output_coffee_2025-09-15.json` |
| explanation | +5.3% day-over-day move (z-score 2.14) on 2025-09-15. Reasons given by 2 reports from that day: - "Café opera em alta na bolsa de Nova York atento ao clima n… | observed | `pipeline_output_coffee_2025-09-15.json` |
| n citations | 2 | observed | `pipeline_output_coffee_2025-09-15.json` |
| readings set aside | none | tested | `pipeline_output_coffee_2025-09-15.json` |

No overall verdict is given. The layers can disagree, and that is information: a day the factor model calls macro-driven, that survives translation into other currencies, and that carries a quoted supply story is a real puzzle, and one word would hide it.