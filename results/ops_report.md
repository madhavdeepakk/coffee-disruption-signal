# Operations Report

99 model calls logged between 2026-09-12 and 2026-10-02 (`results/llm_call_log.csv`).

## Reliability

- Explanation attempts that failed (model unavailable or unparseable response): 24.1% (13/54; 95% CI 14.6-36.9%). Other call types log successes only, so they are not in this figure.
- Successful calls answered by a fallback provider rather than the primary (google-gemini): 52.3% (45/86; 95% CI 41.9-62.6%)

| Provider | Model | Successful calls |
|---|---|---|
| groq | openai/gpt-oss-120b | 45 |
| google-gemini | gemini-3.6-flash | 41 |

Which model answered matters for evaluation: 2021-07-19 was refused by the primary model and explained by the fallback in a later run. Pipeline outputs record the model that produced them.

## Explanation calls

41 calls that returned an explain/refuse decision.

- Prompt tokens: median 2,992, 90th percentile 10,238, max 17,656
- Output tokens: median 274, 90th percentile 848, max 1,320
- Total tokens (including any the provider counts as reasoning): median 4,479, 90th percentile 12,042, max 19,506
- Latency: not recorded for these calls. Calls made from pipeline version 2 onwards log it.

## Cost

The log records tokens, not money. To see what this traffic would cost at a given rate, run `python -m scripts.ops_report --price-in X --price-out Y` with current prices in dollars per million tokens.

Model calls are not where the time goes. The GDELT client's own notes record 20-25 second responses and frequent rate-limiting, which is why GDELT responses and extracted article text are cached on disk.
