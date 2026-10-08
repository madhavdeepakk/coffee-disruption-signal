# RAG Explanation Layer — Quantitative Evaluation

Across **28** pipeline runs found in `results/`:

- **Explained: 14** · **Insufficient-evidence (refused): 12** · Parse-error: 0
- Refusals broken down by cause below — the point of the RAG safety design is that a refusal for "no documents" (a retrieval problem) and a refusal because the LLM judged real evidence didn't fit (a reasoning safeguard) are different and stay labeled.
- **Citation faithfulness** (over 10 run(s) with citations + a faithfulness report): 10/10 had every citation resolve to a retrieved document; 0 fabricated-id citation(s), 0 weak-support flag(s) in total.

| date | decision | refusal cause | retrieved | accepted | best | cites | dir(c/n/i) | cites resolve |
|---|---|---|---|---|---|---|---|---|
| 2026-07-07 | EXPLAINED | - | 23 | 23 | 0.895 | 3 | - | - |
| 2026-08-28 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 34 | 33 | 0.872 | 0 | - | - |
| 2018-05-21 | INSUFFICIENT_EVIDENCE | gate: documents_retrieved_but_below_relevance_threshold | 40 | 0 | 0.847 | 0 | - | - |
| 2020-03-19 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 53 | 1 | 0.837 | 0 | 1/0/0 | - |
| 2021-04-06 | EXPLAINED | - | 26 | 9 | 0.897 | 7 | 9/0/0 | yes |
| 2021-05-05 | EXPLAINED | - | 83 | 7 | 0.881 | 2 | 6/0/1 | yes |
| 2021-07-19 | EXPLAINED | - | 91 | 19 | 0.909 | 2 | 2/4/13 | yes |
| 2021-07-20 | EXPLAINED | - | 76 | 19 | 0.909 | 5 | 16/2/1 | yes |
| 2021-07-22 | EXPLAINED | - | 54 | 17 | 0.909 | 4 | 13/3/1 | yes |
| 2021-07-30 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 77 | 25 | 0.913 | 0 | 1/6/18 | - |
| 2022-05-11 | EXPLAINED | - | 41 | 9 | 0.875 | 3 | 7/1/1 | yes |
| 2023-01-31 | EXPLAINED | - | 80 | 3 | 0.893 | 2 | 2/0/1 | yes |
| 2023-09-20 | EXPLAINED | - | 20 | 9 | 0.890 | 1 | 0/4/5 | yes |
| 2023-11-30 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 80 | 3 | 0.879 | 0 | 1/1/1 | - |
| 2024-04-30 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 85 | 14 | 0.902 | 0 | 1/0/13 | - |
| 2024-09-23 | API_ERROR | model_unavailable: APIStatusError: Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-20b` in organization `org_01m3qksn47eq1sfc1616ske1q9` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested 8428, please reduce your message size and try again | 97 | 25 | 0.902 | 0 | 18/5/2 | - |
| 2024-10-07 | EXPLAINED | - | 37 | 34 | 0.900 | 1 | - | - |
| 2024-12-02 | API_ERROR | model_unavailable: APIStatusError: Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-20b` in organization `org_01m3qksn47eq1sfc1616ske1q9` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested 11051, please reduce your message size and try agai | 77 | 24 | 0.907 | 0 | 0/9/15 | - |
| 2025-02-10 | EXPLAINED | - | 96 | 7 | 0.859 | 4 | 4/3/0 | yes |
| 2025-08-15 | EXPLAINED | - | 25 | 5 | 0.889 | 5 | - | - |
| 2025-09-15 | EXPLAINED | - | 34 | 33 | 0.894 | 4 | - | - |
| 2026-06-16 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 24 | 3 | 0.867 | 0 | 2/0/1 | - |
| 2026-06-30 | EXPLAINED | - | 9 | 9 | 0.891 | 3 | 4/2/3 | yes |
| 2026-07-06 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 27 | 1 | 0.859 | 0 | 0/1/0 | - |
| 2026-07-07 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 29 | 3 | 0.895 | 0 | 0/1/2 | - |
| 2026-07-09 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 9 | 2 | 0.895 | 0 | 2/0/0 | - |
| 2026-08-28 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 27 | 2 | 0.872 | 0 | 2/0/0 | - |
| 2026-09-11 | INSUFFICIENT_EVIDENCE | LLM judged evidence insufficient | 24 | 2 | 0.864 | 0 | 0/0/2 | - |
