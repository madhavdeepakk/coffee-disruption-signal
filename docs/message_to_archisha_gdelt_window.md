Hey Archisha — flagging something before you lock in the evaluation plan.

GDELT's DOC 2.0 API (my main retrieval source) only searches a rolling
3-month window — it's not a historical archive. That means if your
backtest evaluation dataset includes events older than ~3 months at the
time we run retrieval (e.g. anything like the 2021 Brazil frost example in
the team plan), my RAG pipeline literally cannot retrieve any evidence for
it via DOC 2.0 — not weak evidence, none.

Fix: GDELT also publishes its GKG dataset via Google BigQuery, covering
back to April 2013, queryable for free with no credit card (BigQuery
"sandbox" mode). It gives article URLs + dates for a given historical
window; I'd still need my own text-fetch step to pull title/body from
those URLs, same as I'm already doing. It's a different, heavier tool
(SQL, not a simple API call) so it's not going to be my live daily source
— it'd specifically be for pulling evidence around your historical
backtest event dates.

Two things I need from you before I build anything on this:

1. Does your evaluation plan already assume I have full-history retrieval,
   or were you scoping it to only recent/live-flagged anomalies? If the
   former, we need to align on the BigQuery approach now, not discover
   this in Week 9 integration.
2. What's the actual date range of the historical events you're planning
   to put in the backtest/evaluation dataset? That tells me whether this
   is a real blocker or a hypothetical one.

Not blocking anything on my end this week — just don't want your Week 2
evaluation plan getting built on an assumption about my retrieval that
turns out to be wrong.
