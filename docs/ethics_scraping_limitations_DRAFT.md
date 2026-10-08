# Text-Fetch / Scraping — Limitations & Ethical Considerations (DRAFT)

For eventual inclusion in the final report's "Limitations" / "Ethical
Considerations" section (checklist item, Documentation deliverables).

What the system does: the RAG retriever fetches the full text of articles
found via GDELT by directly requesting each article's URL and extracting
readable body text (trafilatura). This is necessary because GDELT itself
returns metadata only (title/url/date), not article content.

Observed failure rate: 28% of fetch attempts (21/75) were blocked or failed
in Week 2 testing, mostly HTTP 403 responses from news sites rejecting
scripted requests (some major outlets among them), plus DNS and connection
failures. We should expect this rate to hold or worsen at scale, not treat
it as a one-off.

Position for the report:

- The project does not check or respect individual sites' `robots.txt` or
  terms of service on a per-site basis before fetching. At the volume and
  frequency involved (a handful of article fetches per flagged anomaly, not
  bulk crawling), we think this is a defensible academic/research-use
  posture, but it is a deliberate choice and should be stated as such rather
  than left unaddressed.
- No paywall bypass, login, CAPTCHA solving, or JS rendering is attempted. A
  site that requires any of that is treated as a failed fetch. This is a
  scope boundary, not a partial attempt at circumvention.
- Extracted text is used only to compute a relevance score and, if accepted,
  to ground a cited explanation that links back to the original article URL.
  The system does not republish or store full article text as a standalone
  product, and every explanation cites its source rather than presenting
  fetched text as original content.
- The system fetches only articles already surfaced by GDELT for a narrow
  query tied to a real flagged anomaly. This is targeted retrieval for a
  specific research question, not general-purpose crawling or scraping for a
  dataset to be redistributed.

Still undecided, and worth deciding before Week 9 integration puts this in
front of a wider evaluation: whether to add a `robots.txt` check before
fetching. It's cheap to add and would reduce the already-high failure rate
somewhat by pre-skipping sites that would reject the request anyway, but it
does not change the underlying ethical posture. We're flagging it as a Week
3-4 nice-to-have rather than committing now. Like the original text/snippet
call, it should be a deliberate decision rather than something added
quietly.
