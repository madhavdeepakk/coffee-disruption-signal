# RAG Design — Relevance Criteria (Week 1 Draft)

Status: draft. This is a first-pass definition based on manual review of 26
articles from 2 successful GDELT queries (see below). It's a small,
non-representative sample. I'll revisit this in Week 4 against a larger
labeled set (20-30 query/document pairs per the Week 4 task list), and again
when semantic (FAISS) retrieval is added in Week 3. Anything below marked
"draft" should be read as a working hypothesis, not a frozen rule.

## Source data

- Script: `scripts/test_gdelt.py`
- Queries tested: `coffee drought Brazil`, `Brazil coffee frost`, `coffee supply disruption`
- Results: first two queries succeeded (75 + 71 = 146 articles); the third
  failed all 3 retry attempts, consistent with GDELT's known-unreliable
  public endpoint (~87% failure rate observed via check-host.net in prior
  testing). I'm treating this as expected infrastructure flakiness rather
  than a code defect: the retry/backoff/timeout logic behaved as designed (3
  attempts, 5s backoff, 60s per-attempt timeout, browser User-Agent header).
- Raw output: `data/raw/gdelt_raw.json`
- Caveat carried forward from Week 1 API research: GDELT DOC 2.0 returns
  title/url/date/metadata only, with no article body text or snippet. The
  manual classification below is therefore based on headlines (translated
  where non-English) plus domain, not full article content. That limits
  classification confidence and is a known constraint for Week 2's schema
  (the `text/snippet` field will need a separate fetch step or will stay
  metadata-only, documented as a limitation).

## Manual classification (26 articles)

Legend: **R** = Relevant, **P** = Partially relevant, **I** = Irrelevant.
"Relevant" here means: specifically discusses a coffee supply and/or price
move tied to an identifiable causal driver (weather/climate event, trade
action, geopolitical event) in a producing region — not just co-occurring
keywords.

| # | Query | Domain | Title (translated where needed) | Label | Why |
|---|---|---|---|---|---|
| 0 | drought Brazil | argumenti.ru | "Drought in Brazil - world prepares for coffee shortage" | R | Direct: Brazil drought → coffee shortage |
| 3 | drought Brazil | biz.heraldcorp.com | "Heatwave raises coffee/bread prices again... Super El Niño hits ag market" | R | El Niño explicitly linked to coffee price rise |
| 6 | drought Brazil | vz.ru | "Bloomberg: Tropical rains in Brazil hit global arabica market" | R | Direct causal claim: Brazil rain → arabica market impact |
| 9 | drought Brazil | esgnow.republika.co.id | "El Nino Strengthens, Global Coffee Supply Threatened" | R | Direct causal claim |
| 11 | drought Brazil | 2merkato.com | "Global Coffee Production, Consumption Hit Record Highs as Ethiopia Reaches Historic Output" | P | About coffee supply, but positive/record output in Ethiopia — not a Brazil disruption; tangential to the anomaly being explained |
| 20 | drought Brazil | g1.globo.com | "From coffee to rice: El Niño threatens production, could raise food prices" | R | Direct, though broader than coffee alone |
| 23 | drought Brazil | g1.globo.com | "Fear of possible super El Niño makes coffee price spike" | R | Direct causal + price move named |
| 27 | drought Brazil | globorural.globo.com | "El Niño brings rains forward, could induce coffee flowering in August" | P | Agronomic detail; effect is ambiguous/positive, not clearly a disruption |
| 32 | drought Brazil | infomoney.com.br | "Rains affect quality, but coffee exports should recover" | P | Mixed signal — quality hit but recovery expected; not a clean disruption narrative |
| 37 | drought Brazil | globorural.globo.com | "Coffee price rises in NY, market watching El Niño" | R | Direct price move + named driver |
| 45 | drought Brazil | noticiabrasil.net.br | "Brazil expects record coffee harvest, but decline in premium beans due to rain" | P | Mixed: record harvest overall, quality-specific rain impact only |
| 50 | drought Brazil | elperiodico.com | "World coffee production wobbles after Colombia earthquake and El Niño" | R | Relevant disruption narrative, though conflates two distinct causes (earthquake + El Niño) — worth flagging for citation care later |
| 52 | drought Brazil | news.tuoitre.vn | "Robusta coffee climate resilience an internet myth, study says" | P | On-topic (coffee + climate) but debunks a narrative rather than reporting an event; useful context, not event evidence |
| 61 | drought Brazil | timesfreepress.com | "Opinion: Climate change isn't taking food off your table" | I | Generic contrarian opinion piece, not about coffee or a specific event; keyword match only |
| 68 | drought Brazil | jmonline.com.br | "Drought becomes coffee's biggest enemy, helps keep price high" | R | Direct, unambiguous |
| 75 | frost Brazil | gazetadopovo.com.br | "The frost that changed a family's life and all Brazilian agribusiness" | P | Reads as a historical/retrospective frost feature, not clearly tied to a current price move — good background, weak as same-day evidence |
| 83 | frost Brazil | businessghana.com | "Coffee futures fall for third day running following USDA report" | R | Direct price move + named cause (USDA report), even though direction is down |
| 89 | frost Brazil | jornaldebrasilia.com.br | "El Niño threatens cocoa, coffee and sugar harvests in tropical countries" | R | Direct |
| 93 | frost Brazil | iz.ru | "2026 chocolate price forecast: how cocoa and coffee costs are changing" | P | Broader forecast piece, coffee only part of it |
| 102 | frost Brazil | vedomosti.ru | "Coffee and cocoa exchange prices rose sharply in July" | R | Direct, quantifiable price move |
| 103 | frost Brazil | globorural.globo.com | "El Niño concern makes coffee spike 16% in New York" | R | Direct, quantified, ideal example of strong evidence |
| 113 | frost Brazil | letemsvetemapplem.eu | "Natural gas, Robusta or orange juice: XTB expands commodities offering" | I | Broker product announcement; "Robusta" keyword match only, no disruption content |
| 116 | frost Brazil | thetelegraphandargus.co.uk | "Ilkley Food and Drink Festival 2026" | I | Local UK festival listing; false keyword match |
| 119 | frost Brazil | riotimesonline.com | "Brazil Coffee Festival Calendar 2026: Dates and Cities" | I | Festival calendar, not disruption-related despite strong keyword overlap ("Brazil", "Coffee") |
| 123 | frost Brazil | lasprovincias.es | "Mercadona ice cream — who really makes it?" | I | No coffee/price content; pure keyword noise |
| 144 | frost Brazil | skepticalscience.com | "Skeptical Science New Research for Week #26 2026" | I | General climate-science digest; no coffee connection found in title |

**Tally: 12 Relevant / 8 Partial / 6 Irrelevant** (of 26 reviewed).

## What this small sample already suggests

Patterns worth carrying into Week 4's larger-sample threshold work:

1. Keyword-only false positives are common and easy to spot by category,
   even without full article text: broker/trading-platform product ads,
   festival/tourism listings, and generic opinion pieces that happen to
   mention "coffee" or a producing country. A keyword-overlap retrieval
   score alone (Week 2's baseline) will likely rate some of these too high
   if the query terms appear verbatim in the title.
2. "Partial" is doing real work as a category. Several articles are clearly
   about coffee and clearly mention a climate/weather driver, but either (a)
   describe a positive/neutral outcome rather than a disruption, (b) mix good
   news and bad news in the same piece, or (c) are retrospective/contextual
   rather than about a current price move. Treating these as equivalent to
   strong evidence would risk the failure mode this track exists to prevent:
   stretching thin or mixed evidence into a confident, one-directional
   explanation.
3. Strong evidence tends to share a shape: a named causal driver (event,
   report, weather phenomenon) plus a described or quantified price/market
   move, both in the headline. Article #103 ("El Niño concern makes coffee
   spike 16% in New York") is the clearest example in this sample.
4. The lack of article text/snippet from GDELT DOC 2.0 limits this pass.
   Several "Partial" labels above might resolve to "Relevant" or "Irrelevant"
   with full body text. This is a genuine limitation: it directly affects how
   confident any keyword or semantic relevance score can be in Week 2-3.

## First-draft definition of "sufficiently relevant evidence" (DRAFT — Week 1, revisit Week 4)

An article is treated as relevant evidence for a flagged anomaly if, based
on this small sample's patterns, it satisfies both of:

- Topical match: the article is substantively about coffee supply,
  production, trade, or pricing in a producing region, not a keyword
  co-occurrence in an unrelated story (festivals, broker ads, unrelated
  commodities, generic opinion).
- Causal/event content: the article names a specific driver (weather event,
  policy/trade action, market report) and describes or implies a market/price
  effect, not just background information about the coffee industry in
  general.

Articles that are topically on-target but lack a clear causal-driver-to-
price-effect link (mixed signals, retrospectives, general forecasts) are
tentatively partial: plausible supporting context, but not sufficient on
their own to ground a specific causal explanation. Under the "insufficient
evidence is safer than a wrong explanation" principle, a retrieval set that
is mostly or entirely "partial" articles should lean toward the gate
rejecting the anomaly rather than the LLM being asked to synthesize a
confident story from mixed signals.

This is a starting hypothesis from a 26-article, title-only, 2-query sample.
It is not a scored/thresholded rule yet; the numeric threshold and scoring
methodology are Week 4 work, done against a proper 20-30 item labeled test
set with full text available where possible.

## Open questions carried to Week 2/4

- How should retrieval score "Partial" articles when multiple partials
  together might collectively support an explanation, even if none does
  individually?
- Does the eventual keyword relevance scorer (Week 2) need a negative list
  for common false-positive patterns identified here (festival/event
  calendars, broker product pages)?
- Should query construction (Week 5) exclude terms that produced obvious
  noise here, or is that better handled at the relevance-gate stage instead
  of retrieval?

Update (Week 2, after running the keyword scorer on real data): the dominant
failure mode turned out to be different from what I anticipated above. See
"Root cause analysis" in `docs/week2_keyword_scoring_comparison.md`. The
keyword score badly under-rates non-English articles (80% of
non-English-titled documents in the 75-document sample scored exactly 0.000,
vs 7% of English-titled ones), because it only matches literal English
query/vocabulary words. No false positives were observed in this run. This
is a language-coverage gap, not a threshold-tuning problem, and it directly
informs the Week 3 embedding model choice (must be multilingual) and Week
4's labeled sample composition (must include non-English articles
proportionally, since they're the majority of what GDELT actually returns
for these queries).

---

# Retrieval Document Schema (Week 2 — FROZEN)

This is the frozen contract for `src/rag/retriever.py` and everything
downstream of it (relevance gate, explainer, dashboard). Changes after this
point should be versioned/noted here explicitly, not made silently.

| Field | Type | Description |
|---|---|---|
| `document_id` | string | Stable unique ID per retrieved document. Derived as a hash of `url` (not row index), so the same article retrieved by two different queries dedupes to one ID. |
| `title` | string | Article headline, from GDELT `title`. |
| `url` | string | Article URL, from GDELT `url`. |
| `publication_date` | string (`YYYY-MM-DD`) | From GDELT `seendate`, per the caveat in the GDELT data-source report: this is crawl/index time, used as a proxy for publication date, not verified original publish time. |
| `source` | string | Constant `"GDELT"` for this retriever. Kept as a field (not hardcoded downstream) so other retrieval sources could plug into the same schema later if ever needed. |
| `text/snippet` | string | See "Text/snippet decision" below. Blank (never "N/A") if unavailable. |
| `query` | string | The exact query string that retrieved this document. A document retrieved by multiple queries keeps one record per query (not deduped away) — this preserves which query paths found it, useful for Week 2's false-positive/negative analysis. |
| `retrieval_score` | float | Raw score from the retrieval step (Week 2: keyword-overlap score; Week 3 will add a semantic/FAISS score alongside or instead). Documented per-run which method produced it. |
| `relevance_score` | float or blank | Reserved for the Week 6 relevance-gating score. Left blank at Week 2 — populated later, not computed by the retriever itself. Keeping it in the schema now (rather than adding it in Week 6) avoids a schema migration later. |

## Text/snippet decision (Week 2)

Decision: add a secondary fetch of the article URL to pull body text, rather
than leaving `text/snippet` blank. Title-only relevance scoring is weak
evidence of relevance on its own; several "Partial" labels in the Week 1
classification above were ambiguous precisely because only the headline was
available. Body text improves both keyword and (later) semantic scoring
quality.

This is a scope increase over "just call the GDELT API," so it gets its own
handling rather than being bolted on quietly:

- Method: fetch each article's `url` directly (plain HTTP GET, browser
  User-Agent, same lesson as the GDELT script), then extract main body text
  with a readability-style extractor (e.g. `trafilatura`) rather than naive
  HTML stripping, to reduce nav/ad/boilerplate noise.
- Note that this adds a second, independent point of failure on top of
  GDELT's own unreliability: dead links, paywalls, non-English extraction
  quality, regional blocking, and sites that refuse scripted requests. We
  should expect a non-trivial fraction of fetches to fail or return unusable
  text.
- Failure handling: a failed or empty fetch does not block the document from
  being retrieved. It just leaves `text/snippet` blank for that document
  (never fabricated, per the team-wide missing-value rule) and the retriever
  falls back to title-only for that document's keyword score. This is logged
  rather than silently dropped, so the fetch failure rate itself becomes a
  measurable stat worth reporting alongside retrieval results.
- Timeout/retry: lighter-weight than the GDELT script's settings. Arbitrary
  news sites shouldn't be as uniformly slow as GDELT's own API, so a shorter
  timeout (e.g. 15-20s) with 1-2 retries is the Week 2 starting point,
  tunable if real data shows otherwise.
- Scope guard: this fetch step is bounded to "grab readable body text for
  scoring," not a general-purpose scraper. No JS rendering, no
  pagination/"read more" handling, no login/paywall bypass attempts. If a
  site needs any of that, it's treated as a failed fetch.
- Rate/politeness: fetches are sequential with a small delay between requests
  to different domains, not parallelized aggressively. This is a 50-100
  document research script, not a crawler, and there's no reason to hammer
  any one news site.
