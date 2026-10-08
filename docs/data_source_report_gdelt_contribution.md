# Data Source Report — GDELT (Madhav's contribution)

*For inclusion in the shared `docs/data_source_report.md` table
(NOAA / GDELT / Freightos / World Bank / FRED × Access / Format / Frequency /
Publication Date Available? / Status), consolidated by Yashika.*

| Field | Value |
|---|---|
| **Source** | GDELT (DOC 2.0 API) |
| **Access / Auth** | None required. Public GET endpoint, no API key, no registration. |
| **Endpoint** | `https://api.gdeltproject.org/api/v2/doc/doc` |
| **Format** | JSON (also CSV, RSS, HTML available via `format` param); we use JSON. |
| **Frequency / Coverage window** | Rolling 3-month window only, so this is not a historical archive. Article "freshness" depends on GDELT's own crawl of global news, effectively near-real-time within that window. |
| **Fields returned** | `url`, `url_mobile`, `title`, `seendate` (crawl/index timestamp, `YYYYMMDDTHHMMSSZ`), `domain`, `language`, `sourcecountry`, `socialimage`. No article body text or snippet. |
| **Publication date available?** | Partially. `seendate` is when GDELT's crawler indexed the article, used here as a proxy for publication date, not necessarily the article's true original publish timestamp. Documented as a known approximation, not treated as authoritative. |
| **Reliability / Status** | Unreliable at the connection level. Testing observed ~87% failure rate (global connection timeouts via check-host.net) and 22-25s response times on successful calls. This is GDELT server-side behavior, not client-side. Mitigated with a 45-60s per-attempt timeout, 3 retry attempts with 5s backoff, and a realistic browser `User-Agent` header (a missing UA is a common cause of silent connection resets). |
| **Live test result (Week 1)** | 2 of 3 test queries succeeded (146 articles total across "coffee drought Brazil" + "Brazil coffee frost"); 1 of 3 ("coffee supply disruption") failed after all 3 retries, consistent with the known failure rate above rather than a query-syntax problem. |
| **Rate limits** | Not formally documented by GDELT; no auth-based quota since there's no key. Treat as best-effort/shared public infrastructure — avoid tight polling loops. |
| **Notes for the team schema** | GDELT gives metadata only, no full text. Any team member relying on GDELT for content (not just headlines/links) will need a secondary fetch step against the article URLs themselves, with its own reliability caveats (paywalls, dead links, non-English content, translation). |

## Raw ingestion schema mapping (for `data/raw/gdelt_raw.json`)

Per the team's shared schema (source, title, url, publication_date, plus
reference_date/known_as_of_date where applicable; missing = blank, never
"N/A"):

| Team schema field | GDELT source field | Notes |
|---|---|---|
| `source` | constant `"GDELT"` | |
| `title` | `title` | |
| `url` | `url` | |
| `publication_date` | `seendate`, converted `YYYYMMDDTHHMMSSZ` → `YYYY-MM-DD` | Approximation — see reliability note above |
| `known_as_of_date` | same as `publication_date` | Week 1 simplification; revisit once Yashika's schema conventions for this field are finalized |
| (additional, not in core schema) | `domain`, `language`, `sourcecountry`, `query` | Kept for RAG-track use (source diversity, dedup, query provenance); not part of the shared minimal schema but harmless extra columns |
