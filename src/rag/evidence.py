"""
Evidence preparation: the steps between "articles came back from retrieval"
and "documents are handed to the LLM".

Three problems showed up when the stored pipeline outputs were audited, and
each one has a small, deterministic fix here:

1. Out-of-window articles. The retrieval window is passed to GDELT as
   startdatetime/enddatetime, but the API is not strict about it: across the
   stored runs that saved their sources, 29 of 182 accepted documents (16%)
   carried a publication date AFTER the anomaly date (2022-05-11 had 7 of 9). The RSS path already
   re-checks dates locally; filter_to_window() applies the same check to
   every source, so the "lookahead-safe" claim is enforced by this code
   rather than trusted to the API.

2. Syndicated copies. De-duplication was by URL only, so one wire story
   republished on ten sites counted as ten documents (2024-09-23: 25 accepted
   documents, about ten of them the same wire article). That inflates the source
   count, skews the direction tally, and wastes prompt tokens - it is what
   pushed two dates over the fallback model's request-size limit.
   collapse_syndicated() keeps one representative per story and records how
   many copies it stood for.

3. Prompt size. The explainer used to send every accepted document at a fixed
   1500 characters regardless of how many there were. pack_evidence() fits
   the evidence into a token budget: it shortens the per-document excerpt
   first and only then drops the lowest-ranked documents, and it reports
   exactly what it did so nothing is cut silently.

Nothing in this module calls the network or a model.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

# Token estimate. Latin-script news text (English, Portuguese, Spanish) runs
# at about 3.3 characters per token on the prompts this pipeline has actually
# sent: the two requests a provider rejected were 8,428 and 11,051 tokens for
# roughly 29,000 and 36,500 characters. Text in other scripts (Cyrillic, Thai,
# Chinese, Arabic - all of which GDELT returns) tokenizes far more densely, so
# those characters are counted at about one token each. That over-counts
# somewhat, which is the safe direction: this only has to keep a request
# under a provider's limit, not predict billing.
CHARS_PER_TOKEN = 3.2
NON_LATIN_CHARS_PER_TOKEN = 1.0

# Per-document excerpt lengths tried in order when fitting a budget. Below
# the last of these an excerpt is often just a headline and a dateline, so
# documents are dropped before going any shorter.
EXCERPT_STEPS = (1500, 1100, 800, 600)
# Used only when the minimum number of documents still does not fit.
LAST_RESORT_EXCERPT_STEPS = (400, 300)

# Two documents whose opening text overlaps this much (Jaccard over 5-word
# shingles) are treated as copies of the same story.
TEXT_DUPLICATE_JACCARD = 0.6
_SHINGLE_WORDS = 5
_SHINGLE_CHARS = 1200

_TRAILING_SOURCE_RE = re.compile(r"\s+[-|–—]\s+[^-|–—]{2,40}$")
# Word characters in ANY script. Retrieval returns titles in Cyrillic, Thai,
# Chinese and Arabic as well as Latin scripts; a Latin-only pattern reduces
# those to whatever digits they contain, and two unrelated articles that both
# mention "2024" then look like the same headline.
_WORD_RE = re.compile(r"[^\W_]+")
# A normalized title shorter than this many words is too little to identify a
# story ("Coffee prices", "Mercado de café") and is not used as a key.
MIN_TITLE_WORDS_FOR_KEY = 3


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def _parse_iso_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def days_before_anomaly(publication_date: str, anomaly_date: str) -> Optional[int]:
    """Whole days between a document's publication date and the anomaly date.
    0 = same day, 3 = three days earlier, negative = published afterwards.
    None when either date is missing or unparseable."""
    pub = _parse_iso_date(publication_date)
    anom = _parse_iso_date(anomaly_date)
    if pub is None or anom is None:
        return None
    return (anom - pub).days


def trading_days_before(publication_date: str, anomaly_date: str) -> Optional[int]:
    """Trading days between a document's publication and the move.

    Counts weekdays after the publication date up to and including the
    anomaly date, so an article from Friday (or the weekend) about a Monday
    move is ONE trading day old, not three. Calendar days badly overstate the
    age of evidence around weekends: a wire story published on the last
    trading day before a move is the freshest news there is.

    Exchange holidays are not known here and are counted as trading days, so
    the figure can be one too high around a holiday. 0 = published on the day
    of the move; negative = published after it; None = a date is missing.
    """
    pub = _parse_iso_date(publication_date)
    anom = _parse_iso_date(anomaly_date)
    if pub is None or anom is None:
        return None
    if pub > anom:
        return -trading_days_before(anomaly_date, publication_date)
    count, day = 0, pub
    while day < anom:
        day += timedelta(days=1)
        if day.weekday() < 5:
            count += 1
    return count


def filter_to_window(documents: list, anomaly_date: str, days_before: int,
                     days_after: int = 0) -> tuple:
    """Keep only documents published inside the retrieval window.

    Window is [anomaly_date - days_before, anomaly_date + days_after],
    inclusive, by calendar date. days_after defaults to 0: nothing published
    after the anomaly date is allowed through.

    A document with no parseable date is kept (it cannot be shown to be out of
    window) but is marked publication_date_unknown=True so it is visible in
    the output rather than silently trusted.

    Returns (kept, dropped) where dropped is a list of
    {"document_id", "publication_date", "reason"} for the audit trail.
    """
    anom = _parse_iso_date(anomaly_date)
    if anom is None:
        return list(documents), []
    earliest = anom - timedelta(days=days_before)
    latest = anom + timedelta(days=days_after)

    kept, dropped = [], []
    for doc in documents:
        pub = _parse_iso_date(doc.get("publication_date", ""))
        if pub is None:
            doc["publication_date_unknown"] = True
            kept.append(doc)
            continue
        if pub > latest:
            dropped.append({"document_id": doc.get("document_id"),
                            "publication_date": doc.get("publication_date"),
                            "reason": "published_after_anomaly"})
        elif pub < earliest:
            dropped.append({"document_id": doc.get("document_id"),
                            "publication_date": doc.get("publication_date"),
                            "reason": "published_before_window"})
        else:
            kept.append(doc)
    return kept, dropped


def annotate_recency(documents: list, anomaly_date: str) -> list:
    """Attach days_before_anomaly (calendar, from the publication date) and
    trading_days_before_anomaly (sessions, from the session date) to each
    document in place."""
    for doc in documents:
        pub = doc.get("publication_date", "")
        doc["days_before_anomaly"] = days_before_anomaly(pub, anomaly_date)
        # Counted from the session the document can be reporting on, where
        # that is known: a document first seen before the exchange opened
        # belongs to the session before (see market_report.session_date).
        # Never later than the publication date, so nothing is made fresher.
        session = doc.get("session_date") or pub
        doc["trading_days_before_anomaly"] = trading_days_before(session, anomaly_date)
    return documents


def recency_summary(documents: list, fresh_within_days: int = 2) -> dict:
    """How recent the evidence is relative to the move it is meant to explain.

    A document published a week before a one-day move can describe the
    background, but it cannot by itself say why the price moved that day, so
    the count of documents from the last `fresh_within_days` days is the
    figure that matters.
    """
    # Age in trading days where it is known (a Friday article about a Monday
    # move is one day old); calendar days only for documents annotated before
    # trading days were recorded.
    def age(d):
        t = d.get("trading_days_before_anomaly")
        return t if t is not None else d.get("days_before_anomaly")

    ages = [age(d) for d in documents if age(d) is not None]
    if not ages:
        return {"n_dated": 0, "n_fresh": 0, "fresh_within_days": fresh_within_days,
                "newest_days_before": None, "median_days_before": None}
    ages_sorted = sorted(ages)
    mid = len(ages_sorted) // 2
    median = (ages_sorted[mid] if len(ages_sorted) % 2
              else (ages_sorted[mid - 1] + ages_sorted[mid]) / 2)
    return {
        "n_dated": len(ages),
        "n_fresh": sum(1 for a in ages if 0 <= a <= fresh_within_days),
        "fresh_within_days": fresh_within_days,
        "newest_days_before": min(ages),
        "median_days_before": median,
    }


# ---------------------------------------------------------------------------
# Syndicated-copy collapsing
# ---------------------------------------------------------------------------

def normalize_title(title: str) -> str:
    """Lowercase, strip a trailing ' - Publisher' suffix and all punctuation,
    so the same headline republished by different outlets compares equal.

    Google News titles already end in ' - Publisher', and the aggregator used
    to append the publisher again, so stored titles can end in
    ' - AP News - AP News'. A repeated final segment is removed first, then
    one publisher suffix.
    """
    t = (title or "").strip()
    parts = re.split(r"\s+[-|–—]\s+", t)
    while len(parts) >= 3 and parts[-1].strip().lower() == parts[-2].strip().lower():
        parts.pop()
        t = " - ".join(parts)
    t = _TRAILING_SOURCE_RE.sub("", t)
    return " ".join(_WORD_RE.findall(t.lower()))


def title_key(title: str) -> str:
    """The normalized title if it is long enough to identify a story, else ""
    (meaning: do not group on this title)."""
    key = normalize_title(title)
    return key if len(key.split()) >= MIN_TITLE_WORDS_FOR_KEY else ""


def _shingles(text: str) -> set:
    words = _WORD_RE.findall((text or "")[:_SHINGLE_CHARS].lower())
    if len(words) < _SHINGLE_WORDS:
        return set()
    return {" ".join(words[i:i + _SHINGLE_WORDS])
            for i in range(len(words) - _SHINGLE_WORDS + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _best_score(doc: dict) -> float:
    return max(doc.get("retrieval_score", 0.0) or 0.0,
               doc.get("semantic_score", 0.0) or 0.0)


def _domain(url: str) -> str:
    m = re.match(r"^[a-z]+://([^/]+)", (url or "").lower())
    host = m.group(1) if m else ""
    return host[4:] if host.startswith("www.") else host


def _prefer(a: dict, b: dict) -> dict:
    """Which of two copies to keep as the representative: the one with body
    text, then the higher score, then the earlier publication date."""
    a_text, b_text = bool(a.get("text")), bool(b.get("text"))
    if a_text != b_text:
        return a if a_text else b
    sa, sb = _best_score(a), _best_score(b)
    if sa != sb:
        return a if sa > sb else b
    da, db = a.get("publication_date") or "9999", b.get("publication_date") or "9999"
    return a if da <= db else b


def collapse_syndicated(documents: list, use_text: bool = True) -> list:
    """Collapse copies of the same story into one document.

    Two documents are the same story when their normalized titles are equal,
    or (use_text) when they come from different sites, both have body text,
    and their openings overlap by TEXT_DUPLICATE_JACCARD or more. Title
    similarity alone is deliberately
    NOT used beyond exact equality: "Coffee Prices Settle Higher on ..." and
    "Coffee Prices Settle Lower on ..." are near-identical strings that say
    opposite things.

    The representative keeps its own fields and gains:
      n_copies          - how many documents it stands for (1 = unique)
      duplicate_urls    - URLs of the copies that were folded into it
      syndication_domains - distinct domains that carried the story

    Order of first appearance is preserved.
    """
    groups = []          # list of {"rep": doc, "members": [doc, ...]}
    by_title = {}        # normalized title -> group index
    shingle_cache = []   # parallel to groups; shingles of the representative

    for doc in documents:
        key = title_key(doc.get("title", ""))
        idx = by_title.get(key) if key else None

        if idx is None and use_text and doc.get("text"):
            sh = _shingles(doc["text"])
            domain = _domain(doc.get("url", ""))
            if sh:
                for i, other in enumerate(shingle_cache):
                    if not other or _jaccard(sh, other) < TEXT_DUPLICATE_JACCARD:
                        continue
                    # Matching text on the SAME site is far more likely to be
                    # that site's boilerplate (a consent page, a paywall
                    # notice, a redirect stub) than a syndicated copy, and
                    # would fold unrelated articles together. A wire story is
                    # the same text on DIFFERENT sites.
                    if domain and any(_domain(m.get("url", "")) == domain
                                      for m in groups[i]["members"]):
                        continue
                    idx = i
                    break

        if idx is None:
            groups.append({"rep": doc, "members": [doc]})
            shingle_cache.append(_shingles(doc.get("text", "")) if use_text else set())
            if key:
                by_title[key] = len(groups) - 1
        else:
            g = groups[idx]
            g["members"].append(doc)
            new_rep = _prefer(g["rep"], doc)
            if new_rep is not g["rep"]:
                g["rep"] = new_rep
                shingle_cache[idx] = _shingles(new_rep.get("text", "")) if use_text else set()
            if key and key not in by_title:
                by_title[key] = idx

    out = []
    for g in groups:
        rep = g["rep"]
        members = g["members"]
        # Counts accumulate across passes: a document that was already a
        # representative from an earlier (title-only) pass keeps the copies it
        # stood for.
        n = sum(int(m.get("n_copies", 1) or 1) for m in members)
        dup_urls, domains = [], set()
        for m in members:
            domains.add(_domain(m.get("url", "")))
            domains.update(m.get("syndication_domains") or [])
            if m is not rep:
                if m.get("url"):
                    dup_urls.append(m["url"])
            dup_urls.extend(u for u in (m.get("duplicate_urls") or [])
                            if u != rep.get("url"))
        rep["n_copies"] = n
        rep["duplicate_urls"] = sorted(set(dup_urls))
        rep["syndication_domains"] = sorted(d for d in domains if d)
        out.append(rep)
    return out


# ---------------------------------------------------------------------------
# Token-budgeted packing
# ---------------------------------------------------------------------------

def _is_latin(ch: str) -> bool:
    # ASCII, Latin-1 Supplement and Latin Extended (covers accented
    # Portuguese/Spanish/French/German letters).
    return ord(ch) < 0x250


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: Latin-script characters at
    CHARS_PER_TOKEN each, every other character at about one token."""
    text = text or ""
    other = sum(1 for ch in text if not _is_latin(ch))
    latin = len(text) - other
    return int(latin / CHARS_PER_TOKEN + other / NON_LATIN_CHARS_PER_TOKEN) + 1


# Labels and layout the evidence block adds around each document
# ("- document_id: ", "  title: ", "  date: ... (N days before the move)",
# "  url: ", "  syndicated: ...", "  text: ", newlines).
_DOC_FRAME_CHARS = 150


def _doc_tokens(doc: dict, excerpt_chars: int) -> int:
    body = (doc.get("text") or doc.get("title") or "")[:excerpt_chars]
    fields = f"{doc.get('document_id', '')}{doc.get('title', '') or ''}{doc.get('url', '') or ''}"
    return estimate_tokens(fields + body) + int(_DOC_FRAME_CHARS / CHARS_PER_TOKEN)


def _block_tokens(documents: list, excerpt_chars: int) -> int:
    return sum(_doc_tokens(doc, excerpt_chars) for doc in documents) + 1


def pack_evidence(documents: list, max_tokens: Optional[int],
                  excerpt_steps: tuple = EXCERPT_STEPS,
                  min_documents: int = 5) -> tuple:
    """Fit an ordered list of documents into a token budget.

    `documents` must already be in priority order (the pipeline passes them
    direction-consistent first, then by score) - when something has to be
    dropped, it is dropped from the END of the list.

    Strategy, in order:
      1. Send everything at the longest excerpt length if it fits.
      2. Otherwise shorten the per-document excerpt step by step. Keeping
         more documents at a shorter excerpt preserves source breadth, which
         matters more for a causal explanation than the tail of any one
         article.
      3. If the shortest of those excerpts still does not fit, drop documents
         from the end until it does (never below min_documents, unless there
         were fewer to begin with).
      4. If even min_documents do not fit, shorten further as a last resort.

    max_tokens=None means no budget: everything is sent at the longest
    excerpt (the original behaviour).

    Returns (packed_documents, report). report is JSON-serializable and goes
    into the pipeline output so a truncation is always visible:
      {n_input, n_packed, excerpt_chars, estimated_tokens, max_tokens,
       dropped_document_ids, truncated}
    """
    docs = list(documents)
    longest = excerpt_steps[0]

    if max_tokens is None:
        return docs, {
            "n_input": len(docs), "n_packed": len(docs), "excerpt_chars": longest,
            "estimated_tokens": _block_tokens(docs, longest), "max_tokens": None,
            "dropped_document_ids": [], "truncated": False,
        }

    chosen_excerpt = None
    for step in excerpt_steps:
        if _block_tokens(docs, step) <= max_tokens:
            chosen_excerpt = step
            break

    dropped = []
    if chosen_excerpt is None:
        chosen_excerpt = excerpt_steps[-1]
        floor = min(min_documents, len(docs))
        while len(docs) > floor and _block_tokens(docs, chosen_excerpt) > max_tokens:
            dropped.append(docs.pop().get("document_id"))
        for step in LAST_RESORT_EXCERPT_STEPS:
            if _block_tokens(docs, chosen_excerpt) <= max_tokens:
                break
            chosen_excerpt = step

    return docs, {
        "n_input": len(documents),
        "n_packed": len(docs),
        "excerpt_chars": chosen_excerpt,
        "estimated_tokens": _block_tokens(docs, chosen_excerpt),
        "max_tokens": max_tokens,
        "dropped_document_ids": dropped,
        "truncated": bool(dropped) or chosen_excerpt != longest,
    }
