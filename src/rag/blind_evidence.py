"""
Direction-blind evidence reading, and the explain/refuse decision built on it.

Why this exists
---------------
The original design hands the model the price move ("+5.1% on 2024-09-23")
together with the retrieved news and asks it to explain the move or decline.
Measured on stored runs, the model declines very little: told a move
happened, it finds a story for it in whatever it was given. A count of which
way the retrieved documents leaned was added as a guard, and turned out to
withhold explanations on days prices fell and never on days they rose,
because the retrieved news was mostly about prices rising.

Both problems have the same root. The model knows the answer it is supposed
to arrive at, and the guard judges all the evidence instead of the evidence
an explanation would rest on.

So the decision is taken away from the model:

1. READ, blind. The model is shown the recent documents and NOT told what
   prices did. For each document it reports which way the document says
   prices moved (or which way the events it reports would push them), the
   reason the document gives, and a verbatim quote stating that reason.
2. VERIFY. A reason only counts if its quote is found in the document. That
   check is string matching, not a model's opinion.
3. DECIDE, by rule. The newest and most direct evidence decides. Readings
   with a verified reason are grouped by day, newest first, and within a day
   documents that report a price move come before documents that only report
   events bearing on price. The first group that is not empty decides:
   explain if the readings in it pointing the way prices went are at least
   as many as those pointing the other way; refuse otherwise, and refuse if
   there is no group at all.
4. WRITE, from the evidence. The explanation is the move, then one line per
   supporting document: its headline, where and when it was published, the
   move it reports, the reason it gives and the quote that states it. No
   second model call, and no sentence that is not one of those facts.

Because step 1 never sees the direction, the same documents produce the same
reading whether the move was up or down, so evidence for a rise cannot be
turned into an explanation of a fall. That property is what the "flip"
condition in src/evaluation/robustness.py tests.

A quiet day's market report still offers reasons ("technical adjustments",
"demand and the dollar"). Tested with invented moves on quiet days, the rule
above explained 5 of the first 10: each time from one routine report of the
day that said nothing about the move being large. So from pipeline version 11
an explanation also needs a report that fits THIS move (see decide):

- about this market, not another grade or a local price;
- giving an event as the reason, not a description of trading;
- describing a move of this size: a stated size at least a quarter of the
  flagged move, or, where no size is stated, words that say it was large.

The reading supplies those three facts and is still made blind: it is asked
how big the document says the move was, never told how big the move was.

What it does not fix: a document can give a reason that is wrong, and the
reading is only as good as the model doing it. Those are measured, not
assumed away.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from src.rag import market_report
from src.rag.evidence import pack_evidence, trading_days_before

BLIND_VERSION = "blind-v4"

# Documents older than this many trading days are not read at all.
MAX_AGE_TRADING_DAYS = 3
# A document this recent can support or oppose an explanation. Same day and
# the two trading days before: a futures price reacts to news within a
# session, and one more day absorbs late date stamps and time zones. Chosen
# before any run was looked at; src/evaluation/metrics.py re-scores the
# stored readings at 1 and 3 so the choice can be seen not to be doing the
# work.
FRESH_TRADING_DAYS = 2

# At most this many documents are read in one call, freshest first. Each
# reading costs output tokens, and an answer cut off mid-way is not valid
# JSON; the newest documents are the ones the rule looks at first anyway.
MAX_DOCUMENTS_READ = 12

EXTRACTION_MAX_OUTPUT_TOKENS = 3000
# The answer can still be cut off at that limit: the fallback models reason
# before they answer and the reasoning counts as output. In the version 9 run
# on the labelled dates that happened on 7 of 18 dates and left 29 of 196
# documents without a reading - always the last ones shown. Documents left
# unread are sent again on their own, to the model that answered, at most
# this many times.
MAX_FOLLOW_UP_READS = 2
# The fallback provider counts prompt and output against one request limit,
# and the reading needs more output than an explanation does.
PROMPT_TOKEN_RESERVE = 1000

DIRECTIONS = ("up", "down")

PROMPT = """You are reading news about the {commodity} market published in the days up to {date}.

You have NOT been told what {commodity} prices did on {date}. Do not guess. Report only what each document itself says. The title is part of the document.

For every document below, return one object with these fields:
  "document_id": the id given for the document.
  "price_move": "up", "down", "mixed" or "none". Does the document report or forecast {commodity} prices on the commodity market (futures or wholesale prices) moving, and which way? "none" if it does not talk about a market price move. The price of a cup or a bag in a shop is not a market price.
  "when": when the price move it reports happened, relative to the document's own date: "same_day", "previous_day", "earlier" (more than a day before, for example a report on last week, last month or the year so far), "forecast" (expected in future) or "unclear". Use "unclear" if price_move is "none".
  "move_pct": the size of that price move as a number of percent, if the document states one (4 for "jumps 4%", 6 for "more than 6%", 4 for "3-5%"), else null. The move of that day, not a change over a week, a month or the year.
  "move_size": how big the document's own words make that move: "large" (jumps, soars, surges, plunges, tumbles, slumps, sharply, the biggest in weeks, or the like in its language), "small" (edges, slightly, adjusts, steady, little changed) or "unstated" (it only says prices rose or fell; also when price_move is "none").
  "this_market": false if the move it reports is only of {other_markets}, not of {market}. Otherwise true.
  "pressure": "up", "down" or "none". What would the events the document reports do to {commodity} prices? Less supply or more demand is "up". More supply, better crop weather or weaker demand is "down". "none" if it reports nothing that bears on prices, or you cannot tell.
  "cause": the reason the document gives, in at most 15 words. Write it in ENGLISH whatever language the document is in. null if the document gives no reason. A document that only says prices rose or fell, without saying why, has cause null.
  "cause_type": "trading" if the reason only describes trading, with no event behind it (technical factors or adjustments, profit-taking, buying or selling by funds, positioning, a contract rollover, "after recent gains"). "event" for any other reason. null if cause is null.
  "quote": at most 25 consecutive words copied exactly from the document, in the document's OWN language, that state that reason. null if cause is null.

Rules:
- One object per document, in the order given. Do not skip a document and do not add any.
- A title alone can be enough. "Coffee tumbles as frost concerns ease" reports a move (down), gives a reason (frost concerns eased), and the title itself is the quote.
- Copy the quote exactly as it appears, from the title or the text. Do not translate it, shorten it with "..." or reword it. A translated quote will be rejected.
- Where the text shows " ... ", passages have been left out; a quote must come from one passage, not across the gap.
- A document that is not about {commodity} prices or supply gets price_move "none", pressure "none", cause null, quote null.

Return ONLY this JSON object: {{"documents": [ ... ]}}

DOCUMENTS:
{evidence_block}
"""


def market_names(commodity: str) -> tuple:
    """(the market whose prices are watched, markets that are not it), as the
    reading is told them. From the commodity's configuration where it says;
    otherwise a wording that fits any futures market."""
    watched, others = f"{commodity} futures", "retail prices, or prices paid locally in one country"
    try:
        from src.config.commodities import COMMODITIES
        cfg = COMMODITIES.get(str(commodity).strip().lower().replace(" ", "_"))
    except Exception:  # noqa: BLE001 - the wording above is a complete fallback
        cfg = None
    if cfg is not None:
        watched = getattr(cfg, "watched_market", "") or watched
        others = getattr(cfg, "other_markets", "") or others
    return watched, others


def build_prompt(commodity: str, anomaly_date: str, evidence_block: str) -> str:
    """The reading prompt. Nothing about the move goes into it: not its
    direction and not its size."""
    watched, others = market_names(commodity)
    return PROMPT.format(commodity=commodity, date=anomaly_date, market=watched,
                         other_markets=others, evidence_block=evidence_block)


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> list:
    return _WORD.findall((text or "").lower())


def shown_text(doc: dict, excerpt_chars: int) -> str:
    """What the model was shown of this document: its title and the excerpt."""
    return f"{doc.get('title') or ''}\n{(doc.get('text') or doc.get('title') or '')[:excerpt_chars]}"


def quote_in_text(quote: str, text: str, min_ngram_share: float = 0.8) -> bool:
    """Is `quote` in `text`, allowing for punctuation, case and spacing?

    Exact match on the text with punctuation and spaces removed, which also
    covers scripts written without spaces. Failing that, at least
    `min_ngram_share` of the quote's word 3-grams must appear in the text,
    which forgives a dropped word or a stray ellipsis but not a paraphrase.
    A quote of fewer than three words is not accepted, since it proves
    nothing - unless it is at least twelve characters long, which is how a
    real quote in a script written without spaces looks to a word splitter.
    """
    q, t = _tokens(quote), _tokens(text)
    joined = "".join(q)
    if not t or (len(q) < 3 and len(joined) < 12):
        return False
    if joined in "".join(t):
        return True
    if len(q) < 3:
        return False
    grams = [tuple(q[i:i + 3]) for i in range(len(q) - 2)]
    have = {tuple(t[i:i + 3]) for i in range(len(t) - 2)}
    return sum(1 for g in grams if g in have) / len(grams) >= min_ngram_share


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _age(doc: dict, anomaly_date: str) -> Optional[int]:
    age = doc.get("trading_days_before_anomaly")
    if age is None:
        age = trading_days_before(doc.get("publication_date", ""), anomaly_date)
    return age


def recent_documents(documents: list, anomaly_date: str,
                     max_age: int = MAX_AGE_TRADING_DAYS) -> list:
    """Documents from the session of the anomaly date or up to max_age
    sessions before it, in the order they are offered to the reading (see
    market_report.reading_order): market reports first, newest session first.
    Undated documents are left out: a document that cannot be placed in time
    cannot be called timely.

    Never ordered by the order the documents arrived in, which upstream steps
    may have arranged by how well each document fits the direction of the
    move."""
    def relevance(doc):
        return max(doc.get("semantic_score") or 0.0, doc.get("retrieval_score") or 0.0)

    ages = {id(d): _age(d, anomaly_date) for d in documents}
    in_range = [d for d in documents if ages[id(d)] is not None and 0 <= ages[id(d)] <= max_age]
    return market_report.reading_order(in_range, lambda d: ages[id(d)], relevance,
                                       limit=MAX_DOCUMENTS_READ)


def as_shown(doc: dict) -> dict:
    """The document as the reading is shown it: its text reduced to the
    opening and the passages that mention the commodity, so that a fixed-length
    excerpt holds the part that matters. The passages are verbatim, so a quote
    from this text is a quote from the document."""
    return {**doc, "text": market_report.focus_text(doc.get("text") or "")}


def _clean_choice(value, allowed: tuple, default: str = "none") -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in allowed else default


WHEN = ("same_day", "previous_day", "earlier", "forecast", "unclear")
MOVE_SIZES = ("large", "small", "unstated")
CAUSE_TYPES = ("event", "trading")


def _clean_flag(value, default: bool = True) -> bool:
    """A true/false field. Anything that is not clearly the other value keeps
    the default, so a model that leaves the field out changes nothing."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text in ("false", "no", "0"):
        return False
    if text in ("true", "yes", "1"):
        return True
    return default


def _clean_percent(value) -> Optional[float]:
    """A stated move size as a positive number of percent, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = abs(float(str(value).replace("%", "").replace(",", ".").strip()))
    except ValueError:
        return None
    return number if 0 < number < 100 else None


def _clean_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in ("", "null", "none", "n/a") else text


def normalize_reading(raw, shown: list, excerpt_chars: int, anomaly_date: str) -> list:
    """Turn the model's JSON into one checked record per shown document.

    Anything the model returned for a document it was not shown is dropped.
    A shown document the model skipped gets an empty record, so the count of
    documents read is always the count shown. The quote is checked here.
    """
    if isinstance(raw, dict):
        raw = raw.get("documents")
    items = raw if isinstance(raw, list) else []
    by_id = {}
    for item in items:
        if isinstance(item, dict) and item.get("document_id") is not None:
            by_id.setdefault(str(item["document_id"]).strip(), item)

    records = []
    for doc in shown:
        doc_id = doc.get("document_id")
        item = by_id.get(str(doc_id), {})
        price_move = _clean_choice(item.get("price_move"), ("up", "down", "mixed", "none"))
        pressure = _clean_choice(item.get("pressure"), ("up", "down", "none"))
        cause, quote = _clean_text(item.get("cause")), _clean_text(item.get("quote"))
        when = _clean_choice(item.get("when"), WHEN, "unclear")
        direction = (price_move if price_move in DIRECTIONS
                     else pressure if price_move == "none" and pressure in DIRECTIONS else None)
        records.append({
            "document_id": doc_id,
            "trading_days_before": _age(doc, anomaly_date),
            "price_move": price_move,
            "pressure": pressure,
            "direction": direction,
            # "report": the document itself says which way prices moved.
            # "pressure": it reports events, and the direction is what those
            # events would do to prices.
            "kind": (None if direction is None
                     else "report" if price_move in DIRECTIONS else "pressure"),
            "when": when,
            "move_pct": _clean_percent(item.get("move_pct")),
            # How big the document's words say the move was. Asked without
            # the model knowing how big the move to be explained is.
            "move_size": (_clean_choice(item.get("move_size"), MOVE_SIZES, "unstated")
                          if price_move in DIRECTIONS else "unstated"),
            # False: the move it reports is of another market only (another
            # grade, a local price). Such a reading is not evidence here.
            "this_market": _clean_flag(item.get("this_market")),
            "cause": cause,
            # "trading": the reason describes trading, with no event behind it.
            "cause_type": (_clean_choice(item.get("cause_type"), CAUSE_TYPES, "event")
                           if cause else None),
            "quote": quote,
            "quote_found": bool(cause and quote
                                and quote_in_text(quote, shown_text(doc, excerpt_chars))),
            "read": bool(item),
            # Did the document clear the relevance gate proper, or was it read
            # only because it is recent and close to the bar? Kept so that
            # the wider net can be scored against the strict one afterwards.
            "passed_gate": bool(doc.get("passed_gate", True)),
            # First seen before the exchange opened, so already counted with
            # the session before its date (market_report.session_date).
            "seen_before_open": bool(doc.get("session_date") and doc.get("publication_date")
                                     and doc["session_date"] != doc["publication_date"]),
        })
    return records


def _read_remaining(readings: list, shown: list, excerpt_chars: int, anomaly_date: str,
                    commodity: str, info: dict) -> tuple:
    """Ask again about the documents the first answer left out.

    Returns (readings, follow_ups). The follow-up goes to the same model and
    provider that gave the first answer, with the same prompt and the same
    text, so every reading of a date comes from one model. A document the
    model already answered for is never asked about twice, so nothing it said
    can change. If a follow-up fails for any reason the documents stay unread,
    as they were, and the failure is recorded.
    """
    from src.rag import explainer

    follow_ups = []
    for _ in range(MAX_FOLLOW_UP_READS):
        unread = [doc for doc, r in zip(shown, readings) if not r["read"]]
        if not unread or len(unread) == len(shown):
            break          # all read; or nothing was, and asking again would repeat it
        prompt = build_prompt(
            commodity, anomaly_date,
            explainer.build_evidence_block(unread, excerpt_chars=excerpt_chars))
        record = {"documents": len(unread), "read": 0}
        follow_ups.append(record)
        try:
            again = explainer.call_one_model(
                info["provider"], info["model"], prompt,
                temperature=explainer.EXPLANATION_TEMPERATURE,
                max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS)
            parsed = json.loads(again["response"].text)
        except Exception as exc:  # noqa: BLE001 - provider errors vary by type
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
            break
        more = {r["document_id"]: r
                for r in normalize_reading(parsed, unread, excerpt_chars, anomaly_date)
                if r["read"]}
        record["read"] = len(more)
        explainer.log_call(anomaly_date, getattr(again["response"], "usage_metadata", None),
                           "READ_MORE", model=again["model"], provider=again["provider"],
                           latency_ms=again["latency_ms"], documents_in_prompt=len(unread),
                           prompt_version=BLIND_VERSION)
        if not more:
            break
        readings = [more.get(r["document_id"], r) for r in readings]
    return readings, follow_ups


def _cache_key(anomaly_date: str, documents: list, providers, commodity: str) -> str:
    payload = [BLIND_VERSION, anomaly_date, commodity, list(providers or []),
               [[d.get("document_id"), d.get("title"),
                 hashlib.sha256((d.get("text") or "").encode("utf-8")).hexdigest()[:16]]
                for d in documents]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def read_documents(anomaly_date: str, documents: list, providers: list = None,
                   commodity: str = "coffee", cache_dir=None) -> dict:
    """Step 1 and 2: have the model read the recent documents without knowing
    the price move, and check its quotes.

    Returns {"status": "ok" | "nothing_recent" | "API_ERROR" | "PARSE_ERROR",
             "readings": [...], and the run details of the call}.
    No recent documents means no call.

    cache_dir: the reading depends only on the date and the documents, never
    on the move, so the evaluation harness reuses one reading for the true
    move and the flipped one instead of paying for it twice.
    """
    # Imported here so that importing this module does not need a model key.
    from src.rag import explainer

    recent = [as_shown(d) for d in recent_documents(documents, anomaly_date)[:MAX_DOCUMENTS_READ]]
    if not recent:
        return {"status": "nothing_recent", "readings": []}

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{_cache_key(anomaly_date, recent, providers, commodity)}.json"
        if cache_path.exists():
            try:
                return {**json.loads(cache_path.read_text(encoding="utf-8")), "from_cache": True}
            except (ValueError, OSError):
                pass

    frame_tokens = int(len(PROMPT) / 3.2) + 60

    def build(max_prompt_tokens):
        budget = (None if max_prompt_tokens is None
                  else max(max_prompt_tokens - PROMPT_TOKEN_RESERVE - frame_tokens, 500))
        packed, report = pack_evidence(recent, budget, min_documents=3)
        report["document_ids"] = [d.get("document_id") for d in packed]
        prompt = build_prompt(
            commodity, anomaly_date,
            explainer.build_evidence_block(packed, excerpt_chars=report["excerpt_chars"]))
        return prompt, report

    try:
        info = explainer._call_model_ex(build, temperature=explainer.EXPLANATION_TEMPERATURE,
                                        max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS,
                                        providers=providers)
    except Exception as exc:  # noqa: BLE001 - provider errors vary by type
        explainer.log_call(anomaly_date, None, "API_ERROR", prompt_version=BLIND_VERSION)
        return {"status": "API_ERROR", "readings": [],
                "reason": f"model_unavailable: {type(exc).__name__}: {exc}"[:300]}

    response = info["response"]
    packing = info.get("packing") or {}
    shown_ids = packing.get("document_ids") or [d.get("document_id") for d in recent]
    run = {
        "model_used": info["model"], "provider": info["provider"],
        "prompt_version": BLIND_VERSION, "latency_ms": info["latency_ms"],
        "evidence_document_ids": shown_ids,
        "evidence_packing": {k: v for k, v in packing.items() if k != "document_ids"},
    }
    log = dict(model=info["model"], provider=info["provider"], latency_ms=info["latency_ms"],
               documents_in_prompt=len(shown_ids), prompt_version=BLIND_VERSION)
    usage = getattr(response, "usage_metadata", None)

    try:
        parsed = json.loads(response.text)
    except (ValueError, TypeError) as exc:
        explainer.log_call(anomaly_date, usage, "PARSE_ERROR", **log)
        return {"status": "PARSE_ERROR", "readings": [],
                "reason": f"model_response_not_valid_json: {exc}",
                "raw_response": (getattr(response, "text", "") or "")[:500], **run}
    if not isinstance(parsed, (dict, list)) or (isinstance(parsed, dict)
                                               and not isinstance(parsed.get("documents"), list)):
        explainer.log_call(anomaly_date, usage, "PARSE_ERROR", **log)
        return {"status": "PARSE_ERROR", "readings": [],
                "reason": "model_response_has_no_documents_list",
                "raw_response": (getattr(response, "text", "") or "")[:500], **run}

    by_id = {d.get("document_id"): d for d in recent}
    shown = [by_id[i] for i in shown_ids if i in by_id]
    excerpt_chars = (packing.get("excerpt_chars") or 1500)
    readings = normalize_reading(parsed, shown, excerpt_chars, anomaly_date)
    explainer.log_call(anomaly_date, usage, "READ", **log)
    left_unread = sum(1 for r in readings if not r["read"])
    readings, follow_ups = _read_remaining(readings, shown, excerpt_chars, anomaly_date,
                                           commodity, info)

    out = {"status": "ok", "readings": readings, **run,
           "reading_calls": {
               "unread_after_first_answer": left_unread,
               "output_tokens": getattr(usage, "candidates_token_count", None),
               "output_limit": EXTRACTION_MAX_OUTPUT_TOKENS,
               "follow_ups": follow_ups,
               "unread_at_the_end": sum(1 for r in readings if not r["read"])}}
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(out), encoding="utf-8")
        except OSError:
            pass
    return out


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

RULES = ("freshest", "pooled")
KINDS = ("report", "pressure")
# The newest evidence points the way prices went, but nothing in it fits this
# move: the reasons only describe trading, or no report describes a move of
# this size.
NO_FITTING_REPORT = "no_report_gives_an_event_as_the_reason_for_a_move_of_this_size"


# Other versions of the rule that the stored readings allow to be scored
# afterwards without calling a model (src/evaluation/metrics.py does). None
# is on in the pipeline: each was thought of before any reading was seen, and
# each should earn its place on the test conditions, not be assumed.
#   gate_only         count only documents that cleared the relevance gate
#                     proper (not those read because they were recent and near
#                     the bar)
#   current_moves     a report of a price move counts only for the session it
#                     describes: one about a move from more than a day before,
#                     or a forecast, is dropped (the year's rally is not a
#                     reason for today's fall), and one about the previous
#                     day's move counts as a session older than its date. On in
#                     the pipeline from version 9 (DEFAULT_VARIANTS); before
#                     that it only dropped the "more than a day before" kind.
#   size_consistent   drop a document whose stated move is under a quarter of
#                     the actual move ("edged up 0.5%" does not explain +8%)
#   reports_only      count only documents that report a price move, not ones
#                     that only report an event that would bear on prices.
#                     Added after the first run on the labelled dates, where
#                     one explanation rested on a single background article
#                     (planted area up 1.9%); chosen with that date in view,
#                     so its score on those dates is not evidence for it.
#   this_market       drop a document whose reported move is only of another
#                     market (robusta for an arabica series, a local price)
#   event_reasons     a report whose reason only describes trading ("technical
#                     adjustments", "profit-taking") cannot support an
#                     explanation. The labelling rule already says such a
#                     reason is no cause (data/labeling/LABELLING_RULE.md).
#   sized_support     a report can support an explanation only if it describes
#                     a move of this size: it states a size of at least
#                     MIN_SIZE_RATIO of the flagged move or, stating none, its
#                     words call the move large. Not applied to a trend
#                     anomaly, where the day's own move is not what was flagged.
# The last three are on in the pipeline from version 11. They were written
# after the first ten invented-move trials (src/evaluation/robustness.py), with
# those trials in view, so the rate on those ten days is not evidence for
# them; quiet days drawn with another seed are.
VARIANTS = ("gate_only", "current_moves", "size_consistent", "reports_only",
            "this_market", "event_reasons", "sized_support")
# The variants the pipeline itself applies.
DEFAULT_VARIANTS = ("current_moves", "this_market", "event_reasons", "sized_support")
# Set when size_consistent was written, before any run; sized_support uses the
# same number so that no new threshold was picked with results in view.
MIN_SIZE_RATIO = 0.25

# Why a reading that points the way prices went was not allowed to support an
# explanation.
TRADING_REASON = "reason_only_describes_trading"
SIZE_NOT_MATCHED = "does_not_describe_a_move_of_this_size"


def matches_size(reading: dict, actual_pct: float = None) -> bool:
    """Does this reading describe a move of the size that was flagged?

    A stated size decides when there is one: at least MIN_SIZE_RATIO of the
    flagged move. With no stated size, the document's own words must call the
    move large. A document that reports no price move at all (background on an
    event) describes no size, so it does not match. With no size to compare
    against (actual_pct None) everything matches.
    """
    if not actual_pct:
        return True
    if reading.get("kind", "report") != "report":
        return False
    if reading.get("move_pct"):
        return reading["move_pct"] >= MIN_SIZE_RATIO * abs(actual_pct)
    return reading.get("move_size") == "large"


def set_aside_reason(reading: dict, variants=(), actual_pct: float = None) -> Optional[str]:
    """Why a reading pointing the way prices went cannot support an
    explanation under these variants, or None if it can."""
    if "event_reasons" in variants and reading.get("cause_type") == "trading":
        return TRADING_REASON
    if "sized_support" in variants and not matches_size(reading, actual_pct):
        return SIZE_NOT_MATCHED
    return None


def usable(readings: list, fresh_days: int = FRESH_TRADING_DAYS, variants=(),
           actual_pct: float = None) -> list:
    """Readings that count as evidence: recent enough, pointing up or down,
    with a reason whose quote was found in the document - and passing any of
    the optional VARIANTS asked for."""
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variant(s) {sorted(unknown)}; expected from {VARIANTS}")
    out = []
    for r in readings:
        if not (r.get("direction") in DIRECTIONS and r.get("cause") and r.get("quote_found")
                and r.get("trading_days_before") is not None
                and 0 <= r["trading_days_before"] <= fresh_days):
            continue
        if "gate_only" in variants and not r.get("passed_gate", True):
            continue
        if "this_market" in variants and r.get("this_market") is False:
            continue
        if "current_moves" in variants and r.get("kind") == "report":
            if r.get("when") in ("earlier", "forecast"):
                continue
            # "Prices rose yesterday" in a document already moved back a
            # session for being seen before the open is about that session;
            # moving it again would count it two sessions old.
            if r.get("when") == "previous_day" and not r.get("seen_before_open"):
                if r["trading_days_before"] + 1 > fresh_days:
                    continue
                r = {**r, "trading_days_before": r["trading_days_before"] + 1,
                     "counted_a_session_older": True}
        if ("size_consistent" in variants and actual_pct and r.get("move_pct")
                and r["move_pct"] < MIN_SIZE_RATIO * abs(actual_pct)):
            continue
        if "reports_only" in variants and r.get("kind") != "report":
            continue
        out.append(r)
    return out


def deciding_group(evidence: list, fresh_days: int = FRESH_TRADING_DAYS):
    """(age, kind, readings) of the first non-empty group, newest day first
    and reports before pressure within a day; None if there is no evidence."""
    for age in range(fresh_days + 1):
        for kind in KINDS:
            group = [r for r in evidence
                     if r["trading_days_before"] == age and r.get("kind", "report") == kind]
            if group:
                return age, kind, group
    return None


def decide(readings: list, direction: str, fresh_days: int = FRESH_TRADING_DAYS,
           rule: str = "freshest", variants=(), actual_pct: float = None) -> dict:
    """Step 3: the explain/refuse rule. A pure function of the stored
    readings, so any other setting can be scored later without calling a
    model.

    rule="freshest" (the pipeline's rule): the newest, most direct group of
    evidence decides - see deciding_group. Yesterday's coverage of a rally
    does not outvote today's report of the fall, and a feature about a
    drought does not outvote a market report published the same day.

    rule="pooled" (kept for comparison): every usable reading in the window
    counts equally. This is the shape of rule that made the earlier
    direction guard refuse almost every down day.

    Either way: explain when the readings pointing the way prices moved are
    at least one and at least as many as those pointing the other way.

    Under the variants event_reasons and sized_support a reading pointing the
    way prices moved supports the explanation only if it gives an event as
    the reason and describes a move of this size (set_aside_reason). The ones
    that do not are set aside: they are not counted for the move and not
    against it, and the rule does not go on to older evidence because of
    them. Readings pointing the other way count as they always did.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    evidence = usable(readings, fresh_days, variants, actual_pct)
    group = deciding_group(evidence, fresh_days) if rule == "freshest" else None
    pool = group[2] if group else (evidence if rule == "pooled" else [])
    with_the_move = [r for r in pool if r["direction"] == direction]
    opposing = [r for r in pool if r["direction"] != direction]
    why_not = {id(r): set_aside_reason(r, variants, actual_pct) for r in with_the_move}
    supporting = [r for r in with_the_move if not why_not[id(r)]]
    set_aside = [{"document_id": r.get("document_id"), "why": why_not[id(r)]}
                 for r in with_the_move if why_not[id(r)]]

    if direction not in DIRECTIONS:
        decision, reason = "REFUSE", "direction_of_the_move_unknown"
    elif not pool:
        decision, reason = "REFUSE", "no_recent_document_gives_a_reason_for_a_price_move"
    elif not with_the_move:
        decision, reason = "REFUSE", "the_freshest_evidence_points_the_other_way"
    elif not supporting:
        decision, reason = "REFUSE", NO_FITTING_REPORT
    elif len(opposing) > len(supporting):
        decision, reason = "REFUSE", "the_freshest_evidence_mostly_points_the_other_way"
    else:
        decision, reason = "EXPLAIN", None
    return {
        "decision": decision, "reason": reason, "rule": rule,
        "supporting": supporting, "opposing": opposing, "set_aside": set_aside,
        "deciding_age": group[0] if group else None,
        "deciding_kind": group[1] if group else None,
        "confidence": ("high" if decision == "EXPLAIN" and len(supporting) >= 2 and not opposing
                       else "medium" if decision == "EXPLAIN" else None),
        "fresh_trading_days": fresh_days,
    }


_WHEN = {0: "that day", 1: "the trading day before", 2: "two trading days before",
         3: "three trading days before"}


def _outlet(url: str) -> str:
    """The site an article is from, as a reader would name it."""
    from urllib.parse import urlparse
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def reported_move(reading: dict) -> str:
    """What the document itself says about the price move, and nothing more:
    the direction, and the size only if the document states one."""
    if reading.get("kind") != "report":
        return "no price move reported"
    way = "up" if reading.get("direction") == "up" else "down"
    if reading.get("move_pct"):
        return f"{way} {reading['move_pct']:g}%"
    return f"{way}, size not stated"


def cited_source(reading: dict, documents: dict) -> dict:
    """One source an explanation rests on, with everything a reader needs to
    check it: where it was published and when, what it reports, the reason it
    gives and the words that state that reason."""
    doc = documents.get(reading.get("document_id")) or {}
    return {
        "document_id": reading.get("document_id"),
        "title": " ".join(str(doc.get("title") or "").split()),
        "outlet": _outlet(doc.get("url")),
        "url": doc.get("url") or "",
        "publication_date": doc.get("publication_date") or "",
        "reported_move": reported_move(reading),
        "supports": reading.get("cause"),
        "quote": reading.get("quote"),
    }


def _source_line(source: dict) -> str:
    where = ", ".join(x for x in (source["outlet"], source["publication_date"]) if x)
    head = f'"{source["title"]}"' if source["title"] else "Untitled article"
    return (f'- {head}{f" ({where})" if where else ""}. '
            f'Reported move: {source["reported_move"]}. '
            f'Reason: {str(source["supports"]).rstrip(". ")}. '
            f'Quote: "{source["quote"]}" (source: {source["document_id"]})')


def compose_explanation(anomaly_date: str, price_description: str, direction: str,
                        verdict: dict, documents: dict = None) -> str:
    """Step 4: the explanation. One line for the move, then one line per
    source: its headline, where and when it was published, the move it
    reports, the reason it gives and the quote that states it. Nothing is
    added to what the sources say; the only words not taken from a source or
    from the price data are the labels.

    documents: the documents that were read, by id.
    """
    documents = documents or {}
    supporting, opposing = verdict["supporting"], verdict["opposing"]
    when = _WHEN.get(verdict.get("deciding_age"), "the days before")

    def counted(n):
        return f"{n} report{'s' if n != 1 else ''} from {when}"

    lines = [f"{price_description} on {anomaly_date}. "
             f"{'Reason' if len(supporting) == 1 else 'Reasons'} given by "
             f"{counted(len(supporting))}:"]
    lines += [_source_line(cited_source(r, documents)) for r in supporting]
    if opposing:
        lines.append(f"Pointing the other way, {counted(len(opposing))}:")
        lines += [_source_line(cited_source(r, documents)) for r in opposing]
    return "\n".join(lines)


def flagged_day_move(anomaly: dict) -> Optional[float]:
    """The size of the day's own move, for the size checks. None for a trend
    anomaly: that is flagged on a move over several days, and a report of one
    day of it is not expected to match."""
    if (anomaly or {}).get("anomaly_type") == "trend":
        return None
    return (anomaly or {}).get("pct_move")


def blind_decision(anomaly: dict, price_description: str, accepted_documents: list,
                   providers: list = None, commodity: str = "coffee", cache_dir=None) -> dict:
    """Run steps 1-4 and return a result in the same shape the explainer's
    generate_explanation returns, so the rest of the pipeline (citation
    audit, outcome tiers, dashboard) reads it unchanged. The readings and the
    rule's inputs are kept under "blind_evidence"."""
    date, direction = anomaly["date"], anomaly.get("direction")
    reading = read_documents(date, accepted_documents, providers=providers,
                             commodity=commodity, cache_dir=cache_dir)
    run = {k: reading[k] for k in ("model_used", "provider", "prompt_version", "latency_ms",
                                   "evidence_document_ids", "evidence_packing",
                                   "reading_calls") if k in reading}
    run.setdefault("prompt_version", BLIND_VERSION)

    if reading["status"] in ("API_ERROR", "PARSE_ERROR"):
        return {"decision": reading["status"], "explanation": None, "citations": [],
                "confidence": None, "reason": reading.get("reason"),
                **({"raw_response": reading["raw_response"]} if "raw_response" in reading else {}),
                **run}

    verdict = decide(reading["readings"], direction, variants=DEFAULT_VARIANTS,
                     actual_pct=flagged_day_move(anomaly))
    blind = {
        "rule": verdict["rule"],
        "variants": list(DEFAULT_VARIANTS),
        "fresh_trading_days": FRESH_TRADING_DAYS,
        "max_age_trading_days": MAX_AGE_TRADING_DAYS,
        "n_recent_documents": len(recent_documents(accepted_documents, date)),
        "readings": reading["readings"],
        "deciding_age": verdict["deciding_age"],
        "deciding_kind": verdict["deciding_kind"],
        "supporting": [r["document_id"] for r in verdict["supporting"]],
        "opposing": [r["document_id"] for r in verdict["opposing"]],
        # Pointing the way prices went, but not allowed to support an
        # explanation, each with why.
        "set_aside": verdict["set_aside"],
    }
    by_id = {d.get("document_id"): d for d in accepted_documents}
    if verdict["decision"] != "EXPLAIN":
        reason = ("no_recent_documents" if reading["status"] == "nothing_recent"
                  else verdict["reason"])
        return {"decision": "INSUFFICIENT_EVIDENCE", "explanation": None, "citations": [],
                "confidence": None, "reason": f"blind_evidence: {reason}",
                "blind_evidence": blind, **run}
    return {
        "decision": "EXPLAINED",
        "explanation": compose_explanation(date, price_description, direction, verdict,
                                           by_id),
        "citations": [cited_source(r, by_id) for r in verdict["supporting"]],
        "confidence": verdict["confidence"],
        "reason": None,
        "blind_evidence": blind,
        **run,
    }


def tally(result: dict) -> dict:
    """The blind reading as a with/against/neutral count, in the shape the
    outcome classifier takes."""
    blind = (result or {}).get("blind_evidence") or {}
    with_, against = len(blind.get("supporting") or []), len(blind.get("opposing") or [])
    return {"consistent": with_, "inconsistent": against,
            "neutral": max(0, len(blind.get("readings") or []) - with_ - against)}
