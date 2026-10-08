"""The blind-evidence decision: reading, quote check, rule, and the pipeline
path built on it. The model is always mocked."""
import json
from unittest import mock

from src import pipeline
from src.rag import blind_evidence as be
from src.rag import explainer
from src.rag import outcome as oc

UP = {"date": "2021-07-20", "price": 180.0, "z_score": 3.1, "anomaly_flag": True,
      "pct_move": 6.7, "direction": "up"}
DOWN = {**UP, "pct_move": -6.7, "z_score": -3.1, "direction": "down"}


def _doc(i, title, text, date="2021-07-20", sem=0.90):
    return {"document_id": f"d{i}", "title": title, "text": text, "url": f"https://s{i}.example/a",
            "publication_date": date, "retrieval_score": 0.0, "semantic_score": sem}


FROST = _doc(1, "Coffee surges as frost hits Brazil",
             "Arabica futures surged on Tuesday after a severe frost damaged crops in Minas Gerais.")
RAIN = _doc(2, "Coffee slips on rain forecast",
            "Coffee prices fell as forecasts of rain eased worries about the Brazilian drought.")
OLD = _doc(3, "Coffee hit a six-year high last week",
           "Prices rose last week as the drought worsened across the coffee belt.",
           date="2021-07-12")
FEATURE = _doc(4, "How drought is changing coffee farming",
               "A long drought has cut yields on farms across the region this year.")


def _reading(doc_id, age, direction, kind="report", cause="a reason", found=True):
    return {"document_id": doc_id, "trading_days_before": age, "direction": direction,
            "kind": kind, "cause": cause, "quote": "q", "quote_found": found}


class _Response:
    usage_metadata = None

    def __init__(self, payload):
        self.text = payload if isinstance(payload, str) else json.dumps(payload)


def _model(payload, seen=None):
    """Stand-in for the provider call: builds the prompt the way a provider
    with a 5,000-token limit would, records it, and answers with `payload`."""
    def call(build, **kwargs):
        prompt, packing = build(5000)
        if seen is not None:
            seen.append(prompt)
        return {"response": _Response(payload), "model": "m", "provider": "groq",
                "latency_ms": 5, "packing": packing}
    return call


def _answer(*items, size="large"):
    """A model answer. Each reported move is one the document's words call
    large, unless a test says otherwise: most tests here are about something
    other than the size check."""
    return {"documents": [{**dict(zip(("document_id", "price_move", "pressure", "cause", "quote"),
                                      item)), "move_size": size} for item in items]}


# --- quote check -----------------------------------------------------------

def test_quote_must_be_in_the_document():
    text = "Coffee: Prices FELL, as rain forecasts eased drought-fears in Brazil."
    assert be.quote_in_text("prices fell as rain forecasts eased drought fears", text)
    assert not be.quote_in_text("prices dropped because the rain finally came", text)   # paraphrase
    assert not be.quote_in_text("on frost", "prices up on frost")                       # too short
    assert be.quote_in_text("ราคากาแฟพุ่ง เซ่นพิษ", "ข่าว: ราคากาแฟพุ่ง เซ่นพิษโลกรวน")      # no spaces
    assert not be.quote_in_text("", text) and not be.quote_in_text("a b c", "")


# --- the rule --------------------------------------------------------------

def test_no_recent_reason_means_refuse():
    v = be.decide([], "up")
    assert v["decision"] == "REFUSE" and v["supporting"] == []
    stale = [_reading("a", 3, "up")]                       # read, but outside the window
    assert be.decide(stale, "up")["decision"] == "REFUSE"
    assert be.decide(stale, "up", fresh_days=3)["decision"] == "EXPLAIN"


def test_a_reason_without_a_found_quote_does_not_count():
    assert be.decide([_reading("a", 0, "up", found=False)], "up")["decision"] == "REFUSE"
    assert be.decide([_reading("a", 0, "up", cause=None)], "up")["decision"] == "REFUSE"


def test_todays_report_of_the_fall_outranks_yesterdays_rally_coverage():
    readings = [_reading("y1", 1, "up"), _reading("y2", 1, "up"), _reading("y3", 1, "up"),
                _reading("t1", 0, "down")]
    v = be.decide(readings, "down")
    assert v["decision"] == "EXPLAIN" and v["deciding_age"] == 0
    assert [r["document_id"] for r in v["supporting"]] == ["t1"]
    # the pooled count is what refused nearly every down day
    assert be.decide(readings, "down", rule="pooled")["decision"] == "REFUSE"


def test_a_market_report_outranks_a_background_feature_from_the_same_day():
    readings = [_reading("report", 0, "down"),
                _reading("f1", 0, "up", kind="pressure"), _reading("f2", 0, "up", kind="pressure")]
    v = be.decide(readings, "down")
    assert v["decision"] == "EXPLAIN" and v["deciding_kind"] == "report"
    # with no report that day, the event coverage decides
    v = be.decide(readings[1:], "up")
    assert v["decision"] == "EXPLAIN" and v["deciding_kind"] == "pressure"


def test_the_same_readings_cannot_explain_both_directions_unless_they_conflict():
    readings = [_reading("a", 0, "up"), _reading("b", 0, "up"), _reading("c", 1, "down")]
    assert be.decide(readings, "up")["decision"] == "EXPLAIN"
    assert be.decide(readings, "down")["decision"] == "REFUSE"
    tied = [_reading("a", 0, "up"), _reading("b", 0, "down")]
    both = [be.decide(tied, d) for d in ("up", "down")]
    assert [v["decision"] for v in both] == ["EXPLAIN", "EXPLAIN"]
    assert all(v["confidence"] == "medium" for v in both)      # and neither is called strong


def test_outnumbered_in_the_deciding_group_means_refuse():
    readings = [_reading("a", 0, "up"), _reading("b", 0, "down"), _reading("c", 0, "down")]
    v = be.decide(readings, "up")
    assert v["decision"] == "REFUSE" and "mostly" in v["reason"]
    assert be.decide(readings, "down")["confidence"] == "medium"


# --- reading ---------------------------------------------------------------

def test_the_prompt_never_mentions_the_move():
    seen = []
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(_answer(), seen)), \
         mock.patch.object(explainer, "log_call"):
        be.read_documents("2021-07-20", [dict(FROST), dict(RAIN)])
    prompt = seen[0]
    assert "NOT been told" in prompt and "2021-07-20" in prompt
    for leak in ("6.7", "z-score", "anomaly", "+6", "-6"):
        assert leak not in prompt


def test_documents_too_old_or_undated_are_not_read_and_no_call_is_made_for_none():
    undated = {**_doc(9, "No date", "text"), "publication_date": ""}
    assert be.recent_documents([dict(OLD), undated], "2021-07-20") == []
    with mock.patch.object(explainer, "_call_model_ex") as call:
        out = be.read_documents("2021-07-20", [dict(OLD), undated])
    call.assert_not_called()
    assert out["status"] == "nothing_recent"


def test_reading_order_does_not_depend_on_input_order():
    a, b = dict(FROST), dict(RAIN, semantic_score=0.95)
    one = [d["document_id"] for d in be.recent_documents([a, b], "2021-07-20")]
    two = [d["document_id"] for d in be.recent_documents([b, a], "2021-07-20")]
    assert one == two == ["d2", "d1"]


def test_readings_are_checked_against_what_was_shown():
    answer = _answer(
        ("d1", "up", "up", "frost damaged crops", "a severe frost damaged crops in Minas Gerais"),
        ("d2", "down", "down", "rain eased drought worries", "rain fixed everything overnight"),
        ("d77", "up", "up", "made up", "not a document that was shown"),
        ("d4", "sideways", "up", "drought cut yields", "A long drought has cut yields on farms"),
    )
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(answer)), \
         mock.patch.object(explainer, "log_call"):
        out = be.read_documents("2021-07-20", [dict(FROST), dict(RAIN), dict(FEATURE)])
    by_id = {r["document_id"]: r for r in out["readings"]}
    assert set(by_id) == {"d1", "d2", "d4"}                       # d77 was never shown
    assert by_id["d1"]["quote_found"] and by_id["d1"]["kind"] == "report"
    assert not by_id["d2"]["quote_found"]                         # quote is not in the document
    assert by_id["d4"]["price_move"] == "none"                    # "sideways" is not a value
    assert by_id["d4"]["direction"] == "up" and by_id["d4"]["kind"] == "pressure"


def test_a_malformed_answer_is_a_fault_not_a_refusal():
    for bad in ("not json", {"something": "else"}):
        with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(bad)), \
             mock.patch.object(explainer, "log_call"):
            result = be.blind_decision(UP, "+6.7% move", [dict(FROST)])
        assert result["decision"] == "PARSE_ERROR"
    with mock.patch.object(explainer, "_call_model_ex", side_effect=RuntimeError("down")), \
         mock.patch.object(explainer, "log_call"):
        assert be.blind_decision(UP, "+6.7% move", [dict(FROST)])["decision"] == "API_ERROR"


def test_a_reading_is_reused_from_the_cache(tmp_path):
    answer = _answer(("d1", "up", "up", "frost damaged crops", "a severe frost damaged crops"))
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(answer)) as call, \
         mock.patch.object(explainer, "log_call"):
        first = be.read_documents("2021-07-20", [dict(FROST)], cache_dir=tmp_path)
        second = be.read_documents("2021-07-20", [dict(FROST)], cache_dir=tmp_path)
    assert call.call_count == 1 and second["from_cache"] is True
    assert first["readings"] == second["readings"]


# --- through the pipeline --------------------------------------------------

ANSWER = _answer(
    ("d1", "up", "up", "frost damaged crops", "a severe frost damaged crops in Minas Gerais"),
    ("d2", "down", "down", "rain eased drought worries",
     "forecasts of rain eased worries about the Brazilian drought"),
)


def _pipeline(anomaly, docs, answer=ANSWER, **kw):
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(answer)), \
         mock.patch.object(explainer, "log_call"), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False):
        return pipeline.explain_from_documents(anomaly, [dict(d) for d in docs],
                                               verbose=False, **kw)


def test_blind_is_the_default_and_explains_from_quoted_evidence():
    frost_only = {"documents": [ANSWER["documents"][0]]}
    out = _pipeline(UP, [FROST], answer=frost_only)
    er = out["explanation_result"]
    assert out["decision_mode"] == "blind" and out["pipeline_version"] == pipeline.PIPELINE_VERSION
    assert er["decision"] == "EXPLAINED" and out["guard_rule"] is None
    assert er["explanation"].splitlines() == [
        "+6.7% day-over-day move (z-score 3.10) on 2021-07-20. Reason given by 1 report from that day:",
        '- "Coffee surges as frost hits Brazil" (s1.example, 2021-07-20). '
        'Reported move: up, size not stated. Reason: frost damaged crops. '
        'Quote: "a severe frost damaged crops in Minas Gerais" (source: d1)']
    assert er["citations"] == [{
        "document_id": "d1", "title": "Coffee surges as frost hits Brazil",
        "outlet": "s1.example", "url": "https://s1.example/a", "publication_date": "2021-07-20",
        "reported_move": "up, size not stated", "supports": "frost damaged crops",
        "quote": "a severe frost damaged crops in Minas Gerais"}]
    assert out["direction_summary"] == {"consistent": 1, "inconsistent": 0, "neutral": 0}
    assert "direction_summary_wordlist" in out
    assert out["outcome"]["tier"] == oc.EXPLAINED_TENTATIVE       # one document is not "strong"


def test_the_same_evidence_does_not_explain_the_opposite_move():
    """The flip test, in miniature: frost news explains a rise and nothing else."""
    up = _pipeline(UP, [FROST], answer={"documents": [ANSWER["documents"][0]]})
    down = _pipeline(DOWN, [FROST], answer={"documents": [ANSWER["documents"][0]]})
    assert up["explanation_result"]["decision"] == "EXPLAINED"
    er = down["explanation_result"]
    assert er["decision"] == "INSUFFICIENT_EVIDENCE"
    assert er["reason"] == "blind_evidence: the_freshest_evidence_points_the_other_way"
    assert down["outcome"]["tier"] == oc.REFUSED_CONFLICTING and down["outcome"]["is_refusal"]
    assert er["blind_evidence"]["readings"] == up["explanation_result"]["blind_evidence"]["readings"]


def test_a_tie_on_the_day_is_explained_but_not_called_strong():
    out = _pipeline(DOWN, [FROST, RAIN])
    assert out["explanation_result"]["decision"] == "EXPLAINED"
    text = out["explanation_result"]["explanation"]
    assert "Pointing the other way, 1 report from that day:" in text   # the opposing report is named
    assert text.count("(source: ") == 2 and '"Coffee surges as frost hits Brazil"' in text
    assert out["outcome"]["tier"] == oc.EXPLAINED_TENTATIVE


def test_nothing_recent_is_refused_without_calling_the_model():
    with mock.patch.object(explainer, "_call_model_ex") as call, \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False):
        out = pipeline.explain_from_documents(UP, [dict(OLD)], verbose=False)
    call.assert_not_called()
    assert out["explanation_result"]["reason"] == "blind_evidence: no_recent_documents"
    assert out["outcome"]["tier"] == oc.REFUSED_WEAK_EVIDENCE


def test_a_reading_fault_is_a_system_error_so_the_date_is_retried():
    out = _pipeline(UP, [FROST], answer="garbage")
    assert out["outcome"]["tier"] == oc.SYSTEM_ERROR and out["outcome"]["is_fault"]


def test_only_the_freshest_documents_are_read_when_there_are_many():
    many = [_doc(i, f"Coffee story number {i}", f"Text of story {i} about coffee prices.",
                 date="2021-07-20" if i < 10 else "2021-07-16") for i in range(20)]
    seen = []
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(_answer(), seen)), \
         mock.patch.object(explainer, "log_call"):
        out = be.read_documents("2021-07-20", many)
    assert len(out["readings"]) == be.MAX_DOCUMENTS_READ
    ages = [r["trading_days_before"] for r in out["readings"]]
    assert ages == sorted(ages) and ages.count(0) == 10      # all ten same-day stories kept


# --- version 6: wider reading net, size and timing of the reported move ------

NEAR = _doc(5, "Kaffee-Preise steigen nach Frost in Brasilien",
            "Arabica futures surged on Tuesday after a severe frost damaged crops in Minas Gerais.",
            sem=0.82)                                   # under the gate's 0.85, over the floor
FAR = _doc(6, "Coffee shop opens downtown", "A new cafe opened on Main Street.", sem=0.60)
NEAR_OLD = _doc(7, "Frost fears lift coffee", "Prices rose on frost fears.", date="2021-07-09",
                sem=0.82)


def _frost_from(doc_id):
    return _answer((doc_id, "up", "up", "frost damaged crops",
                    "a severe frost damaged crops in Minas Gerais"))


def test_a_recent_document_just_under_the_gate_is_read_and_marked():
    """The gate accepts nothing here; before version 6 this date was refused
    without a reading."""
    out = _pipeline(UP, [NEAR, FAR, NEAR_OLD], answer=_frost_from("d5"))
    assert out["gate_result"]["decision"] != "EXPLAIN"
    er = out["explanation_result"]
    assert er["decision"] == "EXPLAINED" and er["evidence_document_ids"] == ["d5"]
    assert [r["passed_gate"] for r in er["blind_evidence"]["readings"]] == [False]
    assert [(s["document_id"], s["passed_gate"]) for s in out["sources"]] == [("d5", False)]
    assert out["outcome"]["tier"] == oc.EXPLAINED_TENTATIVE and not out["outcome"]["is_refusal"]


def test_a_reading_that_finds_nothing_is_a_weak_evidence_refusal_not_a_gate_refusal():
    out = _pipeline(UP, [NEAR], answer=_answer(("d5", "none", "none", None, None)))
    assert out["explanation_result"]["decision"] == "INSUFFICIENT_EVIDENCE"
    assert out["explanation_result"]["reason"].startswith("blind_evidence:")
    assert out["outcome"]["tier"] == oc.REFUSED_WEAK_EVIDENCE


def test_nothing_near_the_bar_is_still_refused_at_the_gate_without_a_call():
    with mock.patch.object(explainer, "_call_model_ex") as call, \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False):
        out = pipeline.explain_from_documents(UP, [dict(FAR), dict(NEAR_OLD)], verbose=False)
    call.assert_not_called()
    assert out["sources"] == [] and out["outcome"]["is_refusal"]
    assert "blind_evidence" not in out["explanation_result"]


def test_documents_that_cleared_the_gate_are_marked_so():
    cands = pipeline.reading_candidates([dict(FROST), dict(NEAR), dict(FAR)], [FROST], "2021-07-20")
    assert [(d["document_id"], d["passed_gate"]) for d in cands] == [("d1", True), ("d5", False)]


def test_two_supporting_documents_are_strong_whatever_the_gate_accepted():
    near2 = {**NEAR, "document_id": "d8", "url": "https://s8.example/a",
             "title": "Frost sends arabica to a six-year high"}
    answer = {"documents": _frost_from("d5")["documents"] + _frost_from("d8")["documents"]}
    out = _pipeline(UP, [NEAR, near2], answer=answer)
    assert out["explanation_result"]["confidence"] == "high"
    assert out["outcome"]["tier"] == oc.EXPLAINED_STRONG


def test_the_size_and_timing_of_the_reported_move_are_kept():
    raw = {"documents": [
        {"document_id": "d1", "price_move": "up", "pressure": "up", "cause": "frost damaged crops",
         "quote": "a severe frost damaged crops in Minas Gerais", "when": "Same_Day",
         "move_pct": "4,5%"},
        {"document_id": "d2", "price_move": "down", "pressure": "down", "cause": "rain",
         "quote": "forecasts of rain eased worries", "when": "last tuesday", "move_pct": -2},
        {"document_id": "d4", "price_move": "none", "pressure": "up", "cause": "drought",
         "quote": "A long drought has cut yields", "move_pct": "a lot"},
    ]}
    by_id = {r["document_id"]: r for r in be.normalize_reading(
        raw, [FROST, RAIN, FEATURE], 1500, "2021-07-20")}
    assert (by_id["d1"]["when"], by_id["d1"]["move_pct"]) == ("same_day", 4.5)
    assert (by_id["d2"]["when"], by_id["d2"]["move_pct"]) == ("unclear", 2.0)   # unknown word, sign dropped
    assert (by_id["d4"]["when"], by_id["d4"]["move_pct"]) == ("unclear", None)
    assert be._clean_percent(True) is None and be._clean_percent(0) is None
    assert be._clean_percent(250) is None                                       # not a percent move


def test_the_prompt_asks_for_market_prices_and_still_never_mentions_the_move():
    seen = []
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(_answer(), seen)), \
         mock.patch.object(explainer, "log_call"):
        be.read_documents("2021-07-20", [FROST])
    assert '"when"' in seen[0] and '"move_pct"' in seen[0] and "in a shop" in seen[0]
    assert "6.7" not in seen[0]


def test_stricter_evidence_rules_can_be_scored_from_stored_readings():
    last_week = {**_reading("a", 0, "up"), "when": "earlier"}
    small = {**_reading("b", 0, "up"), "when": "same_day", "move_pct": 0.5}
    near = {**_reading("c", 0, "up"), "passed_gate": False}
    feature = {**_reading("e", 0, "up", kind="pressure"), "when": "earlier"}

    def explains(readings, variants, pct=8.0):
        return be.decide(readings, "up", variants=variants, actual_pct=pct)["decision"] == "EXPLAIN"

    assert all(explains([r], ()) for r in (last_week, small, near, feature))    # as the pipeline runs
    assert not explains([last_week], ("current_moves",))       # last week's rally is not today's reason
    assert explains([feature], ("current_moves",))             # only reports of a move are dated this way
    assert not explains([small], ("size_consistent",))         # 0.5% reported, 8% to explain
    assert explains([small], ("size_consistent",), pct=1.5)    # a third of the move is enough
    assert explains([small], ("size_consistent",), pct=None)   # no size known: nothing to compare
    assert explains([last_week], ("size_consistent",))         # no size stated: kept
    assert not explains([near], ("gate_only",))
    before_11 = tuple(v for v in be.VARIANTS if v != "sized_support")
    assert explains([_reading("d", 0, "up")], before_11)       # readings from before v6 pass these
    try:
        be.usable([small], variants=("typo",))
    except ValueError as exc:
        assert "typo" in str(exc)
    else:
        raise AssertionError("an unknown variant must not be silently ignored")


def test_background_articles_alone_can_be_kept_from_deciding():
    background = _reading("a", 0, "down", kind="pressure")     # "planted area up 1.9%"
    report = _reading("b", 0, "down")                          # "coffee fell 5% as ..."
    assert be.decide([background], "down")["decision"] == "EXPLAIN"            # as the pipeline runs
    assert be.decide([background], "down", variants=("reports_only",))["decision"] != "EXPLAIN"
    assert be.decide([background, report], "down",
                     variants=("reports_only",))["decision"] == "EXPLAIN"


# --- version 9: what is read, which part of it, and which session it speaks for ---

def test_a_report_counts_for_the_session_it_describes():
    yearly = {**_reading("a", 0, "up"), "when": "earlier"}        # "coffee is up 70% this year"
    forecast = {**_reading("b", 0, "up"), "when": "forecast"}     # "prices expected to rise"
    yesterday = {**_reading("c", 0, "up"), "when": "previous_day"}   # "coffee rose on Thursday", read on Friday
    today = {**_reading("d", 0, "down"), "when": "same_day"}

    v = be.decide([yearly, forecast, yesterday, today], "down", variants=be.DEFAULT_VARIANTS)
    assert v["decision"] == "EXPLAIN" and v["deciding_age"] == 0
    assert [r["document_id"] for r in v["supporting"]] == ["d"] and v["opposing"] == []
    # before version 9 the same four readings refused the fall, three against one
    assert be.decide([yearly, forecast, yesterday, today], "down")["decision"] == "REFUSE"

    # yesterday's report still counts, a session older, when today has nothing
    v = be.decide([yearly, yesterday], "up", variants=be.DEFAULT_VARIANTS)
    assert v["decision"] == "EXPLAIN" and v["deciding_age"] == 1
    assert v["supporting"][0]["counted_a_session_older"]
    # ... but not past the window
    old = {**_reading("e", be.FRESH_TRADING_DAYS, "up"), "when": "previous_day"}
    assert be.decide([old], "up", variants=be.DEFAULT_VARIANTS)["decision"] == "REFUSE"
    # an event that bears on prices is not a report of a move and is not re-dated
    feature = {**_reading("f", 0, "up", kind="pressure"), "when": "earlier"}
    assert be.decide([feature], "up", variants=("current_moves",))["decision"] == "EXPLAIN"


def test_the_pipeline_applies_that_rule_and_records_it():
    old_news = _doc(1, "Coffee prices are up 70% this year",
                    "Coffee prices have risen 70% this year on poor harvests in Brazil.")
    today = _doc(2, "Coffee tumbles as frost concerns ease",
                 "Arabica coffee fell sharply on Friday as frost concerns eased in Brazil.")
    answer = {"documents": [
        {"document_id": "d1", "price_move": "up", "pressure": "up", "when": "earlier",
         "cause": "poor harvests", "quote": "risen 70% this year on poor harvests in Brazil"},
        {"document_id": "d2", "price_move": "down", "pressure": "down", "when": "same_day",
         "move_size": "large", "cause_type": "event",
         "cause": "frost concerns eased", "quote": "Coffee tumbles as frost concerns ease"}]}
    out = _pipeline(DOWN, [old_news, today], answer=answer)
    er = out["explanation_result"]
    assert er["decision"] == "EXPLAINED"
    assert er["blind_evidence"]["variants"] == list(be.DEFAULT_VARIANTS)
    assert be.DEFAULT_VARIANTS == ("current_moves", "this_market", "event_reasons", "sized_support")
    assert er["blind_evidence"]["supporting"] == ["d2"] and er["blind_evidence"]["opposing"] == []
    assert er["citations"][0]["quote"] == "Coffee tumbles as frost concerns ease"   # quoted from the title


def test_market_reports_are_read_before_other_news_of_the_day():
    report = _doc(1, "SOFTS - Coffee prices jump nearly 7% in New York after stocks drawn", "", sem=0.81)
    junk = [_doc(i, f"New coffee shop number {i} opens downtown", "", sem=0.95) for i in range(2, 20)]
    order = be.recent_documents(junk + [report], "2021-07-20")
    assert order[0]["document_id"] == "d1"
    assert len(order) == 19


def test_the_model_is_shown_the_passage_about_coffee_and_can_quote_it():
    filler = "Raw sugar futures climbed to their highest in nearly six years on supply worries. " * 12
    coffee = ("March arabica coffee rose 5% as Brazilian physical coffee premiums hit their "
              "highest in a decade.")
    softs = _doc(1, "Raw sugar climbs to near six-year high, arabica coffee at 1-month peak",
                 filler + coffee)
    assert "premiums" not in softs["text"][:900]
    seen = []
    answer = _answer(("d1", "up", "up", "physical premiums at a decade high",
                      "Brazilian physical coffee premiums hit their highest in a decade"))
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(answer, seen)), \
         mock.patch.object(explainer, "log_call"):
        out = be.read_documents("2021-07-20", [softs])
    assert "premiums hit their highest" in seen[0]              # the passage reached the prompt
    assert " ... " in seen[0]
    assert out["readings"][0]["quote_found"]                    # and its quote checks out
    assert be.as_shown(softs)["text"] != softs["text"] and softs["text"].startswith(filler)


def test_the_prompt_asks_for_english_reasons_and_untranslated_quotes():
    seen = []
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(_answer(), seen)), \
         mock.patch.object(explainer, "log_call"):
        be.read_documents("2021-07-20", [FROST])
    assert "ENGLISH" in seen[0] and "OWN language" in seen[0]
    assert "A title alone can be enough" in seen[0]
    assert be.BLIND_VERSION == "blind-v4"


def test_an_overnight_report_of_yesterday_is_moved_back_once_not_twice():
    overnight = {**_doc(1, "Coffee settles lower on rain forecasts", "Coffee fell on Wednesday as rain was forecast."),
                 "publication_date": "2021-07-20", "session_date": "2021-07-19",
                 "trading_days_before_anomaly": 1}
    raw = _answer(("d1", "down", "down", "rain forecast", "Coffee fell on Wednesday as rain was forecast"))
    raw["documents"][0]["when"] = "previous_day"
    reading = be.normalize_reading(raw, [overnight], 1500, "2021-07-20")[0]
    assert reading["seen_before_open"] and reading["trading_days_before"] == 1
    v = be.decide([reading], "down", variants=be.DEFAULT_VARIANTS)
    assert v["decision"] == "EXPLAIN" and v["deciding_age"] == 1          # one session back, not two
    same_day_doc = {**reading, "seen_before_open": False, "trading_days_before": 0}
    assert be.decide([same_day_doc], "down", variants=be.DEFAULT_VARIANTS)["deciding_age"] == 1


# --- documents the first answer left out are asked about again ----------------

def _one_model(payloads, seen):
    """Stand-in for explainer.call_one_model: answers with each payload in turn."""
    answers = iter(payloads)

    def call(provider, model, prompt, **kwargs):
        seen.append((provider, model, prompt))
        payload = next(answers)
        if isinstance(payload, Exception):
            raise payload
        return {"response": _Response(payload), "model": model, "provider": provider,
                "latency_ms": 3}
    return call


def test_documents_left_unread_are_sent_again_to_the_same_model():
    first = _answer(("d1", "up", "up", "frost damaged crops", "a severe frost damaged crops in Minas Gerais"))
    more = _answer(("d2", "down", "down", "rain eased drought worries",
                    "forecasts of rain eased worries about the Brazilian drought"),
                   ("d1", "down", "down", "changed its mind", "a severe frost damaged crops"))
    seen = []
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(first)), \
         mock.patch.object(explainer, "call_one_model", side_effect=_one_model([more, _answer()], seen)), \
         mock.patch.object(explainer, "log_call") as log:
        out = be.read_documents("2021-07-20", [dict(FROST), dict(RAIN), dict(FEATURE)])
    by_id = {r["document_id"]: r for r in out["readings"]}
    assert [r["document_id"] for r in out["readings"]] == ["d1", "d2", "d4"]     # order kept
    assert by_id["d2"]["read"] and by_id["d2"]["quote_found"] and by_id["d2"]["direction"] == "down"
    assert by_id["d1"]["direction"] == "up"              # an answered document is never re-read
    assert not by_id["d4"]["read"]                       # still left out after a second try
    assert [(p, m) for p, m, _ in seen] == [("groq", "m"), ("groq", "m")]        # no other model
    assert "Coffee slips on rain forecast" in seen[0][2] and "frost hits Brazil" not in seen[0][2]
    assert "Coffee slips on rain forecast" not in seen[1][2]                     # only what is left
    assert out["reading_calls"] == {
        "unread_after_first_answer": 2, "output_tokens": None,
        "output_limit": be.EXTRACTION_MAX_OUTPUT_TOKENS,
        "follow_ups": [{"documents": 2, "read": 1}, {"documents": 1, "read": 0}],
        "unread_at_the_end": 1}
    assert [c.args[2] for c in log.call_args_list] == ["READ", "READ_MORE", "READ_MORE"]


def test_a_failed_follow_up_leaves_the_documents_unread_and_says_so():
    first = _answer(("d1", "up", "up", "frost damaged crops", "a severe frost damaged crops in Minas Gerais"))
    seen = []
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(first)), \
         mock.patch.object(explainer, "call_one_model",
                           side_effect=_one_model([RuntimeError("rate limit reached")], seen)), \
         mock.patch.object(explainer, "log_call"):
        out = be.read_documents("2021-07-20", [dict(FROST), dict(RAIN)])
    assert out["status"] == "ok" and len(seen) == 1
    assert [r["read"] for r in out["readings"]] == [True, False]
    follow_up = out["reading_calls"]["follow_ups"]
    assert len(follow_up) == 1 and "rate limit reached" in follow_up[0]["error"]

    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(first)), \
         mock.patch.object(explainer, "call_one_model", side_effect=_one_model(["not json"], [])), \
         mock.patch.object(explainer, "log_call"):
        out = be.read_documents("2021-07-20", [dict(FROST), dict(RAIN)])
    assert "error" in out["reading_calls"]["follow_ups"][0]


def test_no_follow_up_when_everything_or_nothing_was_answered():
    everything = _answer(("d1", "up", "up", "frost", "a severe frost damaged crops in Minas Gerais"),
                         ("d2", "none", "none", None, None))
    for first in (everything, _answer()):
        with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(first)), \
             mock.patch.object(explainer, "call_one_model") as again, \
             mock.patch.object(explainer, "log_call"):
            out = be.read_documents("2021-07-20", [dict(FROST), dict(RAIN)])
        assert not again.called and out["reading_calls"]["follow_ups"] == []


def test_the_follow_up_count_is_kept_with_the_stored_decision():
    first = _answer(("d1", "up", "up", "frost damaged crops", "a severe frost damaged crops in Minas Gerais"))
    more = _answer(("d2", "down", "down", "rain eased drought worries",
                    "forecasts of rain eased worries about the Brazilian drought"))
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(first)), \
         mock.patch.object(explainer, "call_one_model", side_effect=_one_model([more], [])), \
         mock.patch.object(explainer, "log_call"):
        result = be.blind_decision({"date": "2021-07-20", "direction": "up", "pct_move": 6.0},
                                   "+6%", [dict(FROST), dict(RAIN)])
    assert result["reading_calls"]["unread_at_the_end"] == 0
    # the second document was read after all, and it is a same-day report of a fall
    assert result["blind_evidence"]["opposing"] == ["d2"]


# --- version 11: a report has to fit the move it is offered as the reason for ---

FIT = ("this_market", "event_reasons", "sized_support")


def _report(doc_id, direction, pct=None, size="unstated", cause_type="event", this_market=True,
            age=0, kind="report"):
    return {**_reading(doc_id, age, direction, kind=kind), "when": "same_day", "move_pct": pct,
            "move_size": size, "cause_type": cause_type, "this_market": this_market}


def test_a_report_of_an_ordinary_day_does_not_explain_a_large_move():
    """The five false explanations of the first invented-move run, in miniature."""
    def verdict(reading, pct=4.5, direction="up"):
        return be.decide([reading], direction, variants=FIT, actual_pct=pct)

    # "the market makes technical adjustments": a trading reason, whatever its size
    v = verdict(_report("a", "up", size="large", cause_type="trading"))
    assert v["decision"] == "REFUSE" and v["reason"] == be.NO_FITTING_REPORT
    assert v["set_aside"] == [{"document_id": "a", "why": be.TRADING_REASON}]
    # "closed up 0.6%" against a claimed +3.9%
    v = verdict(_report("b", "up", pct=0.6), pct=3.9)
    assert v["decision"] == "REFUSE" and v["set_aside"][0]["why"] == be.SIZE_NOT_MATCHED
    # a stated size is believed over the adjectives around it
    assert verdict(_report("b2", "up", pct=0.6, size="large"), pct=3.9)["decision"] == "REFUSE"
    # "rose on demand and the dollar", no size and no word for it
    assert verdict(_report("c", "up"))["decision"] == "REFUSE"
    assert verdict(_report("c2", "up", size="small"))["decision"] == "REFUSE"
    # a report about robusta is not evidence about arabica, for the move or against it
    v = verdict(_report("d", "up", size="large", this_market=False))
    assert v["decision"] == "REFUSE" and v["set_aside"] == [] and v["deciding_age"] is None

    # what a real shock's report looks like
    assert verdict(_report("e", "up", pct=4.0), pct=7.96)["decision"] == "EXPLAIN"     # "jumps 4%"
    assert verdict(_report("f", "down", size="large"), pct=-8.6,
                   direction="down")["decision"] == "EXPLAIN"                           # "tumbles most since 2008"
    assert verdict(_report("g", "up", pct=1.0), pct=4.0)["decision"] == "EXPLAIN"      # exactly a quarter


def test_set_aside_reports_do_not_send_the_rule_to_older_evidence():
    today_small = _report("t", "up", pct=0.5)
    yesterday_big = _report("y", "up", pct=5.0, age=1)
    v = be.decide([today_small, yesterday_big], "up", variants=FIT, actual_pct=5.0)
    assert v["decision"] == "REFUSE" and v["deciding_age"] == 0 and v["supporting"] == []
    # with nothing at all from the day, yesterday's report is the newest evidence
    assert be.decide([yesterday_big], "up", variants=FIT, actual_pct=5.0)["decision"] == "EXPLAIN"


def test_reports_against_the_move_count_as_they_always_did():
    fits = _report("s", "up", pct=5.0)
    against = [_report("o1", "down", cause_type="trading"), _report("o2", "down", pct=0.3)]
    v = be.decide([fits] + against, "up", variants=FIT, actual_pct=5.0)
    assert v["decision"] == "REFUSE" and len(v["opposing"]) == 2      # outnumbered, 2 to 1
    v = be.decide([fits, against[0]], "up", variants=FIT, actual_pct=5.0)
    assert v["decision"] == "EXPLAIN" and v["confidence"] == "medium"
    # a report set aside is not counted on either side
    v = be.decide([fits, _report("x", "up", cause_type="trading")], "up", variants=FIT, actual_pct=5.0)
    assert [r["document_id"] for r in v["supporting"]] == ["s"] and v["confidence"] == "medium"


def test_background_coverage_alone_cannot_explain_a_shock_but_can_a_trend():
    feature = _report("f", "up", kind="pressure")          # "drought is cutting yields": no move, so no size
    assert be.decide([feature], "up", variants=FIT, actual_pct=5.0)["decision"] == "REFUSE"
    assert be.decide([feature], "up", variants=FIT, actual_pct=None)["decision"] == "EXPLAIN"
    assert be.flagged_day_move({"pct_move": 0.8, "anomaly_type": "trend"}) is None
    assert be.flagged_day_move({"pct_move": 5.1, "anomaly_type": "shock+trend"}) == 5.1
    assert be.flagged_day_move({"pct_move": 5.1}) == 5.1


def test_the_reading_records_size_in_words_reason_type_and_market():
    raw = {"documents": [
        {"document_id": "d1", "price_move": "up", "pressure": "up", "cause": "frost damaged crops",
         "quote": "a severe frost damaged crops in Minas Gerais", "move_size": "LARGE",
         "cause_type": "event", "this_market": True},
        {"document_id": "d2", "price_move": "down", "pressure": "down", "cause": "technical selling",
         "quote": "forecasts of rain eased worries", "move_size": "tiny",
         "cause_type": "Trading", "this_market": "false"},
        {"document_id": "d4", "price_move": "none", "pressure": "up", "cause": "drought",
         "quote": "A long drought has cut yields", "move_size": "large"},
    ]}
    by_id = {r["document_id"]: r for r in be.normalize_reading(
        raw, [FROST, RAIN, FEATURE], 1500, "2021-07-20")}
    assert (by_id["d1"]["move_size"], by_id["d1"]["cause_type"], by_id["d1"]["this_market"]) == \
        ("large", "event", True)
    assert (by_id["d2"]["move_size"], by_id["d2"]["cause_type"], by_id["d2"]["this_market"]) == \
        ("unstated", "trading", False)                       # "tiny" is not a value
    assert by_id["d4"]["move_size"] == "unstated"            # reports no move, so no size
    assert by_id["d4"]["cause_type"] == "event" and by_id["d4"]["this_market"] is True   # left out: defaults
    empty = be.normalize_reading({"documents": []}, [FROST], 1500, "2021-07-20")[0]
    assert empty["cause_type"] is None and empty["move_size"] == "unstated"


def test_the_prompt_asks_how_big_the_document_says_the_move_was_without_saying_how_big_it_was():
    seen = []
    with mock.patch.object(explainer, "_call_model_ex", side_effect=_model(_answer(), seen)), \
         mock.patch.object(explainer, "log_call"):
        be.read_documents("2021-07-20", [FROST], commodity="coffee")
    prompt = seen[0]
    for field in ('"move_size"', '"cause_type"', '"this_market"'):
        assert field in prompt
    assert "arabica coffee futures in New York" in prompt and "robusta" in prompt
    for leak in ("6.7", "z-score", "anomaly"):
        assert leak not in prompt
    assert be.market_names("wheat")[0] == "wheat futures"       # no wording set: a general one


def test_a_quiet_days_report_is_refused_through_the_pipeline_with_its_own_wording():
    answer = _answer(("d1", "up", "up", "technical adjustments",
                      "Arabica futures surged on Tuesday"), size="unstated")
    out = _pipeline(UP, [FROST], answer=answer)
    er = out["explanation_result"]
    assert er["decision"] == "INSUFFICIENT_EVIDENCE"
    assert er["reason"] == f"blind_evidence: {be.NO_FITTING_REPORT}"
    assert er["blind_evidence"]["set_aside"] == [{"document_id": "d1", "why": be.SIZE_NOT_MATCHED}]
    assert out["outcome"]["tier"] == oc.REFUSED_WEAK_EVIDENCE
    assert "none fits a move of this size" in out["outcome"]["headline"]
    # the same reading is enough when the day's own move was not what was flagged
    trend = {**UP, "pct_move": 0.9, "anomaly_type": "trend"}
    assert _pipeline(trend, [FROST], answer=answer)["explanation_result"]["decision"] == "EXPLAINED"


def test_an_explanation_is_one_exact_line_per_source_and_nothing_else():
    docs = {"a": {"document_id": "a", "title": "Coffee   jumps 7% after\n stocks drawn",
                  "url": "https://www.wire.example/markets/1", "publication_date": "2023-11-30"},
            "b": {"document_id": "b", "title": "", "url": "", "publication_date": ""}}
    sized = {**_report("a", "up", pct=7.0), "cause": "stocks drawn.", "quote": "after stocks drawn"}
    background = {**_report("b", "up", kind="pressure"), "cause": "drought", "quote": "drought cut yields"}
    verdict = {"supporting": [sized, background], "opposing": [], "deciding_age": 1,
               "deciding_kind": "report"}
    lines = be.compose_explanation("2023-11-30", "+7.1% move", "up", verdict, docs).splitlines()
    assert lines[0] == "+7.1% move on 2023-11-30. Reasons given by 2 reports from the trading day before:"
    assert lines[1] == ('- "Coffee jumps 7% after stocks drawn" (wire.example, 2023-11-30). '
                        'Reported move: up 7%. Reason: stocks drawn. '
                        'Quote: "after stocks drawn" (source: a)')
    # nothing is made up for a document with no headline, outlet or date, or with no price move
    assert lines[2] == ('- Untitled article. Reported move: no price move reported. '
                        'Reason: drought. Quote: "drought cut yields" (source: b)')
    assert len(lines) == 3
    assert be.reported_move({"kind": "report", "direction": "down", "move_pct": 4.93}) == "down 4.93%"
    assert be.reported_move({"kind": "report", "direction": "down"}) == "down, size not stated"
