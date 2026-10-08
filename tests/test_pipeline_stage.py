"""The post-retrieval stage (gate -> explain -> guard -> audit -> outcome) and
the retrieval clean-up steps. The model and the network are always mocked."""
import tempfile
from unittest import mock

import pandas as pd

from src import pipeline
from src.rag import outcome as oc


UP = {"date": "2021-07-20", "price": 180.0, "z_score": 3.1, "anomaly_flag": True,
      "pct_move": 6.7, "direction": "up"}
DOWN = {"date": "2021-07-19", "price": 160.0, "z_score": -2.07, "anomaly_flag": True,
        "pct_move": -3.7, "direction": "down"}


def _doc(i, title, text, sem=0.90, date="2021-07-19"):
    return {"document_id": f"d{i}", "title": title, "text": text, "url": f"https://s{i}.example/a",
            "publication_date": date, "retrieval_score": 0.0, "semantic_score": sem}


def _explained(doc_id="d1"):
    return {"decision": "EXPLAINED", "explanation": f"Frost lifted prices (source: {doc_id}).",
            "citations": [{"document_id": doc_id, "supports": "frost lifted coffee prices"}],
            "confidence": "high", "reason": None, "model_used": "m", "provider": "groq"}


RISING = [
    _doc(1, "Coffee surges on Brazil frost", "Arabica futures surged and rose sharply after frost."),
    _doc(2, "Frost lifts coffee to multi-year high", "Prices rose again and jumped higher."),
    _doc(3, "Coffee rallies as frost damage spreads", "Coffee rallied and climbed on supply fears."),
]


def _run(anomaly, docs, model_result, **kw):
    """The earlier decision path: the model is told the move and decides."""
    kw.setdefault("decision_mode", "legacy")
    with mock.patch.object(pipeline, "generate_explanation", return_value=model_result), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False):
        return pipeline.explain_from_documents(anomaly, [dict(d) for d in docs],
                                               verbose=False, **kw)


def test_consistent_evidence_is_explained():
    out = _run(UP, RISING, _explained())
    assert out["explanation_result"]["decision"] == "EXPLAINED"
    assert out["outcome"]["tier"] in oc.EXPLAINED_TIERS
    assert out["pipeline_version"] == pipeline.PIPELINE_VERSION
    assert out["faithfulness_report"]["citations_resolve"] is True


def test_direction_guard_withholds_an_explanation_built_on_opposite_evidence():
    # A DOWN day, evidence all about prices RISING, and a model that explains anyway.
    out = _run(DOWN, RISING, _explained())
    er = out["explanation_result"]
    assert er["decision"] == "INSUFFICIENT_EVIDENCE"
    assert er["reason"].startswith("direction_guard")
    assert er["explanation"] is None and er["citations"] == []
    assert "Frost lifted prices" in er["withheld_explanation"]["explanation"]
    assert er["model_used"] == "m"                      # run metadata is kept
    assert out["outcome"]["tier"] == oc.REFUSED_CONFLICTING
    assert out["outcome"]["is_refusal"] is True
    assert out["faithfulness_report"] is None           # nothing left to audit
    assert out["guard_rule"] == oc.GUARD_RULE and out["decision_mode"] == "legacy"


def test_direction_guard_can_be_switched_off():
    out = _run(DOWN, RISING, _explained(), direction_guard=False)
    assert out["explanation_result"]["decision"] == "EXPLAINED"
    assert out["outcome"]["tier"] == oc.EXPLAINED_TENTATIVE
    assert out["guard_rule"] is None


def test_guard_leaves_a_model_refusal_alone():
    refusal = {"decision": "INSUFFICIENT_EVIDENCE", "explanation": "no", "citations": [],
               "confidence": "low", "reason": "model_judged_evidence_insufficient"}
    assert oc.apply_direction_guard(refusal, {"consistent": 0, "inconsistent": 5}) is refusal


def test_guard_leaves_a_supported_explanation_alone():
    result = _explained()
    assert oc.apply_direction_guard(result, {"consistent": 4, "inconsistent": 3}) is result
    assert oc.apply_direction_guard(result, {}) is result


def test_citation_to_a_document_cut_from_the_prompt_is_flagged():
    result = _explained("d3")
    result["evidence_document_ids"] = ["d1", "d2"]      # d3 was packed out
    out = _run(UP, RISING, result)
    assert out["faithfulness_report"]["n_missing_document"] == 1
    assert out["faithfulness_report"]["citations_resolve"] is False
    assert out["outcome"]["tier"] == oc.EXPLAINED_TENTATIVE


def test_gate_rejection_never_reaches_the_model():
    weak = [_doc(1, "Festival opens", "A food festival.", sem=0.5)]
    with mock.patch.object(pipeline, "generate_explanation") as gen:
        out = pipeline.explain_from_documents(UP, weak, verbose=False)
    gen.assert_not_called()
    assert out["outcome"]["tier"] == oc.REFUSED_WEAK_EVIDENCE


# --- retrieval clean-up ----------------------------------------------------

def _gdelt(articles):
    return lambda q, **k: {"articles": articles}


def _retrieve(articles, date="2022-05-11"):
    with tempfile.TemporaryDirectory() as d, \
         mock.patch.object(pipeline, "fetch_gdelt", side_effect=_gdelt(articles)), \
         mock.patch.object(pipeline, "fetch_article_text", return_value=("", "no text")), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep"):
        return pipeline.retrieve_evidence(date, ["coffee Brazil frost"], "ref", cache_dir=d,
                                          fallback_dir=None, return_meta=True)


def test_articles_dated_after_the_anomaly_are_dropped_locally():
    docs, meta = _retrieve([
        {"url": "http://a/1", "title": "On the day", "seendate": "20220511T140000Z"},
        {"url": "http://b/2", "title": "Next day wrap", "seendate": "20220512T000000Z"},
    ])
    assert [d["title"] for d in docs] == ["On the day"]
    assert meta.dropped_out_of_window == 1
    assert meta.to_dict()["dropped_out_of_window"] == 1


def test_syndicated_copies_are_collapsed_and_counted():
    docs, meta = _retrieve([
        {"url": "http://a/1", "title": "Brazil drought punishes coffee farms", "seendate": "20220510T140000Z"},
        {"url": "http://b/1", "title": "Brazil drought punishes coffee farms", "seendate": "20220510T150000Z"},
        {"url": "http://c/1", "title": "Brazil drought punishes coffee farms - AP", "seendate": "20220510T160000Z"},
        {"url": "http://d/1", "title": "Something else about coffee", "seendate": "20220510T160000Z"},
    ])
    assert len(docs) == 2
    assert docs[0]["n_copies"] == 3 and meta.collapsed_duplicates == 2
    assert docs[0]["days_before_anomaly"] == 1


def test_copies_are_collapsed_before_text_is_fetched():
    articles = [{"url": f"http://s{i}/1", "title": "Same wire story",
                 "seendate": "20220510T140000Z"} for i in range(6)]
    with tempfile.TemporaryDirectory() as d, \
         mock.patch.object(pipeline, "fetch_gdelt", side_effect=_gdelt(articles)), \
         mock.patch.object(pipeline, "fetch_article_text", return_value=("", "x")) as fetch, \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep"):
        pipeline.retrieve_evidence("2022-05-11", ["q"], "ref", cache_dir=d, fallback_dir=None)
    assert fetch.call_count == 1


# --- anomaly description ---------------------------------------------------

def _anomaly_csv(tmp_path, rows):
    cols = ["date", "price", "z_score", "anomaly_flag", "anomaly_type",
            "cumulative_return", "cumulative_z_score"]
    (tmp_path / "results").mkdir()
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "results" / "a.csv", index=False)
    return tmp_path


def test_trend_anomaly_takes_its_direction_from_the_cumulative_move(tmp_path, monkeypatch):
    root = _anomaly_csv(tmp_path, [
        ["2026-06-17", 276.4, 0.2, False, None, 0.05, 1.0],
        ["2026-06-18", 275.1, -0.475, True, "trend", 0.113, 2.32],   # slipped on the day
    ])
    monkeypatch.setattr(pipeline, "REPO_ROOT", root)
    a = pipeline.get_anomaly("2026-06-18", "a.csv")
    assert a["anomaly_type"] == "trend"
    assert a["direction"] == "up"            # the flagged ten-day move was up
    text = pipeline.describe_move(a)
    assert "+11.3% over the last 10 trading days" in text and "-0.5%" in text


def test_shock_anomaly_is_described_by_its_daily_move(tmp_path, monkeypatch):
    root = _anomaly_csv(tmp_path, [
        ["2024-10-04", 250.0, 0.1, False, None, 0.0, 0.0],
        ["2024-10-07", 237.75, -2.22, True, "shock", -0.072, -1.93],
    ])
    monkeypatch.setattr(pipeline, "REPO_ROOT", root)
    a = pipeline.get_anomaly("2024-10-07", "a.csv")
    assert a["direction"] == "down"
    assert pipeline.describe_move(a) == "-4.9% day-over-day move (z-score -2.22)"


def test_describe_move_works_on_outputs_saved_before_anomaly_type_existed():
    assert pipeline.describe_move(UP) == "+6.7% day-over-day move (z-score 3.10)"


def test_google_news_links_are_not_fetched_and_keep_their_headline():
    articles = [
        {"url": "https://news.google.com/rss/articles/CBMiabc", "title": "Coffee jumps on frost - AP",
         "seendate": "20220510T140000Z"},
        {"url": "https://real.example/story", "title": "Brazil frost damages coffee crop badly",
         "seendate": "20220510T140000Z"},
    ]
    with tempfile.TemporaryDirectory() as d, \
         mock.patch.object(pipeline, "fetch_gdelt", side_effect=_gdelt(articles)), \
         mock.patch.object(pipeline, "fetch_article_text", return_value=("body text", None)) as fetch, \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep"):
        docs = pipeline.retrieve_evidence("2022-05-11", ["q"], "ref", cache_dir=d, fallback_dir=None)
    fetch.assert_called_once_with("https://real.example/story")
    by_url = {d["url"]: d for d in docs}
    assert by_url["https://news.google.com/rss/articles/CBMiabc"]["text"] == ""
    assert by_url["https://real.example/story"]["text"] == "body text"


def test_a_fetch_that_raises_does_not_end_the_run():
    articles = [{"url": f"https://site{i}.example/a", "title": f"Distinct coffee story number {i}",
                 "seendate": "20220510T140000Z"} for i in range(12)]

    def flaky(url):
        if "site3" in url:
            raise RuntimeError("connection reset")
        return ("text of " + url, None)

    with tempfile.TemporaryDirectory() as d, \
         mock.patch.object(pipeline, "fetch_gdelt", side_effect=_gdelt(articles)), \
         mock.patch.object(pipeline, "fetch_article_text", side_effect=flaky), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep"):
        docs = pipeline.retrieve_evidence("2022-05-11", ["q"], "ref", cache_dir=d, fallback_dir=None)
    assert len(docs) == 12
    texts = {d["url"]: d["text"] for d in docs}
    assert texts["https://site3.example/a"] == ""
    assert texts["https://site7.example/a"] == "text of https://site7.example/a"   # order kept


def test_what_the_model_was_shown_is_saved_as_shown(tmp_path):
    docs = [{"document_id": "d1", "title": "T1", "publication_date": "2021-07-20",
             "url": "https://a/1", "text": "x" * 5000},
            {"document_id": "d2", "title": "T2", "publication_date": "2021-07-19",
             "url": "https://b/2", "text": ""}]
    result = {"evidence_document_ids": ["d2", "d1"], "evidence_packing": {"excerpt_chars": 800}}
    pipeline.save_shown_documents(tmp_path, "2021-07-20", "coffee", result, docs)
    import json
    pack = json.loads((tmp_path / "coffee_2021-07-20.json").read_text())
    assert pack["excerpt_chars"] == 800
    assert [d["document_id"] for d in pack["documents"]] == ["d2", "d1"]     # prompt order
    assert len(pack["documents"][1]["text"]) == 800 and pack["documents"][0]["text"] == ""


def test_recent_queries_search_only_the_last_few_days_and_ask_the_feed_again():
    seen, feed_windows = [], []

    def gdelt(query, **kwargs):
        seen.append((query, kwargs["startdatetime"], kwargs["enddatetime"]))
        return {"articles": []}

    def feed(commodity, date, window_days=10, **kwargs):
        feed_windows.append(window_days)
        return [], {}

    with tempfile.TemporaryDirectory() as d, \
         mock.patch.object(pipeline, "fetch_gdelt", side_effect=gdelt), \
         mock.patch("src.rag.news_aggregator.aggregate_news", side_effect=feed), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep"):
        _, meta = pipeline.retrieve_evidence(
            "2022-05-11", ["coffee Brazil frost"], "ref", cache_dir=d, fallback_dir=None,
            commodity_key="coffee", recent_queries=["coffee futures"], return_meta=True)
    assert seen == [("coffee Brazil frost", "20220501000000", "20220512000000"),
                    ("coffee futures", "20220506000000", "20220512000000")]
    assert feed_windows == [pipeline.RETRIEVAL_WINDOW_DAYS_BEFORE, pipeline.RECENT_WINDOW_DAYS_BEFORE]
    assert meta.queries_attempted == 2


def test_the_recent_window_always_reaches_three_trading_days_back():
    from datetime import datetime, timedelta
    from src.rag.evidence import trading_days_before
    for offset in range(14):                       # every weekday the move can fall on
        day = datetime(2024, 9, 2) + timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        earliest = day - timedelta(days=pipeline.RECENT_WINDOW_DAYS_BEFORE)
        assert trading_days_before(earliest.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d")) >= 3


def test_recent_queries_ask_for_more_results_and_each_document_records_its_search():
    sizes = {}

    def gdelt(query, **kwargs):
        sizes[query] = kwargs["maxrecords"]
        return {"articles": [
            {"url": "http://shared/1", "title": "Coffee futures settle higher on frost",
             "seendate": "20220510T140000Z"},
            {"url": f"http://only/{len(sizes)}", "title": f"Story only search {len(sizes)} found",
             "seendate": "20220510T140000Z"}]}

    with tempfile.TemporaryDirectory() as d, \
         mock.patch.object(pipeline, "fetch_gdelt", side_effect=gdelt), \
         mock.patch.object(pipeline, "fetch_article_text", return_value=("", "x")), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep"):
        docs = pipeline.retrieve_evidence(
            "2022-05-11", ["coffee Brazil frost"], "ref", cache_dir=d, fallback_dir=None,
            recent_queries=["coffee futures"])
    assert sizes == {"coffee Brazil frost": pipeline.MAX_ARTICLES_PER_QUERY,
                     "coffee futures": pipeline.RECENT_MAX_ARTICLES}
    found = {d["url"]: d["found_by"] for d in docs}
    assert found == {"http://shared/1": "coffee Brazil frost",        # credited to the first search
                     "http://only/1": "coffee Brazil frost",
                     "http://only/2": "coffee futures"}
    assert pipeline._source_record(docs[0])["found_by"] == "coffee Brazil frost"


def test_article_text_is_fetched_only_for_documents_recent_enough_to_be_read():
    articles = [
        {"url": "https://a.example/today", "title": "Coffee jumps on frost damage in Brazil",
         "seendate": "20220511T140000Z"},                                  # Wednesday, the day
        {"url": "https://b.example/friday", "title": "Arabica ends the week lower on rain",
         "seendate": "20220506T140000Z"},                                  # 3 trading days before
        {"url": "https://c.example/old", "title": "Vietnam robusta harvest outlook improves",
         "seendate": "20220503T140000Z"},                                  # 6 trading days before
    ]

    def retrieve(**kw):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(pipeline, "fetch_gdelt", side_effect=_gdelt(articles)), \
             mock.patch.object(pipeline, "fetch_article_text",
                               side_effect=lambda url: ("body of " + url, None)) as fetch, \
             mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
             mock.patch.object(pipeline.time, "sleep"):
            docs = pipeline.retrieve_evidence("2022-05-11", ["q"], "ref", cache_dir=d,
                                              fallback_dir=None, **kw)
        return {d["url"]: d["text"] for d in docs}, fetch.call_count

    texts, calls = retrieve(fetch_text_max_age=3)
    assert calls == 2
    assert texts["https://a.example/today"] and texts["https://b.example/friday"]
    assert texts["https://c.example/old"] == ""                # kept, with its headline only
    assert retrieve()[1] == 3                                  # the default still fetches everything


def test_combined_searches_are_the_only_requests_and_earlier_answers_are_reused():
    from src.rag.gdelt_client import build_query_params, cache_put, DEFAULT_SORT
    combined, combined_recent = "coffee (frost OR drought)", "coffee (prices OR futures)"
    sent = []

    def gdelt(query, **kwargs):
        sent.append((query, kwargs["maxrecords"], kwargs["startdatetime"]))
        return {"articles": [{"url": f"http://new/{len(sent)}", "title": f"Found by request {len(sent)}",
                              "seendate": "20220510T140000Z"},
                             {"url": "http://both/1", "title": "Frost hits Brazil coffee crop",
                              "seendate": "20220510T140000Z"}]}

    with tempfile.TemporaryDirectory() as d, \
         mock.patch.object(pipeline, "fetch_gdelt", side_effect=gdelt), \
         mock.patch.object(pipeline, "fetch_article_text", return_value=("", "x")), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep") as sleep:
        # an earlier run already has the answer to one of the separate queries
        cache_put(d, build_query_params("coffee Brazil frost", pipeline.MAX_ARTICLES_PER_QUERY,
                                        DEFAULT_SORT, "20220501000000", "20220512000000"),
                  {"articles": [{"url": "http://both/1", "title": "Frost hits Brazil coffee crop",
                                 "seendate": "20220510T140000Z"},
                                {"url": "http://old/1", "title": "Only the earlier run found this",
                                 "seendate": "20220509T140000Z"}]})
        docs, meta = pipeline.retrieve_evidence(
            "2022-05-11", ["coffee Brazil frost", "coffee tariff"], "ref", cache_dir=d,
            fallback_dir=None, recent_queries=["coffee futures"], combined_query=combined,
            combined_recent_query=combined_recent, return_meta=True)

    # two requests, the combined ones, at GDELT's largest page; the separate queries are never sent
    assert sent == [(combined, 250, "20220501000000"), (combined_recent, 250, "20220506000000")]
    sleep.assert_called_once_with(pipeline.INTER_QUERY_DELAY_SECONDS)
    assert pipeline.INTER_QUERY_DELAY_SECONDS > 5              # GDELT: one request every 5 seconds
    assert (meta.queries_attempted, meta.network_attempts, meta.network_failures) == (2, 2, 0)
    assert (meta.cache_hits, meta.reused_cache_hits) == (0, 1)
    found = {d["url"]: d["found_by"] for d in docs}
    assert found == {"http://new/1": "combined search, full window",
                     "http://both/1": "combined search, full window",     # credited to the first to return it
                     "http://new/2": "combined search, recent days",
                     "http://old/1": "coffee Brazil frost"}


def test_a_failed_combined_search_still_leaves_the_reused_answers():
    from src.rag.gdelt_client import build_query_params, cache_put, DEFAULT_SORT
    with tempfile.TemporaryDirectory() as d, \
         mock.patch.object(pipeline, "fetch_gdelt", side_effect=RuntimeError("429")), \
         mock.patch.object(pipeline, "fetch_article_text", return_value=("", "x")), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep"):
        cache_put(d, build_query_params("coffee tariff", pipeline.MAX_ARTICLES_PER_QUERY,
                                        DEFAULT_SORT, "20220501000000", "20220512000000"),
                  {"articles": [{"url": "http://old/1", "title": "Tariff talk lifts coffee",
                                 "seendate": "20220509T140000Z"}]})
        docs, meta = pipeline.retrieve_evidence(
            "2022-05-11", ["coffee tariff"], "ref", cache_dir=d, fallback_dir=None,
            combined_query="coffee (tariff OR frost)", return_meta=True)
    assert [d["url"] for d in docs] == ["http://old/1"]
    assert (meta.network_attempts, meta.network_failures, meta.reused_cache_hits) == (1, 1, 1)
    assert meta.retrieval_ran()                                 # not an outage: there was evidence to read


def test_every_caller_gets_the_combined_searches_from_one_place():
    from src.config.commodities import get_commodity
    cfg = get_commodity("coffee")
    args = pipeline.search_arguments(cfg)
    assert args["combined_query"].startswith("coffee (") and " OR " in args["combined_query"]
    assert args["combined_recent_query"].startswith("coffee (")
    # every topic the separate queries searched for is in a combined search
    for word in ("frost", "drought", "Minas Gerais", "tariff", "Vietnam", "shipping"):
        assert word in args["combined_query"]
    for word in ("prices", "futures", "arabica", "settle", "harvest", "rain"):
        assert word in args["combined_recent_query"]
    assert args["combined_day_query"].startswith("coffee (")
    day = [r for r in pipeline.gdelt_query_plan("2022-05-11", [], **args)
           if r.label == "combined search, day of the move"]
    assert len(day) == 1 and day[0].live
    assert (day[0].startdatetime, day[0].enddatetime) == ("20220511000000", "20220512000000")
    # a commodity without combined searches keeps sending its separate ones
    oil = pipeline.search_arguments(get_commodity("crude_oil"))
    assert oil["combined_query"] is None
    assert all(r.live for r in pipeline.gdelt_query_plan("2022-05-11", ["oil price OPEC"], **oil))


def test_a_relevance_model_that_fails_is_a_fault_not_a_refusal():
    """On a real run the model ran out of memory, the gate fell back to
    keyword scores, and four dates were saved as refusals without a reading."""
    articles = [{"url": f"https://s{i}.example/a", "title": f"Coffee jumps on frost, report {i}",
                 "seendate": "20220510T140000Z"} for i in range(5)]

    def retrieve(scorer):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(pipeline, "fetch_gdelt", side_effect=_gdelt(articles)), \
             mock.patch.object(pipeline, "fetch_article_text", return_value=("", "x")), \
             mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", True), \
             mock.patch.object(pipeline, "score_documents_against_query", side_effect=scorer,
                               create=True), \
             mock.patch.object(pipeline.time, "sleep"):
            docs = pipeline.retrieve_evidence("2022-05-11", ["q"], "ref", cache_dir=d,
                                              fallback_dir=None)
            return docs, pipeline.semantic_scoring_failed(docs)

    docs, failed = retrieve(MemoryError("Failed to allocate memory"))
    assert len(docs) == 5 and failed                       # documents came back, unscored
    docs, failed = retrieve(lambda documents, query: [0.9] * len(documents))
    assert not failed and docs[0]["semantic_score"] == 0.9
    assert not pipeline.semantic_scoring_failed([])        # nothing retrieved is not this fault
    with mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False):
        assert not pipeline.semantic_scoring_failed([{"title": "x"}])   # keyword-only mode


def test_an_overnight_report_is_counted_with_the_session_before():
    """2023-11-30: 'Coffee Prices Settle Lower', first seen at 02:00 UTC, is the
    report of the 29th. It was read as evidence against the 30th's rise."""
    docs, _ = _retrieve([
        {"url": "http://a/overnight", "title": "Coffee Prices Settle Lower on Forecasts for Rain in Brazil",
         "seendate": "20220511T020000Z"},
        {"url": "http://b/close", "title": "SOFTS - Coffee prices jump nearly 7% in New York",
         "seendate": "20220511T204500Z"},
        {"url": "http://c/monday", "title": "Arabica futures open the week higher",
         "seendate": "20220509T030000Z"},              # Monday 03:00 UTC: Friday's session
    ])
    by = {d["url"]: d for d in docs}
    assert by["http://a/overnight"]["publication_date"] == "2022-05-11"      # the date is not rewritten
    assert by["http://a/overnight"]["session_date"] == "2022-05-10"
    assert by["http://a/overnight"]["trading_days_before_anomaly"] == 1
    assert by["http://b/close"]["trading_days_before_anomaly"] == 0
    assert by["http://c/monday"]["session_date"] == "2022-05-08"             # a Sunday ...
    assert by["http://c/monday"]["trading_days_before_anomaly"] == 3         # ... which is Friday's session
    record = pipeline._source_record(by["http://b/close"])
    assert record["seen_at"] == "20220511T204500Z" and record["market_report_score"] >= 6


def test_the_preview_shows_what_would_be_read_without_calling_a_model(capsys):
    docs, _ = _retrieve([
        {"url": "http://a/1", "title": "SOFTS - Coffee prices jump nearly 7% in New York after stocks drawn",
         "seendate": "20220511T204500Z"},
        {"url": "http://b/2", "title": "New coffee shop opens downtown", "seendate": "20220511T120000Z"},
    ])
    for d in docs:
        d["semantic_score"] = 0.9
    anomaly = {"date": "2022-05-11", "direction": "up", "pct_move": 7.0}
    with mock.patch("src.rag.explainer._call_model_ex") as call:
        shown = pipeline.preview_reading(anomaly, docs)
    call.assert_not_called()
    assert [d["url"] for d in shown] == ["http://a/1", "http://b/2"]
    out = capsys.readouterr().out
    assert "Reading preview for 2022-05-11" in out and "1 that look like market reports" in out
    assert out.index("SOFTS") < out.index("New coffee shop")
