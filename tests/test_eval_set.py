"""Evaluation-set sampling and run bookkeeping."""
import json
import sys

import pandas as pd

from scripts import run_eval_set as es


def _shocks():
    rows = []
    for year in (2021, 2022, 2023):
        for direction in ("up", "down"):
            for i in range(6):
                rows.append({"date": f"{year}-0{1 + (direction == 'down')}-{10 + i}",
                             "direction": direction, "z_score": 3.0 if direction == "up" else -3.0,
                             "pct_move": 5.0, "anomaly_type": "shock",
                             "stratum": f"{year}-{direction}"})
    return pd.DataFrame(rows)


def test_sample_is_balanced_reproducible_and_keeps_labelled_dates():
    shocks = _shocks()
    always = {"2021-01-10", "2023-02-15"}
    a = es.stratified_sample(shocks, 12, seed=7, always=always)
    b = es.stratified_sample(shocks, 12, seed=7, always=always)
    assert list(a["date"]) == list(b["date"]) and len(a) == 12
    assert always <= set(a["date"])
    assert set(a[a["labelled"]]["date"]) == always
    counts = a["stratum"].value_counts()
    assert counts.max() - counts.min() <= 1          # spread across year x direction
    assert list(a["date"]) != list(es.stratified_sample(shocks, 12, seed=8, always=always)["date"])


def test_output_state(tmp_path, monkeypatch):
    from src import pipeline
    monkeypatch.setattr(es, "RESULTS_DIR", tmp_path)

    def write(date, payload):
        (tmp_path / f"pipeline_output_coffee_{date}.json").write_text(json.dumps(payload))

    assert es.output_state("2021-07-20") == "missing"
    write("2021-07-20", {"outcome": {"tier": "EXPLAINED_STRONG", "is_fault": False}})
    assert es.output_state("2021-07-20") == "stale"                 # no version: predates v2
    write("2021-07-21", {"pipeline_version": pipeline.PIPELINE_VERSION,
                         "outcome": {"tier": "SYSTEM_ERROR", "is_fault": True}})
    assert es.output_state("2021-07-21") == "stale"                 # a fault is retried
    write("2021-07-22", {"pipeline_version": pipeline.PIPELINE_VERSION,
                         "outcome": {"tier": "REFUSED_WEAK_EVIDENCE", "is_fault": False}})
    assert es.output_state("2021-07-22") == "current"


def test_dates_that_lost_news_queries_are_counted_and_can_be_retried(tmp_path, monkeypatch):
    from src import pipeline
    monkeypatch.setattr(es, "RESULTS_DIR", tmp_path)

    def write(date, failures, attempts, version=pipeline.PIPELINE_VERSION):
        (tmp_path / f"pipeline_output_coffee_{date}.json").write_text(json.dumps({
            "pipeline_version": version,
            "outcome": {"tier": "REFUSED_WEAK_EVIDENCE", "is_fault": False},
            "retrieval_meta": {"network_failures": failures, "network_attempts": attempts}}))

    write("2018-05-31", 6, 6)       # every query failed
    write("2018-07-02", 4, 6)
    write("2024-09-23", 0, 1)       # clean
    write("2018-05-21", 2, 4, version="1")   # stale: redone anyway, not counted as incomplete
    dates = ["2018-05-21", "2018-05-31", "2018-07-02", "2024-09-23", "2020-01-01"]

    assert es.failed_queries("2018-07-02") == (4, 6)
    assert es.failed_queries("2020-01-01") == (0, 0)
    states = es.status(dates)
    assert es.incomplete_dates(dates, states) == ["2018-05-31", "2018-07-02"]

    ran = []
    monkeypatch.setattr(es, "run_one", lambda date, commodity: ran.append(date) or 0)
    monkeypatch.setattr(es.time, "sleep", lambda s: None)
    es.run_pending(dates)
    assert ran == ["2018-05-21", "2020-01-01"]
    # The retry first re-asks the failed queries, then re-runs only the dates
    # that now have more of them answered.
    filled = []
    monkeypatch.setattr(es, "fill_cache", lambda dates, commodity: filled.append(list(dates)))
    monkeypatch.setattr(es, "gained_queries",
                        lambda date, commodity: 2 if date == "2018-07-02" else 0)
    ran.clear()
    es.run_pending(dates, retry_incomplete=True)
    assert filled == [["2018-05-31", "2018-07-02"]]
    assert ran == ["2018-05-21", "2020-01-01", "2018-07-02"]      # 05-31 gained nothing


def test_a_run_decided_under_an_older_guard_rule_is_redone(tmp_path, monkeypatch):
    from src import pipeline
    monkeypatch.setattr(es, "RESULTS_DIR", tmp_path)

    def write(date, result, direction):
        (tmp_path / f"pipeline_output_coffee_{date}.json").write_text(json.dumps({
            "pipeline_version": pipeline.PIPELINE_VERSION,
            "outcome": {"tier": "x", "is_fault": False},
            "direction_summary": direction, "explanation_result": result}))

    withheld = {"decision": "INSUFFICIENT_EVIDENCE", "withheld_explanation": {"explanation": "t"}}
    # withheld on a 3-3 tie: the current rule would have let it through
    write("2021-05-05", withheld, {"consistent": 3, "inconsistent": 3})
    # withheld on 2 against 10: still withheld
    write("2021-07-19", withheld, {"consistent": 2, "inconsistent": 10})
    # explained with the evidence against it and no guard applied
    write("2023-09-20", {"decision": "EXPLAINED"}, {"consistent": 0, "inconsistent": 5})
    # explained on a tie: fine under the current rule
    write("2023-01-31", {"decision": "EXPLAINED"}, {"consistent": 1, "inconsistent": 1})
    # the model itself refused: no guard rule has a say
    write("2024-12-02", {"decision": "INSUFFICIENT_EVIDENCE"}, {"consistent": 0, "inconsistent": 9})

    assert es.output_state("2021-05-05") == "stale"
    assert es.output_state("2021-07-19") == "current"
    assert es.output_state("2023-09-20") == "stale"
    assert es.output_state("2023-01-31") == "current"
    assert es.output_state("2024-12-02") == "current"


def _rejection(status=429):
    """A failed fetch as gdelt_client raises it: RuntimeError caused by the HTTP error."""
    import requests
    response = requests.Response()
    response.status_code = status
    try:
        raise requests.HTTPError(f"{status} Client Error", response=response)
    except requests.HTTPError as http_error:
        try:
            raise RuntimeError("All 1 attempts failed") from http_error
        except RuntimeError as exc:
            return exc


def test_cache_fill_asks_for_the_same_requests_the_pipeline_sends(tmp_path, monkeypatch):
    """If these two ever build their request differently, the fill pass would
    cache answers under keys the pipeline never reads."""
    from unittest import mock
    from src import pipeline
    from src.config.commodities import get_commodity

    looked_up, sent = [], []
    with mock.patch.object(pipeline, "cache_get",
                           side_effect=lambda d, params, *a, **k: looked_up.append(params)), \
         mock.patch.object(pipeline, "fetch_gdelt",
                           side_effect=lambda q, **k: sent.append(q) or {"articles": []}), \
         mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
         mock.patch.object(pipeline.time, "sleep"):
        cfg = get_commodity("coffee")
        pipeline.retrieve_evidence("2022-05-11", cfg.gdelt_queries, "ref", cache_dir=tmp_path,
                                   fallback_dir=None, **pipeline.search_arguments(cfg))
    requests = es.gdelt_requests("2022-05-11")
    # three requests are sent for a date: the combined searches, and nothing else
    assert [r[0] for r in requests] == sent == [cfg.gdelt_combined_query,
                                                cfg.gdelt_combined_recent_query,
                                                cfg.gdelt_combined_day_query]
    assert [r[1] for r in requests] == looked_up[:3]
    # the separate queries are still looked up in the cache, after the combined ones
    assert len(looked_up) == 3 + len(cfg.gdelt_queries) + len(cfg.gdelt_recent_queries)
    # the full-window search covers ten days back, the recent one five
    starts = {query: start for query, _, start, _, _ in requests}
    sizes = {query: maxrecords for query, _, _, _, maxrecords in requests}
    assert starts[cfg.gdelt_combined_query] == "20220501000000"
    assert starts[cfg.gdelt_combined_recent_query] == "20220506000000"
    # and one covers the day of the move alone, up to the same end
    assert starts[cfg.gdelt_combined_day_query] == "20220511000000"
    assert {end for _, _, _, end, _ in requests} == {"20220512000000"}
    assert len({cfg.gdelt_combined_query, cfg.gdelt_combined_recent_query,
                cfg.gdelt_combined_day_query}) == 3
    assert set(sizes.values()) == {pipeline.COMBINED_MAX_ARTICLES}


def _two_requests_per_date(monkeypatch):
    """The tests of asking, waiting and giving up count requests; they are
    written for two per date, so the day-of-the-move search is switched off."""
    from src.config.commodities import get_commodity
    monkeypatch.setattr(get_commodity("coffee"), "gdelt_combined_day_query", "")


def test_cache_fill_keeps_asking_and_never_sooner_than_the_gap(tmp_path, monkeypatch):
    from src import pipeline
    from src.rag import gdelt_client

    monkeypatch.setattr(pipeline, "GDELT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(es, "RESULTS_DIR", tmp_path)
    waits = []
    monkeypatch.setattr(es.time, "sleep", waits.append)
    _two_requests_per_date(monkeypatch)
    full, recent = [r[0] for r in es.gdelt_requests("2019-04-03")]
    calls = []

    def fake_fetch(query, **kwargs):
        calls.append((query, kwargs["max_attempts"], kwargs["maxrecords"]))
        if query == full:                           # this one is always rejected
            raise _rejection()
        return {"articles": [{"url": "http://a/1", "title": query}]}

    monkeypatch.setattr(gdelt_client, "fetch_gdelt", fake_fetch)
    out = es.fill_cache(["2019-04-03"], rounds=5)

    assert all(attempts == 1 and size == 250 for _, attempts, size in calls)   # one attempt per ask
    assert out["answered"] == 1 and out["still_missing"] == 1
    # round 1 asks both; rounds 2 to 5 keep asking for the one still missing
    assert [q for q, *_ in calls] == [full, recent] + [full] * 4
    # a wait before every request but the first, never under GDELT's 5 seconds
    assert waits == [10, 6, 10, 10, 10]
    assert min(waits) > 5
    assert out["by_wait"] == {"first": [1, 0], "10s": [4, 1], "6s": [1, 0]}
    assert [q for q, *_ in es.uncached_requests("2019-04-03")] == [full]

    # a stored run that answered none of its requests would now see one more
    (tmp_path / "pipeline_output_coffee_2019-04-03.json").write_text(json.dumps({
        "retrieval_meta": {"cache_hits": 0, "network_attempts": 2, "network_failures": 2}}))
    assert es.gained_queries("2019-04-03") == 1
    assert es.gained_queries("2030-01-01") == 0                    # no stored run


def test_cache_fill_gives_up_after_a_run_of_rejections(tmp_path, monkeypatch, capsys):
    from src import pipeline
    from src.rag import gdelt_client

    monkeypatch.setattr(pipeline, "GDELT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(es.time, "sleep", lambda s: None)
    monkeypatch.setattr(es, "MAX_REJECTIONS_IN_A_ROW", 3)
    _two_requests_per_date(monkeypatch)
    calls = []

    def always_rejected(query, **kwargs):
        calls.append(query)
        raise _rejection()

    monkeypatch.setattr(gdelt_client, "fetch_gdelt", always_rejected)
    out = es.fill_cache(["2019-04-03", "2019-04-11"], rounds=10)
    assert len(calls) == 3 and out["answered"] == 0 and out["still_missing"] == 4
    assert "not answering now" in capsys.readouterr().out


def test_a_search_gdelt_will_not_run_is_reported_and_not_asked_again(tmp_path, monkeypatch, capsys):
    from src import pipeline
    from src.rag import gdelt_client

    monkeypatch.setattr(pipeline, "GDELT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(es.time, "sleep", lambda s: None)
    _two_requests_per_date(monkeypatch)
    full, recent = [r[0] for r in es.gdelt_requests("2019-04-03")]
    calls = []

    def fake_fetch(query, **kwargs):
        calls.append(query)
        if query == full:
            try:
                raise RuntimeError("Response was not valid JSON (first 200 chars): "
                                   "'Parentheses may only be used around OR'd statements.'")
            except RuntimeError as cause:
                raise RuntimeError("All 1 attempts failed") from cause
        return {"articles": []}

    monkeypatch.setattr(gdelt_client, "fetch_gdelt", fake_fetch)
    out = es.fill_cache(["2019-04-03", "2019-04-11"], rounds=4)
    assert calls.count(full) == 1                                   # asked once, across both dates
    assert calls.count(recent) == 2 and out["answered"] == 2
    assert "Parentheses" in out["refused"][full]
    assert "would not run this search" in capsys.readouterr().out
    assert es.why_failed(_rejection())[0] == "rate_limited"
    assert es.why_failed(RuntimeError("boom"))[0] == "error"


def test_earlier_runs_are_archived_before_being_replaced(tmp_path, monkeypatch):
    from src import pipeline
    monkeypatch.setattr(es, "RESULTS_DIR", tmp_path)
    old = tmp_path / "pipeline_output_coffee_2021-07-19.json"
    old.write_text(json.dumps({"pipeline_version": "3", "outcome": {"tier": "REFUSED_CONFLICTING"}}))
    current = tmp_path / "pipeline_output_coffee_2021-07-20.json"
    current.write_text(json.dumps({"pipeline_version": pipeline.PIPELINE_VERSION}))

    kept = es.archive_output("2021-07-19")
    assert kept == tmp_path / "archive" / "pipeline_v3" / old.name
    assert json.loads(kept.read_text())["outcome"]["tier"] == "REFUSED_CONFLICTING"
    assert es.archive_output("2021-07-20") is None                 # same version: nothing to keep
    assert es.archive_output("2030-01-01") is None                 # no output
    # a second replacement must not overwrite the first archived copy
    old.write_text(json.dumps({"pipeline_version": "3", "outcome": {"tier": "CHANGED"}}))
    assert es.archive_output("2021-07-19") is None
    assert json.loads(kept.read_text())["outcome"]["tier"] == "REFUSED_CONFLICTING"


def test_a_blind_decision_is_never_stale_on_account_of_the_guard(tmp_path, monkeypatch):
    from src import pipeline
    monkeypatch.setattr(es, "RESULTS_DIR", tmp_path)
    (tmp_path / "pipeline_output_coffee_2021-07-20.json").write_text(json.dumps({
        "pipeline_version": pipeline.PIPELINE_VERSION, "decision_mode": "blind",
        "outcome": {"tier": "EXPLAINED_TENTATIVE", "is_fault": False},
        "direction_summary": {"consistent": 1, "inconsistent": 4},
        "explanation_result": {"decision": "EXPLAINED"}}))
    assert es.output_state("2021-07-20") == "current"


def test_a_batch_stops_when_dates_keep_failing_the_same_way(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(es, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(es, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()
    for d in ("2020-01-01", "2020-01-02"):
        (tmp_path / "logs" / f"coffee_{d}.log").write_text("retrieving...\nNo LLM provider is configured\n\n")
    ran = []
    monkeypatch.setattr(es, "run_one", lambda date, commodity: ran.append(date) or 1)
    monkeypatch.setattr(es.time, "sleep", lambda s: None)
    es.run_pending(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    out = capsys.readouterr().out
    assert ran == ["2020-01-01", "2020-01-02"]                     # the other two were not wasted
    assert "DID NOT FINISH" in out and "No LLM provider is configured" in out
    assert "Stopping: 2 dates in a row did not finish" in out


def test_a_short_check_fills_the_cache_only_for_the_dates_it_will_run(monkeypatch):
    filled, ran = [], []
    monkeypatch.setattr(es, "read_manifest", lambda: ["2020-01-01", "2020-01-02", "2020-01-03"])
    monkeypatch.setattr(es, "labelled_dates", lambda: {"2020-01-03"})
    monkeypatch.setattr(es, "fill_cache", lambda dates, rounds=3: filled.append((dates, rounds)))
    monkeypatch.setattr(es, "run_pending", lambda dates, *a, **k: ran.append(dates))
    monkeypatch.setattr("src.rag.explainer.check_providers", lambda: ["groq"])
    monkeypatch.setattr(sys, "argv", ["run_eval_set", "--fill-cache", "--rounds", "2", "--run",
                                      "--max", "2", "--labelled-first"])
    es.main()
    assert filled == [(["2020-01-03", "2020-01-01"], 2)]          # labelled first, two dates only
    assert ran == [["2020-01-03", "2020-01-01", "2020-01-02"]]    # --run applies --max itself


def test_the_preview_prints_the_reading_list_and_runs_no_evaluation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(es, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(es, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(es, "read_manifest", lambda: ["2020-01-01", "2020-01-02", "2020-01-03"])
    monkeypatch.setattr(es, "labelled_dates", lambda: {"2020-01-03"})
    monkeypatch.setattr(es, "run_pending", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no run")))
    commands = []

    class _Done:
        def __init__(self, code): self.returncode = code

    def fake_run(cmd, stdout=None, **kwargs):
        commands.append(cmd)
        date = cmd[cmd.index("--date") + 1]
        if date == "2020-01-01":
            stdout.write("Traceback\nERROR: the relevance model failed\n")
            return _Done(1)
        stdout.write(f"retrieval noise\n--- Reading preview for {date} (no model call) ---\n   1. a report\n")
        return _Done(0)

    monkeypatch.setattr(es.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_eval_set", "--preview-reading", "--max", "2", "--labelled-first"])
    es.main()
    out = capsys.readouterr().out
    assert [c[c.index("--date") + 1] for c in commands] == ["2020-01-03", "2020-01-01"]
    assert all("--preview-reading" in c for c in commands)
    assert "Reading preview for 2020-01-03" in out and "retrieval noise" not in out
    assert "did not finish (exit 1)" in out and "the relevance model failed" in out

    commands.clear()
    monkeypatch.setattr(sys, "argv", ["run_eval_set", "--preview-reading", "--dates", "2021-07-30, 2023-11-30"])
    es.main()
    assert [c[c.index("--date") + 1] for c in commands] == ["2021-07-30", "2023-11-30"]


def test_days_that_are_not_real_moves_are_dropped_everywhere(tmp_path, monkeypatch):
    manifest = tmp_path / "eval_dates.csv"
    manifest.write_text("date,direction\n2023-09-19,up\n2023-09-20,down\n2023-11-30,up\n")
    key = tmp_path / "key.csv"
    key.write_text("date,expected_outcome\n2023-09-20,EXCLUDE\n2023-11-30,EXPLAIN\n2024-01-05,REFUSE\n")
    monkeypatch.setattr(es, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(es, "LABELS_PATH", key)
    monkeypatch.setattr(es, "not_real_moves", lambda commodity="coffee": {"2023-09-19", "2023-09-20"})

    assert es.read_manifest() == ["2023-11-30"]                       # an old list still naming them
    assert es.labelled_dates() == {"2023-11-30", "2024-01-05"}        # EXCLUDE is not a label


def test_the_stored_date_list_holds_no_contract_switch_day():
    import csv
    from src.config.commodities import load_contract_switches
    with open(es.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        dates = [row["date"] for row in csv.DictReader(f)]
    assert not set(dates) & set(load_contract_switches("coffee"))
    assert es.labelled_dates() <= set(dates) and len(es.labelled_dates()) == 16
