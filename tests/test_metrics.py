"""Evaluation metrics: intervals, outcome accounting, decision accuracy."""
import json

import pytest

from src.evaluation import metrics as m
from src.rag import outcome as oc


def _run(date, decision, tier=None, direction=None, gate="EXPLAIN", accepted=3,
         citations=None, sources=None, faith=None):
    run = {
        "anomaly": {"date": date, "direction": "up"},
        "gate_result": {"decision": gate, "reason": f"{accepted}_document(s)_above_threshold",
                        "accepted_document_count": accepted, "total_documents_considered": 20,
                        "accepted_document_ids": []},
        "documents_retrieved": 20,
        "direction_summary": direction or {"consistent": 3, "neutral": 0, "inconsistent": 0},
        "explanation_result": {"decision": decision, "confidence": "high",
                               "citations": citations or []},
        "faithfulness_report": faith,
        "sources": sources or [],
    }
    if tier:
        run["outcome"] = {"tier": tier}
    return run


def test_wilson_interval_is_not_degenerate_at_the_extremes():
    lo, hi = m.wilson_interval(33, 33)
    assert hi == 100.0 and 89 < lo < 90          # 33/33 is not "100% +/- 0"
    lo, hi = m.wilson_interval(8, 10)
    assert 48 < lo < 50 and 94 < hi < 95
    assert m.wilson_interval(0, 0) == (None, None)


def test_every_run_is_counted_in_exactly_one_tier():
    runs = [
        _run("2021-07-20", "EXPLAINED", tier=oc.EXPLAINED_STRONG),
        _run("2021-07-30", "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_CONFLICTING),
        _run("2024-09-23", "API_ERROR", tier=oc.SYSTEM_ERROR),
        _run("2025-08-15", "EXPLAINED"),              # saved before the outcome block existed
        _run("2026-09-11", "INSUFFICIENT_EVIDENCE"),  # likewise
    ]
    out = m.outcome_section(runs)
    assert sum(out["tiers"].values()) == len(runs)
    assert out["tiers_backfilled"] == 2
    assert out["explained"]["k"] == 2 and out["refused"]["k"] == 2 and out["faults"]["k"] == 1
    assert out["explained_excluding_faults"]["n"] == 4


def test_decision_accuracy_against_labels():
    labels = {
        "2021-07-20": {"date": "2021-07-20", "expected_outcome": "EXPLAIN"},
        "2021-07-19": {"date": "2021-07-19", "expected_outcome": "REFUSE"},
        "2021-07-30": {"date": "2021-07-30", "expected_outcome": "REFUSE"},
        "2024-09-23": {"date": "2024-09-23", "expected_outcome": "EXPLAIN"},
        "2025-04-07": {"date": "2025-04-07", "expected_outcome": "EXPLAIN"},
    }
    against = {"consistent": 2, "neutral": 4, "inconsistent": 13}
    runs = [
        _run("2021-07-20", "EXPLAINED", tier=oc.EXPLAINED_STRONG),
        _run("2021-07-19", "EXPLAINED", tier=oc.EXPLAINED_TENTATIVE, direction=against),
        _run("2021-07-30", "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_CONFLICTING),
        _run("2024-09-23", "API_ERROR", tier=oc.SYSTEM_ERROR),
    ]
    acc = m.decision_accuracy_section(runs, labels)
    assert acc["n_labelled"] == 5 and acc["n_graded"] == 3
    assert acc["n_fault"] == 1 and acc["n_no_run"] == 1
    assert acc["as_run"]["overall"]["k"] == 2
    assert acc["as_run"]["refuse_labelled"] == m.rate(1, 2)
    assert [x["date"] for x in acc["misses"]] == ["2021-07-19"]
    # the guard would have withheld the 2-vs-13 explanation
    assert acc["with_direction_guard"]["refuse_labelled"]["k"] == 2
    assert acc["with_direction_guard"]["overall"]["k"] == 3


def test_guard_only_fires_on_an_explanation_with_conflicting_evidence():
    against = {"consistent": 0, "inconsistent": 5}
    assert m.guard_would_fire(_run("d", "EXPLAINED", direction=against))
    assert not m.guard_would_fire(_run("d", "INSUFFICIENT_EVIDENCE", direction=against))
    assert not m.guard_would_fire(_run("d", "EXPLAINED"))


def test_citation_scores_are_read_from_the_report():
    faith = {"n_citations": 2, "n_missing_document": 0, "n_weak_support": 0,
             "per_citation": [{"support_score": 0.86}, {"support_score": 0.90}]}
    cit = m.citation_section([_run("d", "EXPLAINED", faith=faith)])
    assert cit["total_citations"] == 2 and cit["resolve"]["pct"] == 100.0
    assert cit["avg_faithfulness_score"] == 0.88     # used to be reported as None
    assert cit["support_score_min"] == 0.86


def test_copies_and_late_documents_are_counted():
    sources = [
        {"title": "Brazil drought punishes coffee farms", "url": "https://a.example/1",
         "publication_date": "2024-09-20"},
        {"title": "Brazil drought punishes coffee farms - Yahoo", "url": "https://b.example/1",
         "publication_date": "2024-09-20"},
        {"title": "Next-day wrap", "url": "https://c.example/1", "publication_date": "2024-09-24"},
    ]
    run = _run("2024-09-23", "EXPLAINED", tier=oc.EXPLAINED_STRONG, sources=sources)
    ev = m.evidence_section([run], [run], index={})
    assert ev["accepted_documents"] == 3 and ev["distinct_stories"] == 2
    assert ev["duplicate_share"]["k"] == 1
    assert ev["published_after_anomaly"] == m.rate(1, 3)
    assert ev["avg_unique_domains"] == 3.0 and ev["avg_distinct_stories"] == 2.0


def test_report_renders_and_states_the_refusal_failures():
    labels = {"2021-07-19": {"date": "2021-07-19", "expected_outcome": "REFUSE"}}
    runs = [_run("2021-07-19", "EXPLAINED", tier=oc.EXPLAINED_TENTATIVE,
                 direction={"consistent": 2, "inconsistent": 13})]
    report = m.generate_report(m.compute_metrics(runs, labels))
    assert "1 date(s) labelled REFUSE were explained" in report
    assert "| **Total** | **1** |" in report


def test_dashboard_keys_are_still_present():
    met = m.compute_metrics([_run("2021-07-20", "EXPLAINED", tier=oc.EXPLAINED_STRONG)], {})
    assert {"gate_explain_rate", "llm_explain_rate", "avg_documents_retrieved",
            "zero_retrieval_rate"} <= set(met["retrieval"])
    assert "citations_resolve_rate" in met["citation_accuracy"]
    assert "direction_consistent_rate" in met["direction_alignment"]
    assert {"avg_unique_domains", "avg_accepted_documents",
            "single_source_rate"} <= set(met["source_diversity"])
    assert met["outcome_distribution"] == {oc.EXPLAINED_STRONG: 1}


def test_runs_that_lost_news_queries_are_reported_separately():
    def with_meta(run, failures, attempts, cache_hits=0):
        run["retrieval_meta"] = {"network_failures": failures, "network_attempts": attempts,
                                 "cache_hits": cache_hits}
        return run

    runs = [
        with_meta(_run("2018-05-31", "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_WEAK_EVIDENCE), 6, 6),
        with_meta(_run("2018-07-02", "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_CONFLICTING), 4, 6),
        with_meta(_run("2018-05-21", "EXPLAINED", tier=oc.EXPLAINED_STRONG), 2, 4, cache_hits=2),
        with_meta(_run("2024-09-23", "EXPLAINED", tier=oc.EXPLAINED_STRONG), 0, 1, cache_hits=5),
        _run("2021-07-20", "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_WEAK_EVIDENCE),   # no meta saved
    ]
    h = m.retrieval_health_section(runs)
    assert h["runs_with_failed_queries"]["k"] == 3 and h["runs_with_failed_queries"]["n"] == 5
    assert h["runs_with_no_query_answered"] == 1
    assert h["refused_among_those"]["k"] == 2 and h["refused_among_those"]["n"] == 3
    assert h["refused_among_the_rest"]["k"] == 1 and h["refused_among_the_rest"]["n"] == 2
    assert h["dates"] == ["2018-05-21", "2018-05-31", "2018-07-02"]

    report = m.generate_report(m.compute_metrics(runs, labels={}))
    assert "at least one news-API query failed" in report and "--retry-incomplete" in report
    clean = m.generate_report(m.compute_metrics(runs[3:], labels={}))
    assert "news-API query failed" not in clean


def test_model_alone_and_both_guard_rules_are_scored_from_the_same_stored_runs():
    def withheld(date, direction):
        run = _run(date, "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_CONFLICTING, direction=direction)
        run["explanation_result"]["withheld_explanation"] = {"explanation": "text"}
        return run

    runs = [
        withheld("2021-05-05", {"consistent": 3, "inconsistent": 3}),     # EXPLAIN, tie
        withheld("2021-07-19", {"consistent": 2, "inconsistent": 10}),    # REFUSE, clear
        _run("2023-09-20", "EXPLAINED", tier=oc.EXPLAINED_TENTATIVE,      # REFUSE, run unguarded
             direction={"consistent": 0, "inconsistent": 5}),
        _run("2021-07-20", "EXPLAINED", tier=oc.EXPLAINED_STRONG),        # EXPLAIN
        _run("2024-12-02", "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_WEAK_EVIDENCE),   # REFUSE
    ]
    labels = {d: {"expected_outcome": e} for d, e in [
        ("2021-05-05", "EXPLAIN"), ("2021-07-19", "REFUSE"), ("2023-09-20", "REFUSE"),
        ("2021-07-20", "EXPLAIN"), ("2024-12-02", "REFUSE")]}

    assert m.model_verdict(runs[0]) == m.EXPLAINED and m.verdict(runs[0]) == m.REFUSED
    assert m.verdict_with_guard(runs[0]) == m.EXPLAINED            # a tie no longer withholds
    assert m.verdict_with_guard(runs[0], "ties") == m.REFUSED
    assert m.verdict_with_guard(runs[2]) == m.REFUSED              # re-scored though run without it

    acc = m.decision_accuracy_section(runs, labels)
    assert acc["as_run"]["overall"]["k"] == 3                      # misses 05-05 and 09-20
    assert acc["model_alone"]["refuse_labelled"]["k"] == 1         # only the model's own refusal
    assert acc["model_alone"]["explain_labelled"]["k"] == 2
    assert acc["with_direction_guard"]["overall"]["k"] == 5
    assert acc["with_direction_guard_ties"]["explain_labelled"]["k"] == 1
    changed = {c["date"]: c for c in acc["guard_changed"]}
    assert set(changed) == {"2021-05-05", "2021-07-19", "2023-09-20"}
    assert changed["2021-05-05"]["tie_only"] and not changed["2021-05-05"]["correct"]
    assert changed["2021-07-19"]["correct"] and not changed["2021-07-19"]["tie_only"]

    report = m.generate_report(m.compute_metrics(runs, labels=labels))
    assert "Model alone" in report and "tie rule only" in report


def test_outcomes_are_split_by_answering_model_when_more_than_one_answered():
    def answered(run, model):
        run["explanation_result"]["model_used"] = model
        return run

    runs = [answered(_run("2018-04-27", "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_WEAK_EVIDENCE), "a"),
            answered(_run("2018-05-21", "INSUFFICIENT_EVIDENCE", tier=oc.REFUSED_WEAK_EVIDENCE), "a"),
            answered(_run("2021-07-20", "EXPLAINED", tier=oc.EXPLAINED_STRONG), "b"),
            answered(_run("2021-07-22", "EXPLAINED", tier=oc.EXPLAINED_STRONG), "b"),
            _run("2020-01-01", "EXPLAINED", tier=oc.EXPLAINED_STRONG)]      # model not recorded
    by = m.by_model_section(runs)
    assert by["a"]["explained_rate"]["k"] == 0 and by["a"]["explained_rate"]["n"] == 2
    assert by["b"]["explained_rate"]["k"] == 2 and set(by) == {"a", "b"}
    assert "not interchangeable" in m.generate_report(m.compute_metrics(runs, labels={}))
    assert "not interchangeable" not in m.generate_report(m.compute_metrics(runs[2:4], labels={}))


def test_guard_effect_is_reported_by_direction_of_the_move():
    def run(date, direction, decision, tally, withheld=False):
        r = _run(date, "INSUFFICIENT_EVIDENCE" if withheld else decision,
                 tier=(oc.REFUSED_CONFLICTING if withheld or decision != "EXPLAINED"
                       else oc.EXPLAINED_TENTATIVE), direction=tally)
        r["anomaly"]["direction"] = direction
        if withheld:
            r["explanation_result"]["withheld_explanation"] = {"explanation": "t"}
        return r

    against, with_ = {"consistent": 1, "inconsistent": 5}, {"consistent": 5, "inconsistent": 1}
    runs = [run("d1", "down", "EXPLAINED", against, withheld=True),
            run("d2", "down", "EXPLAINED", against, withheld=True),
            run("d3", "down", "EXPLAINED", with_),
            run("d4", "down", "INSUFFICIENT_EVIDENCE", against),
            run("u1", "up", "EXPLAINED", with_),
            run("u2", "up", "EXPLAINED", with_),
            run("u3", "up", "INSUFFICIENT_EVIDENCE", with_)]
    g = m.guard_effect_section(runs)
    down, up = g["by_direction"]["down"], g["by_direction"]["up"]
    assert (down["runs"], down["model_explained"], down["guard_withholds"]) == (4, 3, 2)
    assert (up["runs"], up["model_explained"], up["guard_withholds"]) == (3, 2, 0)
    assert down["explained_after_guard"]["k"] == 1 and up["explained_after_guard"]["k"] == 2
    assert g["withheld_overall"]["k"] == 2 and g["withheld_overall"]["n"] == 5
    report = m.generate_report(m.compute_metrics(runs, labels={}))
    assert "| down | 4 | 3 |" in report and "which way the" in report


def _blind_run(date, direction, readings, decision, tier):
    r = _run(date, decision, tier=tier)
    r["anomaly"]["direction"] = direction
    r["decision_mode"] = "blind"
    r["explanation_result"]["blind_evidence"] = {"readings": readings}
    return r


def _reading(age, direction, found=True, kind="report"):
    return {"document_id": f"x{age}{direction}", "trading_days_before": age, "direction": direction,
            "kind": kind, "cause": "a reason", "quote": "q", "quote_found": found, "read": True}


def test_blind_runs_are_reported_by_direction_and_rescored_at_other_settings():
    runs = [
        # explained from a same-day report; yesterday's coverage went the other way
        _blind_run("2024-10-07", "down",
                   [_reading(0, "down"), _reading(1, "up"), _reading(1, "up", found=False)],
                   "EXPLAINED", oc.EXPLAINED_TENTATIVE),
        # only a three-day-old reason: refused at 2 trading days, explained at 3
        _blind_run("2021-07-19", "down", [_reading(3, "down")],
                   "INSUFFICIENT_EVIDENCE", oc.REFUSED_WEAK_EVIDENCE),
        _blind_run("2021-07-20", "up", [_reading(0, "up"), _reading(0, "up")],
                   "EXPLAINED", oc.EXPLAINED_STRONG),
    ]
    labels = {"2024-10-07": {"expected_outcome": "EXPLAIN"},
              "2021-07-19": {"expected_outcome": "REFUSE"},
              "2021-07-20": {"expected_outcome": "EXPLAIN"}}
    b = m.blind_section(runs, labels)
    assert b["n_runs"] == 3
    assert b["reasons_with_quote_found"]["k"] == 5 and b["reasons_with_quote_found"]["n"] == 6
    assert b["by_direction"]["down"]["explained"] == 1 and b["by_direction"]["up"]["explained"] == 1
    by = {(row["rule"], row["fresh_trading_days"]): row for row in b["settings"]}
    assert by[("freshest", 2)]["explain_labelled"]["k"] == 2
    assert by[("freshest", 2)]["refuse_labelled"]["k"] == 1
    assert by[("freshest", 3)]["refuse_labelled"]["k"] == 0        # the stale reason now counts
    # two verified readings in the window: one down today, one up yesterday. Pooled, that
    # is a tie and still explained; it takes a majority against to refuse.
    assert by[("pooled", 2)]["explain_labelled"]["k"] == 2

    # the word-list guard has no say over a blind run
    assert not m.guard_would_fire({**runs[0], "direction_summary": {"consistent": 0,
                                                                   "inconsistent": 9}})
    met = m.compute_metrics(runs, labels=labels)
    assert met["guard_effect"]["withheld_overall"]["n"] == 0
    report = m.generate_report(met)
    assert "Runs decided by the blind-evidence rule" in report and "| pooled |" in report


def test_an_archived_set_gets_its_own_report(tmp_path, monkeypatch):
    import json
    archive = tmp_path / "archive" / "pipeline_v3"
    archive.mkdir(parents=True)
    run = _run("2021-07-20", "EXPLAINED", tier=oc.EXPLAINED_STRONG)
    (archive / "pipeline_output_coffee_2021-07-20.json").write_text(json.dumps(run))
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(m, "load_labels", lambda *a, **k: {})
    out = m.run_evaluation(verbose=False, outputs_dir=archive, name="v3")
    assert out["dates_evaluated"] == 1
    assert (tmp_path / "evaluation_report_v3.md").exists()
    assert not (tmp_path / "evaluation_report.md").exists()


def test_the_direction_test_is_computed_from_stored_readings():
    runs = [
        _blind_run("2021-07-20", "up", [_reading(0, "up"), _reading(0, "up")],
                   "EXPLAINED", oc.EXPLAINED_STRONG),
        # split evenly on the day: explained either way, which is the only way this can happen
        _blind_run("2021-07-21", "up", [_reading(0, "up"), _reading(0, "down")],
                   "EXPLAINED", oc.EXPLAINED_TENTATIVE),
        _blind_run("2021-07-19", "down", [_reading(0, "up")],
                   "INSUFFICIENT_EVIDENCE", oc.REFUSED_CONFLICTING),
    ]
    b = m.blind_section(runs, {})
    assert b["explained_either_direction"]["k"] == 1 and b["explained_either_direction"]["n"] == 2
    assert "Direction test on every explained date" in m.generate_report(
        m.compute_metrics(runs, labels={}))


def test_refused_dates_are_sorted_by_where_they_stopped():
    def refused(date, retrieved_recent, sources, readings=None, read=True):
        r = _blind_run(date, "up", readings or [], "INSUFFICIENT_EVIDENCE", oc.REFUSED_WEAK_EVIDENCE)
        r["documents_retrieved"] = 50
        r["documents_retrieved_recent"] = retrieved_recent
        r["sources"] = sources
        if not read:
            del r["explanation_result"]["blind_evidence"]
        return r

    old, fresh = {"trading_days_before_anomaly": 6}, {"trading_days_before_anomaly": 1}
    runs = [
        refused("2020-01-01", 0, [old], read=False),                       # search found nothing recent
        refused("2020-01-02", 7, [old], read=False),                       # filter dropped the recent ones
        refused("2020-01-03", 7, [fresh], [_reading(1, "up", found=False)]),   # no quote that checks out
        refused("2020-01-04", 7, [fresh], [_reading(0, "down")]),          # reason for the other direction
        _blind_run("2020-01-05", "up", [_reading(0, "up")], "EXPLAINED", oc.EXPLAINED_TENTATIVE),
    ]
    assert [m.stop_stage(r) for r in runs] == [
        "nothing_recent_found", "recent_found_none_relevant", "read_no_quoted_reason",
        "reason_points_other_way", "explained"]
    labels = {"2020-01-02": {"expected_outcome": "EXPLAIN"}}
    b = m.blind_section(runs, labels)
    assert all(b["refusal_stages"][key]["k"] == 1 for key, _ in m.STOP_STAGES)
    assert b["refused_but_should_explain"] == [{
        "date": "2020-01-02", "stage": "recent_found_none_relevant", "retrieved": 50,
        "retrieved_recent": 7, "accepted": 1}]
    report = m.generate_report(m.compute_metrics(runs, labels=labels))
    assert "Where the refused dates stopped" in report
    assert "| 2020-01-02 | Recent articles found, none relevant enough to read | 50 | 7 | 1 |" in report


def test_stricter_evidence_rules_are_scored_from_the_stored_readings():
    stale = {**_reading(0, "up"), "when": "earlier"}                 # reports last week's rally
    tiny = {**_reading(0, "up"), "when": "same_day", "move_pct": 0.4}
    near = {**_reading(0, "down"), "passed_gate": False}
    runs = [
        _blind_run("2022-01-03", "up", [stale], "EXPLAINED", oc.EXPLAINED_TENTATIVE),
        _blind_run("2022-01-04", "up", [tiny], "EXPLAINED", oc.EXPLAINED_TENTATIVE),
        _blind_run("2022-01-05", "down", [near], "EXPLAINED", oc.EXPLAINED_TENTATIVE),
        _blind_run("2022-01-06", "up", [_reading(0, "up")], "EXPLAINED", oc.EXPLAINED_TENTATIVE),
    ]
    for r in runs:
        r["anomaly"]["pct_move"] = 6.0
    runs[1]["anomaly"]["pct_move"] = -6.0                             # sign must not matter
    labels = {"2022-01-03": {"expected_outcome": "REFUSE"},
              "2022-01-04": {"expected_outcome": "REFUSE"},
              "2022-01-05": {"expected_outcome": "EXPLAIN"},
              "2022-01-06": {"expected_outcome": "EXPLAIN"}}
    b = m.blind_section(runs, labels)
    by = {tuple(row["variants"]): row for row in b["variants"]}
    assert b["settings"][0]["refuse_labelled"]["k"] == 0             # as run: both wrongly explained
    assert by[("current_moves",)]["refuse_labelled"]["k"] == 1
    assert by[("size_consistent",)]["refuse_labelled"]["k"] == 1
    assert by[("current_moves", "size_consistent")]["refuse_labelled"]["k"] == 2
    assert by[("current_moves", "size_consistent")]["explain_labelled"]["k"] == 2
    assert by[("gate_only",)]["explain_labelled"]["k"] == 1          # loses the near-the-bar date
    assert (b["near_gate_read"], b["near_gate_usable"]) == (1, 1)

    # a trend anomaly is not held to the size of one day's move
    runs[1]["anomaly"]["anomaly_type"] = "trend"
    assert m.blind_verdict(runs[1], "freshest", 2, ("size_consistent",)) == m.EXPLAINED

    report = m.generate_report(m.compute_metrics(runs, labels=labels))
    assert "| As the pipeline runs |" in report and "| Both of the last two |" in report
    assert "just under the relevance filter's bar: 1, of which 1" in report


def test_each_search_is_credited_with_the_documents_it_found():
    explained = _blind_run("2022-02-01", "up",
                           [{**_reading(0, "up"), "document_id": "a"},
                            {**_reading(0, "up"), "document_id": "b", "quote_found": False}],
                           "EXPLAINED", oc.EXPLAINED_TENTATIVE)
    explained["explanation_result"]["blind_evidence"]["supporting"] = ["a"]
    explained["sources"] = [{"document_id": "a", "found_by": "coffee futures"},
                            {"document_id": "b", "found_by": "coffee Brazil frost"},
                            {"document_id": "c", "found_by": "coffee Brazil frost"}]
    refused = _blind_run("2022-02-02", "down", [{**_reading(0, "up"), "document_id": "d"}],
                         "INSUFFICIENT_EVIDENCE", oc.REFUSED_CONFLICTING)
    refused["explanation_result"]["blind_evidence"]["supporting"] = []
    refused["sources"] = [{"document_id": "d", "found_by": "news feed"}]
    before = _blind_run("2022-02-03", "up", [_reading(0, "up")], "EXPLAINED", oc.EXPLAINED_TENTATIVE)
    before["sources"] = [{"document_id": "x0up"}]                     # made before searches were recorded

    rows = m.search_yield([explained, refused, before])
    assert rows == [
        {"search": "coffee futures", "candidates": 1, "usable": 1, "deciding": 1, "dates_decided": 1},
        {"search": "news feed", "candidates": 1, "usable": 1, "deciding": 0, "dates_decided": 0},
        {"search": "coffee Brazil frost", "candidates": 2, "usable": 0, "deciding": 0,
         "dates_decided": 0}]
    report = m.generate_report(m.compute_metrics([explained, refused, before], labels={}))
    assert "| coffee futures | 1 | 1 | 1 | 1 |" in report


def test_the_report_can_be_scored_against_another_answer_key(tmp_path, monkeypatch):
    runs = [_blind_run("2022-03-01", "down", [_reading(0, "down")], "EXPLAINED", oc.EXPLAINED_TENTATIVE),
            _blind_run("2022-03-02", "down", [_reading(0, "up")], "INSUFFICIENT_EVIDENCE",
                       oc.REFUSED_CONFLICTING),
            _blind_run("2022-03-03", "up", [_reading(0, "up")], "EXPLAINED", oc.EXPLAINED_TENTATIVE)]
    out = tmp_path / "outputs"; out.mkdir()
    for r in runs:
        (out / f"pipeline_output_coffee_{r['anomaly']['date']}.json").write_text(json.dumps(r))
    key = tmp_path / "other_key.csv"
    key.write_text("date,expected_outcome\n2022-03-01,EXPLAIN\n2022-03-02,EXPLAIN\n2022-03-03,EXCLUDE\n")
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path)
    metrics = m.run_evaluation(verbose=False, outputs_dir=out, name="other", labels_path=key)
    acc = metrics["decision_accuracy"]
    assert (acc["as_run"]["overall"]["k"], acc["as_run"]["overall"]["n"]) == (1, 2)   # EXCLUDE is not scored
    assert acc["as_run"]["refuse_labelled"]["n"] == 0
    assert (tmp_path / "evaluation_report_other.md").exists()
    with pytest.raises(SystemExit):
        m.run_evaluation(verbose=False, outputs_dir=out, name="x", labels_path=tmp_path / "missing.csv")


def test_a_run_is_rescored_under_the_rule_it_was_decided_by():
    yearly = {**_reading(0, "up"), "document_id": "y", "when": "earlier"}
    today = {**_reading(0, "down"), "document_id": "t", "when": "same_day"}
    v9 = _blind_run("2022-04-01", "down", [yearly, today], "EXPLAINED", oc.EXPLAINED_TENTATIVE)
    v9["explanation_result"]["blind_evidence"]["variants"] = ["current_moves"]
    v8 = _blind_run("2022-04-04", "down", [yearly, today], "EXPLAINED", oc.EXPLAINED_TENTATIVE)

    assert m.run_variants(v9) == ("current_moves",) and m.run_variants(v8) == ()
    assert m.blind_verdict(v9, "freshest", 2) == m.EXPLAINED                    # as it ran
    assert m.blind_verdict(v9, "freshest", 2, without=("current_moves",)) == m.EXPLAINED   # 1-1 tie
    b = m.blind_section([v9], {"2022-04-01": {"expected_outcome": "EXPLAIN"}})
    # had prices risen instead, the yearly article is not a report of the day: no explanation
    assert b["explained_either_direction"]["k"] == 0
    names = [tuple(row["variants"]) for row in b["variants"]]
    assert ("current_moves",) not in names                      # already how it ran
    assert ("without current_moves",) in names
    report = m.generate_report(m.compute_metrics([v9], labels={}))
    assert "the rule before version 9" in report
    # the same readings under the old rule would also explain a rise: the tie that version 9 removes
    b8 = m.blind_section([v8], {})
    assert b8["explained_either_direction"]["k"] == 1


def test_runs_on_days_that_are_not_real_moves_are_left_out(monkeypatch):
    runs = [_blind_run("2022-03-01", "down", [_reading(0, "down")], "EXPLAINED", oc.EXPLAINED_TENTATIVE),
            _blind_run("2022-03-02", "down", [_reading(0, "up")], "INSUFFICIENT_EVIDENCE",
                       oc.REFUSED_CONFLICTING),
            _blind_run("2022-03-03", "up", [_reading(0, "up")], "EXPLAINED", oc.EXPLAINED_TENTATIVE),
            _blind_run("2022-03-04", "up", [_reading(0, "up")], "EXPLAINED", oc.EXPLAINED_TENTATIVE)]
    labels = {"2022-03-01": {"expected_outcome": "EXPLAIN"}, "2022-03-02": {"expected_outcome": "EXPLAIN"},
              "2022-03-03": {"expected_outcome": "EXCLUDE"}}
    from src.config import commodities
    monkeypatch.setattr(commodities, "load_contract_switches", lambda key="coffee": {"2022-03-04": 0.01})

    metrics = m.compute_metrics(runs, labels=labels)
    assert metrics["dates_evaluated"] == 2
    assert metrics["provenance"]["runs_left_out_not_real_moves"] == ["2022-03-03", "2022-03-04"]
    acc = metrics["decision_accuracy"]
    assert (acc["always_explain"]["k"], acc["always_explain"]["n"]) == (2, 2)
    assert acc["n_should_refuse"] == 0
    report = m.generate_report(metrics)
    assert "2022-03-03, 2022-03-04" in report
    assert "No graded date in this answer key is one where refusing is right" in report


def test_the_default_answer_key_is_the_one_made_under_the_written_rule():
    labels = m.load_labels()
    assert m.LABELS_PATH.name == "labels_by_written_rule.csv" and m.ORIGINAL_LABELS_PATH.exists()
    wanted = [label["expected_outcome"] for label in labels.values()]
    assert (wanted.count("EXPLAIN"), wanted.count("REFUSE"), wanted.count("EXCLUDE")) == (16, 0, 2)
