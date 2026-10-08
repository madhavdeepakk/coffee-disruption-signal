"""Robustness harness: date selection, input construction, trial bookkeeping
and the summary. The pipeline stage and the model are replaced throughout."""
import json
import types

import pandas as pd
import pytest

from src.evaluation import robustness as rb
from src.evaluation.metrics import EXPLAINED, REFUSED, FAULT, rate


def _detections():
    rows = []
    for i in range(60):
        date = (pd.Timestamp("2021-01-04") + pd.tseries.offsets.BDay(i)).strftime("%Y-%m-%d")
        rows.append({"date": date, "price": 100.0 + i, "z_score": 0.1, "cumulative_z_score": 0.2,
                     "anomaly_flag": False, "anomaly_type": None})
    df = pd.DataFrame(rows)
    for idx, z, kind in ((20, 3.0, "shock"), (40, -2.6, "shock")):
        df.loc[idx, ["z_score", "anomaly_flag", "anomaly_type"]] = [z, True, kind]
    df.loc[5, "z_score"] = 1.4            # not flagged, but not quiet either
    return df


def test_placebo_days_are_quiet_and_clear_of_anomalies():
    df = _detections()
    picked = rb.select_placebo_dates(df, 10, seed=1)
    assert len(picked) == 10
    flagged = set(df.index[df["anomaly_flag"]])
    for date in picked:
        i = df.index[df["date"] == date][0]
        assert abs(df.loc[i, "z_score"]) < rb.PLACEBO_MAX_ABS_Z
        assert all(abs(i - j) > rb.PLACEBO_CLEAR_TRADING_DAYS for j in flagged)
    assert df.loc[5, "date"] not in picked
    assert rb.select_placebo_dates(df, 10, seed=1) == picked     # reproducible


def test_fabricated_move_copies_a_real_anomaly_onto_a_quiet_day():
    df = _detections()
    quiet = df.loc[30, "date"]
    fake = rb.fabricated_move(df, quiet, seed=3)
    assert fake["date"] == quiet and fake["placebo"] is True
    assert fake["move_copied_from"] in (df.loc[20, "date"], df.loc[40, "date"])
    assert abs(fake["z_score"]) >= 2.6 and fake["true_z_score"] == 0.1
    assert fake["direction"] == ("up" if fake["z_score"] > 0 else "down")
    assert rb.fabricated_move(df, quiet, seed=3) == fake


def test_flipped_reverses_the_move_and_nothing_else():
    a = {"date": "2021-07-20", "price": 180.0, "z_score": 3.1, "pct_move": 6.7,
         "cumulative_pct_move": 11.0, "cumulative_z_score": 2.5, "direction": "up"}
    f = rb.flipped(a)
    assert (f["pct_move"], f["z_score"], f["direction"]) == (-6.7, -3.1, "down")
    assert f["cumulative_pct_move"] == -11.0 and f["price"] == 180.0
    assert a["direction"] == "up"                                  # original untouched


def test_transplant_donor_is_same_direction_and_far_away():
    anomalies = {
        "2021-07-20": {"direction": "up"}, "2021-07-22": {"direction": "up"},
        "2022-05-11": {"direction": "up"}, "2021-07-30": {"direction": "down"},
        "2024-12-02": {"direction": "down"},
    }
    pairs = rb.transplant_pairs(anomalies, seed=7)
    assert pairs["2021-07-20"] == "2022-05-11"        # 07-22 is the same direction but too close
    assert pairs["2021-07-30"] == "2024-12-02"
    assert all(anomalies[a]["direction"] == anomalies[b]["direction"] for a, b in pairs.items())


def test_redated_documents_keep_their_distance_from_the_move():
    docs = [{"document_id": "a", "publication_date": "2022-05-09"},
            {"document_id": "b", "publication_date": ""}]
    out = rb.redate_documents(docs, donor_date="2022-05-11", target_date="2021-07-20")
    assert out[0]["publication_date"] == "2021-07-18" and out[0]["days_before_anomaly"] == 2
    assert out[1]["days_before_anomaly"] is None
    assert docs[0]["publication_date"] == "2022-05-09"


def _stage_output(tier, decision, withheld=False):
    er = {"decision": decision, "explanation": "text" if decision == "EXPLAINED" else None,
          "model_used": "m", "provider": "groq", "prompt_version": "v2", "reason": None}
    if withheld:
        er["withheld_explanation"] = {"explanation": "withheld text"}
    return {"outcome": {"tier": tier}, "explanation_result": er, "price_description": "+6.7%",
            "gate_result": {"accepted_document_count": 3, "decision": "EXPLAIN"},
            "direction_summary": {}, "evidence_recency": {}}


def test_trial_records_the_models_decision_separately_from_the_guards(monkeypatch):
    from src import pipeline
    anomaly = {"date": "2021-07-19", "direction": "down"}

    monkeypatch.setattr(pipeline, "explain_from_documents",
                        lambda *a, **k: _stage_output("REFUSED_CONFLICTING",
                                                      "INSUFFICIENT_EVIDENCE", withheld=True))
    t = rb.pipeline_trial("flip", anomaly, [], {}, "v2")
    assert (t["verdict"], t["model_verdict"], t["guard_fired"]) == (REFUSED, EXPLAINED, True)
    assert t["explanation"] == "withheld text"

    monkeypatch.setattr(pipeline, "explain_from_documents",
                        lambda *a, **k: _stage_output("SYSTEM_ERROR", "API_ERROR"))
    assert rb.pipeline_trial("true", anomaly, [], {}, "v2")["verdict"] == FAULT


def test_faults_are_not_cached_but_results_are(tmp_path):
    calls = {"n": 0}

    def run_fault():
        calls["n"] += 1
        return {"experiment": "true", "date": "d", "verdict": FAULT}
    path = tmp_path / "t.json"
    rb.cached_trial(path, run_fault)
    rb.cached_trial(path, run_fault)
    assert calls["n"] == 2 and not path.exists()

    def run_ok():
        calls["n"] += 1
        return {"experiment": "true", "date": "d", "verdict": REFUSED}
    rb.cached_trial(path, run_ok)
    rb.cached_trial(path, run_ok)
    assert calls["n"] == 3 and json.loads(path.read_text())["verdict"] == REFUSED


def test_closed_book_refuses_when_the_model_says_unknown(monkeypatch, tmp_path):
    from src.rag import explainer
    monkeypatch.setattr(explainer, "LOG_PATH", tmp_path / "log.csv")
    anomaly = {"date": "2026-07-09", "z_score": 3.3, "pct_move": 10.1, "direction": "up"}

    def answer(text):
        return lambda *a, **k: {"response": types.SimpleNamespace(text=text, usage_metadata=None),
                                "model": "m", "provider": "groq", "latency_ms": 5, "packing": None}

    monkeypatch.setattr(explainer, "_call_model_ex", answer('{"unknown": true, "cause": ""}'))
    assert rb.closed_book_trial(anomaly)["verdict"] == REFUSED
    monkeypatch.setattr(explainer, "_call_model_ex",
                        answer('{"unknown": false, "cause": "Frost in Minas Gerais"}'))
    t = rb.closed_book_trial(anomaly, source="placebo")
    assert t["verdict"] == EXPLAINED and t["source"] == "placebo"

    def boom(*a, **k):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(explainer, "_call_model_ex", boom)
    assert rb.closed_book_trial(anomaly)["verdict"] == FAULT


def _t(experiment, date, verdict, model_verdict=None, version="v2", **extra):
    return {"experiment": experiment, "date": date, "prompt_version": version,
            "verdict": verdict, "model_verdict": model_verdict or verdict,
            "gate_decision": "EXPLAIN", "model_used": "m", **extra}


def test_summary_rates_and_paired_direction_check():
    trials = [
        _t("true", "d1", EXPLAINED), _t("true", "d2", EXPLAINED), _t("true", "d3", REFUSED),
        _t("true", "d4", FAULT),
        # d1: model explains the opposite move too, guard catches it. d2: model refuses.
        _t("flip", "d1", REFUSED, model_verdict=EXPLAINED), _t("flip", "d2", REFUSED),
        _t("flip", "d3", REFUSED),
        _t("placebo", "q1", EXPLAINED), _t("placebo", "q2", REFUSED),
        _t("placebo", "q3", REFUSED), _t("placebo", "q4", REFUSED),
        _t("closed_book", "d1", EXPLAINED, version="closed_book", source="real"),
        _t("closed_book", "d3", REFUSED, version="closed_book", source="real"),
        _t("closed_book", "q1", EXPLAINED, version="closed_book", source="placebo"),
    ]
    labels = {"d1": {"expected_outcome": "EXPLAIN"}, "d3": {"expected_outcome": "REFUSE"}}
    s = rb.summarize(trials, labels, cutoff="d2")
    v2 = s["by_prompt_version"]["v2"]

    assert s["n_faults"] == 1
    assert v2["true"]["with_guard"] == rate(2, 3)                 # the fault is excluded
    assert v2["flip"]["model_only"] == rate(1, 3)
    assert v2["flip"]["with_guard"] == rate(0, 3)
    assert v2["placebo"]["with_guard"] == rate(1, 4)
    both = v2["explained_both_directions"]
    assert both["model_only"]["k"] == 1 and both["model_only"]["n"] == 2
    assert both["model_only"]["dates"] == ["d1"] and both["with_guard"]["k"] == 0
    assert v2["labelled"]["with_guard"]["explain_recall"] == rate(1, 1)
    assert v2["labelled"]["with_guard"]["refusal_specificity"] == rate(1, 1)
    assert s["closed_book"]["real"] == rate(1, 2)
    assert s["closed_book"]["placebo"] == rate(1, 1)
    assert s["closed_book"]["real_before_cutoff"] == rate(1, 1)
    assert s["closed_book"]["real_after_cutoff"] == rate(0, 1)

    report = rb.generate_report(s, trials, labels,
                                {"n_real": 4, "n_placebo": 4, "retrieval": "cache", "seed": 7})
    assert "False-explanation rate" in report and "Direction sensitivity" in report


def test_run_end_to_end_with_everything_mocked(tmp_path, monkeypatch):
    from src import pipeline
    df = _detections()
    flagged_date = df.loc[20, "date"]
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rb, "TRIAL_CACHE", tmp_path / "trials")
    monkeypatch.setattr(rb, "EVIDENCE_CACHE", tmp_path / "evidence")
    monkeypatch.setattr(rb, "load_detections", lambda *a: df)
    monkeypatch.setattr(rb, "load_labels", lambda: {})
    monkeypatch.setattr(rb, "real_dates", lambda labels, limit=None, df=None: [flagged_date])
    monkeypatch.setattr(rb, "evidence_for", lambda date, retrieval="live", use_cache=True, **k: ([], {}))
    monkeypatch.setattr(rb, "news_complete", lambda date: True)
    monkeypatch.setattr(pipeline, "get_anomaly", lambda date, f: {
        "date": date, "price": 1.0, "z_score": 3.0, "pct_move": 6.0, "direction": "up",
        "anomaly_flag": True})

    seen = []

    def fake_stage(anomaly, documents, meta, **kwargs):
        seen.append((anomaly["date"], anomaly.get("direction"), anomaly.get("placebo", False)))
        explain = anomaly.get("direction") == "up" and not anomaly.get("placebo")
        return _stage_output("EXPLAINED_STRONG" if explain else "REFUSED_WEAK_EVIDENCE",
                             "EXPLAINED" if explain else "INSUFFICIENT_EVIDENCE")
    monkeypatch.setattr(pipeline, "explain_from_documents", fake_stage)

    out = rb.run(experiments=["true", "flip", "placebo"], n_placebo=3, sleep_seconds=0,
                 verbose=False)
    block = out["summary"]["by_prompt_version"][rb.BLIND]     # the default decision path
    assert block["true"]["with_guard"] == rate(1, 1)
    assert block["flip"]["with_guard"] == rate(0, 1)
    assert block["placebo"]["with_guard"]["n"] == 3
    assert (tmp_path / "robustness_report.md").exists()
    assert len(seen) == 5

    # a second run is served entirely from the trial cache
    seen.clear()
    rb.run(experiments=["true", "flip", "placebo"], n_placebo=3, sleep_seconds=0, verbose=False)
    assert seen == []
    assert rb.run(report_only=True, verbose=False)["summary"]["n_trials"] == 5

    # trials made under other settings are not pooled into this report
    with pytest.raises(SystemExit) as err:
        rb.run(report_only=True, providers=["groq"], verbose=False)
    assert "other settings" in str(err.value)
    only_flip = rb.run(report_only=True, experiments=["flip"], verbose=False)
    assert only_flip["summary"]["n_trials"] == 1


def test_real_dates_exclude_days_the_detector_does_not_flag(tmp_path, monkeypatch):
    df = _detections()
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    flagged, quiet = df.loc[20, "date"], df.loc[30, "date"]
    labels = {flagged: {}, quiet: {}}
    assert rb.real_dates(labels, df=df) == [flagged]
    assert rb.real_dates(labels) == sorted([flagged, quiet])     # no detections given


def test_a_crash_inside_one_trial_is_recorded_as_a_fault(monkeypatch):
    from src import pipeline

    def boom(*a, **k):
        raise AttributeError("'str' object has no attribute 'get'")
    monkeypatch.setattr(pipeline, "explain_from_documents", boom)
    t = rb.pipeline_trial("true", {"date": "2021-07-20", "direction": "up"}, [], {}, "v2")
    assert t["verdict"] == FAULT and "AttributeError" in t["reason"]


def test_failed_retrieval_is_not_remembered_as_evidence(tmp_path, monkeypatch):
    from src import pipeline
    from src.rag.outcome import RetrievalMeta
    monkeypatch.setattr(rb, "EVIDENCE_CACHE", tmp_path)
    calls = {"n": 0}

    def outage(*a, **k):
        calls["n"] += 1
        return [], RetrievalMeta(queries_attempted=6, network_attempts=6, network_failures=6)
    monkeypatch.setattr(pipeline, "retrieve_evidence", outage)
    rb.evidence_for("2021-07-20", "live")
    rb.evidence_for("2021-07-20", "live")
    assert calls["n"] == 2 and list(tmp_path.glob("*.json")) == []

    def answered(*a, **k):
        calls["n"] += 1
        return ([{"document_id": "a", "semantic_score": 0.9}],
                RetrievalMeta(network_attempts=6, network_failures=2, network_successes=4))
    monkeypatch.setattr(pipeline, "retrieve_evidence", answered)
    rb.evidence_for("2021-07-20", "live")
    docs, meta = rb.evidence_for("2021-07-20", "live")            # served from the cache
    assert calls["n"] == 3 and meta["network_failures"] == 2 and len(docs) == 1


def test_report_only_without_trials_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "TRIAL_CACHE", tmp_path / "none")
    monkeypatch.setattr(rb, "load_detections", _detections)
    with pytest.raises(SystemExit):
        rb.run(report_only=True, verbose=False)


def test_cached_trials_are_rescored_under_the_current_guard_rule():
    from src.evaluation.robustness import with_current_guard
    tie = {"experiment": "true", "date": "2023-01-31", "prompt_version": "v2",
           "verdict": REFUSED, "model_verdict": EXPLAINED, "guard_fired": True,
           "direction_summary": {"consistent": 1, "inconsistent": 1}}
    clear = {**tie, "date": "2021-07-19", "direction_summary": {"consistent": 2, "inconsistent": 8}}
    refused = {**tie, "model_verdict": REFUSED, "guard_fired": False}
    closed = {"experiment": "closed_book", "date": "d", "verdict": EXPLAINED,
              "model_verdict": EXPLAINED}

    out = with_current_guard(tie)
    assert (out["verdict"], out["guard_fired"]) == (EXPLAINED, False)
    assert tie["verdict"] == REFUSED                    # the cached record is not edited
    assert with_current_guard(clear)["verdict"] == REFUSED
    assert with_current_guard(refused) is refused
    assert with_current_guard(closed) is closed         # no evidence tally, no guard


def test_the_blind_rule_runs_as_its_own_version_on_the_same_evidence(monkeypatch):
    from src import pipeline
    calls = []

    def stage(anomaly, documents, meta, **kwargs):
        calls.append(kwargs)
        out = _stage_output("EXPLAINED_TENTATIVE", "EXPLAINED")
        out["explanation_result"]["prompt_version"] = (
            "blind-v1" if kwargs["decision_mode"] == "blind" else kwargs["prompt_version"])
        out["direction_summary"] = {"consistent": 0, "inconsistent": 3}   # would trip the old guard
        return out

    monkeypatch.setattr(pipeline, "explain_from_documents", stage)
    anomaly = {"date": "2021-07-20", "direction": "up"}
    blind = rb.pipeline_trial("true", anomaly, [], {}, rb.BLIND)
    legacy = rb.pipeline_trial("true", anomaly, [], {}, "v2")

    assert calls[0]["decision_mode"] == "blind" and calls[0]["prompt_version"] is None
    assert calls[0]["reading_cache_dir"] == rb.READING_CACHE
    assert calls[1]["decision_mode"] == "legacy" and calls[1]["prompt_version"] == "v2"
    assert blind["prompt_version"] == rb.BLIND and blind["decision_mode"] == "blind"
    # the word-list guard belongs to the earlier path only
    assert rb.with_current_guard(blind) is blind
    assert rb.with_current_guard({**legacy, "model_verdict": EXPLAINED})["verdict"] == REFUSED

    summary = rb.summarize([blind, {**legacy, "model_verdict": EXPLAINED}], labels={})
    assert set(summary["by_prompt_version"]) == {rb.BLIND, "v2"}
    report = rb.generate_report(summary, [blind, legacy], {}, {
        "n_real": 1, "n_placebo": 0, "retrieval": "cache", "seed": 7, "cutoff": None})
    assert "## Blind-evidence rule" in report and "## Prompt v2" in report


# --- the refusal test waits for complete news, and scores stricter rules ------

def _reading(doc_id, direction, pct=None, kind="report", age=0, passed_gate=True):
    return {"document_id": doc_id, "trading_days_before": age, "direction": direction,
            "kind": kind, "when": "same_day", "move_pct": pct, "cause": "a reason",
            "quote": "q", "quote_found": True, "read": True, "passed_gate": passed_gate}


def test_a_quiet_day_is_not_run_until_all_its_news_has_arrived(tmp_path, monkeypatch):
    from src import pipeline
    df = _detections()
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rb, "TRIAL_CACHE", tmp_path / "trials")
    monkeypatch.setattr(rb, "load_detections", lambda *a: df)
    monkeypatch.setattr(rb, "load_labels", lambda: {})
    monkeypatch.setattr(rb, "evidence_for", lambda date, retrieval="live", use_cache=True, **k: ([], {}))
    days = rb.select_placebo_dates(df, 3, 7)
    answered = set()                                        # nothing in the cache to begin with
    asked = []

    def fill(dates):
        asked.append(list(dates))
        answered.update(dates[:2])                          # the API answers for two of the three

    monkeypatch.setattr(rb, "news_complete", lambda date: date in answered)
    monkeypatch.setattr(rb, "fill_news", fill)
    ran = []
    monkeypatch.setattr(pipeline, "explain_from_documents",
                        lambda anomaly, *a, **k: ran.append(anomaly["date"]) or _stage_output(
                            "REFUSED_WEAK_EVIDENCE", "INSUFFICIENT_EVIDENCE"))

    out = rb.run(experiments=["placebo"], n_placebo=3, sleep_seconds=0, verbose=False)
    assert asked == [days] and ran == days[:2]              # the third is not run on partial news
    assert out["config"]["placebo_waiting"] == days[2:]
    assert out["summary"]["by_prompt_version"][rb.BLIND]["placebo"]["with_guard"]["n"] == 2
    report = (tmp_path / "robustness_report.md").read_text(encoding="utf-8")
    assert "Quiet days not run yet" in report and days[2] in report

    # asked not to fetch: nothing is fetched and the day still waits
    asked.clear()
    rb.run(experiments=["placebo"], n_placebo=3, sleep_seconds=0, verbose=False, fill=False)
    assert asked == []


def test_stricter_rules_are_scored_from_the_readings_a_trial_stored():
    def trial(experiment, date, direction, pct, readings, verdict):
        return {**_t(experiment, date, verdict, version=rb.BLIND), "direction": direction,
                "pct_move": pct, "readings": readings, "rule_variants": ["current_moves"]}

    trials = [
        # invented +6%: the day's report says prices edged up 0.4% - too small to be this move
        trial("placebo", "2021-02-01", "up", 6.0, [_reading("a", "up", pct=0.4)], EXPLAINED),
        # invented -5%: only a background article points down
        trial("placebo", "2021-02-02", "down", -5.0, [_reading("b", "down", kind="pressure")], EXPLAINED),
        # invented +5%: a report with no size stated - no stricter rule here catches it
        trial("placebo", "2021-02-03", "up", 5.0, [_reading("c", "up")], EXPLAINED),
        # a real +6.7% move reported as +6.6%: every rule keeps it
        trial("true", "2021-07-20", "up", 6.7, [_reading("d", "up", pct=6.6)], EXPLAINED),
    ]
    rows = {r["key"]: r for r in rb.rules_on_stored_readings(trials)}
    assert (rows["as_run"]["placebo"]["k"], rows["as_run"]["placebo"]["n"]) == (3, 3)
    assert rows["size"]["placebo"]["k"] == 2                 # the 0.4% report no longer counts
    assert rows["reports"]["placebo"]["k"] == 2              # the background article no longer counts
    # the version 11 rule, on readings that recorded a size only as a number: none
    # of the three describes a move of the claimed size
    assert rows["fit"]["placebo"]["k"] == 0
    assert "before_fit" not in rows                          # these trials already ran without it
    assert all(r["true"]["k"] == 1 for r in rows.values())   # and the real move is kept throughout
    assert rb.rule_verdict({"verdict": REFUSED}) == REFUSED  # no readings stored: verdict stands
    assert not rb.sizes_in_words_recorded(trials)

    summary = rb.summarize(trials, labels={})
    report = rb.generate_report(summary, trials, {}, {"n_real": 1, "n_placebo": 3,
                                                      "retrieval": "live", "seed": 7})
    assert "The same readings under other versions of the rule" in report
    assert "Invented moves explained (right answer: none)" in report
    assert "did not record how big" in report

    # trials run under version 11: the row that matters is the rule without its checks
    v11 = [{**t, "rule_variants": ["current_moves", *rb.FIT_THE_MOVE],
            "readings": [{**r, "move_size": "large", "cause_type": "event", "this_market": True}
                         for r in t["readings"]]} for t in trials]
    rows = {r["key"]: r for r in rb.rules_on_stored_readings(v11)}
    assert "fit" not in rows and rb.sizes_in_words_recorded(v11)
    assert rows["as_run"]["placebo"]["k"] == 1               # only the report whose words say "large"
    assert rows["before_fit"]["placebo"]["k"] == 3


def test_a_blind_trial_keeps_its_readings(monkeypatch):
    from src import pipeline
    out = _stage_output("EXPLAINED_TENTATIVE", "EXPLAINED")
    out["explanation_result"]["blind_evidence"] = {"readings": [_reading("a", "up", pct=4.0)],
                                                   "variants": ["current_moves"]}
    monkeypatch.setattr(pipeline, "explain_from_documents", lambda *a, **k: out)
    t = rb.pipeline_trial("placebo", {"date": "2021-02-01", "direction": "up", "pct_move": 6.0},
                          [], {}, rb.BLIND)
    assert t["readings"][0]["document_id"] == "a" and t["pct_move"] == 6.0
    assert t["rule_variants"] == ["current_moves"]
    legacy = rb.pipeline_trial("placebo", {"date": "2021-02-01", "direction": "up"}, [], {}, "v2")
    assert "readings" not in legacy


def test_the_report_of_an_earlier_run_is_kept_before_it_is_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    now = {"providers": "groq", "retrieval": "live", "seed": 11, "pipeline_version": "11"}
    assert rb.archive_previous_report(now) is None                     # nothing there yet
    earlier = {**now, "seed": 7, "pipeline_version": "10"}
    (tmp_path / "robustness.json").write_text(
        json.dumps({"trials": [{"experiment": "placebo", "run": earlier}]}), encoding="utf-8")
    (tmp_path / "robustness_report.md").write_text("the version 10 report", encoding="utf-8")

    assert rb.archive_previous_report(earlier) is None                 # same settings: just overwritten
    kept = rb.archive_previous_report(now)
    assert kept == tmp_path / "archive" / "robustness" / "groq__live__s7__p10"
    assert (kept / "robustness_report.md").read_text(encoding="utf-8") == "the version 10 report"
    assert (kept / "robustness.json").exists()
    assert rb.archive_previous_report(now) is None                     # kept once, never replaced
