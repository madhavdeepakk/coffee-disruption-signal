"""The parts that let the pipeline run on a commodity other than coffee:
which headlines count as that commodity's market report, the detector's
score against chance, the harness keeping commodities apart, and the
evaluation that needs no answer key."""

import json

import pandas as pd
import pytest

from scripts import evaluate_detector as ed
from scripts import run_commodity_eval as rce
from src.config.commodities import COMMODITIES, get_commodity
from src.evaluation import robustness as rb
from src.rag import market_report as mr


@pytest.fixture(autouse=True)
def _coffee_afterwards():
    yield
    mr.set_subject("coffee")


# --- which headlines are a commodity's market report -------------------------

def test_coffee_reads_exactly_what_it_read_before():
    assert mr.SUBJECTS["coffee"].name is mr.COFFEE
    assert mr.SUBJECTS["coffee"].exchange is None and mr.SUBJECTS["coffee"].others is None
    mr.set_subject("coffee")
    assert mr.market_report_score(
        {"title": "Coffee prices jump nearly 7% in New York after stocks drawn"}) == 7


@pytest.mark.parametrize("commodity, report, not_a_report", [
    ("gold", "PRECIOUS-Gold settles lower as Fed signals patience",
     "Best gold jewellery shops in Dubai"),
    ("silver", "Spot silver slides from 8-year peak as retail frenzy fizzles",
     "Silver screen legends remembered"),
    ("crude_oil", "Brent crude settles above $100 a barrel",
     "Petrol prices at the pump rise for consumers"),
    ("wheat", "Wheat futures surge in Chicago after Russia quits grain deal",
     "Bread prices rise in supermarkets"),
])
def test_each_commodity_recognises_its_own_market_report(commodity, report, not_a_report):
    mr.set_subject(commodity)
    assert mr.market_report_score({"title": report}) >= mr.REPORT_SCORE
    assert mr.market_report_score({"title": not_a_report}) < mr.REPORT_SCORE


def test_a_headline_scores_the_same_whichever_way_prices_moved():
    mr.set_subject("gold")
    up = mr.market_report_score({"title": "Gold jumps 5% as dollar slides"})
    down = mr.market_report_score({"title": "Gold plunges 5% as dollar jumps"})
    assert up == down


def test_the_key_with_a_space_and_an_unknown_commodity(capsys):
    assert mr.set_subject("crude oil") is mr.SUBJECTS["crude_oil"]
    assert mr.set_subject("cocoa") is mr.SUBJECTS["coffee"]       # said out loud, not silent
    assert "no word lists for 'cocoa'" in capsys.readouterr().out


def test_every_configured_commodity_has_word_lists_and_market_wording():
    for key, cfg in COMMODITIES.items():
        assert key in mr.SUBJECTS, key
        assert cfg.watched_market and cfg.other_markets, key
        assert cfg.gdelt_recent_queries, key


def test_a_round_up_brings_up_the_passage_about_the_commodity():
    mr.set_subject("gold")
    opening = ("Stocks fell on Wall Street and crude oil slid as traders weighed the outlook. "
               * 9)
    passage = "Gold fell 5.4% to settle at $4,100 an ounce. Traders cited a stronger dollar."
    shown = mr.focus_text(opening + passage)
    assert passage.split(". ")[0] in shown and mr.SEPARATOR in shown
    # an article that is about gold from the start is left exactly as it is
    about_gold = "Gold fell 5.4% on Tuesday. " + "Analysts discussed the outlook at length. " * 30
    assert mr.focus_text(about_gold) == about_gold


# --- the detector against chance ---------------------------------------------

def test_chance_of_a_flag_in_a_window():
    assert ed.chance_of_a_flag([0, 0, 0, 0], 2) == 0.0
    assert ed.chance_of_a_flag([1, 1, 1, 1], 2) == 1.0
    # one flag in ten days: 3 of the 8 three-day windows hold it
    assert ed.chance_of_a_flag([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 3) == pytest.approx(3 / 8)
    assert ed.chance_of_a_flag([1, 0], 5) == 0.0                 # window longer than the series


def test_the_detector_report_gives_recall_next_to_chance():
    row = {"commodity": "x", "display_name": "X", "events_detected": 2, "n_events": 2,
           "event_recall": 1.0, "control_specificity": 1.0, "control_false_positives": 0,
           "control_days": 60, "control_window": "a..b", "total_flagged": 10, "total_days": 100,
           "flag_rate": 0.1, "flagged_in_event": 5, "flagged_in_control": 0,
           "flagged_elsewhere": 5, "expected_by_chance": 1.4, "event_flag_density": 0.5,
           "judged_flag_rate": 0.1,
           "event_rows": [{"label": "e", "window": "a..b", "trading_days": 5, "flagged_days": 3,
                           "detected": True, "types": ["shock"], "chance": 0.7}],
           "sensitivity": [{"threshold": 3.0, "flagged": 3, "judged": 100, "flag_rate": 0.03,
                            "shock_rate": 0.02, "detected": 2, "n_events": 2,
                            "expected_by_chance": 0.4, "control_fp": 0, "lift": 9.0}]}
    text = "\n".join(ed.format_commodity(row))
    assert "expected to 'detect' 1.4 of these 2 events" in text
    assert "5.0 times the base rate" in text and "| 70% |" in text
    assert "| 3.0 | 3/100 (3.0%) | 2.0% | 2/2 | 0.4 | 0 | 9.0x |" in text


# --- the harness keeps commodities apart -------------------------------------

def test_another_commodity_gets_its_own_report_names(monkeypatch):
    assert rb._named("robustness_report.md") == "robustness_report.md"
    monkeypatch.setattr(rb, "COMMODITY", "gold")
    assert rb._named("robustness_report.md") == "robustness_report_gold.md"
    assert rb._named("robustness.json") == "robustness_gold.json"


def test_a_gold_run_is_scored_without_coffees_answer_key(tmp_path, monkeypatch):
    from src import pipeline
    days = pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": days, "price": 100.0, "z_score": 0.1,
                       "cumulative_z_score": 0.1, "anomaly_flag": False,
                       "anomaly_type": None})
    df.loc[5, ["z_score", "anomaly_flag", "anomaly_type"]] = [4.0, True, "shock"]
    monkeypatch.setattr(rb, "COMMODITY", "gold")
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rb, "TRIAL_CACHE", tmp_path / "trials")
    monkeypatch.setattr(rb, "EVIDENCE_CACHE", tmp_path / "evidence")
    monkeypatch.setattr(rb, "load_detections", lambda *a: df)
    monkeypatch.setattr(rb, "load_labels",
                        lambda: pytest.fail("coffee's answer key was read for gold"))
    monkeypatch.setattr(rb, "evidence_for", lambda date, retrieval="live", use_cache=True, **k: ([], {}))
    monkeypatch.setattr(rb, "news_complete", lambda date: True)
    seen = []

    def fake_stage(anomaly, documents, meta, **kwargs):
        seen.append(kwargs.get("commodity_name"))
        # Every field the trial record reads. The stand-in carries the real
        # pipeline's shape on purpose: robustness.py indexes these directly so
        # that a commodity path which drops one fails loudly here rather than
        # writing a trial row with a silent None in it.
        return {"outcome": {"tier": "REFUSED_WEAK_EVIDENCE", "is_fault": False},
                "explanation_result": {"decision": "INSUFFICIENT_EVIDENCE", "model_used": "m",
                                       "reason": "blind_evidence: none"},
                "price_description": "+4.0% day-over-day move (z-score 4.00)",
                "gate_result": {"decision": "EXPLAIN", "accepted_document_count": 0},
                "direction_summary": {}, "evidence_recency": {}}
    monkeypatch.setattr(pipeline, "explain_from_documents", fake_stage)

    out = rb.run(experiments=["placebo"], n_placebo=4, sleep_seconds=0, verbose=False)
    assert seen and set(seen) == {"gold"}
    assert out["config"]["commodity"] == "gold"
    assert (tmp_path / "robustness_report_gold.md").exists()
    assert not (tmp_path / "robustness_report.md").exists()
    assert "Robustness Evaluation - gold" in (tmp_path / "robustness_report_gold.md").read_text(
        encoding="utf-8")


# --- the evaluation that needs no answer key ---------------------------------

def _detections(rows):
    return pd.DataFrame(rows, columns=["date", "z_score", "anomaly_flag", "anomaly_type"])


def test_documented_dates_are_the_largest_shock_days_inside_cited_events():
    cfg = get_commodity("gold")
    label = "2025 October reversal from the record"
    assert any(e[0] == label for e in cfg.known_events)
    df = _detections([
        ("2013-04-15", -10.0, True, "shock"),          # a cited event, but before the news search
        ("2025-10-20", 2.9, True, "shock"),
        ("2025-10-21", -5.4, True, "shock"),
        ("2025-10-22", -2.4, True, "trend"),           # flagged for a ten-day move: left out
        ("2025-10-23", 3.5, True, "shock"),
        ("2025-11-14", -6.0, True, "shock"),           # a shock, but inside no cited event
        ("2025-10-17", 9.0, False, ""),                # not flagged
    ])
    picked = rce.documented_dates("gold", per_event=2, detections=df)
    assert [(p["date"], p["event"]) for p in picked] == [
        ("2025-10-21", label), ("2025-10-23", label)]
    assert "2013 April gold crash" in rce.events_before_the_news_search("gold")


def test_the_report_counts_only_decided_dates_and_prints_what_was_said(tmp_path, monkeypatch):
    from src.pipeline import PIPELINE_VERSION
    monkeypatch.setattr(rce, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rce, "REPORT_MD", tmp_path / "m.md")
    monkeypatch.setattr(rce, "REPORT_JSON", tmp_path / "m.json")
    entries = [{"date": "2025-10-21", "event": "E", "z_score": -5.4},
               {"date": "2025-10-23", "event": "E", "z_score": 3.5},
               {"date": "2026-01-30", "event": "F", "z_score": -6.8},
               {"date": "2026-02-03", "event": "F", "z_score": 2.9}]
    monkeypatch.setattr(rce, "documented_dates", lambda c, per_event=2, detections=None: entries)

    def stored(date, decision, tier, fault=False, version=PIPELINE_VERSION):
        (tmp_path / f"pipeline_output_gold_{date}.json").write_text(json.dumps({
            "pipeline_version": version, "anomaly": {"pct_move": -5.7},
            "outcome": {"tier": tier, "is_fault": fault},
            "explanation_result": {"decision": decision, "model_used": "m",
                                   "explanation": "Gold fell after a record. | Quote",
                                   "reason": "blind_evidence: nothing recent"}}), encoding="utf-8")
    stored("2025-10-21", "EXPLAINED", "EXPLAINED_STRONG")
    stored("2025-10-23", "INSUFFICIENT_EVIDENCE", "REFUSED_WEAK_EVIDENCE")
    stored("2026-01-30", "INSUFFICIENT_EVIDENCE", "SYSTEM_ERROR", fault=True)
    # 2026-02-03 has no stored run
    (tmp_path / "robustness_gold.json").write_text(json.dumps({
        "config": {"placebo_waiting": ["2024-05-01"]},
        "trials": [
            {"experiment": "placebo", "prompt_version": "blind", "date": "q1", "verdict": "EXPLAINED",
             "explanation": "said something", "run": {"pipeline_version": "11", "seed": 7}},
            {"experiment": "placebo", "prompt_version": "blind", "date": "q2", "verdict": "REFUSED"},
            {"experiment": "placebo", "prompt_version": "blind", "date": "q3", "verdict": "REFUSED"},
            {"experiment": "placebo", "prompt_version": "blind", "date": "q4", "verdict": "FAULT"},
        ]}), encoding="utf-8")

    collected = rce.write_report(["gold"], verbose=False)
    gold = collected["gold"]
    assert (gold["explained"], gold["decided"], gold["not_decided"]) == (1, 2, 2)
    assert [r["state"] for r in gold["documented"]] == ["explained", "refused", "fault", "not run"]
    assert (gold["placebo"]["explained"], gold["placebo"]["n"], gold["placebo"]["faults"]) == (1, 3, 1)
    report = (tmp_path / "m.md").read_text(encoding="utf-8")
    assert "50.0% (1/2" in report and "2 more not decided" in report      # faults are not refusals
    assert "33.3% (1/3" in report and "1 fault(s) left out" in report
    assert "still waiting for news" in report
    assert "Gold fell after a record. / Quote" in report                  # the table is not broken
    assert "False explanation on q1: said something" in report
    assert "no answer key" in report.lower()


def test_an_older_versions_run_is_not_counted_as_a_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(rce, "RESULTS_DIR", tmp_path)
    (tmp_path / "pipeline_output_gold_2025-10-21.json").write_text(json.dumps({
        "pipeline_version": "3", "outcome": {"tier": "EXPLAINED_STRONG"},
        "explanation_result": {"decision": "EXPLAINED", "explanation": "x"}}), encoding="utf-8")
    row = rce.documented_outcome("gold", {"date": "2025-10-21", "event": "E", "z_score": 1.0})
    assert row["state"] == "older version"
