"""Offline tests for the explanation-vs-baseline evaluation."""
from scripts import evaluate_explanation_vs_baseline as ev


def test_baseline_never_refuses_when_docs_present():
    assert ev.baseline_decision({"documents_retrieved": 5}) == "EXPLAINED"
    assert ev.baseline_decision({"documents_retrieved": 0}) == "REFUSED"


def test_system_decision_marks_faults_none():
    assert ev.system_decision({"explanation_result": {"decision": "EXPLAINED"}}) == "EXPLAINED"
    assert ev.system_decision({"explanation_result": {"decision": "INSUFFICIENT_EVIDENCE"}}) == "REFUSED"
    assert ev.system_decision({"explanation_result": {"decision": "API_ERROR"}}) is None


def test_coverage_counts_cause_terms():
    terms = ev._content_terms("Brazil frost destroyed the crop")  # 'coffee'/'price' are stopped
    # explanation mentions frost + brazil -> partial coverage
    cov = ev._coverage("A severe frost hit Brazil overnight", terms)
    assert 0.0 < cov <= 1.0
    assert ev._coverage("totally unrelated text", terms) == 0.0


def test_direction_only_baseline_refuses_exactly_the_down_days():
    from src.evaluation import baseline_comparison as bc
    assert bc.direction_only({"anomaly": {"direction": "up"}}) == bc.EXPLAINED
    assert bc.direction_only({"anomaly": {"direction": "down"}}) == bc.REFUSED
    assert bc.direction_only({}) == bc.REFUSED
    labels = {"a": {"expected_outcome": "EXPLAIN"}, "b": {"expected_outcome": "REFUSE"},
              "c": {"expected_outcome": "EXPLAIN"}}
    verdicts = {"a": bc.EXPLAINED, "b": bc.REFUSED, "c": bc.REFUSED}   # c: an explainable down day
    out = bc.score(verdicts, labels)
    assert out["refusal_specificity"]["k"] == 1 and out["explain_recall"]["k"] == 1
    assert "direction_only" in bc.SYSTEM_NOTES
