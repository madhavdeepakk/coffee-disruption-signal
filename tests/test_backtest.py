"""Explanation-backtest scoring logic (no network, no files)."""
from scripts import explanation_backtest as bt


def _run(outcome_tier=None, model_decision=None, direction_summary=None, is_fault=False):
    run = {"anomaly": {"direction": "up"}, "explanation_result": {}}
    if outcome_tier is not None:
        run["outcome"] = {"tier": outcome_tier, "is_fault": is_fault}
    if model_decision is not None:
        run["explanation_result"] = {"decision": model_decision, "confidence": "high"}
    if direction_summary is not None:
        run["direction_summary"] = direction_summary
    return run


def test_explain_label_hit():
    label = {"date": "d", "expected_outcome": "EXPLAIN", "expected_direction": "up",
             "label_confidence": "high"}
    s = bt.score_row(label, _run(outcome_tier="EXPLAINED_STRONG"))
    assert s["correct"] is True


def test_explain_but_conflicted_evidence_is_not_a_hit():
    label = {"date": "d", "expected_outcome": "EXPLAIN", "expected_direction": "up",
             "label_confidence": "high"}
    s = bt.score_row(label, _run(outcome_tier="EXPLAINED_TENTATIVE",
                                 direction_summary={"consistent": 0, "inconsistent": 2}))
    assert s["evidence_dir"] == "conflicted"
    assert s["correct"] is False


def test_refuse_label_hit():
    label = {"date": "d", "expected_outcome": "REFUSE", "expected_direction": "down",
             "label_confidence": "high"}
    s = bt.score_row(label, _run(outcome_tier="REFUSED_CONFLICTING"))
    assert s["correct"] is True


def test_refuse_label_miss_when_system_explains():
    label = {"date": "d", "expected_outcome": "REFUSE", "expected_direction": "down",
             "label_confidence": "high"}
    s = bt.score_row(label, _run(outcome_tier="EXPLAINED_STRONG"))
    assert s["correct"] is False


def test_fault_is_excluded_not_a_miss():
    label = {"date": "d", "expected_outcome": "EXPLAIN", "expected_direction": "up",
             "label_confidence": "high"}
    s = bt.score_row(label, _run(outcome_tier="SYSTEM_ERROR", is_fault=True))
    assert s["correct"] is None
    assert s["status"] == "EXCLUDED_FAULT"


def test_missing_output_is_none():
    label = {"date": "d", "expected_outcome": "EXPLAIN", "expected_direction": "up",
             "label_confidence": "high"}
    s = bt.score_row(label, None)
    assert s["correct"] is None
    assert s["status"] == "NO_OUTPUT"


def test_legacy_output_without_outcome_block_still_scores():
    # A pre-refactor cached file: no 'outcome', only raw explanation_result.
    label = {"date": "d", "expected_outcome": "EXPLAIN", "expected_direction": "up",
             "label_confidence": "high"}
    s = bt.score_row(label, _run(model_decision="EXPLAINED"))
    assert s["verdict"] == "EXPLAINED"
    assert s["correct"] is True


def test_summarize_counts():
    scored = [
        {"expected_outcome": "EXPLAIN", "correct": True, "status": "scored"},
        {"expected_outcome": "EXPLAIN", "correct": False, "status": "scored"},
        {"expected_outcome": "REFUSE", "correct": True, "status": "scored"},
        {"expected_outcome": "EXPLAIN", "correct": None, "status": "NO_OUTPUT"},
    ]
    summ = bt.summarize(scored)
    assert summ["n_graded"] == 3
    assert summ["n_correct"] == 2
    assert summ["explain_correct"] == 1 and summ["n_explain"] == 2
    assert summ["refuse_correct"] == 1 and summ["n_refuse"] == 1
    assert summ["n_no_output"] == 1
