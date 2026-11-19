"""The definition module exists to stop three specific overclaims.

Each test guards one of them: counting a build-up as ten anomalies, quoting a
normal-distribution false-alarm rate for a fat-tailed series, and comparing a
flagged day only against an ordinary day rather than against a large move the
detector declined to flag.
"""
import csv
import inspect

import numpy as np
import pandas as pd
import pytest

from scripts import anomaly_definition as ad


def _frame(prices, flags=None, kinds=None, start="2015-01-01"):
    n = len(prices)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": idx,
        "price": [float(p) for p in prices],
        "z_score": 0.0,
        "rolling_std": 0.01,
        "anomaly_flag": flags if flags is not None else [False] * n,
        "anomaly_type": kinds if kinds is not None else [""] * n,
        "contract_switch": [False] * n,
    })


def test_a_ten_day_build_up_counts_as_one_episode_not_ten():
    """The cumulative check tags every day inside a build-up. Counting days
    multiplies one disruption by the length of the window that found it."""
    flags = [False] * 20 + [True] * 10 + [False] * 20
    eps = ad.episodes(pd.Series(flags))
    assert len(eps) == 1
    assert eps[0] == (20, 29)


def test_two_runs_separated_by_a_quiet_week_are_two_episodes():
    flags = [False] * 10 + [True] * 3 + [False] * 10 + [True] * 3 + [False] * 10
    assert len(ad.episodes(pd.Series(flags))) == 2


def test_a_one_day_gap_does_not_split_an_episode():
    """A build-up that dips below the bar for a day is still one event."""
    flags = [False] * 5 + [True, True, False, True, True] + [False] * 5
    assert len(ad.episodes(pd.Series(flags))) == 1


def test_calibration_reports_the_measured_rate_beside_the_normal_one():
    """Quoting 4.55% for a fat-tailed series is the error this prevents."""
    rng = np.random.default_rng(0)
    # Student-t: fatter tails than normal, so more days beyond 2 sigma.
    z = pd.Series(rng.standard_t(4, size=20000))
    df = _frame([100.0] * 20000)
    df["z_score"] = z.to_numpy()
    out = ad.calibration(df)
    assert out["normal_at_2"] == pytest.approx(0.0455)
    assert out["measured_at_2"] > out["normal_at_2"]
    assert out["excess_kurtosis"] > 1          # signed, not folded
    assert out["z_for_0.01"] > out["z_for_0.05"]


def test_the_move_statistics_use_shock_days_not_every_flagged_day():
    """A trend-only day's own move is often near zero, so a minimum over all
    flagged days reads 0.00% and means nothing."""
    source = inspect.getsource(ad.definitions)
    assert "median_shock_move_pct" in source
    assert "move[daily]" in source


def test_the_regime_test_compares_against_a_large_unflagged_move_too():
    """Against an ordinary day almost any large move looks consequential. The
    demanding comparison is against a large move the detector did not flag."""
    source = inspect.getsource(ad.regime_test)
    assert "big_unflagged" in source
    assert "lift_over_large_move" in source


def test_the_regime_test_needs_no_labels_and_no_event_list():
    """Its whole value is that nothing in it was chosen by hand."""
    source = inspect.getsource(ad.regime_test)
    for forbidden in ("known_events", "labels", "control_period"):
        assert forbidden not in source


def test_the_sample_is_drawn_at_random_not_from_the_biggest_moves(tmp_path, monkeypatch):
    """Sampling the largest flagged moves would flatter any precision figure
    computed from the labels that come back."""
    source = inspect.getsource(ad.sample_for_labelling)
    assert "shuffle" in source
    assert "nlargest" not in source and "sort_values" not in source


def test_the_sample_carries_empty_columns_for_a_person_to_fill(tmp_path, monkeypatch):
    df = _frame([100 + i * 0.5 for i in range(300)],
                flags=[i % 20 == 0 for i in range(300)],
                kinds=["shock" if i % 20 == 0 else "" for i in range(300)])
    monkeypatch.setattr(ad, "load", lambda key: df)
    rows = ad.sample_for_labelling(["coffee"], 5, seed=7)
    assert rows
    assert rows[0]["verdict_real_or_noise"] == ""
    assert "source_url" in rows[0] and "labelled_by" in rows[0]


def test_the_report_refuses_to_state_a_precision():
    """No label exists for most flagged days. A precision figure here would be
    invented, and the absence has to be stated rather than left blank."""
    text = ad.report(
        [{"commodity": "coffee", "scorable_days": 2122, "flagged_days": 279,
          "shock_only": 101, "trend_only": 149, "both_checks": 29,
          "episodes": 102, "episodes_per_year": 11.6, "median_episode_days": 1.0,
          "median_daily_move_pct": 1.40, "materiality_bar_pct": 4.21,
          "median_shock_move_pct": 4.85, "smallest_shock_move_pct": 2.34,
          "daily_check_days": 130, "material_days": 143, "both_days": 103,
          "material_not_flagged": 38, "flagged_not_material": 174,
          "jaccard_daily_material": 0.61, "calibration": {}}],
        [{"commodity": "coffee", "events": 4, "flagged": 4, "daily": 4,
          "material": 4, "both": 4}],
        [{"commodity": "coffee", "window": 20, "after_a_flag": 0.02368,
          "after_any_day": 0.02132, "after_a_large_unflagged_move": 0.02184,
          "lift_over_baseline": 1.11, "lift_over_large_move": 1.08,
          "n_large_unflagged": 143}])
    assert "Precision" in text
    assert "no answer here" in text
    assert "one episode a month" in text
