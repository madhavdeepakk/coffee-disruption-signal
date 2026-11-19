"""The strategy layer turns measurements into statements someone might act on.

Every test here guards the same property: a statement must not outrun the
measurement under it. The failure mode this module invites is a confident
recommendation built on a number with no control, no interval, or no run
behind it at all.
"""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import strategy as st


def _series(prices, flags=None, types=None):
    n = len(prices)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({
        "date": idx,
        "price": [float(p) for p in prices],
        "z_score": 0.0,
        "anomaly_flag": flags if flags is not None else [False] * n,
        "anomaly_type": types if types is not None else [""] * n,
        "contract_switch": [False] * n,
    })


def test_a_shock_that_never_reverts_is_counted_not_dropped():
    """Dropping them makes every shock look temporary, which is the single
    easiest way to turn a timing table into a false reassurance."""
    prices = [100.0] * 50 + [110.0] * 200          # +10%, never given back
    df = _series(prices)
    price = df["price"]
    move = price.pct_change() * 100
    stat = st._half_life(price, move, [50])
    assert stat["n"] == 1
    assert stat["never_retraced"] == 1
    assert stat["share_never"] == 1.0
    assert stat["median_days"] is None


def test_a_shock_that_reverts_is_measured_in_trading_days_off_the_series():
    prices = [100.0] * 50 + [110.0, 109.0, 108.0, 104.0] + [104.0] * 50
    df = _series(prices)
    price = df["price"]
    move = price.pct_change() * 100
    stat = st._half_life(price, move, [50])
    # Half of a +10% move from 100 is a return to 105; that is day 3 after.
    assert stat["median_days"] == 3
    assert stat["never_retraced"] == 0


def test_a_move_through_zero_is_never_measured():
    """A percentage change is undefined when a price is not positive at both
    ends: WTI settled at -$37.63 on 20 April 2020."""
    df = _series([50.0, 20.0, -37.63, 10.01, 12.0, 13.0])
    usable = st._usable(df)
    assert not usable.iloc[2]
    assert not usable.iloc[3]
    assert usable.iloc[1]


def test_a_contract_switch_day_is_never_measured():
    df = _series([100.0, 101.0, 140.0, 141.0])
    df.loc[2, "contract_switch"] = True
    assert not st._usable(df).iloc[2]


def test_the_timing_table_always_carries_its_control():
    """Commodity prices mean-revert after any large move. Without the
    unflagged comparison the table would credit the detector with a property
    of the series."""
    source = inspect.getsource(st.timing)
    assert "control" in source
    report = st.report([], [], {"available": False}, [])
    assert "control" in report.lower()
    assert "did NOT flag" in report


def test_no_p_value_is_computed_from_overlapping_windows():
    """Forward windows from nearby flagged days share most of their days, so
    a p-value from them would be wrong. None is produced."""
    source = inspect.getsource(st)
    for forbidden in ("p_value", "pvalue", "ttest", "scipy.stats"):
        assert forbidden not in source


def test_the_spread_is_reported_and_the_mean_is_not():
    """One April 2020 dominates a mean. The quartiles show the spread a
    median hides."""
    out = st._spread(pd.Series([-6.0, -1.0, 1.0, 3.0, 8.0, 2.0, 4.0, -2.0]))
    assert {"median", "q1", "q3", "share_up"} <= set(out)
    assert "mean" not in out
    # `.mean()` does appear, computing share_up; what must never appear is a
    # reported mean of the returns themselves.
    assert '"mean"' not in inspect.getsource(st._spread)


def test_a_group_too_small_to_read_reports_its_size_and_no_median():
    out = st._spread(pd.Series([1.0, 2.0, 3.0]))
    assert out == {"n": 3}


def test_a_layer_that_has_not_run_says_so_rather_than_showing_a_zero(tmp_path, monkeypatch):
    """A blank is a gap in what is known, not a negative finding."""
    monkeypatch.setattr(st, "RESULTS_DIR", tmp_path)
    out = st.exposure("coffee")
    assert out["available"] is False
    assert "macro_share" not in out
    text = st.report([], [{"commodity": "coffee", "available": False}],
                     {"available": False}, [])
    assert "not run" in text


def test_pass_through_verdicts_use_the_corrected_p_value(tmp_path, monkeypatch):
    """25 pairs were tested. A raw p-value under 0.05 among them is expected
    by chance, so only the Holm column may decide a verdict."""
    monkeypatch.setattr(st, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(st, "INFLATION_JSON", tmp_path / "inflation_report.json")
    (tmp_path / "inflation_report.json").write_text(json.dumps({
        "settings": {"significance": 0.05},
        "pairs": [
            {"country": "United States", "target_label": "gasoline",
             "driver_label": "crude oil (WTI)",
             "pass_through": {"sum_beta": 0.6, "p": 0.0001, "p_holm": 0.0001, "n": 308},
             "nowcast": {"p": 0.003, "p_holm": 0.068}},
            {"country": "India", "target_label": "all items",
             "driver_label": "wheat in rupees",
             "pass_through": {"sum_beta": 0.3, "p": 0.04, "p_holm": 0.9, "n": 290},
             "nowcast": {"p": 0.04, "p_holm": 1.0}},
        ]}), encoding="utf-8")
    out = st.pass_through()
    assert len(out["measured"]) == 1
    assert out["measured"][0]["target"] == "gasoline"
    # Striking on a raw p-value, not after correction: a null, and reported so.
    assert len(out["no_relationship"]) == 1
    # The nowcast row clears 0.05 raw and fails Holm, so it is not counted.
    assert out["nowcast_better"] == []


def test_the_report_refuses_to_forecast_and_says_it_is_not_advice():
    text = st.report([], [], {"available": False}, [])
    assert "nothing here is a forecast" in text.lower()
    assert "at chance" in text.lower()
    assert "this is not advice" in text.lower()
    assert "Not financial advice." in text


def test_the_report_states_what_each_section_does_not_support():
    """A recommendation without its boundary is the thing this module exists
    to avoid producing."""
    text = st.report([], [], {"available": False}, [])
    assert text.lower().count("does not support") >= 2
    assert "What nothing here supports" in text


def test_the_crude_oil_caveat_travels_with_the_number():
    """Crude's macro share is inflated because the broad factor contains
    energy. Quoted beside the others without that, it is misleading."""
    text = st.report([], [], {"available": False}, [])
    assert "weighted to energy" in text


def test_a_numeraire_layer_that_has_not_run_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "RESULTS_DIR", tmp_path)
    assert st.numeraire_mix("coffee") == {"commodity": "coffee", "available": False}


def test_days_with_no_currency_data_are_left_out_of_the_denominator(tmp_path, monkeypatch):
    """A day the translation could not be computed is not evidence that the
    move was commodity-specific. It is left out, not counted as a pass."""
    monkeypatch.setattr(st, "RESULTS_DIR", tmp_path)
    (tmp_path / "gold_numeraire.json").write_text(json.dumps(
        [{"reading": "specific to the commodity"}] * 8
        + [{"reading": "mostly the numeraire"}] * 2
        + [{"reading": "not enough currency data"}] * 5), encoding="utf-8")
    out = st.numeraire_mix("gold")
    assert out["judged"] == 10          # not 15
    assert out["numeraire_share"] == 0.2
    assert out["no_data"] == 5


def test_the_two_exposure_methods_are_printed_side_by_side_not_merged():
    """Collapsing a regression verdict and a translation verdict into one
    number would hide the days on which they disagree, which are the days
    worth looking at."""
    text = st.report([], [{"commodity": "gold", "available": True, "assessed": 397,
                           "macro": 167, "macro_share": 167 / 397,
                           "median_window_r2": 0.30}],
                     {"available": False}, [],
                     [{"commodity": "gold", "available": True, "judged": 488,
                       "specific": 377, "mixed": 67, "numeraire": 44,
                       "no_data": 3, "numeraire_share": 44 / 488}])
    assert "42.1%" in text and "9.0%" in text
    assert "the disagreement is the finding" in text
