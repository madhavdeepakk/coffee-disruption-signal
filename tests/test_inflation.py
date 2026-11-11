"""The inflation objective: reading the downloaded series, and the two
measurements (pass-through and the nowcast test) on made-up data whose
answer is known. Nothing here touches the network."""

import numpy as np
import pandas as pd
import pytest

from src.data import cpi
from src.modeling import inflation as inf


def _months(n, start="2000-01-01"):
    return pd.date_range(start, periods=n, freq="MS")


def _levels(changes_pct, index):
    """A price level whose month-on-month log change is the given series."""
    return pd.Series(100.0 * np.exp(np.cumsum(np.asarray(changes_pct) / 100.0)), index=index)


# --- reading the download ----------------------------------------------------

def test_a_missing_observation_is_left_out_not_read_as_zero():
    text = "observation_date,X\n2020-01-01,1.5\n2020-02-01,.\n2020-03-01,\n2020-04-01,2\n"
    assert cpi.parse_fred_csv(text, "X") == [("2020-01-01", 1.5), ("2020-04-01", 2.0)]


def test_an_error_page_is_not_mistaken_for_an_empty_series():
    with pytest.raises(ValueError):
        cpi.parse_fred_csv("<html><body>Not Found</body></html>", "X")
    with pytest.raises(ValueError):
        cpi.parse_fred_csv("", "X")


def test_the_older_header_name_is_read_too():
    assert cpi.parse_fred_csv("DATE,X\n2020-01-01,3\n", "X") == [("2020-01-01", 3.0)]


def test_a_figure_is_usable_only_after_it_is_published():
    # a price average: the first of the next month; consumer prices: mid next month
    assert cpi.known_from("2024-12-01", "commodity") == "2025-01-01"
    assert cpi.known_from("2024-12-01", "cpi") == "2025-01-15"
    assert cpi.known_from("2024-05-01", "cpi_yoy") == "2024-06-15"


def test_the_table_round_trips_and_every_series_has_a_name(tmp_path):
    rows = {sid: [("2020-01-01", 1.0), ("2020-02-01", 2.0)] for sid in cpi.SERIES}
    path = tmp_path / "t.csv"
    assert cpi.write_table(rows, path) == 2 * len(cpi.SERIES)
    table = cpi.load_table(path)
    assert set(table) == {name for name, _w, _k in cpi.SERIES.values()}
    assert list(table["us_cpi"].values) == [1.0, 2.0]
    assert len(set(cpi.NAME_TO_ID)) == len(cpi.SERIES)        # no two series share a name


# --- the numerical tools -----------------------------------------------------

def test_a_missing_month_is_not_bridged():
    index = _months(6).delete(3)
    change = inf.pct_change_log(pd.Series([100, 101, 102, 104, 105.0], index=index))
    assert len(change) == 6
    assert np.isnan(change.iloc[3]) and np.isnan(change.iloc[4])   # either side of the gap
    assert change.iloc[1] == pytest.approx(100 * np.log(1.01))


def test_holm_keeps_order_and_never_lowers_a_p_value():
    adjusted = inf.holm([0.01, None, 0.04, 0.03])
    assert adjusted[1] is None
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[3] == pytest.approx(0.06) and adjusted[2] == pytest.approx(0.06)
    assert all(a >= p for a, p in zip([adjusted[0], adjusted[2], adjusted[3]], [0.01, 0.04, 0.03]))


# --- pass-through ------------------------------------------------------------

def test_a_known_pass_through_is_recovered():
    rng = np.random.default_rng(3)
    n = 300
    index = _months(n)
    driver = pd.Series(rng.normal(0, 5, n), index=index)
    # 2% of a move arrives at once, 3% a month later, 1% six months later: 6% in all
    target = (0.2 + 0.02 * driver + 0.03 * driver.shift(1).fillna(0)
              + 0.01 * driver.shift(6).fillna(0) + rng.normal(0, 0.05, n))
    result = inf.pass_through(target, driver)
    assert not result["too_short"]
    assert result["sum_beta"] == pytest.approx(0.06, abs=0.01)
    assert result["early_beta"] == pytest.approx(0.05, abs=0.01)
    assert result["p"] < 0.001


def test_an_unrelated_commodity_shows_no_pass_through():
    rng = np.random.default_rng(11)
    n = 300
    index = _months(n)
    driver = pd.Series(rng.normal(0, 5, n), index=index)
    target = pd.Series(0.2 + rng.normal(0, 0.3, n), index=index)
    result = inf.pass_through(target, driver)
    assert abs(result["sum_beta"]) < 0.05 and result["p"] > 0.05


def test_too_few_months_is_said_not_estimated():
    index = _months(40)
    series = pd.Series(np.arange(40.0), index=index)
    assert inf.pass_through(series, series)["too_short"] is True
    assert inf.nowcast_test(series, series)["too_short"] is True


def test_seasonal_terms_absorb_a_seasonal_pattern():
    rng = np.random.default_rng(5)
    n = 300
    index = _months(n)
    driver = pd.Series(rng.normal(0, 5, n), index=index)
    season = pd.Series(np.where(index.month == 7, 2.0, 0.0), index=index)   # every July +2
    target = 0.2 + 0.03 * driver + season + rng.normal(0, 0.05, n)
    with_terms = inf.pass_through(target, driver, seasonal=True)
    assert with_terms["sum_beta"] == pytest.approx(0.03, abs=0.01)
    assert with_terms["r2"] > inf.pass_through(target, driver)["r2"]


# --- the nowcast test --------------------------------------------------------

def test_a_commodity_that_drives_inflation_lowers_the_error_out_of_sample():
    rng = np.random.default_rng(7)
    n = 320
    index = _months(n)
    driver = pd.Series(rng.normal(0, 5, n), index=index)
    target = 0.2 + 0.05 * driver + pd.Series(rng.normal(0, 0.05, n), index=index)
    result = inf.nowcast_test(target, driver)
    assert result["n_test"] == n - max(inf.NOWCAST_AR_LAGS) - inf.MIN_TRAIN_MONTHS
    assert result["rmse_model"] < result["rmse_baseline"]
    assert result["improvement_pct"] > 50 and result["p"] < 0.01


def test_an_unrelated_commodity_does_not_help_and_is_not_called_better():
    rng = np.random.default_rng(13)
    n = 320
    index = _months(n)
    driver = pd.Series(rng.normal(0, 5, n), index=index)
    target = pd.Series(0.2 + rng.normal(0, 0.3, n), index=index)
    result = inf.nowcast_test(target, driver)
    assert result["improvement_pct"] < 2 and result["p"] > 0.05


def test_the_nowcast_never_sees_the_month_it_estimates(monkeypatch):
    """Each fit must use only earlier months: the rows handed to the fit for
    test month t must number exactly t."""
    rng = np.random.default_rng(1)
    n = 140
    index = _months(n)
    driver = pd.Series(rng.normal(0, 5, n), index=index)
    target = pd.Series(rng.normal(0, 1, n), index=index)
    sizes = []
    real = inf.ols
    monkeypatch.setattr(inf, "ols", lambda y, X: (sizes.append(len(y)), real(y, X))[1])
    result = inf.nowcast_test(target, driver, min_train=100)
    rows = n - max(inf.NOWCAST_AR_LAGS)
    assert sizes[::2] == list(range(100, rows)) and result["n_test"] == rows - 100


# --- the whole analysis ------------------------------------------------------

def _made_up_world(n=300):
    rng = np.random.default_rng(21)
    index = _months(n)
    oil = rng.normal(0, 6, n)
    noise = lambda scale: rng.normal(0, scale, n)        # noqa: E731
    series = {
        "oil_wti": _levels(oil, index), "oil_brent": _levels(oil + noise(1), index),
        "wheat": _levels(noise(5), index), "coffee": _levels(noise(6), index),
        "food_index": _levels(noise(3), index), "energy_index": _levels(oil + noise(1), index),
        "all_commodities": _levels(0.5 * oil + noise(2), index),
        "inr_per_usd": _levels(noise(1), index),
        "us_cpi": _levels(0.2 + 0.05 * oil + noise(0.05), index),
        "us_core": _levels(0.2 + noise(0.1), index), "us_food": _levels(0.2 + noise(0.2), index),
        "us_energy": _levels(0.4 * oil + noise(0.5), index),
        "us_gasoline": _levels(0.6 * oil + noise(0.5), index),
        "us_coffee": _levels(0.2 + noise(0.8), index),
        "india_cpi": _levels(0.4 + noise(0.5), index),
    }
    series["g20_inflation"] = inf.yoy(series["us_cpi"]).dropna()
    return series


def test_every_listed_pair_is_reported_whatever_it_shows():
    result = inf.analyse(_made_up_world())
    listed = sum(len([d for d in drivers if d not in ("gold", "silver")])
                 for _c, _t, _s, drivers in inf.PAIRS) + len(inf.GLOBAL_DRIVERS)
    assert result["n_pairs"] == len(result["pairs"]) == listed
    by = {(r["target"], r["driver"]): r for r in result["pairs"]}
    # built in: oil drives US headline, energy and gasoline prices
    assert by[("us_cpi", "oil_wti")]["verdict"].startswith("pass-through measured")
    assert by[("us_gasoline", "oil_wti")]["pass_through"]["sum_beta"] == pytest.approx(0.6, abs=0.1)
    # not built in: nothing drives India's index, and it must be reported as nothing
    assert by[("india_cpi", "wheat")]["verdict"] == "no relationship that can be told from chance"
    assert by[("india_cpi", "wheat")]["currency"] == "INR"
    assert by[("india_cpi", "wheat")]["seasonal_terms"] is True
    assert all("p_holm" in r["pass_through"] for r in result["pairs"])


def test_the_report_names_every_pair_and_states_what_it_is_not():
    result = inf.analyse(_made_up_world())
    report = inf.generate_report(result)
    assert "None of this is a forecast" in report
    assert report.count("| United States: gasoline | crude oil (WTI)") == 2   # both tables
    assert "in rupees" in report and "Holm" in report
    for row in result["tracking"]:
        assert row["month"] in report


def test_tracking_gives_the_latest_month_and_its_rate():
    index = _months(40)
    series = {"us_cpi": pd.Series(100.0 * 1.002 ** np.arange(40), index=index)}
    row = inf.tracking(series)[0]
    assert row["month"] == index[-1].strftime("%Y-%m")
    assert row["yoy"] == pytest.approx(100 * (1.002 ** 12 - 1), abs=1e-6)
    assert row["three_month_annualised"] == pytest.approx(100 * (1.002 ** 12 - 1), abs=1e-6)


def test_the_unfinished_month_of_a_daily_series_is_left_out(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "RAW_DIR", tmp_path)
    days = pd.bdate_range("2024-01-01", "2024-03-12")
    pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "price": 10.0}).to_csv(
        tmp_path / "gold_prices.csv", index=False)
    averaged = inf.monthly_average_of_daily("gold_prices.csv")
    assert [d.strftime("%Y-%m") for d in averaged.index] == ["2024-01", "2024-02"]
    assert inf.monthly_average_of_daily("absent.csv") is None
