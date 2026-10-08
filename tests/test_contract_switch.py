"""Days on which the price series switches futures contract."""
import numpy as np
import pandas as pd

from src.config import commodities
from src.modeling import anomaly_detector as ad


def _prices(n=140, jump_at=120, jump=1.08, seed=3):
    rng = np.random.default_rng(seed)
    steps = 1 + rng.normal(0, 0.01, n)
    steps[0] = 1.0
    steps[jump_at] = jump                      # a move far outside the recent range
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "price": 100 * np.cumprod(steps)})


def test_a_listed_day_is_measured_by_its_reported_move():
    prices = _prices()
    day = prices["date"][120]
    plain = ad.compute_anomalies(prices).set_index("date")
    fixed = ad.compute_anomalies(prices, reported_returns={day: 0.01}).set_index("date")

    assert plain.loc[day, "daily_anomaly_flag"] and not plain.loc[day, "contract_switch"]
    assert not fixed.loc[day, "anomaly_flag"] and fixed.loc[day, "contract_switch"]
    assert abs(fixed.loc[day, "z_score"]) < 2 < abs(plain.loc[day, "z_score"])
    assert fixed["contract_switch"].sum() == 1
    assert (fixed["price"] == plain["price"]).all()             # quoted prices are not rewritten
    before = prices["date"][:120]                               # earlier days saw no jump either way
    assert np.allclose(fixed.loc[before, "z_score"].dropna(), plain.loc[before, "z_score"].dropna())


def test_the_jump_no_longer_inflates_later_baselines():
    prices = _prices()
    day, later = prices["date"][120], prices["date"][125]      # inside the ten-day window
    plain = ad.compute_anomalies(prices).set_index("date")
    fixed = ad.compute_anomalies(prices, reported_returns={day: 0.01}).set_index("date")
    assert fixed.loc[later, "rolling_std"] < plain.loc[later, "rolling_std"]
    assert abs(fixed.loc[later, "cumulative_return"]) < abs(plain.loc[later, "cumulative_return"])


def test_adjusted_prices_change_only_the_listed_steps():
    prices = _prices()
    prices["date"] = pd.to_datetime(prices["date"])
    days = {prices["date"][120].strftime("%Y-%m-%d"): 0.01,
            prices["date"][121].strftime("%Y-%m-%d"): -0.02,     # two days running, as in 2023-09
            "1999-01-01": 0.5}                                   # not in the series: ignored
    raw = prices["price"].pct_change()
    adj = ad.adjusted_prices(prices, days).pct_change()
    assert abs(adj[120] - 0.01) < 1e-12 and abs(adj[121] + 0.02) < 1e-12
    others = [i for i in range(1, len(prices)) if i not in (120, 121)]
    assert np.allclose(adj[others], raw[others])
    assert ad.adjusted_prices(prices, days).iloc[-1] == prices["price"].iloc[-1]


def test_only_checked_exclusions_are_loaded(tmp_path, monkeypatch):
    (tmp_path / "contract_switch_dates.csv").write_text(
        "date,reported_pct,verdict\n"
        "2023-09-20,-1.7,EXCLUDE\n"
        "2020-03-19,4.1,CHECK\n"            # not settled: left alone
        "2021-03-22,,EXCLUDE\n")            # no reported move: cannot be used
    monkeypatch.setattr(commodities, "CONTRACT_SWITCH_DIR", tmp_path)
    assert commodities.load_contract_switches("coffee") == {"2023-09-20": -0.017}
    assert commodities.load_contract_switches("wheat") == {}


def test_the_coffee_list_matches_the_answer_key():
    import csv
    from src.evaluation import metrics as m
    switches = commodities.load_contract_switches("coffee")
    assert len(switches) == 6 and all(abs(v) < 0.05 for v in switches.values())
    with open(m.LABELS_PATH, newline="", encoding="utf-8") as f:
        excluded = {r["date"] for r in csv.DictReader(f) if r["expected_outcome"] == "EXCLUDE"}
    assert excluded and excluded <= set(switches)
