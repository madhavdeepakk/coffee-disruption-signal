"""Price outlook: analog scenarios + large-move risk (synthetic data)."""
import numpy as np
import pandas as pd

from src.modeling import outlook as ol


def _synthetic_prices(n=700, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"date": pd.date_range("2019-01-01", periods=n), "price": prices})


def test_build_outlook_frame_price_only():
    feat, cols, weather_used = ol.build_outlook_frame(_synthetic_prices(), use_weather=False)
    assert set(ol.FEATURE_COLUMNS).issubset(feat.columns)
    assert weather_used is False
    assert cols == list(ol.FEATURE_COLUMNS)


def test_analog_scenarios_shape():
    feat, cols, _ = ol.build_outlook_frame(_synthetic_prices(), use_weather=False)
    last = int(np.where(~feat[cols].isna().any(axis=1).to_numpy())[0][-1])
    res = ol.analog_scenarios(feat, cols, last, horizon=5)
    assert "error" not in res
    assert res["n_analogs"] > 0
    # percentiles must be ordered
    assert res["min_pct"] <= res["p10_pct"] <= res["median_pct"] <= res["p90_pct"] <= res["max_pct"]
    assert 0.0 <= res["share_up"] <= 1.0


def test_large_move_backtest_metrics_valid():
    feat, cols, _ = ol.build_outlook_frame(_synthetic_prices(), use_weather=False)
    res = ol.large_move_backtest(feat, cols, horizon=5)
    assert "error" not in res
    for k in ("base_rate", "precision", "recall", "flag_rate"):
        assert 0.0 <= res[k] <= 1.0
    assert res["lift"] >= 0.0
    assert res["n_test"] > 0


def test_latest_risk_probability_in_range():
    feat, cols, _ = ol.build_outlook_frame(_synthetic_prices(), use_weather=False)
    res = ol.latest_large_move_risk(feat, cols, horizon=5)
    assert "error" not in res
    assert 0.0 <= res["prob_large_move"] <= 1.0
    assert isinstance(res["elevated"], (bool, np.bool_))


def test_evaluate_runs_end_to_end(monkeypatch, tmp_path):
    # point the price file at a synthetic CSV via a fake commodity config
    import types
    csv = tmp_path / "data" / "raw" / "coffee_prices.csv"
    csv.parent.mkdir(parents=True)
    _synthetic_prices().to_csv(csv, index=False)
    fake_cfg = types.SimpleNamespace(display_name="Fake", price_file="coffee_prices.csv")
    monkeypatch.setattr(ol, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ol, "WEATHER_PATH", tmp_path / "data" / "raw" / "nowx.csv")
    monkeypatch.setattr(ol, "get_commodity", lambda k: fake_cfg)
    res = ol.evaluate("coffee", use_weather=False)
    assert "error" not in res
    assert set(res["horizons"].keys()) == set(ol.HORIZONS)
