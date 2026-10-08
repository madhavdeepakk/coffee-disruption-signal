"""Directional forecaster: model, features, labels, walk-forward eval, news signal."""
import numpy as np
import pandas as pd

from src.modeling import forecast as fc
from src.rag import direction as d


def test_logreg_learns_separable_pattern():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (100, 1)), rng.normal(2, 0.5, (100, 1))])
    y = np.array([0] * 100 + [1] * 100, dtype=float)
    model = fc.LogisticRegressionNP(n_iter=500).fit(X, y)
    p = model.predict_proba(np.array([[-2.0], [2.0]]))
    assert p[0] < 0.5 < p[1]


def test_make_labels_no_lookahead_semantics():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5),
                       "price": [10, 11, 9, 12, 8]})
    lab = fc.make_labels(df, horizon=1)
    # up, down, up, down, then NaN (no next day)
    assert list(lab[:4]) == [1.0, 0.0, 1.0, 0.0]
    assert pd.isna(lab.iloc[4])


def test_build_features_present_and_finite_after_warmup():
    rng = np.random.default_rng(1)
    prices = 100 + np.cumsum(rng.normal(0, 1, 400))
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=400), "price": prices})
    feat = fc.build_features(df)
    for col in fc.FEATURE_COLUMNS:
        assert col in feat.columns
    # after the 75-day warmup, features should be finite
    assert feat[fc.FEATURE_COLUMNS].iloc[100:].notna().all().all()


def test_walk_forward_eval_returns_valid_metrics():
    rng = np.random.default_rng(2)
    prices = 100 + np.cumsum(rng.normal(0, 1, 600))
    df = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=600), "price": prices})
    res = fc.walk_forward_eval(df, horizon=1)
    assert "error" not in res
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["baseline_accuracy"] <= 1.0
    assert res["n_test"] > 0


def test_predict_latest_shape():
    rng = np.random.default_rng(3)
    prices = 100 + np.cumsum(rng.normal(0, 1, 400))
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=400), "price": prices})
    out = fc.train_and_predict_latest(df, horizon=5)
    assert "error" not in out
    assert 0.0 <= out["proba_up"] <= 1.0
    assert out["direction"] in ("up", "down")


def test_news_lean_labels():
    up = d.news_lean([{"title": "Coffee surges on frost", "text": ""},
                      {"title": "Café dispara", "text": ""}])
    assert up["lean"].startswith("bullish")
    down = d.news_lean([{"title": "Coffee falls as rain returns", "text": ""},
                        {"title": "Preço do café cai", "text": ""}])
    assert down["lean"].startswith("bearish")
    none = d.news_lean([{"title": "A coffee festival opens", "text": ""}])
    assert none["lean"] == "no signal"


def test_news_direction_signal_offline_no_cache(tmp_path):
    # allow_live=False and an empty cache dir -> no network, unavailable.
    sig = fc.news_direction_signal("coffee", "2025-08-15", allow_live=False, cache_dir=tmp_path)
    assert sig["available"] is False
