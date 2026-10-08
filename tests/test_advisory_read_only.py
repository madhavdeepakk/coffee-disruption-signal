"""Read-only mode must not call a model or the network.

The public dashboard runs with READ_ONLY=1. Before this was enforced, that
setting skipped the news retrieval but still called the model with an empty
evidence block: every page view spent the project's model quota and produced
a briefing resting on no news.
"""
from unittest import mock

from src.advisory import daily_briefing, outlook_rag


def _no_model_allowed(*args, **kwargs):
    raise AssertionError("a model was called with live access off")


def test_the_briefing_calls_no_model_when_live_access_is_off(monkeypatch):
    from src.rag import explainer
    monkeypatch.setattr(explainer, "_call_model", _no_model_allowed)
    with mock.patch.object(daily_briefing, "_retrieve_briefing_news",
                           side_effect=AssertionError("news was fetched")):
        out = daily_briefing.generate_daily_briefing(allow_live=False)
    assert out.get("model_generated") is not True
    assert "allow_live=False" in out.get("retrieval_info", "")


def test_the_outlook_calls_no_model_when_live_access_is_off(monkeypatch):
    from src.rag import explainer
    monkeypatch.setattr(explainer, "_call_model", _no_model_allowed)
    with mock.patch.object(outlook_rag, "_retrieve_outlook_news",
                           side_effect=AssertionError("news was fetched")):
        out = outlook_rag.generate_outlook(allow_live=False)
    assert out["label"] == "Signals only"
    assert out["citations"] == [] and out["accepted_documents"] == []
    # the signal data still reaches the page
    assert "signals_collected" in out
