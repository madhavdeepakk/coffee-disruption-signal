"""GDELT response cache + Retry-After handling (mocked network)."""
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest
import requests

from src.rag import gdelt_client as gc
from src import pipeline


def test_cache_roundtrip_and_miss():
    with tempfile.TemporaryDirectory() as d:
        params = gc.build_query_params("coffee frost", 20, gc.DEFAULT_SORT,
                                       "20250805000000", "20250816000000")
        assert gc.cache_get(d, params) is None
        resp = {"articles": [{"url": "http://x", "title": "t"}]}
        gc.cache_put(d, params, resp)
        assert gc.cache_get(d, params) == resp
        other = gc.build_query_params("coffee frost", 20, gc.DEFAULT_SORT,
                                      "20240805000000", "20240816000000")
        assert gc.cache_get(d, other) is None
        assert gc.cache_get(None, params) is None
        gc.cache_put(None, params, resp)  # no-op, must not raise


def test_cache_corrupt_file_is_miss():
    with tempfile.TemporaryDirectory() as d:
        params = gc.build_query_params("q", 20, gc.DEFAULT_SORT, "1", "2")
        (Path(d) / (gc._cache_key(params) + ".json")).write_text("{ bad json", encoding="utf-8")
        assert gc.cache_get(d, params) is None


def _exc(value):
    resp = types.SimpleNamespace(headers={"Retry-After": value} if value is not None else {})
    e = requests.exceptions.HTTPError("429")
    e.response = resp
    return e


@pytest.mark.parametrize("value,expected", [
    ("7", 7), ("9999", gc.MAX_RETRY_AFTER_SECONDS), (None, None),
    ("Wed, 21 Oct 2025 07:28:00 GMT", None), ("-5", None),
])
def test_parse_retry_after(value, expected):
    assert gc._parse_retry_after(_exc(value)) == expected


def test_parse_retry_after_no_response():
    assert gc._parse_retry_after(requests.exceptions.HTTPError("x")) is None


def _mock_429(headers):
    r = mock.Mock()
    r.status_code = 429
    r.headers = headers
    err = requests.exceptions.HTTPError("429")
    err.response = r
    r.raise_for_status.side_effect = err
    return r


def test_fetch_gdelt_honors_retry_after():
    sleeps = []
    with mock.patch.object(gc.requests, "get", return_value=_mock_429({"Retry-After": "12"})), \
         mock.patch.object(gc.time, "sleep", side_effect=sleeps.append):
        with pytest.raises(RuntimeError):
            gc.fetch_gdelt("q", max_attempts=3, backoff_seconds=5, verbose=False)
    assert sleeps == [12, 12]


def test_fetch_gdelt_growing_backoff_without_header():
    sleeps = []
    with mock.patch.object(gc.requests, "get", return_value=_mock_429({})), \
         mock.patch.object(gc.time, "sleep", side_effect=sleeps.append):
        with pytest.raises(RuntimeError):
            gc.fetch_gdelt("q", max_attempts=3, backoff_seconds=5, verbose=False)
    assert sleeps == [5, 10]


def _fake_fetch(query, **kw):
    return {"articles": [{"url": f"http://{query.replace(' ', '')}/a",
                          "title": f"{query} headline", "seendate": "20250810T120000Z"}]}


def test_retrieve_evidence_cold_then_warm_cache():
    queries = ["coffee Brazil frost", "coffee Brazil drought", "coffee Minas Gerais"]
    with tempfile.TemporaryDirectory() as d:
        calls = []
        with mock.patch.object(pipeline, "fetch_gdelt", side_effect=lambda q, **k: (calls.append(q) or _fake_fetch(q))), \
             mock.patch.object(pipeline, "fetch_article_text", return_value=("body", None)), \
             mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
             mock.patch.object(pipeline.time, "sleep"):
            cold = pipeline.retrieve_evidence("2025-08-15", queries, "ref", cache_dir=d, fallback_dir=None)
        assert len(calls) == 3 and len(cold) == 3

        calls2 = []
        sleeps = []
        with mock.patch.object(pipeline, "fetch_gdelt", side_effect=lambda q, **k: (calls2.append(q) or _fake_fetch(q))), \
             mock.patch.object(pipeline, "fetch_article_text", return_value=("body", None)), \
             mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
             mock.patch.object(pipeline.time, "sleep", side_effect=sleeps.append):
            warm = pipeline.retrieve_evidence("2025-08-15", queries, "ref", cache_dir=d, fallback_dir=None)
        assert calls2 == [] and len(warm) == 3 and sleeps == []
        assert sorted(x["document_id"] for x in warm) == sorted(x["document_id"] for x in cold)


def test_failed_queries_not_cached():
    queries = ["a", "b"]
    with tempfile.TemporaryDirectory() as d:
        def boom(q, **k):
            raise RuntimeError("fail")
        with mock.patch.object(pipeline, "fetch_gdelt", side_effect=boom), \
             mock.patch.object(pipeline, "fetch_article_text", return_value=("", None)), \
             mock.patch.object(pipeline, "SEMANTIC_SCORING_AVAILABLE", False), \
             mock.patch.object(pipeline.time, "sleep"):
            docs = pipeline.retrieve_evidence("2025-08-15", queries, "ref", cache_dir=d, fallback_dir=None)
        assert docs == []
        assert list(Path(d).glob("*.json")) == []
