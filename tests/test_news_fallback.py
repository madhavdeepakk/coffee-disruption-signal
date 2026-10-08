"""Local news-fallback provider."""
import json
import tempfile
from pathlib import Path

from src.rag import news_fallback as nf


def _write(dir_, key, date, items):
    p = nf.fallback_path(dir_, key, date)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items), encoding="utf-8")
    return p


def test_loads_valid_and_drops_future_and_malformed():
    with tempfile.TemporaryDirectory() as dd:
        _write(dd, "coffee", "2025-08-15", [
            {"url": "http://a", "title": "on time", "seendate": "20250810T000000Z"},
            {"url": "http://b", "title": "future - drop", "seendate": "2025-08-20"},
            {"title": "no url - drop"},
        ])
        arts = nf.load_fallback_articles(dd, "coffee", "2025-08-15")
        assert [a["url"] for a in arts] == ["http://a"]
        assert arts[0]["_source"] == "local_fallback"


def test_missing_file_returns_empty():
    with tempfile.TemporaryDirectory() as dd:
        assert nf.load_fallback_articles(dd, "coffee", "2099-01-01") == []


def test_none_dir_returns_empty():
    assert nf.load_fallback_articles(None, "coffee", "2025-08-15") == []


def test_plain_date_seendate_accepted():
    with tempfile.TemporaryDirectory() as dd:
        _write(dd, "coffee", "2025-08-15", [
            {"url": "http://c", "title": "plain date", "seendate": "2025-08-15"}])
        arts = nf.load_fallback_articles(dd, "coffee", "2025-08-15")
        assert len(arts) == 1
