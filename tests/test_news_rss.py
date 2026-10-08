"""Second retrieval source (Google News RSS): parsing + lookahead safety.

Network is mocked - these test that the parser shapes items like GDELT
articles and, critically, enforces the same lookahead window (nothing dated
after the anomaly, nothing undated).
"""
from datetime import datetime, timezone
from unittest import mock

from src.rag import news_rss as rss

_FEED = b"""<?xml version="1.0"?><rss><channel>
<item><title>Coffee jumps on Brazil frost</title><link>http://ex.com/a</link>
<pubDate>Mon, 13 Jul 2021 09:00:00 GMT</pubDate><source>Reuters</source></item>
<item><title>Way out of window (too old)</title><link>http://ex.com/old</link>
<pubDate>Tue, 01 Jun 2021 09:00:00 GMT</pubDate></item>
<item><title>After the anomaly (lookahead violation)</title><link>http://ex.com/b</link>
<pubDate>Fri, 30 Jul 2021 09:00:00 GMT</pubDate></item>
<item><title>No date</title><link>http://ex.com/c</link></item>
</channel></rss>"""


class _FakeResp:
    def __init__(self, b):
        self.b = b
    def read(self):
        return self.b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_rss_keeps_only_in_window_dated_items():
    with mock.patch("urllib.request.urlopen", return_value=_FakeResp(_FEED)):
        arts = rss.fetch_rss_articles("coffee", "2021-07-19", 10)
    assert len(arts) == 1
    a = arts[0]
    assert a["url"] == "http://ex.com/a"
    assert a["seendate"].startswith("20210713")  # GDELT seendate shape
    assert "Reuters" in a["title"]
    assert a["_source"] == "google_news_rss"


def test_rss_network_failure_returns_empty_not_raise():
    with mock.patch("urllib.request.urlopen", side_effect=OSError("blocked")):
        arts = rss.fetch_rss_articles("coffee", "2026-01-01", 10)
    assert arts == []


def test_rss_unknown_commodity_returns_empty():
    assert rss.fetch_rss_articles("gold", "2026-01-01", 10) == []


def test_rss_malformed_xml_returns_empty():
    with mock.patch("urllib.request.urlopen", return_value=_FakeResp(b"not xml <<<")):
        assert rss.fetch_rss_articles("coffee", "2026-01-01", 10) == []


def test_aggregator_scopes_the_google_news_query_to_the_window():
    from datetime import datetime, timezone
    from src.rag import news_aggregator as agg
    start, end = agg._make_window("2021-07-19", 10)
    q = agg._date_scoped_query("coffee prices OR arabica", start, end)
    assert q == "(coffee prices OR arabica) after:2021-07-09 before:2021-07-20"
    # the day after the anomaly is outside the window, even at exactly midnight
    midnight_next_day = datetime(2021, 7, 20, 0, 0, tzinfo=timezone.utc)
    assert agg._in_window(midnight_next_day, start, end) is False
    assert agg._in_window(datetime(2021, 7, 19, 23, 59, tzinfo=timezone.utc), start, end) is True
