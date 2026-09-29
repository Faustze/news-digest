"""
Tests for news.fetch module (no network: downloads are mocked).
"""

from datetime import datetime, timezone

import httpx

from news import fetch
from news.fetch import fetch_rss_items, parse_entries

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
FEED_CFG = {"name": "Example", "url": "https://example.com/rss", "tags": ["AI"]}


def rss(*items: str) -> bytes:
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>'
        + "".join(items)
        + "</channel></rss>"
    ).encode()


def item(title: str, link: str, pub: str = "Tue, 29 Sep 2026 10:00:00 GMT") -> str:
    return (
        f"<item><title>{title}</title><link>{link}</link>"
        f"<pubDate>{pub}</pubDate><description>Body</description></item>"
    )


class TestParseEntries:
    def test_parses_fresh_item(self):
        items = parse_entries(FEED_CFG, rss(item("A", "https://e.com/a")), 24, NOW)
        assert len(items) == 1
        assert items[0]["title"] == "A"
        assert items[0]["source"] == "Example"
        assert items[0]["tags"] == ["AI"]
        assert len(items[0]["news_id"]) == 64

    def test_skips_items_older_than_cutoff(self):
        old = item("Old", "https://e.com/o", "Sat, 26 Sep 2026 10:00:00 GMT")
        items = parse_entries(FEED_CFG, rss(old), 24, NOW)
        assert items == []

    def test_broken_content_gives_no_items(self):
        assert parse_entries(FEED_CFG, b"not xml at all <<<", 24, NOW) == []


class TestFetchRssItems:
    def test_failed_feed_is_skipped_and_order_kept(self, monkeypatch):
        feeds = [
            {"name": "One", "url": "https://one/rss"},
            {"name": "Down", "url": "https://down/rss"},
            {"name": "Two", "url": "https://two/rss"},
        ]
        bodies = {
            "https://one/rss": rss(item("First", "https://one/1")),
            "https://two/rss": rss(item("Second", "https://two/2")),
        }
        monkeypatch.setattr(
            fetch, "download_feed", lambda url, timeout: bodies.get(url)
        )
        items = fetch_rss_items({"feeds": feeds}, cutoff_hours=10**6)
        assert [i["title"] for i in items] == ["First", "Second"]

    def test_download_feed_returns_none_on_timeout(self, monkeypatch):
        def boom(*args, **kwargs):
            raise httpx.ReadTimeout("slow")

        monkeypatch.setattr(httpx, "get", boom)
        assert fetch.download_feed("https://slow/rss", timeout=1) is None

    def test_download_feed_returns_none_on_http_error(self, monkeypatch):
        request = httpx.Request("GET", "https://gone/rss")
        monkeypatch.setattr(
            httpx, "get", lambda *a, **k: httpx.Response(404, request=request)
        )
        assert fetch.download_feed("https://gone/rss") is None
