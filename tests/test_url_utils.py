"""
Tests for news.url_utils module.
"""

from news.url_utils import normalize_url


class TestNormalizeUrl:
    def test_drops_query_and_fragment(self):
        assert normalize_url("https://x.com/a?utm_source=rss#top") == "https://x.com/a"

    def test_lowercases_host_and_path(self):
        assert normalize_url("HTTPS://X.com/Path") == "https://x.com/path"

    def test_drops_trailing_slash(self):
        assert normalize_url("https://x.com/a/") == "https://x.com/a"
