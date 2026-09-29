"""
Tests for news.url_utils module.
"""

from news.url_utils import normalize_url


class TestNormalizeUrl:
    def test_drops_tracking_params_and_fragment(self):
        assert normalize_url("https://x.com/a?utm_source=rss#top") == "https://x.com/a"

    def test_keeps_identifying_query(self):
        assert (
            normalize_url("https://x.com/item?id=2&fbclid=z&UTM_MEDIUM=y&a=1")
            == "https://x.com/item?a=1&id=2"
        )

    def test_different_query_ids_stay_distinct(self):
        assert normalize_url("https://x.com/item?id=1") != normalize_url(
            "https://x.com/item?id=2"
        )

    def test_lowercases_scheme_and_host_keeps_path_case(self):
        assert normalize_url("HTTPS://X.com/Path") == "https://x.com/Path"

    def test_drops_trailing_slash(self):
        assert normalize_url("https://x.com/a/") == "https://x.com/a"
