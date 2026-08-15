"""
Tests for news.deduplicate module.
"""

from news.deduplicate import (
    _normalize_title,
    _normalize_url,
    _title_similarity,
    deduplicate,
)


class TestNormalizeUrl:
    def test_drops_query_and_fragment(self):
        assert _normalize_url("https://x.com/a?utm_source=rss#top") == "https://x.com/a"

    def test_lowercases_host_and_path(self):
        assert _normalize_url("HTTPS://X.com/Path") == "https://x.com/path"

    def test_drops_trailing_slash(self):
        assert _normalize_url("https://x.com/a/") == "https://x.com/a"


class TestNormalizeTitle:
    def test_lowercase(self):
        assert _normalize_title("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert _normalize_title("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize_title("  hello   world  ") == "hello world"


class TestTitleSimilarity:
    def test_identical_titles(self):
        assert _title_similarity("Hello World", "Hello World") == 1.0

    def test_no_overlap(self):
        assert _title_similarity("Cat Dog", "Fish Bird") == 0.0

    def test_partial_overlap(self):
        sim = _title_similarity("New AI Model Released", "New AI Tool Released")
        assert 0.5 < sim < 1.0

    def test_empty_titles(self):
        assert _title_similarity("", "") == 0.0


class TestDeduplicate:
    def test_removes_same_url(self):
        items = [
            {"news_id": "a", "title": "Title A", "link": "https://example.com/1"},
            {"news_id": "b", "title": "Title B", "link": "https://example.com/1"},
        ]
        result = deduplicate(items)
        assert len(result) == 1

    def test_removes_url_with_tracking_params(self):
        items = [
            {"news_id": "a", "title": "Title A", "link": "https://example.com/1"},
            {
                "news_id": "b",
                "title": "Title B",
                "link": "https://example.com/1?utm_source=rss&utm_medium=feed",
            },
        ]
        result = deduplicate(items)
        assert len(result) == 1

    def test_removes_same_title(self):
        items = [
            {"news_id": "a", "title": "Breaking News Today", "link": "https://a.com"},
            {"news_id": "b", "title": "Breaking News Today", "link": "https://b.com"},
        ]
        result = deduplicate(items)
        assert len(result) == 1

    def test_removes_similar_titles(self):
        items = [
            {
                "news_id": "a",
                "title": "New AI Model Released by OpenAI",
                "link": "https://a.com",
            },
            {
                "news_id": "b",
                "title": "OpenAI Releases New AI Model",
                "link": "https://b.com",
            },
        ]
        result = deduplicate(items, similarity_threshold=0.5)
        assert len(result) == 1

    def test_keeps_different_articles(self):
        items = [
            {"news_id": "a", "title": "AI News", "link": "https://a.com"},
            {"news_id": "b", "title": "Running Tips", "link": "https://b.com"},
            {"news_id": "c", "title": "Movie Review", "link": "https://c.com"},
        ]
        result = deduplicate(items)
        assert len(result) == 3

    def test_removes_same_news_id(self):
        items = [
            {"news_id": "abc", "title": "Title A", "link": "https://a.com"},
            {"news_id": "abc", "title": "Title B", "link": "https://b.com"},
        ]
        result = deduplicate(items)
        assert len(result) == 1

    def test_preserves_order(self):
        items = [
            {"news_id": "a", "title": "First", "link": "https://a.com"},
            {"news_id": "b", "title": "Second", "link": "https://b.com"},
            {"news_id": "c", "title": "Third", "link": "https://c.com"},
        ]
        result = deduplicate(items)
        assert [i["title"] for i in result] == ["First", "Second", "Third"]
