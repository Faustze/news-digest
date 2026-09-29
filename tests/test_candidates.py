"""
Tests for news.candidates (pre-LLM selection).
"""

from news.candidates import (
    drop_disabled_categories,
    limit_per_source,
    select_candidates,
)


def item(source: str, published: str, categories=("ai",), title: str = "") -> dict:
    return {
        "source": source,
        "published": published,
        "categories": list(categories),
        "title": title or f"{source}-{published}",
    }


class TestDropDisabledCategories:
    def test_drops_items_whose_feed_categories_are_all_disabled(self):
        items = [item("A", "1", ["games"]), item("B", "1", ["ai"])]
        assert [i["source"] for i in drop_disabled_categories(items, {"ai"})] == ["B"]

    def test_keeps_item_if_any_feed_category_enabled(self):
        items = [item("A", "1", ["business", "finance"])]
        assert drop_disabled_categories(items, {"finance"}) == items

    def test_keeps_untagged_items(self):
        items = [item("A", "1", [])]
        assert drop_disabled_categories(items, set()) == items


class TestLimitPerSource:
    def test_keeps_newest_per_source_in_input_order(self):
        items = [
            item("A", "2026-09-29T08:00:00+00:00"),
            item("A", "2026-09-29T10:00:00+00:00"),
            item("B", "2026-09-29T01:00:00+00:00"),
            item("A", "2026-09-29T09:00:00+00:00"),
        ]
        kept = limit_per_source(items, 2)
        assert [i["published"][11:13] for i in kept] == ["10", "01", "09"]

    def test_no_limit_keeps_everything(self):
        items = [item("A", str(n)) for n in range(5)]
        assert limit_per_source(items, None) == items
        assert limit_per_source(items, 0) == items


def test_select_candidates_combines_both_cuts():
    items = [item("A", str(n), ["ai"]) for n in range(5)] + [item("G", "1", ["games"])]
    result = select_candidates(items, {"ai"}, 3)
    assert len(result) == 3
    assert all(i["source"] == "A" for i in result)
