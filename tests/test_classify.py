"""
Tests for news.classify module.
"""

import asyncio
import json

from news.classify import _build_categories_json, classify_batch


class TestBuildCategoriesJson:
    def test_returns_valid_json(self):
        result = _build_categories_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_all_categories(self):
        from news.profile import CATEGORIES

        result = _build_categories_json()
        parsed = json.loads(result)
        for cat_id in CATEGORIES:
            assert cat_id in parsed

    def test_category_has_label_and_subtopics(self):
        result = _build_categories_json()
        parsed = json.loads(result)
        ai = parsed["ai"]
        assert "label" in ai
        assert "subtopics" in ai
        assert len(ai["subtopics"]) == 6


class _BadLLM:
    """Stub chain target that returns content which is not a JSON array."""

    def __call__(self, _payload):
        return type("Msg", (), {"content": "sorry, no JSON here"})()

    async def ainvoke(self, _payload):
        return type("Msg", (), {"content": "sorry, no JSON here"})()


class TestClassifyBatch:
    def test_failed_batch_marks_every_item_unaccepted(self):
        items = [{"news_id": str(n), "title": f"t{n}"} for n in range(3)]
        result = asyncio.run(classify_batch(items, _BadLLM(), batch_size=3))
        assert len(result) == 3
        assert all(i["accepted"] is False for i in result)
