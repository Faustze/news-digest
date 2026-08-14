"""
Tests for news.classify module.
"""

from news.classify import _build_categories_json


class TestBuildCategoriesJson:
    def test_returns_valid_json(self):
        import json

        result = _build_categories_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_all_categories(self):
        import json

        from news.profile import CATEGORIES

        result = _build_categories_json()
        parsed = json.loads(result)
        for cat_id in CATEGORIES:
            assert cat_id in parsed

    def test_category_has_label_and_subtopics(self):
        import json

        result = _build_categories_json()
        parsed = json.loads(result)
        ai = parsed["ai"]
        assert "label" in ai
        assert "subtopics" in ai
        assert len(ai["subtopics"]) == 6
