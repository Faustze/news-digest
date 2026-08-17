"""
Tests for news.classify module.
"""

import asyncio
import json

import groq
import httpx

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


class _JsonLLM:
    """Returns a valid JSON classification for a single-item batch."""

    def _respond(self):
        content = json.dumps(
            [
                {
                    "news_id": "1",
                    "category": "ai",
                    "subtopics": ["new_models"],
                    "importance": 0.5,
                    "accepted": True,
                }
            ]
        )
        return type("Msg", (), {"content": content})()

    def __call__(self, _payload):
        return self._respond()

    async def ainvoke(self, _payload):
        return self._respond()


def _rate_limit_error() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/chat/completions")
    response = httpx.Response(429, request=request)
    return groq.RateLimitError(
        "Rate limit reached", response=response, body={"error": {"message": "TPD"}}
    )


class _RateLimitedLLM(_JsonLLM):
    def _respond(self):
        raise _rate_limit_error()


class TestClassifyBatchErrors:
    def test_rate_limit_marks_remaining_unaccepted_without_crashing(self):
        items = [{"news_id": str(n), "title": f"t{n}"} for n in range(3)]
        result = asyncio.run(classify_batch(items, _RateLimitedLLM(), batch_size=1))
        assert len(result) == 3
        assert all(i["accepted"] is False for i in result)

    def test_rate_limit_keeps_already_classified_batches(self):
        class _RateLimitAfterFirst(_JsonLLM):
            def __init__(self):
                self.calls = 0

            def _respond(self):
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps(
                        [
                            {
                                "news_id": "0",
                                "category": "ai",
                                "subtopics": ["new_models"],
                                "importance": 0.5,
                                "accepted": True,
                            }
                        ]
                    )
                    return type("Msg", (), {"content": content})()
                raise _rate_limit_error()

        items = [{"news_id": str(n), "title": f"t{n}"} for n in range(3)]
        result = asyncio.run(
            classify_batch(items, _RateLimitAfterFirst(), batch_size=1)
        )
        assert len(result) == 3
        assert result[0]["accepted"] is True
        assert result[1]["accepted"] is False
        assert result[2]["accepted"] is False

    def test_transient_groq_error_skips_batch_and_continues(self):
        class _FlakyLLM(_JsonLLM):
            def __init__(self):
                self.calls = 0

            def _respond(self):
                self.calls += 1
                if self.calls == 1:
                    raise httpx.ConnectError("boom")
                return super()._respond()

        items = [{"news_id": str(n), "title": f"t{n}"} for n in range(2)]
        result = asyncio.run(classify_batch(items, _FlakyLLM(), batch_size=1))
        assert len(result) == 2
        assert result[0]["accepted"] is False
        assert result[1]["accepted"] is True
