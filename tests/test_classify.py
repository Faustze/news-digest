"""
Tests for news.classify module.
"""

import asyncio
import json

import groq
import httpx

from news.classify import (
    _build_categories_text,
    _build_items_json,
    _pause_seconds,
    apply_classification,
    classify_batch,
)
from news.llm import TokenCounter
from news.profile import CATEGORIES


class TestPromptPayload:
    def test_categories_text_lists_every_category_with_six_subtopics(self):
        lines = _build_categories_text().splitlines()
        assert len(lines) == len(CATEGORIES)
        for line in lines:
            cat_id, subtopics = line.split(": ")
            assert cat_id in CATEGORIES
            assert subtopics.split(", ") == list(CATEGORIES[cat_id])

    def test_items_use_short_batch_index_not_news_id(self):
        items = [{"news_id": "f" * 64, "title": "A", "categories": ["ai"]}] * 2
        payload = json.loads(_build_items_json(items))
        assert [p["id"] for p in payload] == [0, 1]
        assert "f" * 64 not in _build_items_json(items)
        assert payload[0]["feed"] == ["ai"]

    def test_summary_truncated(self):
        payload = json.loads(_build_items_json([{"summary": "x" * 1000}]))
        assert len(payload[0]["summary"]) == 250


class TestApplyClassification:
    def test_valid_classification_accepted(self):
        item = {}
        apply_classification(
            item, {"category": "ai", "subtopics": ["robotics"], "importance": 0.8}
        )
        assert item == {
            "category": "ai",
            "subtopics": ["robotics"],
            "importance": 0.8,
            "accepted": True,
        }

    def test_unknown_or_null_category_rejected(self):
        for category in ("sports", None):
            item = {}
            apply_classification(item, {"category": category})
            assert item["accepted"] is False

    def test_foreign_subtopics_dropped_and_importance_clamped(self):
        item = {}
        apply_classification(
            item,
            {"category": "ai", "subtopics": ["robotics", "usa"], "importance": 7},
        )
        assert item["subtopics"] == ["robotics"]
        assert item["importance"] == 1.0

    def test_bad_importance_defaults(self):
        item = {}
        apply_classification(item, {"category": "ai", "importance": "high"})
        assert item["importance"] == 0.5


class TestPauseSeconds:
    def test_paces_to_tokens_per_minute(self):
        # 4000 tokens at 8000 TPM = 30 s, minus 5 s already spent on the call.
        assert _pause_seconds(4000, 5, 8000, 2) == 25

    def test_min_pause_without_budget_or_usage(self):
        assert _pause_seconds(4000, 0, None, 2) == 2
        assert _pause_seconds(0, 0, 8000, 2) == 2


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
                    "id": 0,
                    "category": "ai",
                    "subtopics": ["new_models"],
                    "importance": 0.5,
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
                                "id": 0,
                                "category": "ai",
                                "subtopics": ["new_models"],
                                "importance": 0.5,
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


class TestClassifyBatchMapping:
    def test_maps_answers_by_index_and_rejects_missing(self):
        class _TwoOfThree:
            def __call__(self, _payload):
                content = json.dumps(
                    [
                        {"id": 2, "category": "space", "importance": 0.9},
                        {"id": 0, "category": "games"},
                        {"id": 7, "category": "ai"},
                    ]
                )
                msg = type("Msg", (), {"content": content})()
                msg.usage_metadata = {"total_tokens": 123}
                return msg

        items = [{"news_id": str(n), "title": f"t{n}"} for n in range(3)]
        usage = TokenCounter()
        result = asyncio.run(
            classify_batch(items, _TwoOfThree(), batch_size=3, usage=usage)
        )
        assert [i["accepted"] for i in result] == [True, False, True]
        assert result[0]["category"] == "games"
        assert result[2]["category"] == "space"
        assert usage.total == 123
