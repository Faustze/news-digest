"""
Article classification: assign category + subtopics via LLM.

The prompt is kept compact because the Groq free tier limits tokens per
minute and per day: items are referenced by a short batch-local index instead
of their 64-char news_id, and categories are sent as bare ids (the ids are
self-descriptive English words, labels are only needed for humans).
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import groq
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from news.llm import TokenCounter
from news.profile import CATEGORIES

SUMMARY_CHARS = 250

# ── Prompt ────────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты классификатор новостей. Для каждой новости определи категорию и подтемы.

Категории и их подтемы (id):
{categories}

Верни только JSON-массив, по объекту на каждую новость:
{{"id": <id новости>, "category": "<id категории или null>", "subtopics": ["<id подтемы>"], "importance": <0.0-1.0>}}

Правила:
- category и subtopics — строго id из списка; подтемы только из выбранной категории.
- category: null, если новость не подходит ни под одну категорию или это реклама.
- importance — насколько новость важна сама по себе, без учёта чьих-либо интересов.
- Не выдумывай факты. Без пояснений и markdown.""",
        ),
        ("human", "{items_json}"),
    ]
)


# ── Prompt payload ────────────────────────────────────────────────────────────


def _build_categories_text() -> str:
    """One line per category: `ai: new_models, ai_tools, ...`."""
    return "\n".join(
        f"{cat_id}: {', '.join(subtopics)}" for cat_id, subtopics in CATEGORIES.items()
    )


def _build_items_json(items: list[dict]) -> str:
    """
    Compact serialization for the classification prompt.

    `id` is the item's index in the batch; `feed` is the category hint from
    config.yaml. Summaries are truncated to keep each batch small.
    """
    compact = [
        {
            "id": idx,
            "title": item.get("title", ""),
            "summary": (item.get("summary", "") or "")[:SUMMARY_CHARS],
            "source": item.get("source", ""),
            "feed": item.get("categories", []),
        }
        for idx, item in enumerate(items)
    ]
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


# ── Response validation ───────────────────────────────────────────────────────


def _parse_response(content: str) -> list:
    clean = re.sub(r"```(?:json)?|```", "", content).strip()
    parsed = json.loads(clean)
    if not isinstance(parsed, list):
        raise TypeError("classification output is not a JSON array")
    return parsed


def apply_classification(item: dict, cls: dict) -> None:
    """
    Merge one validated classification into an item.

    Unknown categories reject the item; subtopics outside the chosen category
    are dropped; importance is clamped to 0..1 (default 0.5).
    """
    category = cls.get("category")
    if category not in CATEGORIES:
        item["accepted"] = False
        return

    subtopics = cls.get("subtopics")
    if not isinstance(subtopics, list):
        subtopics = []

    try:
        importance = float(cls.get("importance", 0.5))
    except (TypeError, ValueError):
        importance = 0.5

    item["category"] = category
    item["subtopics"] = [s for s in subtopics if s in CATEGORIES[category]]
    item["importance"] = min(max(importance, 0.0), 1.0)
    item["accepted"] = True


def _merge_batch(batch: list[dict], parsed: list) -> None:
    for item in batch:
        item["accepted"] = False
    for cls in parsed:
        if not isinstance(cls, dict):
            continue
        idx = cls.get("id")
        if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(batch):
            apply_classification(batch[idx], cls)


# ── Classification ────────────────────────────────────────────────────────────


def _pause_seconds(
    tokens: int, elapsed: float, tokens_per_minute: int | None, min_pause: float
) -> float:
    """Wait long enough that this request's tokens fit the per-minute budget."""
    if not tokens_per_minute or tokens <= 0:
        return min_pause
    return max(min_pause, tokens / tokens_per_minute * 60 - elapsed)


async def classify_batch(
    items: list[dict],
    llm: BaseChatModel,
    batch_size: int = 12,
    *,
    usage: TokenCounter | None = None,
    tokens_per_minute: int | None = None,
    min_pause: float = 2.0,
) -> list[dict]:
    """
    Classify items in batches. Returns items with added classification fields.

    A failed batch marks its items `accepted: False` and the run continues; a
    rate-limit error stops classification and rejects the remaining items.
    """
    if not items:
        return []

    chain = CLASSIFY_PROMPT | llm
    results = []
    categories = _build_categories_text()

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_no = i // batch_size + 1
        started = time.monotonic()
        tokens = 0
        try:
            raw = await chain.ainvoke(
                {"categories": categories, "items_json": _build_items_json(batch)}
            )
            if usage is not None:
                tokens = usage.add(raw)
            content = raw.content if hasattr(raw, "content") else str(raw)
            _merge_batch(batch, _parse_response(content))
        except (ValueError, TypeError) as e:  # malformed model output
            print(f"[WARN] Classification batch {batch_no} failed: {e}")
            for item in batch:
                item["accepted"] = False
        except groq.RateLimitError as e:
            print(
                f"[WARN] Groq rate limit reached; stopping classification at "
                f"batch {batch_no}: {e}"
            )
            for item in items[i:]:
                item["accepted"] = False
            results.extend(items[i:])
            break
        except (groq.GroqError, httpx.HTTPError, TimeoutError) as e:
            print(f"[WARN] Classification batch {batch_no} failed, continuing: {e}")
            for item in batch:
                item["accepted"] = False
        results.extend(batch)

        if i + batch_size < len(items):
            elapsed = time.monotonic() - started
            await asyncio.sleep(
                _pause_seconds(tokens, elapsed, tokens_per_minute, min_pause)
            )

    return results
