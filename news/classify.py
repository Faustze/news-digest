"""
Article classification: assign category + subtopics via LLM.
"""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from news.profile import CATEGORIES, CATEGORY_LABELS

# ── Prompt ────────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты классификатор новостей. Определи категорию и подтемы для каждой новости.

Доступные категории и подтемы:
{categories_json}

Для каждой новости верни JSON с полями:
{{
    "news_id": "оригинальный news_id",
    "category": "id_категории",
    "subtopics": ["id_подтемы1"],
    "importance": 0.0-1.0,
    "accepted": true/false
}}

Правила:
- Категория должна быть строго из списка.
- Подтемы должны принадлежать выбранной категории.
- Если новость не подходит ни под одну категорию — accepted: false.
- importance: насколько новость важна/интересна сама по себе (без учёта профиля).
- Не выдумывай факты.

Верни JSON-массив.""",
        ),
        ("human", "Новости:\n{items_json}"),
    ]
)


# ── Classification ────────────────────────────────────────────────────────────


def _build_categories_json() -> str:
    """Build a compact representation of categories for the prompt."""
    result = {}
    for cat_id, subtopics in CATEGORIES.items():
        label = CATEGORY_LABELS.get(cat_id, cat_id)
        result[cat_id] = {
            "label": label,
            "subtopics": dict(subtopics),
        }
    return json.dumps(result, ensure_ascii=False, indent=2)


async def classify_batch(
    items: list[dict],
    llm: ChatGroq,
    batch_size: int = 12,
) -> list[dict]:
    """
    Classify a batch of items. Returns items with added classification fields.
    """
    if not items:
        return []

    chain = CLASSIFY_PROMPT | llm
    results = []
    categories_json = _build_categories_json()

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        try:
            raw = await chain.ainvoke(
                {
                    "categories_json": categories_json,
                    "items_json": json.dumps(batch, ensure_ascii=False, indent=2),
                }
            )
            content = raw.content if hasattr(raw, "content") else str(raw)
            clean = re.sub(r"```(?:json)?|```", "", content).strip()
            parsed = json.loads(clean)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] Classification batch {i // batch_size + 1} failed: {e}")
            for item in batch:
                item["accepted"] = False
            results.extend(batch)
        else:
            if not isinstance(parsed, list):
                print(
                    f"[WARN] Classification batch {i // batch_size + 1} "
                    "returned non-list output"
                )
                for item in batch:
                    item["accepted"] = False
                results.extend(batch)
            else:
                # Merge classification back into items
                by_id = {item.get("news_id"): item for item in batch}
                for cls in parsed:
                    nid = cls.get("news_id")
                    if nid and nid in by_id:
                        by_id[nid]["category"] = cls.get("category", "")
                        by_id[nid]["subtopics"] = cls.get("subtopics", [])
                        by_id[nid]["importance"] = cls.get("importance", 0.5)
                        by_id[nid]["accepted"] = cls.get("accepted", False)
                results.extend(by_id.values())

        if i + batch_size < len(items):
            await asyncio.sleep(2)

    return results
