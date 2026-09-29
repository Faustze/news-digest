"""
Pre-LLM candidate selection: cheap deterministic cuts before classification.

Every item sent to the LLM costs Groq tokens, while a digest shows only a
handful of items. Dropping items the profile can never show and capping how
many fresh items one source contributes keeps the token bill small without
losing category coverage (sources map to categories in config.yaml).
"""

from __future__ import annotations

from collections import defaultdict


def drop_disabled_categories(items: list[dict], enabled: set[str]) -> list[dict]:
    """
    Drop items whose feed is tagged only with disabled categories.

    Items from feeds without category tags are kept: the classifier decides.
    """
    return [
        item
        for item in items
        if not item.get("categories")
        or any(cat in enabled for cat in item["categories"])
    ]


def limit_per_source(items: list[dict], max_per_source: int | None) -> list[dict]:
    """Keep the `max_per_source` newest items of every source, in input order."""
    if not max_per_source or max_per_source <= 0:
        return list(items)

    by_source: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        by_source[item.get("source", "")].append(idx)

    keep: set[int] = set()
    for indexes in by_source.values():
        newest = sorted(
            indexes, key=lambda i: items[i].get("published") or "", reverse=True
        )
        keep.update(newest[:max_per_source])
    return [item for idx, item in enumerate(items) if idx in keep]


def select_candidates(
    items: list[dict], enabled: set[str], max_per_source: int | None
) -> list[dict]:
    return limit_per_source(drop_disabled_categories(items, enabled), max_per_source)
