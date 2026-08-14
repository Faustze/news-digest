"""
Deduplication: remove duplicate articles based on URL, title, and similarity.
"""

from __future__ import annotations

import re


def _normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, remove punctuation, collapse whitespace."""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _title_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity between two titles."""
    words_a = set(_normalize_title(a).split())
    words_b = set(_normalize_title(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def deduplicate(items: list[dict], similarity_threshold: float = 0.6) -> list[dict]:
    """
    Remove duplicate articles.

    Deduplication criteria (in order):
    1. Same news_id
    2. Same normalized URL
    3. Same normalized title
    4. Title similarity above threshold
    """
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict] = []

    for item in items:
        news_id = item.get("news_id", "")
        url = item.get("link", "").strip().lower()
        title = item.get("title", "")
        norm_title = _normalize_title(title)

        # Check news_id
        if news_id and news_id in seen_ids:
            continue

        # Check URL
        if url and url in seen_urls:
            continue

        # Check normalized title
        if norm_title and norm_title in seen_titles:
            continue

        # Check title similarity against all seen titles
        is_similar = False
        for existing_title in seen_titles:
            if _title_similarity(title, existing_title) >= similarity_threshold:
                is_similar = True
                break

        if is_similar:
            continue

        # All checks passed — keep this item
        if news_id:
            seen_ids.add(news_id)
        if url:
            seen_urls.add(url)
        if norm_title:
            seen_titles.add(norm_title)
        unique.append(item)

    return unique
