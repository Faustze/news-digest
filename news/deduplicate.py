"""
Deduplication: remove duplicate articles based on URL, title, and similarity.
"""

from __future__ import annotations

import re

from news.url_utils import normalize_url


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
    return _set_similarity(words_a, words_b)


def _set_similarity(words_a: set[str], words_b: set[str]) -> float:
    """Jaccard similarity between two precomputed word sets."""
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _word_set(title: str) -> set[str]:
    """Normalize a title once and return its word set for reuse."""
    return set(_normalize_title(title).split())


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
    seen_word_sets: list[set[str]] = []
    unique: list[dict] = []

    for item in items:
        news_id = item.get("news_id", "")
        url = normalize_url(item.get("link", "") or "")
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
        words = _word_set(title)
        is_similar = any(
            _set_similarity(words, existing) >= similarity_threshold
            for existing in seen_word_sets
        )

        if is_similar:
            continue

        # All checks passed — keep this item
        if news_id:
            seen_ids.add(news_id)
        if url:
            seen_urls.add(url)
        if norm_title:
            seen_titles.add(norm_title)
            seen_word_sets.append(words)
        unique.append(item)

    return unique
