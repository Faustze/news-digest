"""
Ranking: score articles based on user profile, feedback, and other signals.
"""

from __future__ import annotations

from news.feedback import FeedbackStore
from news.profile import UserProfile

# ── Ranking ───────────────────────────────────────────────────────────────────

_FEEDBACK_DELTAS = {"useful": 0.1, "not_interesting": -0.1, "hide_similar": -0.2}


def _category_feedback_scores(feedback: FeedbackStore | None) -> dict[str, float]:
    """
    Aggregate one feedback score per category, in the range 0.0-1.0.

    Scanned once per run instead of once per item; order-independent because
    the result for a category no longer depends on the reaction list order.
    """
    if not feedback:
        return {}
    scores: dict[str, float] = {}
    for reaction in feedback.reactions:
        delta = _FEEDBACK_DELTAS.get(reaction.reaction)
        if delta is None:
            continue
        current = scores.get(reaction.category, 0.5)
        scores[reaction.category] = min(max(current + delta, 0.0), 1.0)
    return scores


def rank_item(
    item: dict,
    profile: UserProfile,
    feedback: FeedbackStore | None = None,
    category_scores: dict[str, float] | None = None,
) -> float:
    """
    Compute a final relevance score for an item based on multiple factors.

    Factors:
    1. Category relevance (is the category enabled?)
    2. Subtopic interest (0-5)
    3. Explicit exclusions (interest=0 and profile.general.exclusions → filter out)
    4. Source reliability
    5. Feedback signals
    6. Freshness (already filtered by cutoff)
    7. Importance (from LLM classification)
    8. Personal context (minor boost)
    """
    if item.get("accepted") is not True:
        return 0.0

    category = item.get("category", "")
    subtopics = item.get("subtopics", [])

    # Hard exclusions from the profile. Nothing later may override these.
    haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    for term in profile.general.exclusions:
        if term.strip() and term.strip().lower() in haystack:
            return 0.0

    # Check explicit exclusions (interest=0). Unconfigured subtopics are neutral.
    for st in subtopics:
        if profile.get_interest(category, st) == 0:
            return 0.0

    # Check category exclusion. An unknown category never contributes candidates.
    cat = profile.categories.get(category)
    if cat is None or not cat.enabled:
        return 0.0

    # Subtopic interest score (max of matched subtopics, 0-5 → 0-1)
    interests = []
    for st in subtopics:
        interest = profile.get_interest(category, st)
        interests.append((interest if interest is not None else 3) / 5.0)
    interest_score = max(interests) if interests else 0.5

    # Importance score (from LLM classification)
    importance = item.get("importance", 0.5)

    # Source reliability (simple heuristic based on source name)
    source = item.get("source", "").lower()
    high_reliability = {
        "reuters",
        "bbc",
        "associated press",
        "the verge",
        "ars technica",
        "techcrunch",
    }
    medium_reliability = {"hacker news", "dev.to", "habr"}

    if any(r in source for r in high_reliability):
        source_score = 0.9
    elif any(r in source for r in medium_reliability):
        source_score = 0.7
    else:
        source_score = 0.5

    # Adjust source score based on user preference
    reliability_pref = profile.general.source_reliability.value
    if reliability_pref == "verified":
        source_score = min(source_score * 1.2, 1.0)
    elif reliability_pref == "broad":
        source_score = max(source_score * 0.8, 0.3)

    # Feedback adjustment (precomputed per-category score; 0.5 = neutral)
    if category_scores is None:
        category_scores = _category_feedback_scores(feedback)
    feedback_score = category_scores.get(category, 0.5)

    # Personal context boost
    context_boost = 0.0
    if profile.general.personal_context:
        context_lower = profile.general.personal_context.lower()
        for st in subtopics:
            if st.replace("_", " ") in context_lower:
                context_boost = 0.1
                break

    # Weighted combination
    final_score = (
        interest_score * 0.40
        + importance * 0.25
        + source_score * 0.15
        + feedback_score * 0.15
        + context_boost
    )

    return round(final_score, 3)


def rank_items(
    items: list[dict],
    profile: UserProfile,
    feedback: FeedbackStore | None = None,
) -> list[dict]:
    """Rank all items and return sorted by final score descending."""
    category_scores = _category_feedback_scores(feedback)

    for item in items:
        item["final_score"] = rank_item(item, profile, category_scores=category_scores)

    # Filter out zero-score items
    items = [i for i in items if i["final_score"] > 0]

    # Sort by score
    items.sort(key=lambda x: x["final_score"], reverse=True)
    return items
