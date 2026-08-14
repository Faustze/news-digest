"""
Ranking: score articles based on user profile, feedback, and other signals.
"""

from __future__ import annotations

from news.feedback import FeedbackStore
from news.profile import UserProfile

# ── Ranking ───────────────────────────────────────────────────────────────────


def rank_item(
    item: dict,
    profile: UserProfile,
    feedback: FeedbackStore | None = None,
) -> float:
    """
    Compute a final relevance score for an item based on multiple factors.

    Factors:
    1. Category relevance (is the category enabled?)
    2. Subtopic interest (0-5)
    3. Explicit exclusions (interest=0 → filter out)
    4. Region relevance
    5. Source reliability
    6. Feedback signals
    7. Freshness (already filtered by cutoff)
    8. Importance (from LLM classification)
    9. Personal context (minor boost)
    """
    if not item.get("accepted", True):
        return 0.0

    category = item.get("category", "")
    subtopics = item.get("subtopics", [])

    # Check explicit exclusions (interest=0)
    for st in subtopics:
        if profile.get_interest(category, st) == 0:
            return 0.0

    # Check category exclusion
    if not profile.categories.get(category, profile.categories.get("ai")).enabled:
        return 0.0

    # Subtopic interest score (average of matched subtopics, 0-5 → 0-1)
    interests = []
    for st in subtopics:
        interest = profile.get_interest(category, st)
        interests.append(interest / 5.0)
    interest_score = max(interests) if interests else 0.5

    # Importance score (from LLM classification)
    importance = item.get("importance", 0.5)

    # Source reliability (simple heuristic based on source name)
    source = item.get("source", "").lower()
    high_reliability = {
        "reuters",
        "bbc",
        "ap",
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

    # Feedback adjustment
    feedback_score = 0.5  # neutral
    if feedback:
        for reaction in feedback.reactions:
            if reaction.category == category:
                if reaction.reaction == "useful":
                    feedback_score = min(feedback_score + 0.1, 1.0)
                elif reaction.reaction == "not_interesting":
                    feedback_score = max(feedback_score - 0.1, 0.0)
                elif reaction.reaction == "hide_similar":
                    feedback_score = max(feedback_score - 0.2, 0.0)

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
        + context_boost * 0.05
    )

    return round(final_score, 3)


def rank_items(
    items: list[dict],
    profile: UserProfile,
    feedback: FeedbackStore | None = None,
) -> list[dict]:
    """Rank all items and return sorted by final score descending."""
    for item in items:
        item["final_score"] = rank_item(item, profile, feedback)

    # Filter out zero-score items
    items = [i for i in items if i["final_score"] > 0]

    # Sort by score
    items.sort(key=lambda x: x["final_score"], reverse=True)
    return items
