"""
Feedback schema and persistence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

# ── Schema ────────────────────────────────────────────────────────────────────


class Reaction(str):
    pass


USEFUL = "useful"
NOT_INTERESTING = "not_interesting"
HIDE_SIMILAR = "hide_similar"

VALID_REACTIONS = {USEFUL, NOT_INTERESTING, HIDE_SIMILAR}


class FeedbackEntry(BaseModel):
    news_id: str
    reaction: str
    category: str
    subtopics: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class FeedbackStore(BaseModel):
    version: int = 1
    reactions: list[FeedbackEntry] = Field(default_factory=list)
    last_update_id: int = 0


# ── News ID generation ───────────────────────────────────────────────────────


def generate_news_id(title: str, url: str) -> str:
    """Generate a stable news ID from title and canonical URL."""
    raw = f"{title.strip().lower()}|{url.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Loader ────────────────────────────────────────────────────────────────────

DEFAULT_FEEDBACK_PATH = Path("feedback.json")


def load_feedback(path: Path | str = DEFAULT_FEEDBACK_PATH) -> FeedbackStore:
    """Load feedback from JSON file, or return empty store."""
    p = Path(path)
    if not p.exists():
        return FeedbackStore()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return FeedbackStore.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARN] Invalid feedback at {p}: {e}. Starting fresh.")
        return FeedbackStore()


def save_feedback(
    store: FeedbackStore, path: Path | str = DEFAULT_FEEDBACK_PATH
) -> None:
    """Save feedback store to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(store.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_reaction(
    store: FeedbackStore,
    news_id: str,
    reaction: str,
    category: str,
    subtopics: list[str] | None = None,
) -> FeedbackStore:
    """Add a reaction to the feedback store."""
    if reaction not in VALID_REACTIONS:
        raise ValueError(
            f"Invalid reaction: {reaction}. Must be one of {VALID_REACTIONS}"
        )

    entry = FeedbackEntry(
        news_id=news_id,
        reaction=reaction,
        category=category,
        subtopics=subtopics or [],
    )
    store.reactions.append(entry)
    return store


def get_reactions_for_category(
    store: FeedbackStore, category: str
) -> list[FeedbackEntry]:
    """Get all reactions for a specific category."""
    return [r for r in store.reactions if r.category == category]


def get_reactions_for_subtopic(
    store: FeedbackStore, subtopic: str
) -> list[FeedbackEntry]:
    """Get all reactions that include a specific subtopic."""
    return [r for r in store.reactions if subtopic in r.subtopics]
