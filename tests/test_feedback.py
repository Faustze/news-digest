"""
Tests for news.feedback module.
"""

import pytest

from news.feedback import (
    HIDE_SIMILAR,
    NOT_INTERESTING,
    USEFUL,
    VALID_REACTIONS,
    FeedbackStore,
    add_reaction,
    generate_news_id,
    get_reactions_for_category,
    get_reactions_for_subtopic,
    load_feedback,
    save_feedback,
)

# ── News ID generation ────────────────────────────────────────────────────────


class TestNewsId:
    def test_deterministic(self):
        id1 = generate_news_id("Title", "https://example.com")
        id2 = generate_news_id("Title", "https://example.com")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = generate_news_id("Title A", "https://example.com/a")
        id2 = generate_news_id("Title B", "https://example.com/b")
        assert id1 != id2

    def test_case_insensitive(self):
        id1 = generate_news_id("Hello World", "https://Example.COM")
        id2 = generate_news_id("hello world", "https://example.com")
        assert id1 == id2

    def test_is_hex_string(self):
        nid = generate_news_id("Test", "https://example.com")
        assert len(nid) == 64  # SHA-256 hex
        int(nid, 16)  # should not raise


# ── Feedback store ────────────────────────────────────────────────────────────


class TestFeedbackStore:
    def test_empty_store(self):
        store = FeedbackStore()
        assert store.version == 1
        assert len(store.reactions) == 0
        assert store.last_update_id == 0

    def test_add_reaction(self):
        store = FeedbackStore()
        store = add_reaction(
            store,
            news_id="abc123",
            reaction=USEFUL,
            category="ai",
            subtopics=["new_models"],
        )
        assert len(store.reactions) == 1
        assert store.reactions[0].news_id == "abc123"
        assert store.reactions[0].reaction == USEFUL

    def test_invalid_reaction_raises(self):
        store = FeedbackStore()
        with pytest.raises(ValueError, match="Invalid reaction"):
            add_reaction(store, "abc", "invalid", "ai")

    def test_get_reactions_for_category(self):
        store = FeedbackStore()
        add_reaction(store, "a", USEFUL, "ai")
        add_reaction(store, "b", NOT_INTERESTING, "running")
        add_reaction(store, "c", HIDE_SIMILAR, "ai")

        ai_reactions = get_reactions_for_category(store, "ai")
        assert len(ai_reactions) == 2

        running_reactions = get_reactions_for_category(store, "running")
        assert len(running_reactions) == 1

    def test_get_reactions_for_subtopic(self):
        store = FeedbackStore()
        add_reaction(store, "a", USEFUL, "ai", ["new_models", "research"])
        add_reaction(store, "b", NOT_INTERESTING, "ai", ["robotics"])

        new_models_reactions = get_reactions_for_subtopic(store, "new_models")
        assert len(new_models_reactions) == 1

        research_reactions = get_reactions_for_subtopic(store, "research")
        assert len(research_reactions) == 1


# ── Persistence ───────────────────────────────────────────────────────────────


class TestFeedbackPersistence:
    def test_roundtrip(self, tmp_path):
        store = FeedbackStore()
        add_reaction(store, "a", USEFUL, "ai", ["new_models"])
        add_reaction(store, "b", HIDE_SIMILAR, "running", ["recovery"])

        path = tmp_path / "feedback.json"
        save_feedback(store, path)

        loaded = load_feedback(path)
        assert len(loaded.reactions) == 2
        assert loaded.reactions[0].news_id == "a"
        assert loaded.reactions[1].reaction == HIDE_SIMILAR

    def test_load_missing_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        store = load_feedback(path)
        assert len(store.reactions) == 0

    def test_load_invalid_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        store = load_feedback(path)
        assert len(store.reactions) == 0


# ── Constants ─────────────────────────────────────────────────────────────────


class TestConstants:
    def test_valid_reactions(self):
        assert USEFUL == "useful"
        assert NOT_INTERESTING == "not_interesting"
        assert HIDE_SIMILAR == "hide_similar"
        assert len(VALID_REACTIONS) == 3
