"""
Tests for news.profile module.
"""

import json

import pytest

from news.profile import (
    CATEGORIES,
    CATEGORY_LABELS,
    Category,
    DetailLevel,
    Frequency,
    Language,
    LanguageLevel,
    Priority,
    SourceReliability,
    _empty_profile,
    _migration_profile_from_config,
    load_profile,
    save_profile,
)

# ── Schema validation ─────────────────────────────────────────────────────────


class TestUserProfileSchema:
    def test_empty_profile_has_all_categories(self):
        profile = _empty_profile()
        assert len(profile.categories) == 12
        for cat_id in CATEGORIES:
            assert cat_id in profile.categories

    def test_empty_profile_all_enabled(self):
        profile = _empty_profile()
        assert len(profile.enabled_categories()) == 12

    def test_empty_profile_interests_in_range(self):
        profile = _empty_profile()
        for cat in profile.categories.values():
            for score in cat.interests.values():
                assert 0 <= score <= 5

    def test_category_interest_0_excluded(self):
        profile = _empty_profile()
        profile.categories["ai"].interests["robotics"] = 0
        assert profile.get_interest("ai", "robotics") == 0

    def test_unconfigured_subtopic_is_not_an_exclusion(self):
        profile = _empty_profile()
        del profile.categories["ai"].interests["robotics"]
        # An absent subtopic must not read as an explicit zero-interest setting.
        assert profile.get_interest("ai", "robotics") is None

    def test_category_interest_5_high_priority(self):
        profile = _empty_profile()
        profile.categories["ai"].interests["new_models"] = 5
        assert profile.get_interest("ai", "new_models") == 5

    def test_disabled_category_returns_0(self):
        profile = _empty_profile()
        profile.categories["ai"].enabled = False
        assert profile.get_interest("ai", "new_models") == 0

    def test_invalid_interest_raises(self):
        with pytest.raises(ValueError):
            Category(enabled=True, interests={"test": 6})

    def test_general_defaults(self):
        profile = _empty_profile()
        assert profile.general.detail_level == DetailLevel.normal
        assert profile.general.language_level == LanguageLevel.standard
        assert profile.general.reading_time == 10
        assert profile.general.frequency == Frequency.daily
        assert profile.general.priority == Priority.balanced
        assert profile.general.language == Language.ru
        assert profile.general.source_reliability == SourceReliability.balanced

    def test_version_default(self):
        profile = _empty_profile()
        assert profile.version == 1


# ── Serialization roundtrip ───────────────────────────────────────────────────


class TestSerialization:
    def test_roundtrip(self, tmp_path):
        profile = _empty_profile()
        profile.categories["ai"].interests["new_models"] = 5
        profile.categories["ai"].interests["robotics"] = 0
        profile.general.language = Language.en
        profile.general.exclusions = ["clickbait", "advertising"]

        path = tmp_path / "profile.json"
        save_profile(profile, path)

        loaded = load_profile(path)
        assert loaded.version == 1
        assert loaded.categories["ai"].interests["new_models"] == 5
        assert loaded.categories["ai"].interests["robotics"] == 0
        assert loaded.general.language == Language.en
        assert loaded.general.exclusions == ["clickbait", "advertising"]


# ── Loader ────────────────────────────────────────────────────────────────────


class TestLoader:
    def test_load_creates_default_when_missing(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        profile = load_profile(path)
        assert len(profile.categories) == 12
        assert path.exists()

    def test_load_invalid_file_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        with pytest.raises(ValueError, match="Invalid profile"):
            load_profile(path)

    def test_invalid_file_is_preserved(self, tmp_path):
        path = tmp_path / "bad.json"
        original = "not valid json {{{"
        path.write_text(original)
        with pytest.raises(ValueError):
            load_profile(path)
        assert path.exists()
        assert path.read_text() == original

    def test_load_valid_profile(self, tmp_path):
        data = {
            "version": 1,
            "categories": {
                "ai": {"enabled": True, "interests": {"new_models": 5}},
            },
            "general": {"language": "en"},
        }
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(data))

        profile = load_profile(path)
        assert profile.categories["ai"].interests["new_models"] == 5
        assert profile.general.language == Language.en


# ── Migration ─────────────────────────────────────────────────────────────────


class TestMigration:
    def test_migration_enables_matching_topics(self):
        config = {"topics": ["Vue.js ecosystem", "Nuxt.js framework", "AI tools"]}
        profile = _migration_profile_from_config(config)
        assert profile.categories["technology"].enabled is True
        assert profile.categories["ai"].enabled is True

    def test_migration_fallback_to_technology(self):
        config = {"topics": ["something unrelated"]}
        profile = _migration_profile_from_config(config)
        # At least technology should be enabled as fallback
        assert profile.categories["technology"].enabled is True


# ── Constants ─────────────────────────────────────────────────────────────────


class TestConstants:
    def test_all_categories_have_subtopics(self):
        for cat_id, subtopics in CATEGORIES.items():
            assert len(subtopics) == 6, f"Category {cat_id} should have 6 subtopics"

    def test_all_categories_have_labels(self):
        for cat_id in CATEGORIES:
            assert cat_id in CATEGORY_LABELS

    def test_category_count(self):
        assert len(CATEGORIES) == 12
        assert len(CATEGORY_LABELS) == 12
