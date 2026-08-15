"""
Tests for news.rank module.
"""

from news.feedback import HIDE_SIMILAR, USEFUL, FeedbackStore, add_reaction
from news.profile import _empty_profile
from news.rank import rank_item, rank_items


class TestRankItem:
    def test_accepted_item_scores_positive(self):
        profile = _empty_profile()
        item = {
            "news_id": "abc",
            "title": "Test",
            "category": "ai",
            "subtopics": ["new_models"],
            "accepted": True,
            "importance": 0.8,
            "source": "TechCrunch",
        }
        score = rank_item(item, profile)
        assert score > 0

    def test_rejected_item_scores_zero(self):
        profile = _empty_profile()
        item = {
            "news_id": "abc",
            "accepted": False,
            "category": "ai",
            "subtopics": ["new_models"],
        }
        score = rank_item(item, profile)
        assert score == 0.0

    def test_excluded_subtopic_scores_zero(self):
        profile = _empty_profile()
        profile.categories["ai"].interests["new_models"] = 0
        item = {
            "news_id": "abc",
            "accepted": True,
            "category": "ai",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        score = rank_item(item, profile)
        assert score == 0.0

    def test_disabled_category_scores_zero(self):
        profile = _empty_profile()
        profile.categories["ai"].enabled = False
        item = {
            "news_id": "abc",
            "accepted": True,
            "category": "ai",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        score = rank_item(item, profile)
        assert score == 0.0

    def test_unknown_category_scores_zero(self):
        profile = _empty_profile()
        item = {
            "news_id": "abc",
            "accepted": True,
            "category": "not_a_category",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        score = rank_item(item, profile)
        assert score == 0.0

    def test_missing_accepted_is_rejected(self):
        profile = _empty_profile()
        item = {
            "news_id": "abc",
            "category": "ai",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        assert rank_item(item, profile) == 0.0

    def test_general_exclusion_filters_item(self):
        profile = _empty_profile()
        profile.general.exclusions = ["clickbait"]
        item = {
            "news_id": "abc",
            "accepted": True,
            "category": "ai",
            "subtopics": ["new_models"],
            "title": "Some clickbait title",
            "summary": "Body text",
            "importance": 0.9,
            "source": "Reuters",
        }
        assert rank_item(item, profile) == 0.0

    def test_unconfigured_subtopic_uses_neutral_interest(self):
        profile = _empty_profile()
        del profile.categories["ai"].interests["new_models"]
        item = {
            "news_id": "abc",
            "accepted": True,
            "category": "ai",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        assert rank_item(item, profile) > 0

    def test_high_interest_scores_higher(self):
        profile = _empty_profile()
        profile.categories["ai"].interests["new_models"] = 5
        item_high = {
            "news_id": "abc",
            "accepted": True,
            "category": "ai",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        score_high = rank_item(item_high, profile)

        profile.categories["ai"].interests["new_models"] = 1
        item_low = {
            "news_id": "abc",
            "accepted": True,
            "category": "ai",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        score_low = rank_item(item_low, profile)

        assert score_high > score_low

    def test_feedback_useful_increases_score(self):
        profile = _empty_profile()
        feedback = FeedbackStore()
        add_reaction(feedback, "xyz", USEFUL, "ai", ["new_models"])

        item = {
            "news_id": "abc",
            "accepted": True,
            "category": "ai",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        score_with = rank_item(item, profile, feedback)
        score_without = rank_item(item, profile, None)
        assert score_with > score_without

    def test_feedback_hide_similar_decreases_score(self):
        profile = _empty_profile()
        feedback = FeedbackStore()
        add_reaction(feedback, "xyz", HIDE_SIMILAR, "ai", ["new_models"])

        item = {
            "news_id": "abc",
            "accepted": True,
            "category": "ai",
            "subtopics": ["new_models"],
            "importance": 0.5,
            "source": "Test",
        }
        score_with = rank_item(item, profile, feedback)
        score_without = rank_item(item, profile, None)
        assert score_with < score_without


class TestRankItems:
    def test_returns_sorted_by_score(self):
        profile = _empty_profile()
        items = [
            {
                "news_id": "a",
                "accepted": True,
                "category": "ai",
                "subtopics": ["new_models"],
                "importance": 0.9,
                "source": "Reuters",
            },
            {
                "news_id": "b",
                "accepted": True,
                "category": "ai",
                "subtopics": ["robotics"],
                "importance": 0.3,
                "source": "Unknown Blog",
            },
        ]
        ranked = rank_items(items, profile)
        assert len(ranked) == 2
        assert ranked[0]["final_score"] >= ranked[1]["final_score"]

    def test_filters_out_zero_scores(self):
        profile = _empty_profile()
        profile.categories["ai"].interests["new_models"] = 0
        items = [
            {
                "news_id": "a",
                "accepted": True,
                "category": "ai",
                "subtopics": ["new_models"],
                "importance": 0.9,
                "source": "Test",
            },
        ]
        ranked = rank_items(items, profile)
        assert len(ranked) == 0
