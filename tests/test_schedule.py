"""
Tests for news.schedule module.
"""

import pytest

from news.profile import Frequency, UserProfile
from news.schedule import (
    cutoff_hours_for_frequency,
    digest_filename,
    slot_from_env,
    slots_for_frequency,
)


def _profile_with_frequency(frequency: Frequency | None = None) -> UserProfile:
    profile = UserProfile()
    if frequency is not None:
        profile.general.frequency = frequency
    return profile


class TestCutoffHoursForFrequency:
    def test_daily(self):
        assert (
            cutoff_hours_for_frequency(_profile_with_frequency(Frequency.daily)) == 24
        )

    def test_morning(self):
        assert (
            cutoff_hours_for_frequency(_profile_with_frequency(Frequency.morning)) == 24
        )

    def test_evening(self):
        assert (
            cutoff_hours_for_frequency(_profile_with_frequency(Frequency.evening)) == 24
        )

    def test_weekly(self):
        assert (
            cutoff_hours_for_frequency(_profile_with_frequency(Frequency.weekly)) == 168
        )

    def test_important_only(self):
        assert (
            cutoff_hours_for_frequency(
                _profile_with_frequency(Frequency.important_only)
            )
            == 24
        )

    def test_missing_frequency_defaults_to_24(self):
        assert cutoff_hours_for_frequency(_profile_with_frequency()) == 24


def test_twice_daily_cutoff_is_about_half_a_day():
    profile = _profile_with_frequency(Frequency.twice_daily)
    assert 12 <= cutoff_hours_for_frequency(profile) < 24


class TestSlotsForFrequency:
    @pytest.mark.parametrize(
        ("frequency", "slots"),
        [
            (Frequency.twice_daily, {"morning", "evening"}),
            (Frequency.evening, {"evening"}),
            (Frequency.morning, {"morning"}),
            (Frequency.daily, {"morning"}),
            (Frequency.weekly, {"morning"}),
            (Frequency.important_only, {"morning"}),
        ],
    )
    def test_slots(self, frequency, slots):
        assert slots_for_frequency(_profile_with_frequency(frequency)) == slots


class TestSlotFromEnv:
    def test_unset_means_manual_run(self, monkeypatch):
        monkeypatch.delenv("DIGEST_SLOT", raising=False)
        assert slot_from_env() is None

    def test_normalizes_case(self, monkeypatch):
        monkeypatch.setenv("DIGEST_SLOT", " Evening ")
        assert slot_from_env() == "evening"

    def test_rejects_unknown_slot(self, monkeypatch):
        monkeypatch.setenv("DIGEST_SLOT", "noon")
        with pytest.raises(ValueError, match="DIGEST_SLOT"):
            slot_from_env()


def test_digest_filename():
    assert digest_filename("2026-09-29", None) == "digest_2026-09-29.txt"
    assert digest_filename("2026-09-29", "evening") == "digest_2026-09-29_evening.txt"
