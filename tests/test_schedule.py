"""
Tests for news.schedule module.
"""

from news.profile import Frequency, UserProfile
from news.schedule import cutoff_hours_for_frequency


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
