"""
Schedule-related helpers: derive pipeline timing knobs from the user profile.
"""

from __future__ import annotations

from news.profile import Frequency, UserProfile


def cutoff_hours_for_frequency(profile: UserProfile) -> int:
    """
    Map the profile's digest frequency to a news-age cutoff in hours.

    A weekly digest looks back seven days; every other frequency (including
    a missing/unset one, which defaults to `daily`) gets a 24-hour window.
    """
    if profile.general.frequency == Frequency.weekly:
        return 7 * 24
    return 24
