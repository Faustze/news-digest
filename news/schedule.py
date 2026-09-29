"""
Schedule-related helpers: derive pipeline timing knobs from the user profile.

The workflow runs twice a day and tells the pipeline which run this is via
the DIGEST_SLOT environment variable ("morning" or "evening"). The profile's
frequency decides which slots actually produce a digest; the other runs exit
before fetching anything, so they cost no LLM tokens.
"""

from __future__ import annotations

import os

from news.profile import Frequency, UserProfile

MORNING = "morning"
EVENING = "evening"
SLOTS = (MORNING, EVENING)

# Twice-daily runs are ~11 and ~13 hours apart (plus GitHub cron delays), so
# each run looks back a little further; overlap is removed by the sent-news log.
TWICE_DAILY_CUTOFF_HOURS = 14


def cutoff_hours_for_frequency(profile: UserProfile) -> int:
    """
    Map the profile's digest frequency to a news-age cutoff in hours.

    A weekly digest looks back seven days, a twice-daily one ~half a day;
    every other frequency (including the default `daily`) gets 24 hours.
    """
    if profile.general.frequency == Frequency.weekly:
        return 7 * 24
    if profile.general.frequency == Frequency.twice_daily:
        return TWICE_DAILY_CUTOFF_HOURS
    return 24


def slots_for_frequency(profile: UserProfile) -> set[str]:
    """Which scheduled runs produce a digest for this profile."""
    frequency = profile.general.frequency
    if frequency == Frequency.twice_daily:
        return {MORNING, EVENING}
    if frequency == Frequency.evening:
        return {EVENING}
    return {MORNING}


def slot_from_env() -> str | None:
    """
    The current run's slot from DIGEST_SLOT, or None for a manual local run
    (which always produces a digest).
    """
    slot = os.environ.get("DIGEST_SLOT", "").strip().lower()
    if not slot:
        return None
    if slot not in SLOTS:
        raise ValueError(f"DIGEST_SLOT must be one of {SLOTS}, got {slot!r}")
    return slot


def digest_filename(date_str: str, slot: str | None) -> str:
    """`digest_2026-09-29.txt` for local runs, `digest_2026-09-29_evening.txt` per slot."""
    suffix = f"_{slot}" if slot else ""
    return f"digest_{date_str}{suffix}.txt"
