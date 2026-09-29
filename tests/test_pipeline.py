"""
Tests for news.pipeline orchestration that need no network or LLM.
"""

import asyncio
import json

import pytest

from news import pipeline
from news.profile import UserProfile


@pytest.fixture
def profile_path(tmp_path):
    def write(frequency: str):
        profile = UserProfile().model_dump(mode="json")
        profile["general"]["frequency"] = frequency
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(profile))
        return str(path)

    return write


@pytest.fixture
def no_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("skipped slot must not fetch feeds or build the LLM")

    monkeypatch.setattr(pipeline, "build_llm", boom)
    monkeypatch.setattr(pipeline, "fetch_rss_items", boom)


@pytest.mark.parametrize(
    ("frequency", "slot"),
    [("daily", "evening"), ("evening", "morning"), ("morning", "evening")],
)
def test_run_outside_profile_slots_exits_before_llm(
    tmp_path, profile_path, no_network, frequency, slot
):
    result = asyncio.run(
        pipeline.run_pipeline(
            "config.yaml",
            profile_path(frequency),
            str(tmp_path / "feedback.json"),
            str(tmp_path / "sent.json"),
            slot=slot,
        )
    )
    assert result == ""
    assert not (tmp_path / "sent.json").exists()


def test_render_title_per_slot():
    profile = UserProfile()
    assert "Утренний дайджест" in pipeline.render_telegram([], "s", profile, "morning")
    assert "Вечерний дайджест" in pipeline.render_telegram([], "s", profile, "evening")
    assert "📰 *Дайджест" in pipeline.render_telegram([], "s", profile)
