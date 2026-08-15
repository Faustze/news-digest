"""
Tests for send_telegram module.
"""

import json
from datetime import datetime, timezone

import pytest

from send_telegram import (
    _escape_markdown,
    build_callback_data,
    build_inline_keyboard,
    latest_digest,
    parse_items_from_digest,
)

SAMPLE_DIGEST = """📰 *Дайджест 14.08.2026*

Сегодня важные новости в мире технологий.

─────────────────────

🤖 [New AI Model Released](https://example.com/ai)
OpenAI выпустила новую модель GPT-5.
#AI  `abc123def456`

💻 [Web Framework Update](https://example.com/web)
Вышел обновлённый фреймворк для веб-разработки.
#Technology  `xyz789ghi012`

_Источников: 2 · Новостей: 2_"""


class TestParseItemsFromDigest:
    def test_parses_items_correctly(self):
        items = parse_items_from_digest(SAMPLE_DIGEST)
        assert len(items) == 2

    def test_extracts_title(self):
        items = parse_items_from_digest(SAMPLE_DIGEST)
        assert items[0]["title"] == "New AI Model Released"
        assert items[1]["title"] == "Web Framework Update"

    def test_extracts_link(self):
        items = parse_items_from_digest(SAMPLE_DIGEST)
        assert items[0]["link"] == "https://example.com/ai"
        assert items[1]["link"] == "https://example.com/web"

    def test_extracts_category(self):
        items = parse_items_from_digest(SAMPLE_DIGEST)
        assert items[0]["category"] == "AI"
        assert items[1]["category"] == "Technology"

    def test_extracts_news_id_prefix(self):
        items = parse_items_from_digest(SAMPLE_DIGEST)
        assert items[0]["news_id_prefix"] == "abc123def456"
        assert items[1]["news_id_prefix"] == "xyz789ghi012"

    def test_empty_digest(self):
        items = parse_items_from_digest("")
        assert len(items) == 0


class TestBuildCallbackData:
    def test_compact_json(self):
        data = build_callback_data("abc123", "useful", "AI")
        parsed = json.loads(data)
        assert parsed["id"] == "abc123"
        assert parsed["r"] == "useful"
        assert parsed["c"] == "AI"

    def test_within_telegram_limit(self):
        data = build_callback_data("a" * 64, "not_interesting", "Technology")
        assert len(data) <= 64


class TestBuildInlineKeyboard:
    def test_has_three_buttons(self):
        item = {
            "title": "Test",
            "link": "https://example.com",
            "category": "AI",
            "news_id_prefix": "abc123",
        }
        keyboard = build_inline_keyboard(item)
        rows = keyboard["inline_keyboard"]
        assert isinstance(rows, list) and len(rows) == 1
        assert len(rows[0]) == 3

    def test_button_texts(self):
        item = {
            "title": "Test",
            "link": "https://example.com",
            "category": "AI",
            "news_id_prefix": "abc123",
        }
        keyboard = build_inline_keyboard(item)
        texts = [b["text"] for b in keyboard["inline_keyboard"][0]]
        assert "👍 Полезно" in texts
        assert "👎 Неинтересно" in texts
        assert "🔕 Больше такого" in texts


class TestEscapeMarkdown:
    def test_escapes_special_chars(self):
        assert _escape_markdown("a_b *c* [d]") == r"a\_b \*c\* \[d\]"

    def test_leaves_plain_text(self):
        assert _escape_markdown("plain text 123") == "plain text 123"

    def test_escapes_link_brackets(self):
        assert (
            _escape_markdown("[title](https://x.com)") == r"\[title\]\(https://x\.com\)"
        )


class TestLatestDigest:
    def test_raises_when_no_digest_today(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            latest_digest(str(tmp_path))

    def test_reads_today_digest(self, tmp_path):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        f = tmp_path / f"digest_{date_str}.txt"
        f.write_text("hello", encoding="utf-8")
        assert latest_digest(str(tmp_path)) == "hello"
