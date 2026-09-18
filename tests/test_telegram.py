"""
Tests for send_telegram module.
"""

import json
from datetime import datetime, timezone

import pytest

from send_telegram import (
    TELEGRAM_MAX_LEN,
    build_callback_data,
    build_inline_keyboard,
    latest_digest,
    md_to_html,
    parse_items_from_digest,
    split_message,
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

    def test_parses_items_with_tags(self):
        digest = """📰 *Дайджест 14.08.2026*

🤖 [New AI Model Released](https://example.com/ai)
OpenAI выпустила новую модель.
#AI  #ml  #research  `abc123def456`

_Источников: 1 · Новостей: 1_"""
        items = parse_items_from_digest(digest)
        assert len(items) == 1
        assert items[0]["category"] == "AI"
        assert items[0]["news_id_prefix"] == "abc123def456"

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


class TestMdToHtml:
    def test_escapes_html_special_chars(self):
        assert md_to_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"

    def test_bold(self):
        assert md_to_html("**важно** и *тоже*") == "<b>важно</b> и <b>тоже</b>"

    def test_heading_becomes_bold(self):
        assert md_to_html("## 1. AI") == "<b>1. AI</b>"

    def test_drops_horizontal_rules(self):
        assert md_to_html("a\n\n---\n\nb") == "a\n\nb"

    def test_list_marker_is_not_bold(self):
        assert md_to_html("* item one\n* item two") == "* item one\n* item two"

    def test_link(self):
        assert (
            md_to_html("[x](https://e.com/?a=1&b=2)")
            == '<a href="https://e.com/?a=1&amp;b=2">x</a>'
        )

    def test_decodes_feed_entities_once(self):
        assert md_to_html("Q&amp;A") == "Q&amp;A"


class TestSplitMessage:
    def test_short_text_is_single_chunk(self):
        assert split_message("hello\nworld") == ["hello\nworld"]

    def test_chunks_respect_limit_and_keep_lines(self):
        lines = [f"line {i} " + "x" * 50 for i in range(200)]
        chunks = split_message("\n".join(lines))
        assert len(chunks) > 1
        assert all(len(c) <= TELEGRAM_MAX_LEN for c in chunks)
        assert "\n".join(chunks).split("\n") == lines

    def test_hard_cuts_overlong_line(self):
        chunks = split_message("y" * 10000, limit=4096)
        assert [len(c) for c in chunks] == [4096, 4096, 1808]


class TestLatestDigest:
    def test_raises_when_no_digest_today(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            latest_digest(str(tmp_path))

    def test_reads_today_digest(self, tmp_path):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        f = tmp_path / f"digest_{date_str}.txt"
        f.write_text("hello", encoding="utf-8")
        assert latest_digest(str(tmp_path)) == "hello"


class TestSendMessage:
    def test_400_fallback_keeps_reply_markup(self, monkeypatch):
        import httpx

        import send_telegram

        req = httpx.Request("POST", "https://api.telegram.org/x")
        responses = [
            httpx.Response(400, text="bad", request=req),
            httpx.Response(200, json={"ok": True}, request=req),
        ]
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(dict(json))
            return responses.pop(0)

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
        monkeypatch.setattr(send_telegram.httpx, "post", fake_post)

        keyboard = {"inline_keyboard": [[{"text": "👍", "callback_data": "x"}]]}
        send_telegram.send_message(
            "<b>hello</b>", reply_markup=keyboard, plain_text="hello"
        )

        assert calls[0]["parse_mode"] == "HTML"
        assert "parse_mode" not in calls[1]
        assert calls[1]["text"] == "hello"
        assert calls[1]["reply_markup"] == json.dumps(keyboard)

    def test_error_does_not_leak_token(self, monkeypatch):
        import httpx

        import send_telegram

        req = httpx.Request("POST", "https://api.telegram.org/botSECRET/x")
        resp = httpx.Response(
            400,
            json={"ok": False, "description": "Bad Request: message is too long"},
            request=req,
        )
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRET")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
        monkeypatch.setattr(send_telegram.httpx, "post", lambda *a, **k: resp)

        with pytest.raises(send_telegram.TelegramError) as exc:
            send_telegram.send_message("hello")
        assert "message is too long" in str(exc.value)
        assert "SECRET" not in str(exc.value)


class TestSendDigest:
    def test_long_header_is_split_and_items_sent(self, monkeypatch):
        import send_telegram

        sent = []
        monkeypatch.setattr(
            send_telegram,
            "send_message",
            lambda text, **kw: sent.append((text, kw.get("reply_markup"))),
        )
        long_summary = "\n".join("- пункт " + "я" * 80 for _ in range(100))
        digest = SAMPLE_DIGEST.replace(
            "Сегодня важные новости в мире технологий.", long_summary
        )

        send_telegram.send_digest(digest)

        header_parts = [t for t, kb in sent if kb is None]
        item_msgs = [t for t, kb in sent if kb is not None]
        assert len(header_parts) >= 2
        assert all(len(t) <= TELEGRAM_MAX_LEN for t in header_parts)
        assert len(item_msgs) == 2
        assert '<a href="https://example.com/ai">' in item_msgs[0]
