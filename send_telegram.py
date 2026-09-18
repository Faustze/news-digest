"""
Send digest to Telegram with inline feedback buttons.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
"""

import html
import json
import os
import re
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from news.profile import category_id_from_label

# Telegram rejects callback_data over 64 *bytes* with BUTTON_DATA_INVALID.
CALLBACK_DATA_MAX_BYTES = 64


def _get_bot_token() -> str:
    return os.environ["TELEGRAM_BOT_TOKEN"].strip()


def _get_chat_id() -> str:
    return os.environ["TELEGRAM_CHAT_ID"].strip()


def _api_url() -> str:
    return f"https://api.telegram.org/bot{_get_bot_token()}"


def latest_digest(output_dir: str = "output") -> str:
    """Return today's digest text, raising if it has not been produced yet."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(output_dir, f"digest_{date_str}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No digest for {date_str}; nothing to send.")
    return open(path, encoding="utf-8").read()


def parse_items_from_digest(digest_text: str) -> list[dict]:
    """
    Parse news items from the rendered digest text.
    Extracts title, link, category, and news_id from the formatted output.
    """
    items = []
    lines = digest_text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match news item pattern: emoji [Title](link)
        match = re.match(r"^[🤖💻🔬🚀📱🎮💼💰🏃🎬🎵🌍📌]\s*\[(.+?)\]\((.+?)\)", line)
        if match:
            title = match.group(1)
            link = match.group(2)

            # Next line is summary, then category tag with news_id
            summary = ""
            category = ""
            news_id = ""

            if i + 1 < len(lines):
                summary = lines[i + 1].strip()
            if i + 2 < len(lines):
                tag_line = lines[i + 2].strip()
                # Parse #Category [#tag ...]  `news_id_prefix`
                cat_match = re.match(r"#(\S+)", tag_line)
                id_match = re.search(r"`(\w+)`", tag_line)
                if cat_match:
                    category = cat_match.group(1)
                if id_match:
                    news_id = id_match.group(1)

            items.append(
                {
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "category": category,
                    "news_id_prefix": news_id,
                }
            )
        i += 1

    return items


def build_callback_data(news_id: str, reaction: str, category: str) -> str:
    """Build compact callback_data for inline keyboard buttons.

    The category is sent as its ASCII id: a Cyrillic label, JSON-escaped to
    ``\\uXXXX``, overflows the 64-byte limit. If the payload still does not
    fit, the category is dropped rather than failing the whole message.
    """
    payload = {"id": news_id[:16], "r": reaction, "c": category_id_from_label(category)}
    data = json.dumps(payload, separators=(",", ":"))
    if len(data.encode()) > CALLBACK_DATA_MAX_BYTES:
        del payload["c"]
        data = json.dumps(payload, separators=(",", ":"))
    return data


def build_inline_keyboard(item: dict) -> dict:
    """Build an inline keyboard with feedback buttons for a news item."""
    news_id = item.get("news_id_prefix", "")
    category = item.get("category", "")

    buttons = [
        {
            "text": "👍 Полезно",
            "callback_data": build_callback_data(news_id, "useful", category),
        },
        {
            "text": "👎 Неинтересно",
            "callback_data": build_callback_data(news_id, "not_interesting", category),
        },
        {
            "text": "🔕 Больше такого",
            "callback_data": build_callback_data(news_id, "hide_similar", category),
        },
    ]

    return {"inline_keyboard": [buttons]}


TELEGRAM_MAX_LEN = 4096


def split_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Split text into chunks of at most ``limit`` chars on line boundaries.

    Lines longer than ``limit`` are hard-cut as a last resort.
    """
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return [c.strip("\n") for c in chunks if c.strip()]


def _escape_html(text: str) -> str:
    """Escape untrusted feed text for Telegram HTML (entities decoded first)."""
    return html.escape(html.unescape(text), quote=False)


def md_to_html(text: str) -> str:
    """Convert the LLM's light Markdown into Telegram-safe HTML.

    Telegram supports only a small tag subset, so headings become bold,
    horizontal rules are dropped and everything else is escaped.
    """
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"[-*_]{3,}", stripped):
            continue
        heading = re.match(r"^#{1,6}\s+(.*)$", stripped)
        line = _escape_html(heading.group(1) if heading else line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(
            r"(?<![*\w])\*(?!\s)([^*]+?)(?<!\s)\*(?![*\w])", r"<b>\1</b>", line
        )
        line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
        # The URL is already HTML-escaped along with the rest of the line.
        line = re.sub(
            r"\[([^\]]+)\]\((https?://[^)\s\"]+)\)", r'<a href="\2">\1</a>', line
        )
        out.append(f"<b>{line}</b>" if heading else line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


class TelegramError(RuntimeError):
    """Telegram Bot API rejected a request."""


def _describe(resp: httpx.Response) -> str:
    try:
        return f"{resp.status_code} {resp.json().get('description', '')}".strip()
    except ValueError:
        return f"{resp.status_code} {resp.text[:200]}".strip()


def send_message(
    text: str,
    parse_mode: str | None = "HTML",
    reply_markup: dict | None = None,
    plain_text: str | None = None,
):
    """Send a single message to Telegram.

    If Telegram rejects the formatting (400), retry once without
    ``parse_mode`` using ``plain_text`` (or ``text``), keeping the buttons.
    """
    payload = {
        "chat_id": _get_chat_id(),
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)

    resp = httpx.post(f"{_api_url()}/sendMessage", json=payload, timeout=30)
    if resp.status_code == 400 and parse_mode:
        print(f"[WARN] Telegram rejected formatted message: {_describe(resp)}")
        payload.pop("parse_mode", None)
        payload["text"] = plain_text or text
        resp = httpx.post(f"{_api_url()}/sendMessage", json=payload, timeout=30)
    if resp.is_error:
        # Do not use raise_for_status(): its message contains the bot token URL.
        raise TelegramError(f"sendMessage failed: {_describe(resp)}")
    return resp.json()


def send_digest(digest_text: str) -> None:
    """
    Send digest to Telegram, splitting into header + individual news items
    each with its own feedback keyboard.

    A failed message does not stop the rest of the delivery; the first error
    is re-raised at the end so the job still reports failure.
    """
    items = parse_items_from_digest(digest_text)

    # Do not deliver an empty digest just to satisfy a schedule.
    if not items:
        print("No news items in digest; nothing to send.")
        return

    # Find the header (everything before the first news item)
    header_lines = []
    for line in digest_text.split("\n"):
        if re.match(r"^[🤖💻🔬🚀📱🎮💼💰🏃🎬🎵🌍📌]\s*\[", line.strip()):
            break
        header_lines.append(line)
    header = "\n".join(header_lines).strip()

    errors: list[TelegramError] = []

    # Telegram caps a message at 4096 chars, and the LLM summary can exceed it.
    for part in split_message(header):
        try:
            send_message(md_to_html(part), plain_text=part)
        except TelegramError as e:
            print(f"[ERROR] Header part not sent: {e}")
            errors.append(e)

    # Send each news item with its own feedback keyboard
    for item in items:
        title = _escape_html(item["title"])
        summary = _escape_html(item["summary"])
        link = html.escape(item["link"])
        text = f'<b>{title}</b>\n\n{summary}\n\n<a href="{link}">Читать источник</a>'
        plain = f"{item['title']}\n\n{item['summary']}\n\n{item['link']}"
        try:
            send_message(
                text, reply_markup=build_inline_keyboard(item), plain_text=plain
            )
        except TelegramError as e:
            print(f"[ERROR] Item {item['news_id_prefix']} not sent: {e}")
            errors.append(e)

    if errors:
        raise errors[0]


if __name__ == "__main__":
    load_dotenv()
    try:
        digest = latest_digest()
    except FileNotFoundError as e:
        print(e)
        print("Skipping delivery: no digest was produced for today.")
        raise SystemExit(0)
    send_digest(digest)
    print("Digest sent to Telegram ✓")
