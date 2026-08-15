"""
Send digest to Telegram with inline feedback buttons.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
"""

import json
import os
import re
from datetime import datetime, timezone

import httpx


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
                # Parse #Category  `news_id_prefix`
                cat_match = re.match(r"#(\S+)\s+`(\w+)`", tag_line)
                if cat_match:
                    category = cat_match.group(1)
                    news_id = cat_match.group(2)

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
    """Build compact callback_data for inline keyboard buttons."""
    payload = {"id": news_id[:16], "r": reaction, "c": category}
    return json.dumps(payload, separators=(",", ":"))


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


def chunk(text: str, size: int = 4000):
    """Telegram messages max out at 4096 chars."""
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _escape_markdown(text: str) -> str:
    """Escape Telegram Markdown special characters in untrusted feed text."""
    for ch in "_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def send_message(
    text: str,
    parse_mode: str = "Markdown",
    reply_markup: dict | None = None,
):
    """Send a single message to Telegram."""
    payload = {
        "chat_id": _get_chat_id(),
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)

    resp = httpx.post(f"{_api_url()}/sendMessage", json=payload)
    if resp.status_code == 400 and parse_mode:
        payload.pop("parse_mode", None)
        payload.pop("reply_markup", None)
        resp = httpx.post(f"{_api_url()}/sendMessage", json=payload)
    resp.raise_for_status()
    return resp.json()


def send_digest(digest_text: str):
    """
    Send digest to Telegram, splitting into header + individual news items
    each with its own feedback keyboard.
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

    # Send header
    if header:
        send_message(header, parse_mode="Markdown")

    # Send each news item with its own feedback keyboard
    for item in items:
        title = _escape_markdown(item["title"])
        summary = _escape_markdown(item["summary"])
        text = f"{title}\n\n{summary}\n\n[Читать источник]({item['link']})"
        keyboard = build_inline_keyboard(item)
        send_message(text, parse_mode="Markdown", reply_markup=keyboard)


if __name__ == "__main__":
    try:
        digest = latest_digest()
    except FileNotFoundError as e:
        print(e)
        print("Skipping delivery: no digest was produced for today.")
        raise SystemExit(0)
    send_digest(digest)
    print("Digest sent to Telegram ✓")
