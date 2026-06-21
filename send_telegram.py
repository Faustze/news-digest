"""
Optional: send latest digest to a Telegram chat.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
"""

import os
import glob
import httpx


BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"].strip()
API_URL   = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

def latest_digest() -> str:
    files = sorted(glob.glob("output/digest_*.txt") + glob.glob("output/digest_*.md"))
    if not files:
        raise FileNotFoundError("No digest file found in output/")
    return open(files[-1]).read()

def chunk(text: str, size: int = 4000):
    """Telegram messages max out at 4096 chars."""
    for i in range(0, len(text), size):
        yield text[i : i + size]

def send(text: str, parse_mode: str = "Markdown"):
    for part in chunk(text):
        payload = {
            "chat_id":    CHAT_ID,
            "text":       part,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        resp = httpx.post(API_URL, json=payload)
        if resp.status_code == 400 and parse_mode:
            # Telegram couldn't parse the message (e.g. invalid Markdown).
            # Retry without parse_mode as plain text.
            payload.pop("parse_mode", None)
            resp = httpx.post(API_URL, json=payload)
        resp.raise_for_status()

if __name__ == "__main__":
    digest = latest_digest()
    send(digest)
    print("Digest sent to Telegram ✓")
