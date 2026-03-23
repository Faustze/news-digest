"""
Optional: send latest digest to a Telegram chat.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
"""

import os
import glob
import httpx

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
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
        resp = httpx.post(API_URL, json={
            "chat_id":    CHAT_ID,
            "text":       part,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        })
        resp.raise_for_status()

if __name__ == "__main__":
    digest = latest_digest()
    send(digest)
    print("Digest sent to Telegram ✓")
