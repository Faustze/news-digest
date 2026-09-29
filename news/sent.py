"""
Sent-news log: remember which items already went out, so the next digest
(e.g. the evening one after the morning one) does not repeat them.

Stored compactly as {news_id prefix: ISO date} and pruned after a few days;
the workflow keeps the file on the digest-data branch with the feedback state.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

DEFAULT_SENT_PATH = Path("sent_news.json")
ID_PREFIX = 16
KEEP_DAYS = 3


def _key(news_id: str) -> str:
    return news_id[:ID_PREFIX]


def load_sent(path: Path | str = DEFAULT_SENT_PATH) -> dict[str, str]:
    """Load the log; a missing or corrupt file means nothing was sent yet."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Ignoring unreadable {p}: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def drop_already_sent(items: list[dict], sent: dict[str, str]) -> list[dict]:
    return [item for item in items if _key(item.get("news_id", "")) not in sent]


def record_sent(
    sent: dict[str, str], items: list[dict], today: date, keep_days: int = KEEP_DAYS
) -> dict[str, str]:
    """Add today's items and drop entries older than `keep_days`."""
    oldest = (today - timedelta(days=keep_days)).isoformat()
    updated = {k: v for k, v in sent.items() if v >= oldest}
    for item in items:
        if item.get("news_id"):
            updated[_key(item["news_id"])] = today.isoformat()
    return updated


def save_sent(sent: dict[str, str], path: Path | str = DEFAULT_SENT_PATH) -> None:
    Path(path).write_text(
        json.dumps(sent, ensure_ascii=False, indent=0, sort_keys=True) + "\n",
        encoding="utf-8",
    )
