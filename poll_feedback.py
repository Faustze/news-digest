"""
Poll Telegram for inline feedback callbacks using getUpdates.
No webhook server needed — runs during GitHub Actions.

Reads TELEGRAM_BOT_TOKEN from environment.
Uses last_update_id to avoid processing the same update twice.
"""

import json
import os
from pathlib import Path

import httpx

from news.feedback import (
    add_reaction,
    load_feedback,
    save_feedback,
)

FEEDBACK_PATH = Path("feedback.json")
STATE_PATH = Path("feedback_state.json")


def _api_url() -> str:
    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    return f"https://api.telegram.org/bot{token}"


def _load_state() -> int:
    """Load the last processed update_id."""
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return data.get("last_update_id", 0)
        except (json.JSONDecodeError, ValueError):
            return 0
    return 0


def _save_state(update_id: int) -> None:
    """Save the last processed update_id."""
    STATE_PATH.write_text(
        json.dumps({"last_update_id": update_id}),
        encoding="utf-8",
    )


def _parse_callback_data(callback_data: str) -> dict | None:
    """
    Parse callback_data from inline keyboard.
    Format: {"id":"news_id_prefix","r":"reaction","c":"category"}
    """
    try:
        payload = json.loads(callback_data)
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def poll_updates() -> tuple[list[dict], int]:
    """
    Poll Telegram for new callback_query updates.

    Returns a tuple of (reactions, max_update_id). The caller must persist
    the cursor only after the returned reactions have been applied.
    """
    last_id = _load_state()
    reactions = []

    try:
        resp = httpx.get(
            f"{_api_url()}/getUpdates",
            params={
                "offset": last_id + 1,
                "timeout": 5,
                "allowed_updates": json.dumps(["callback_query"]),
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        print(f"[WARN] Failed to poll Telegram: {e}")
        return reactions, last_id

    if not data.get("ok"):
        print(f"[WARN] Telegram API error: {data}")
        return reactions, last_id

    updates = data.get("result", [])
    max_update_id = last_id

    for update in updates:
        update_id = update.get("update_id", 0)
        max_update_id = max(max_update_id, update_id)

        callback_query = update.get("callback_query")
        if not callback_query:
            continue

        callback_data = callback_query.get("data", "")
        parsed = _parse_callback_data(callback_data)
        if not parsed:
            continue

        # Answer the callback query to remove loading indicator
        query_id = callback_query.get("id")
        if query_id:
            try:
                httpx.post(
                    f"{_api_url()}/answerCallbackQuery",
                    json={"callback_query_id": query_id},
                    timeout=5,
                )
            except httpx.HTTPError:
                pass

        reactions.append(parsed)

    return reactions, max_update_id


def apply_reactions(reactions: list[dict]) -> None:
    """Apply parsed reactions to the feedback store."""
    if not reactions:
        return

    store = load_feedback(FEEDBACK_PATH)

    for r in reactions:
        news_id = r.get("id", "")
        reaction = r.get("r", "")
        category = r.get("c", "")

        if not news_id or not reaction:
            continue

        # Map category label back to ID if needed
        category_id = category.lower()

        try:
            add_reaction(
                store,
                news_id=news_id,
                reaction=reaction,
                category=category_id,
            )
            print(f"  + Reaction: {news_id[:12]}... → {reaction} ({category_id})")
        except ValueError as e:
            print(f"  [WARN] Invalid reaction: {e}")

    save_feedback(store, FEEDBACK_PATH)
    print(f"  Total reactions in store: {len(store.reactions)}")


def main() -> None:
    print("[feedback] Polling Telegram for reactions…")
    reactions, max_update_id = poll_updates()
    print(f"  Found {len(reactions)} new reaction(s)")

    if max_update_id <= _load_state():
        print("  No new updates")
        return

    # Advance the poll cursor only after the reactions were persisted, so a
    # failure in apply_reactions does not lose feedback permanently.
    try:
        if reactions:
            apply_reactions(reactions)
        _save_state(max_update_id)
    except OSError as e:
        print(f"  [WARN] Failed to persist reactions: {e}. Cursor not advanced.")


if __name__ == "__main__":
    main()
