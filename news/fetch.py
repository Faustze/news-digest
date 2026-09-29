"""
RSS fetching: download feeds with a timeout and turn entries into raw items.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import feedparser
import httpx

from news.feedback import generate_news_id

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_WORKERS = 8
USER_AGENT = "news-digest/1.0 (+https://github.com/Faustze/news-digest)"


def download_feed(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bytes | None:
    """Download one feed. Returns None (and logs) on any network/HTTP failure."""
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
    except httpx.HTTPError as e:
        print(f"[WARN] Could not fetch {url}: {type(e).__name__}: {e}")
        return None
    if resp.is_error:
        print(f"[WARN] Could not fetch {url}: HTTP {resp.status_code}")
        return None
    return resp.content


def parse_entries(
    feed_cfg: dict,
    content: bytes,
    cutoff_hours: int,
    now: datetime | None = None,
) -> list[dict]:
    """Parse downloaded feed content into raw items newer than the cutoff."""
    now = now or datetime.now(timezone.utc)
    feed = feedparser.parse(content)

    if not feed.entries and feed.get("bozo"):
        print(
            f"[WARN] Empty feed {feed_cfg['url']}: "
            f"{getattr(feed, 'bozo_exception', 'parse error')}"
        )

    items = []
    for entry in feed.entries:
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published:
            pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
            age_hours = (now - pub_dt).total_seconds() / 3600
            if age_hours > cutoff_hours:
                continue

        title = entry.get("title", "")
        link = entry.get("link", "")

        items.append(
            {
                "news_id": generate_news_id(title, link),
                "title": title,
                "summary": re.sub(
                    r"<[^>]+>",
                    "",
                    entry.get("summary", entry.get("description", ""))[:600],
                ),
                "link": link,
                "source": feed_cfg.get("name", feed.feed.get("title", "Unknown")),
                "tags": feed_cfg.get("tags", []),
                "categories": feed_cfg.get("categories", []),
            }
        )
    return items


def fetch_rss_items(config: dict, cutoff_hours: int = 24) -> list[dict]:
    """
    Fetch raw entries from all configured feeds in parallel.

    A slow or broken feed costs at most `fetch_timeout` seconds and is skipped;
    items keep the order of feeds in config.yaml.
    """
    feeds = config["feeds"]
    timeout = float(config.get("fetch_timeout", DEFAULT_TIMEOUT_SECONDS))
    workers = int(config.get("fetch_workers", DEFAULT_WORKERS))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        contents = list(pool.map(lambda f: download_feed(f["url"], timeout), feeds))

    items = []
    for feed_cfg, content in zip(feeds, contents):
        if content is not None:
            items.extend(parse_entries(feed_cfg, content, cutoff_hours))
    return items
