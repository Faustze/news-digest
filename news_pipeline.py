"""
News Feed Pipeline — LangChain + Groq (free tier)
Fetches, filters, classifies, ranks, and summarises news from RSS feeds.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from news.classify import classify_batch
from news.deduplicate import deduplicate
from news.feedback import generate_news_id, load_feedback
from news.profile import CATEGORY_LABELS, UserProfile, load_profile
from news.rank import rank_items

# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Step 1: Fetch RSS items ───────────────────────────────────────────────────


def fetch_rss_items(config: dict) -> list[dict]:
    """Fetch raw entries from all configured RSS feeds."""
    items = []
    cutoff_hours = config.get("cutoff_hours", 24)

    for feed_cfg in config["feeds"]:
        try:
            feed = feedparser.parse(feed_cfg["url"])
        except (OSError, KeyError) as e:
            print(f"[WARN] Could not fetch {feed_cfg['url']}: {e}")
            continue

        for entry in feed.entries:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
                if age_hours > cutoff_hours:
                    continue

            title = entry.get("title", "")
            link = entry.get("link", "")
            news_id = generate_news_id(title, link)

            items.append(
                {
                    "news_id": news_id,
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


# ── Step 2: Executive summary ─────────────────────────────────────────────────

DIGEST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты пишешь краткий ежедневный дайджест новостей.
Язык: {language}.
Уровень детализации: {detail_level}.
Уровень языка: {language_level}.

Напиши 3-4 предложения: что важного произошло сегодня.
Будь конкретным и полезным. Без воды.""",
        ),
        ("human", "Топ новостей:\n{items_json}"),
    ]
)


async def generate_digest_summary(
    items: list[dict],
    profile: UserProfile,
    llm: ChatGroq,
) -> str:
    chain = DIGEST_PROMPT | llm | StrOutputParser()
    language = "русский" if profile.general.language.value == "ru" else "English"
    detail_map = {"short": "кратко", "normal": "обычно", "detailed": " подробно"}
    lang_map = {
        "simple": "простыми словами",
        "standard": "обычный уровень",
        "advanced": "технический язык",
    }

    return await chain.ainvoke(
        {
            "items_json": json.dumps(
                [
                    {"title": i["title"], "summary": i.get("summary", "")}
                    for i in items[:10]
                ],
                ensure_ascii=False,
            ),
            "language": language,
            "detail_level": detail_map.get(
                profile.general.detail_level.value, "обычно"
            ),
            "language_level": lang_map.get(
                profile.general.language_level.value, "обычный уровень"
            ),
        }
    )


# ── Step 3: Render Telegram message ──────────────────────────────────────────

CATEGORY_EMOJI = {
    "ai": "🤖",
    "technology": "💻",
    "science": "🔬",
    "space": "🚀",
    "gadgets": "📱",
    "games": "🎮",
    "business": "💼",
    "finance": "💰",
    "running": "🏃",
    "movies": "🎬",
    "music": "🎵",
    "world": "🌍",
}


def render_telegram(items: list[dict], summary: str, profile: UserProfile) -> str:
    date_str = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    lines = [
        f"📰 *Дайджест {date_str}*",
        "",
        summary,
        "",
        "─────────────────────",
        "",
    ]

    for item in items:
        category = item.get("category", "")
        emoji = CATEGORY_EMOJI.get(category, "📌")
        title = item.get("title", "").strip()
        link = item.get("link", "")
        text = item.get("summary", "")[:280].strip()
        cat_label = CATEGORY_LABELS.get(category, category)
        news_id = item.get("news_id", "")

        lines += [
            f"{emoji} [{title}]({link})",
            text,
            f"#{cat_label}  `{news_id[:12]}`",
            "",
        ]

    lines.append(
        f"_Источников: {len({i['source'] for i in items})} · Новостей: {len(items)}_"
    )
    return "\n".join(lines)


# ── Reading time budget ───────────────────────────────────────────────────────


def _reading_time_budget(profile: UserProfile) -> int:
    """Estimate how many items fit in the reading time budget."""
    reading_time = profile.general.reading_time
    # Rough estimate: 2-3 min per item
    if reading_time <= 5:
        return 5
    if reading_time <= 10:
        return 8
    if reading_time <= 20:
        return 12
    return 15


# ── Orchestrator ──────────────────────────────────────────────────────────────


async def run_pipeline(
    config_path: str = "config.yaml",
    profile_path: str = "user-profile.json",
    feedback_path: str = "feedback.json",
) -> str:
    config = load_config(config_path)
    profile = load_profile(profile_path, config)
    feedback = load_feedback(feedback_path)

    llm = ChatGroq(
        model=config.get("model", "llama-3.3-70b-versatile"),
        temperature=0,
        max_tokens=4096,
    )
    batch_size = config.get("batch_size", 12)

    print(f"[1/5] Fetching RSS feeds ({len(config['feeds'])} sources)…")
    raw_items = fetch_rss_items(config)
    print(f"      → {len(raw_items)} raw items")

    print("[2/5] Deduplicating…")
    unique_items = deduplicate(raw_items)
    print(f"      → {len(unique_items)} unique items")

    print("[3/5] Classifying with Groq…")
    classified = await classify_batch(unique_items, llm, batch_size)
    accepted = [i for i in classified if i.get("accepted", True)]
    print(f"      → {len(accepted)} accepted items")

    print("[4/5] Ranking by profile…")
    ranked = rank_items(accepted, profile, feedback)
    budget = _reading_time_budget(profile)
    top_items = ranked[:budget]
    print(f"      → {len(top_items)} items (budget: {budget})")

    print("[5/5] Generating summary…")
    try:
        summary = (
            await generate_digest_summary(top_items, profile, llm)
            if top_items
            else "Сегодня новостей по твоим темам не нашлось."
        )
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARN] Summary failed: {e}")
        summary = "⚠️ Саммари недоступно. Смотри новости ниже."

    print("      Rendering Telegram message…")
    output = render_telegram(top_items, summary, profile)

    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"digest_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.txt"
    out_file.write_text(output, encoding="utf-8")
    print(f"      → Saved to {out_file}")

    return output


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    result = asyncio.run(run_pipeline(cfg))
    print("\n" + "─" * 60)
    print(result)
