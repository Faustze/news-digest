"""
News Feed Pipeline — LangChain + Groq (free tier)
Fetches, filters, and summarises news from RSS feeds.
"""

import asyncio
import json
import re
import feedparser
import yaml
from datetime import datetime, timezone
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


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
        except Exception as e:
            print(f"[WARN] Could not fetch {feed_cfg['url']}: {e}")
            continue

        for entry in feed.entries:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
                if age_hours > cutoff_hours:
                    continue

            items.append({
                "title":   entry.get("title", ""),
                "summary": re.sub(r"<[^>]+>", "", entry.get("summary", entry.get("description", ""))[:600]),
                "link":    entry.get("link", ""),
                "source":  feed_cfg.get("name", feed.feed.get("title", "Unknown")),
                "tags":    feed_cfg.get("tags", []),
            })

    return items


# ── Step 2: Filter & score items ─────────────────────────────────────────────

FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты куратор новостей для веб-разработчика (Vue, Nuxt, Vuetify, Tailwind, UnoCSS, Python).

Отфильтруй список новостей — оставь только те, что полезны разработчику по темам: {topics}

Верни ТОЛЬКО валидный JSON-массив без markdown-обёрток и объяснений.

Для каждой подходящей новости:
{{
    "title": "оригинальный заголовок",
    "summary": "2-3 предложения на русском: что нового, зачем полезно разработчику",
    "link": "...",
    "source": "...",
    "tags": ["Vue", "Python", ...],
    "priority": "high|medium|low",
    "relevance_score": 0-10
}}

Игнорируй: вакансии, маркетинг, opinion без технической ценности, новости не по теме.
Если ничего подходящего — верни [].
"""),
    ("human", "Новости:\n{items_json}"),
])


async def filter_and_score(items: list[dict], config: dict, llm: ChatGroq) -> list[dict]:
    if not items:
        return []

    chain = FILTER_PROMPT | llm | StrOutputParser()
    batch_size = config.get("batch_size", 12)  # smaller batches for Groq rate limits
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        try:
            raw = await chain.ainvoke({
                "topics":     ", ".join(config["topics"]),
                "items_json": json.dumps(batch, ensure_ascii=False, indent=2),
            })
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(clean)
            results.extend(parsed if isinstance(parsed, list) else [])
        except Exception as e:
            print(f"[WARN] Batch {i // batch_size + 1} failed: {e}")

        # Small delay to stay within Groq free tier rate limits (30 RPM)
        if i + batch_size < len(items):
            await asyncio.sleep(2)

    results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return results[:config.get("max_items", 15)]


# ── Step 3: Executive summary ─────────────────────────────────────────────────

DIGEST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты пишешь краткий ежедневный дайджест для веб-разработчика (Vue/Nuxt/Python).
Напиши 3-4 предложения: что важного произошло сегодня в мире фронтенда и Python.
Будь конкретным и полезным. Без воды. Язык: русский."""),
    ("human", "Топ новостей:\n{items_json}"),
])


async def generate_digest_summary(items: list[dict], llm: ChatGroq) -> str:
    chain = DIGEST_PROMPT | llm | StrOutputParser()
    return await chain.ainvoke({
        "items_json": json.dumps(
            [{"title": i["title"], "summary": i.get("summary", "")} for i in items[:10]],
            ensure_ascii=False,
        ),
    })


# ── Step 4: Render Telegram message ──────────────────────────────────────────

TAG_EMOJI = {
    "Vue": "💚", "Nuxt": "💚", "Vuetify": "💚",
    "Tailwind": "🎨", "CSS": "🎨", "UnoCSS": "🎨",
    "JavaScript": "🟨", "TypeScript": "🔷",
    "Python": "🐍", "FastAPI": "🐍", "backend": "⚙️",
    "frontend": "🖥️", "webdev": "🌐",
}

def tag_emoji(tags: list[str]) -> str:
    for t in tags:
        if t in TAG_EMOJI:
            return TAG_EMOJI[t]
    return "📌"


def render_telegram(items: list[dict], summary: str) -> str:
    date_str = datetime.now().strftime("%d.%m.%Y")

    lines = [
        f"📰 *Дайджест {date_str}*",
        "",
        summary,
        "",
        "─────────────────────",
        "",
    ]

    for item in items:
        emoji  = tag_emoji(item.get("tags", []))
        title  = item.get("title", "").strip()
        link   = item.get("link", "")
        text   = item.get("summary", "")[:280].strip()
        tags   = " ".join(f"#{t}" for t in item.get("tags", [])[:3])

        lines += [
            f"{emoji} [{title}]({link})",
            text,
            tags,
            "",
        ]

    lines.append(
        f"_Источников: {len(set(i['source'] for i in items))} · "
        f"Новостей: {len(items)}_"
    )
    return "\n".join(lines)


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_pipeline(config_path: str = "config.yaml") -> str:
    config = load_config(config_path)

    llm = ChatGroq(
        model=config.get("model", "llama-3.3-70b-versatile"),
        temperature=0,
        max_tokens=4096,
    )

    print(f"[1/4] Fetching RSS feeds ({len(config['feeds'])} sources)…")
    raw_items = fetch_rss_items(config)
    print(f"      → {len(raw_items)} raw items")

    print("[2/4] Filtering & scoring with Groq…")
    scored = await filter_and_score(raw_items, config, llm)
    print(f"      → {len(scored)} relevant items")

    print("[3/4] Generating summary…")
    summary = await generate_digest_summary(scored, llm) if scored else "Сегодня новостей по твоим темам не нашлось."

    print("[4/4] Rendering Telegram message…")
    output = render_telegram(scored, summary)

    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"digest_{datetime.now().strftime('%Y-%m-%d')}.txt"
    out_file.write_text(output, encoding="utf-8")
    print(f"      → Saved to {out_file}")

    return output


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    result = asyncio.run(run_pipeline(cfg))
    print("\n" + "─" * 60)
    print(result)
