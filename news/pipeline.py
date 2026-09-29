"""
News Feed Pipeline — LangChain + Groq (free tier)
Fetches, filters, classifies, ranks, and summarises news from RSS feeds.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from news.candidates import select_candidates
from news.classify import classify_batch
from news.deduplicate import deduplicate
from news.feedback import load_feedback
from news.fetch import fetch_rss_items
from news.llm import TokenCounter, build_llm
from news.profile import CATEGORY_LABELS, UserProfile, load_profile
from news.rank import rank_items
from news.schedule import cutoff_hours_for_frequency

# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Step 2: Executive summary ─────────────────────────────────────────────────

DIGEST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты пишешь краткий ежедневный дайджест новостей.
Язык: {language}.
Уровень детализации: {detail_level}.
Уровень языка: {language_level}.
Время чтения: {reading_time}.
Приоритет: {priority}.

Напиши дайджест: что важного произошло сегодня.
Уложись в заданное время чтения и расставь акценты согласно приоритету.
Будь конкретным и полезным. Без воды.

Формат — сообщение в Telegram:
- не длиннее 3000 символов;
- без markdown-заголовков (#), таблиц и горизонтальных линий;
- выделяй жирным через **текст**, списки — через «- ».""",
        ),
        ("human", "Топ новостей:\n{items_json}"),
    ]
)


async def generate_digest_summary(
    items: list[dict],
    profile: UserProfile,
    llm: BaseChatModel,
    usage: TokenCounter | None = None,
) -> str:
    chain = DIGEST_PROMPT | llm
    language = "русский" if profile.general.language.value == "ru" else "English"
    detail_map = {"short": "кратко", "normal": "обычно", "detailed": "подробно"}
    lang_map = {
        "simple": "простыми словами",
        "standard": "обычный уровень",
        "advanced": "технический язык",
    }
    priority_map = {
        "important_only": "только самое важное",
        "balanced": "сбалансированный акцент",
        "everything": "максимум новостей",
    }

    message = await chain.ainvoke(
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
            "reading_time": f"{profile.general.reading_time} минут",
            "priority": priority_map.get(
                profile.general.priority.value, "сбалансированный акцент"
            ),
        }
    )
    if usage is not None:
        usage.add(message)
    return StrOutputParser().invoke(message)


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

        tags = [t for t in (item.get("tags") or []) if t]
        tag_suffix = "  " + " ".join(f"#{t}" for t in tags) if tags else ""

        lines += [
            f"{emoji} [{title}]({link})",
            text,
            f"#{cat_label}{tag_suffix}  `{news_id[:12]}`",
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
    # Use the legacy migration only while config.yaml still defines `topics`.
    legacy_config = config if config.get("topics") else None
    profile = load_profile(profile_path, legacy_config)
    feedback = load_feedback(feedback_path)

    llm = build_llm(config)
    batch_size = config.get("batch_size", 12)
    usage = TokenCounter()

    print(f"[1/5] Fetching RSS feeds ({len(config['feeds'])} sources)…")
    raw_items = fetch_rss_items(config, cutoff_hours_for_frequency(profile))
    print(f"      → {len(raw_items)} raw items")

    print("[2/5] Deduplicating…")
    unique_items = deduplicate(raw_items)
    print(f"      → {len(unique_items)} unique items")

    candidates = select_candidates(
        unique_items,
        set(profile.enabled_categories()),
        config.get("max_items_per_source"),
    )
    print(f"      → {len(candidates)} candidates for classification")

    print("[3/5] Classifying with Groq…")
    classified = await classify_batch(
        candidates,
        llm,
        batch_size,
        usage=usage,
        tokens_per_minute=config.get("tokens_per_minute"),
    )
    accepted = [i for i in classified if i.get("accepted") is True]
    print(f"      → {len(accepted)} accepted items")

    print("[4/5] Ranking by profile…")
    ranked = rank_items(accepted, profile, feedback)
    budget = _reading_time_budget(profile)
    top_items = ranked[:budget]
    print(f"      → {len(top_items)} items (budget: {budget})")

    print("[5/5] Generating summary…")
    try:
        summary = (
            await generate_digest_summary(top_items, profile, llm, usage)
            if top_items
            else "Сегодня новостей по твоим темам не нашлось."
        )
    except Exception as e:  # noqa: BLE001
        # Best-effort step: whatever the provider raises (network, rate limit,
        # malformed output) must not sink the whole digest.
        print(f"[WARN] Summary failed: {e}")
        summary = "⚠️ Саммари недоступно. Смотри новости ниже."

    print(f"      LLM tokens used this run: {usage.total}")
    print("      Rendering Telegram message…")

    if not top_items:
        print("      → No items matched the profile; skipping the empty digest.")
        return ""

    output = render_telegram(top_items, summary, profile)

    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"digest_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.txt"
    out_file.write_text(output, encoding="utf-8")
    print(f"      → Saved to {out_file}")

    return output


def main() -> None:
    """CLI entry point: ``news-digest [config.yaml]``."""
    load_dotenv()
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    result = asyncio.run(run_pipeline(cfg))
    print("\n" + "─" * 60)
    print(result)


if __name__ == "__main__":
    main()
