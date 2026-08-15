"""
User profile schema, loader, defaults, and validation.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# ── Enums ─────────────────────────────────────────────────────────────────────


class DetailLevel(str, Enum):
    short = "short"
    normal = "normal"
    detailed = "detailed"


class LanguageLevel(str, Enum):
    simple = "simple"
    standard = "standard"
    advanced = "advanced"


class ReadingTime(int, Enum):
    five = 5
    ten = 10
    twenty = 20
    thirty = 30


class Frequency(str, Enum):
    morning = "morning"
    evening = "evening"
    daily = "daily"
    weekly = "weekly"
    important_only = "important_only"


class Priority(str, Enum):
    important_only = "important_only"
    balanced = "balanced"
    everything = "everything"


class Language(str, Enum):
    ru = "ru"
    en = "en"


class SourceReliability(str, Enum):
    verified = "verified"
    balanced = "balanced"
    broad = "broad"


# ── Canonical categories & subtopics ──────────────────────────────────────────

CATEGORIES: dict[str, dict[str, str]] = {
    "ai": {
        "new_models": "Новые модели",
        "ai_tools": "AI-инструменты и сервисы",
        "research": "Исследования",
        "generative_ai": "Генеративный AI",
        "ai_business": "AI и бизнес",
        "robotics": "Робототехника",
    },
    "technology": {
        "internet_web": "Интернет и веб",
        "mobile": "Мобильные технологии",
        "computers_hardware": "Компьютеры и железо",
        "cybersecurity": "Кибербезопасность",
        "cloud_infrastructure": "Облака и инфраструктура",
        "programming": "Программирование",
    },
    "science": {
        "medicine_biology": "Медицина и биология",
        "brain_psychology": "Мозг и психология",
        "earth_nature": "Земля и природа",
        "physics": "Физика и фундаментальная наука",
        "chemistry_materials": "Химия и новые материалы",
        "scientific_discoveries": "Научные открытия и исследования",
    },
    "space": {
        "space_missions": "Космические миссии",
        "rockets_launches": "Ракеты и запуски",
        "astronomy_observation": "Астрономия и наблюдения",
        "planets_objects": "Планеты и космические объекты",
        "new_discoveries": "Новые открытия",
        "human_spaceflight": "Пилотируемая космонавтика",
    },
    "gadgets": {
        "smartphones": "Смартфоны",
        "laptops_tablets": "Ноутбуки и планшеты",
        "headphones_audio": "Наушники и аудио",
        "smartwatches_wearables": "Умные часы и носимые устройства",
        "smart_home": "Умный дом",
        "new_devices": "Новые устройства и технологии",
    },
    "games": {
        "new_games_releases": "Новые игры и релизы",
        "gaming_technology_hardware": "Игровые технологии и железо",
        "esports": "Киберспорт",
        "game_companies": "Новости игровых компаний",
        "indie_games": "Инди-игры",
        "gaming_trends_industry": "Игровые тренды и индустрия",
    },
    "business": {
        "precious_metals": "Драгоценные металлы",
        "savings_deposits": "Вклады и сбережения",
        "large_companies": "Крупные компании",
        "markets_economy": "Рынки и экономика",
        "safe_investing": "Безопасное инвестирование",
        "entrepreneurs_leaders": "Предприниматели и руководители",
    },
    "finance": {
        "currencies": "Валюты",
        "cryptocurrencies": "Криптовалюты",
        "stock_market": "Фондовый рынок",
        "banks_rates": "Банки и ставки",
        "taxes_financial_rules": "Налоги и финансовые правила",
        "personal_finance": "Личные финансы",
    },
    "running": {
        "recovery": "Восстановление после бега",
        "motivation_habits": "Мотивация и привычка бегать",
        "beginners_running_technique": "Бег для новичков и техника",
        "marathon_half_marathon": "Марафон и полумарафон",
        "running_gear": "Экипировка бегуна",
        "running_nutrition": "Питание и бег",
    },
    "movies": {
        "new_movies": "Новые фильмы",
        "new_series": "Новые сериалы",
        "trailers_announcements": "Трейлеры и анонсы",
        "actors_directors": "Актёры и режиссёры",
        "what_to_watch": "Что посмотреть",
        "streaming": "Стриминги",
    },
    "music": {
        "new_releases": "Новые релизы",
        "artists_bands": "Исполнители и группы",
        "concerts_festivals": "Концерты и фестивали",
        "music_recommendations": "Музыкальные рекомендации",
        "awards_charts": "Музыкальные премии и чарты",
        "music_technology": "Музыкальные технологии",
    },
    "world": {
        "world_events": "Главные события в мире",
        "usa": "США",
        "europe": "Европа",
        "asia_other_regions": "Азия и другие регионы",
        "global_economy": "Мировая экономика",
        "international_relations": "Международные отношения",
    },
}

CATEGORY_LABELS: dict[str, str] = {
    "ai": "AI",
    "technology": "Технологии",
    "science": "Наука",
    "space": "Космос",
    "gadgets": "Гаджеты",
    "games": "Игры",
    "business": "Бизнес",
    "finance": "Финансы",
    "running": "Бег и тренировки",
    "movies": "Кино и сериалы",
    "music": "Музыка",
    "world": "Мир",
}


# ── Models ────────────────────────────────────────────────────────────────────


class Category(BaseModel):
    enabled: bool = True
    interests: dict[str, int] = Field(default_factory=dict)

    @field_validator("interests")
    @classmethod
    def validate_interests(cls, v: dict[str, int]) -> dict[str, int]:
        for key, val in v.items():
            if not 0 <= val <= 5:
                raise ValueError(f"Interest for {key} must be 0-5, got {val}")
        return v


class General(BaseModel):
    detail_level: DetailLevel = DetailLevel.normal
    language_level: LanguageLevel = LanguageLevel.standard
    reading_time: int = Field(default=10, ge=1, le=120)
    frequency: Frequency = Frequency.daily
    priority: Priority = Priority.balanced
    language: Language = Language.ru
    source_reliability: SourceReliability = SourceReliability.balanced
    regions: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    personal_context: str = ""


class UserProfile(BaseModel):
    version: int = 1
    categories: dict[str, Category] = Field(default_factory=dict)
    general: General = Field(default_factory=General)

    def enabled_categories(self) -> list[str]:
        return [cid for cid, cat in self.categories.items() if cat.enabled]

    def get_interest(self, category: str, subtopic: str) -> int | None:
        """
        Return the configured interest for a subtopic.

        Returns None when the subtopic is absent from the profile (neutral),
        and 0 only when the profile explicitly stores a zero interest.
        """
        cat = self.categories.get(category)
        if not cat or not cat.enabled:
            return 0
        return cat.interests.get(subtopic)

    def is_excluded(self, subtopic: str) -> bool:
        """Check if a subtopic has interest=0 (explicit negative preference)."""
        for cat in self.categories.values():
            if (
                cat.enabled
                and subtopic in cat.interests
                and cat.interests[subtopic] == 0
            ):
                return True
        return False


# ── Loader ────────────────────────────────────────────────────────────────────

DEFAULT_PROFILE_PATH = Path("user-profile.json")
FALLBACK_PROFILE_PATH = Path("config.yaml")


def _empty_profile() -> UserProfile:
    """Return a minimal valid profile with all categories enabled at default interest."""
    cats = {}
    for cid, subtopics in CATEGORIES.items():
        interests = {st: 3 for st in subtopics}
        cats[cid] = Category(enabled=True, interests=interests)
    return UserProfile(categories=cats)


def _migration_profile_from_config(config: dict) -> UserProfile:
    """
    Create a profile from the old config.yaml topics.
    Maps known keywords to categories.
    """
    topic_map: dict[str, list[str]] = {
        "ai": ["ai"],
        "vue": ["technology"],
        "nuxt": ["technology"],
        "javascript": ["technology"],
        "typescript": ["technology"],
        "css": ["technology"],
        "html": ["technology"],
        "python": ["technology"],
        "web": ["technology"],
        "frontend": ["technology"],
        "backend": ["technology"],
    }
    old_topics = [t.lower() for t in config.get("topics", [])]

    cats = {}
    for cid, subtopics in CATEGORIES.items():
        interests = {st: 3 for st in subtopics}
        cats[cid] = Category(enabled=False, interests=interests)

    # Enable categories that match old topics
    for keyword, cat_ids in topic_map.items():
        for topic in old_topics:
            if keyword in topic:
                for cid in cat_ids:
                    cats[cid].enabled = True

    # If nothing matched, enable technology as fallback
    if not any(c.enabled for c in cats.values()):
        cats["technology"].enabled = True

    return UserProfile(categories=cats)


def load_profile(
    path: Path | str = DEFAULT_PROFILE_PATH,
    config: dict | None = None,
) -> UserProfile:
    """
    Load user profile from JSON file.

    If the file doesn't exist, create a default one (optionally from config.yaml migration).
    If the file is invalid, raise an actionable error — never silently fall back
    to unrelated default interests.
    """
    p = Path(path)

    if not p.exists():
        print(f"[INFO] Profile not found at {p}, creating default profile.")
        if config:
            profile = _migration_profile_from_config(config)
        else:
            profile = _empty_profile()
        save_profile(profile, p)
        return profile

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return UserProfile.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"Invalid profile at {p}: {e}. Fix or delete the file, then re-run."
        ) from e


def save_profile(profile: UserProfile, path: Path | str = DEFAULT_PROFILE_PATH) -> None:
    """Save user profile to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(profile.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
