# News Digest

Персональный агрегатор новостей с AI-персонализацией.

## Как это работает

1. **Web UI** — настрой интересы через простой интерфейс (без технических терминов)
2. **RSS** — агрегация новостей из 42 источников по 12 категориям
3. **AI классификация** — Groq LLM классифицирует и оценивает каждую новость
4. **Персонализация** — ранжирование по твоим интересам, исключениям, регионам
5. **Telegram** — дайджест с кнопками обратной связи

## Категории

| Категория | Подтемы |
|-----------|---------|
| 🤖 AI | Новые модели, инструменты, исследования, генеративный AI, бизнес, робототехника |
| 💻 Технологии | Веб, мобильные, железо, кибербезопасность, облака, программирование |
| 🔬 Наука | Медицина, психология, природа, физика, химия, открытия |
| 🚀 Космос | Миссии, ракеты, астрономия, планеты, открытия, пилотируемые полёты |
| 📱 Гаджеты | Смартфоны, ноутбуки, аудио, часы, умный дом, новые устройства |
| 🎮 Игры | Релизы, технологии, киберспорт, компании, инди, тренды |
| 💼 Бизнес | Металлы, вклады, компании, рынки, инвестирование, руководители |
| 💰 Финансы | Валюты, крипто, фондовый рынок, банки, налоги, личные финансы |
| 🏃 Бег | Восстановление, мотивация, техника, марафоны, экипировка, питание |
| 🎬 Кино | Фильмы, сериалы, трейлеры, актёры, рекомендации, стриминги |
| 🎵 Музыка | Релизы, исполнители, концерты, рекомендации, чарты, технологии |
| 🌍 Мир | События, США, Европа, Азия, экономика, отношения |

## Быстрый старт

### 1. Настрой профиль через Web UI

```bash
cd web-ui
pnpm install
pnpm run dev
```

Открой http://localhost:3000 и пройди onboarding.

### 2. Экспортируй профиль

Нажми «Экспорт JSON» и сохрани файл `user-profile.json` в корень проекта.

### 3. Запусти пайплайн

```bash
uv sync
export GROQ_API_KEY=your_groq_key
uv run python -m news.pipeline      # или: uv run news-digest
```

### 4. Отправь в Telegram

```bash
export TELEGRAM_BOT_TOKEN=your_token
export TELEGRAM_CHAT_ID=your_chat_id
uv run python -m news.send_telegram
```

## Структура проекта

```text
news-digest/
├── news/                    # Весь Python-код
│   ├── pipeline.py          # Основной пайплайн (точка входа)
│   ├── send_telegram.py     # Отправка в Telegram с feedback кнопками
│   ├── poll_feedback.py     # Опрос Telegram callback'ов
│   ├── fetch.py             # Загрузка RSS и очистка текста
│   ├── deduplicate.py       # Дедупликация статей
│   ├── classify.py          # Классификация новостей через LLM
│   ├── rank.py              # Ранжирование по профилю
│   ├── profile.py           # Схема профиля, загрузка, валидация
│   ├── feedback.py          # Схема feedback, persistence
│   ├── llm.py               # Фабрика LLM-провайдеров (groq/openai/anthropic/ollama)
│   ├── schedule.py          # Динамический cutoff по частоте
│   ├── url_utils.py         # Нормализация URL
│   └── db/                  # Опциональная PostgreSQL: session, repositories, console
├── db/                      # alembic.ini, миграции, docker-compose.yml (опционально)
├── web-ui/                  # Nuxt Web UI (onboarding + редактор профиля)
├── tests/                   # Тесты
├── docs/                    # Рабочие документы (PLAN, STATE, TODO, DESIGN)
├── .github/                 # Workflows, CONTRIBUTING, SECURITY, шаблоны
├── config.yaml              # RSS feeds, модель, настройки
└── user-profile.json        # Профиль пользователя (создаётся через Web UI)
```

## GitHub Actions

Два запуска в день: утром в 04:17 UTC (07:17 МСК) и вечером в 15:17 UTC
(18:17 МСК); не в :00 — GitHub сильно задерживает такие cron. Какие из них
присылают дайджест, решает частота в профиле («Когда присылать новости»):
раз в день утром, раз в день вечером или утром и вечером. Лишний запуск
завершается сразу и не тратит токены. Уже отправленные новости не повторяются
(`sent_news.json`).

1. Опрос Telegram feedback
2. Запуск пайплайна
3. Отправка в Telegram
4. Сохранение digest, feedback и `sent_news.json` в ветку `digest-data`
   (`main` защищён, поэтому сгенерированное состояние живёт отдельно)

Ручной запуск: `gh workflow run daily_digest.yml -f slot=evening`.

Токены Groq: классифицируются только до `max_items_per_source` свежих новостей
с источника, в логе прогона есть строка `LLM tokens used this run`.

Секреты: `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Тесты

```bash
uv run pytest tests/ -v
```

## Локальная PostgreSQL (опционально)

Пайплайну база не нужна. Слой PostgreSQL — локальный эксперимент, его
зависимости вынесены в группу `db`:

```bash
cp .env.example .env   # задай POSTGRES_PASSWORD, DATABASE_URL, TEST_DATABASE_URL
docker compose -f db/docker-compose.yml --env-file .env up -d
uv sync --group db
uv run alembic -c db/alembic.ini upgrade head
uv run pytest tests/test_feedback_repository.py   # без TEST_DATABASE_URL — skip
```

## Линтер

```bash
uv run ruff check .
uv run ruff format .
```

## Архитектура

- **Нет backend** — всё работает через GitHub Actions cron
- **Нет VPS** — статический Web UI + GitHub Actions
- **Single-user** — один профиль, без авторизации
- **Groq Free Tier** — batching, rate limiting, минимум запросов
- **PostgreSQL** — опциональный локальный слой (группа зависимостей `db`); пайплайн, CI и cron его не используют

> Примечание: проект изначально задуман без базы данных (профиль/feedback в JSON). Слой PostgreSQL (`news/db/`, `db/`) добавлен как локальный инструмент разработки и не является обязательным для работы пайплайна.

## Лицензия

[CC BY-NC 4.0](LICENSE.md) — можно форкать, изменять и распространять с указанием автора, но не в коммерческих целях.
