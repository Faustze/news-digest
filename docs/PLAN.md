# PLAN.md — План задач news-digest

## Текущая волна: Инфраструктура и стабильность

| # | Задача | Статус |
|---|--------|--------|
| C-01 | Добавить ruff линтер и форматтер | ✅ |
| C-02 | Добавить type hints во все функции | ✅ |
| C-03 | Добавить pydantic-модели для конфига и вывода | ✅ |
| C-04 | Добавить unit-тесты (pytest) | ✅ (74 теста) |
| C-05 | Добавить CI (GitHub Actions) для lint/test | ✅ (2026-08-14) |

## Бэклог

| # | Задача | Источник | Статус |
|---|--------|----------|--------|
| B-01 | Добавить новые темы/фиды (51 фид, 12 категорий) | TODO.md | ✅ |
| B-02 | Дедупликация статей (URL + title similarity) | TODO.md | ✅ |
| B-03 | Мульти-провайдер LLM (OpenAI, Anthropic, Ollama) | TODO.md | ☐ |
| B-04 | HTML-формат дайджеста (email-рассылка) | TODO.md | ☐ |
| B-05 | Веб-интерфейс архива дайджестов | TODO.md | ☐ |
| B-06 | Метрики пайплайна | TODO.md | ☐ |

## Деплой

| # | Задача | Статус |
|---|--------|--------|
| D-01 | Настроить GitHub Actions (lint/test) | ✅ (2026-08-14) |
| D-02 | Настроить GitHub Actions CI/CD для деплоя Web UI на GitHub Pages | ✅ (2026-08-16) |

---

## Дорожная карта: Персонализация

> **Цель:** Превратить существующий RSS + LangChain + Groq + GitHub Actions + Telegram проект в персонализированный дайджест новостей без введения бэкенда, VPS, базы данных или платной инфраструктуры.

### Non-goals

- Нет VPS
- Нет постоянно работающего бэкенда
- Нет базы данных
- Нет мульти-пользовательской аутентификации (v1)
- Нет пользовательской настройки RSS/источников
- Нет обязательной интеграции с GitHub API для сохранения профиля

### Целевая архитектура

```text
Статический Nuxt Web UI
         |
         | export/import
         v
  user-profile.json
         |
         v
  GitHub repository
         |
         v
  GitHub Actions (cron)
         |
         +--> RSS sources
         |
         +--> LangChain + Groq
         |       |
         |       +--> classify
         |       +--> personalize
         |       +--> rank
         |       +--> summarize
         |
         v
  Telegram Bot
         |
         +--> 👍 useful
         +--> 👎 not interesting
         +--> 🔕 hide similar
         |
         v
  feedback.json
```

### Phase 0 — База и безопасность ✅

- [x] Проверить текущий репозиторий перед изменением архитектуры
- [x] Сохранить существующее локальное выполнение: `python news_pipeline.py`
- [x] Сохранить выполнение через GitHub Actions
- [x] Сохранить текущее поведение RSS как фолбэк
- [x] Добавить тесты перед большими рефакторингами
- [x] Хранить секреты только в environment/GitHub Secrets
- [x] Не помещать credentials Telegram/Groq в состояние Web UI

### Phase 1 — Модель конфигурации ✅

#### 1.1 Разделить техническую конфигурацию и пользовательские предпочтения

- [x] Рефакторить `config.yaml` — оставить только технические настройки
- [x] Убрать пользовательские темы из `config.yaml`

#### 1.2 Добавить `user-profile.json`

- [x] Профиль — единый источник истины для предпочтений пользователя
- [x] Категория: `enabled: boolean` + 6 подтем с interest 0–5
- [x] General: detail_level, language_level, reading_time, frequency, priority, language, source_reliability, regions, exclusions, personal_context

#### 1.3 Добавить валидацию профиля

- [x] Pydantic-модель (`news/profile.py`)
- [x] enum значения, числовые диапазоны
- [x] Malformed JSON → fail early с понятной ошибкой (без fallback на unrelated interests)
- [x] Обратная совместимость: миграция из `config.yaml` topics

### Phase 2 — Категории и интересы ✅

- [x] 12 категорий с 6 подтемами каждая
- [x] Канонические ID и labels (`news/profile.py` → CATEGORIES, CATEGORY_LABELS)
- [x] Interest 0 = явный запрет
- [x] Категории в Web UI (`web-ui/lib/categories.ts`)

### Phase 3 — Web UI / Nuxt ✅

#### First-run онбординг

- [x] Пошаговый степпер: категории → подтемы (0-5) → detail → lang → time → exclusions → context
- [x] Прогресс-бар, кнопка «Назад», сохранение в localStorage
- [x] Нет технического языка

#### Профиль UI

- [x] `/profile` — редактирование всех настроек
- [x] Export/Import/Reset
- [x] Mobile-first, responsive

### Phase 4 — Движок персонализации ✅

- [x] RSS fetch → deduplicate → classify (LLM) → rank → time-budget → summary → Telegram
- [x] LLM classification: category, subtopics, importance, accepted
- [x] Ranking: interest × importance × source_reliability × feedback × personal_context
- [x] Explicit zero: interest=0 → фильтр
- [x] Модули: `news/classify.py`, `news/rank.py`, `news/deduplicate.py`

### Phase 5 — Расширение фидов ✅

- [x] 51 RSS фид на все 12 категорий
- [x] Метаданные: name, url, tags, categories
- [x] Качественные источники (Reuters, BBC, Nature, NASA, etc.)

### Phase 6 — Telegram обратная связь ✅

- [x] Inline-кнопки: 👍 Полезно, 👎 Неинтересно, 🔕 Больше такого
- [x] SHA-256 news_id (title + url)
- [x] Callback data ≤ 64 bytes
- [x] Каждая новость со своей клавиатурой

### Phase 7 — GitHub Actions ✅

- [x] Cron 04:00 UTC + workflow_dispatch
- [x] Feedback polling → pipeline → send → commit
- [x] Коммит: output/, feedback.json, feedback_state.json, user-profile.json

### Phase 8 — Семантика расписания ✅

- [x] Динамический cutoff: daily=24h, weekly=7d, important_only=24h
- [x] Не отправлять пустой дайджест

### Phase 9 — UX дайджеста ✅

- [x] Персонализированное summary
- [x] Категория + теги в каждой новости
- [x] Fallback для Markdown

### Phase 10 — Тесты и качество

- [x] Profile: loading, validation, defaults, missing fields
- [x] Categories: subtopic count, labels
- [x] Ranking: interest 0/5, exclusions, feedback
- [x] Deduplication: URL, title, similarity
- [x] Feedback: news_id, reactions, persistence
- [x] Telegram: parsing, callback data, keyboard
- [x] Scheduling: daily/weekly cutoff, important-only
- [ ] LLM mocking fixtures

---

## Что осталось (приоритет)

1. **B-03** — Мульти-провайдер LLM
2. **B-05** — Веб-интерфейс архива дайджестов
