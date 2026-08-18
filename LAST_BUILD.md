# Last Build

**Date:** 2026-08-18
**Branch:** feat/personalized-news-digest
**Commit:** 2b09f9f

## Current State

News Digest — персональный агрегатор новостей с AI-персонализацией на основе RSS + Groq LLM. Веб-UI (Nuxt) для настройки профиля, Telegram-бот для доставки дайджеста и фидбека. Добавлен слой PostgreSQL-персистентности: Alembic-миграции, SQLAlchemy engine, репозитории для users/articles/feedback.

## Recent Progress

С момента предыдущего коммита (`b753728`, 17.08.2026):

- Добавлен пакет `news/repositories/` — `UserRepository`, `FeedbackRepository`, `ArticleRepository`; весь SQL вынесен из консольных скриптов, живёт только в репозиториях (raw `text()`, ORM пока не вводится).
- `news/console.py` переписан на репозитории вместо прямых `text()`-запросов; импорты как пакет (`news.db`, `news.repositories`), запуск через `python -m news.console`.
- Миграция `initial_schema` дополнена: `UniqueConstraint(user_id, article_id)` на `feedback` и `UniqueConstraint` на `users.telegram_id` (защита от дубликатов, база пересоздана через downgrade/upgrade).
- `ArticleRepository.get_or_create` и `UserRepository.get_by_telegram_id` — идемпотентные (INSERT → fallback SELECT при конфликте/отсутствии).
- Новый `tests/test_feedback_repository.py`: session-scoped фикстура с поднятием тестовой БД (`newsdigest_test`) через Alembic, тесты insert/get-or-create, upsert (без дубликатов) и JOIN через репозитории.

Изменения **закоммичены и запушены** в `feat/personalized-news-digest`.

## Verification

- Тесты: `uv run pytest tests/test_feedback_repository.py` — **3 passed** (интеграционные, против `newsdigest_test` в Docker).
- ruff: `uv run ruff check news/ tests/` — **All checks passed**.
- Консоль: `uv run python -m news.console` — отрабатывает, выводит JOIN-строки.
- Полный прогон всех тестов (`uv run pytest`) не выполнялся в этой сессии.

## Project Statistics

- Files: 27 (12 py src + 8 тестовых + 7 vue/ts исходников)
- Lines of code: 3672 (2161 py + 1511 web-ui)
- Tests: 97 (8 файлов)
- TODOs в исходниках: 0 (10 — в docs)

## Next Focus

1. Связать Web UI с персистентностью (сохранение профиля/фидбека через репозитории).
2. Решить судьбу серверного слоя (FastAPI/uvicorn уже в зависимостях) относительно ограничения «без бэкенда».
3. Переход на ORM-модели SQLAlchemy (следующий уровень после raw `text()`).