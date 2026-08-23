# Last Build

**Date:** 2026-08-23
**Branch:** main
**Commit:** a6e5266

## Current State

News Digest — персональный агрегатор новостей с AI‑персонализацией на основе RSS + Groq LLM. Веб‑UI (Nuxt) для настройки профиля, Telegram‑бот для доставки дайджеста и фидбека. PostgreSQL‑персистентность реализована через Alembic‑миграции и SQLAlchemy engine; репозитории предоставляют CRUD‑операции для users, articles и feedback (raw `text()`). Недавно добавлен слой DELETE‑операций в репозитории, workflow `daily-commit` для автоматических коммитов, а также приостановлен `daily_digest` до 2026‑08‑28. Переключён Groq‑модель на `gpt-oss-120b` и включён автоматический мёрдж PR.

## Recent Progress

- **2026‑08‑22** Merge pull request #17 – daily‑commit automation (commit a6e5266).

## Verification

- `uv run pytest`: **не запускался** (нет новых результатов).

## Project Statistics

(см. отдельный файл `PROJECT_STATS.md`)

## Next Focus

1. Мониторинг стабильности workflow `daily-commit`.
2. Реактивация процесса `daily_digest` после 2026‑08‑28.
3. Дальнейшее тестирование и отладка слоя DELETE‑операций в репозиториях.
