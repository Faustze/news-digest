# Last Build

**Date:** 2026-08-20
**Branch:** main
**Commit:** f4611f9

## Current State

News Digest — персональный агрегатор новостей с AI‑персонализацией на основе RSS + Groq LLM. Веб‑UI (Nuxt) для настройки профиля, Telegram‑бот для доставки дайджеста и фидбека. PostgreSQL‑персистентность реализована через Alembic‑миграции и SQLAlchemy engine; репозитории предоставляют CRUD‑операции для users, articles и feedback (raw `text()`). Недавно добавлен слой DELETE‑операций в репозитории, workflow `daily-commit` для автоматических коммитов, а также приостановлен `daily_digest` до 2026‑08‑28.

## Recent Progress

- **2026‑08‑20** docs: update project state (commit f4611f9).
- **2026‑08‑20** Merge PR #10 – интегрирован агент ежедневных коммитов (`daily_commit.py`, workflow `.github/workflows/daily-commit.yml`).
- **2026‑08‑20** Добавлен workflow `daily-commit` и приостановлен `daily_digest` до 2026‑08‑28.
- **2026‑08‑19** Merge PR #9 – нормализация URL статьи перед вставкой в dedupe‑модуль.
- **2026‑08‑19** Добавлены DELETE‑операции в репозитории базы данных.
- **2026‑08‑19** Merge PR #7 – обновлена документация журнала разработки (`docs/STATE.md`).
- **2026‑08‑18** Merge PR #4 – обновления зависимостей GitHub Actions через Dependabot.

## Verification

Automated test suite has not been executed after the latest changes. No new test results are available at this time.

## Project Statistics

- Python source files: **32**
- Lines of Python code (including tests): **3696**
- Test functions: **121**
- TODO/FIXME comments in source: **2**

## Next Focus

1. Run the full test suite and address any failures.
2. Verify the `daily-commit` workflow execution and integrate its results into monitoring.
3. Prepare for re‑enabling `daily_digest` after 2026‑08‑28.
