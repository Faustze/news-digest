# Last Build

**Date:** 2026-08-20
**Branch:** main
**Commit:** 97d9afc

## Current State

News Digest — персональный агрегатор новостей с AI‑персонализацией на основе RSS + Groq LLM. Веб‑UI (Nuxt) для настройки профиля, Telegram‑бот для доставки дайджеста и фидбека. PostgreSQL‑персистентность реализована через Alembic‑миграции и SQLAlchemy engine; репозитории предоставляют CRUD‑операции для users, articles и feedback (raw `text()`). Недавно добавлен слой DELETE‑операций в репозитории, workflow `daily-commit` для автоматических коммитов, а также приостановлен `daily_digest` до 2026‑08‑28. Переключён Groq‑модель на `gpt-oss-120b` и включён автоматический мёрдж PR.

## Recent Progress

- **2026‑08‑20** Merge pull request #12 – исправлен процесс модели и PR‑flow для `daily-commit` (commit 97d9afc).
- **2026‑08‑20** fix(ci): switch Groq model to gpt-oss-120b and publish via PR with auto-merge (commit ec77c30).
- **2026‑08‑20** Merge pull request #11 – ежедневный коммит за 2026‑08‑20 (commit f95347c).
- **2026‑08‑20** docs: update project state (commit c917a4b).
- **2026‑08‑20** docs: update project state (commit f4611f9).
- **2026‑08‑20** Merge pull request #10 – интегрирован агент ежедневных коммитов (`daily_commit.py`, workflow `.github/workflows/daily-commit.yml`) (commit 7a47c81).
- **2026‑08‑20** ci: add daily-commit workflow and pause digest until 2026‑08‑28 (commit 09973df).
- **2026‑08‑19** Merge pull request #9 – нормализация URL статьи перед вставкой в dedupe‑модуль (commit 2fd5dfc).
- **2026‑08‑19** feat(db): normalize article URL before insert to dedupe RSS duplicates (commit 42d6459).
- **2026‑08‑19** Merge pull request #8 – подготовка CRUD‑операций репозиториев базы данных (commit e8de2e1).

## Verification

Automated test suite has not been executed after the latest changes. No new test results are available at this time.

## Project Statistics

- Python source files: **32**
- Lines of Python code (including tests): **3720**
- Test functions: **121**
- TODO/FIXME comments in source: **2**

## Next Focus

1. Execute the full pytest suite and capture results.
2. Unpause `daily_digest` after 2026‑08‑28 and monitor its behavior.
3. Extend documentation for the new `daily-commit` agent and CI workflow.
