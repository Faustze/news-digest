# Last Build

**Date:** 2026-08-18
**Branch:** main
**Commit:** ffaa12e

## Current State

News Digest — персональный агрегатор новостей с AI-персонализацией на основе RSS + Groq LLM. Веб-UI (Nuxt) для настройки профиля, Telegram-бот для доставки дайджеста и фидбека. Добавлен слой PostgreSQL-персистентности: Alembic-миграции, SQLAlchemy engine, репозитории для users/articles/feedback (raw `text()`, ORM пока не вводится).

## Recent Progress

С момента предыдущего билда (`2b09f9f`, ветка `feat/personalized-news-digest`):

- Ветка `feat/personalized-news-digest` смёржена в `main` через PR #2.
- Разрешён конфликт с `main`: принят multi-provider LLM (`news/llm.py`, `build_llm`), `config.yaml` хранит `max_tokens: 2048` (фикс free-tier бюджета), `uv.lock` перегенерирован.
- Исправлен отступ `llm = build_llm(config)` в `news_pipeline.py` (сломан при разрешении конфликта); `ruff`-автофикс для `alembic/` (сортировка импортов, `collections.abc.Sequence`, `X | Y` аннотации).
- Удалён `deploy_ui.yml`: сайт сервится с корня `news.faustze.tech`, но билд линковал ассеты как `/news-digest/_nuxt/...` → 404, страница пустая. Web UI пока не нужен в проде (localStorage-прототип без backend-интеграции); файлы `web-ui/` оставлены в репо.
- `requirements.txt` перегенерирован через `uv pip compile` (добавлены sqlalchemy, alembic, psycopg, fastapi, uvicorn, langchain-openai/anthropic/ollama).

## Verification

- Тесты: `uv run pytest` — **116 passed** (включая 3 интеграционных против `newsdigest_test` в Docker).
- ruff: `uv run ruff check .` — **All checks passed**; `uv run ruff format --check .` — **46 files already formatted**.
- Web UI: `pnpm build` — успешно (SPA, `ssr: false`).

## Project Statistics

- Files: 31 (14 py src + 10 тестовых + 7 vue/ts исходников)
- Lines of code: 3922 (2427 py + 1495 web-ui)
- Tests: 116 (10 файлов)
- TODOs в исходниках: 0 (12 — в docs)

## Next Focus

1. Связать Web UI с персистентностью (сохранение профиля/фидбека через репозитории).
2. Решить судьбу серверного слоя (FastAPI/uvicorn уже в зависимостях) относительно ограничения «без бэкенда».
3. Переход на ORM-модели SQLAlchemy (следующий уровень после raw `text()`).