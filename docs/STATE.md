# STATE.md — Текущее состояние

## Позиция

- **Активная задача:** Слой PostgreSQL-персистентности (репозитории) — смёржен в `main`; дальше — связка Web UI с персистентностью
- **Последний коммит:** ci: remove web-ui deploy to GitHub Pages (ffaa12e)
- **Следующий шаг:** Связать Web UI с персистентностью или решить судьбу серверного слоя (FastAPI/uvicorn)

## Выполнено

- [x] Базовый пайплайн (RSS → LangChain+Groq → output)
- [x] Telegram-бот для отправки
- [x] GitHub Actions (cron 04:00 UTC)
- [x] Профиль пользователя (`user-profile.json`) с 12 категориями и 0-5 интересами
- [x] Pydantic-валидация профиля и feedback
- [x] Классификация, ранжирование и дедупликация новостей
- [x] Telegram inline feedback (👍👎🔕) и polling через `getUpdates`
- [x] 42 RSS фида на все 12 категорий (все проверены, dead-фиды удалены)
- [x] Nuxt Web UI: onboarding + редактор профиля
- [x] CI: lint (ruff) + test (pytest) в GitHub Actions
- [x] README + AGENTS.md + docs-структура
- [x] Локальный скилл task-workflow в `.agents/skills/`
- [x] Phase 8: динамический cutoff по frequency (`daily/evening/morning/important_only=24h`, `weekly=7d`)
- [x] Phase 9: персонализированное summary (reading_time/priority), категория + теги в новостях, Markdown fallback с сохранением кнопок
- [x] D-02: деплой Web UI на GitHub Pages (`deploy_ui.yml`, `pnpm generate`, `deploy-pages`)
- [x] B-03: мульти-провайдер LLM (`news/llm.py`, провайдер из `config.yaml`)
- [x] PostgreSQL-персистентность: Alembic-миграции, SQLAlchemy engine, docker-compose (Postgres 16, порт 5434)
- [x] Repository pattern: `news/repositories/` (User/Article/Feedback, raw `text()`), SQL вынесен из консольных скриптов
- [x] Интеграционные тесты против `newsdigest_test` (`tests/test_feedback_repository.py`)
- [x] PR #2 смёржен в `main` (конфликт с multi-provider LLM разрешён: `build_llm` + `max_tokens: 2048` в конфиге)
- [x] Deploy web-ui убран: `news.faustze.tech` сервит из корня, а билд линковал `/news-digest/_nuxt/...` → 404, страница пустая

## Правки по ревью CodeRabbit (PR #1)

- [x] `classify_batch`: каждый item получает `accepted: false` при сбое батча; фильтр в pipeline требует `accepted is True`
- [x] `rank_item`: неизвестная категория → 0.0 (без fallback на `ai`); применены `general.exclusions` как hard filter
- [x] `get_interest`: отсутствующая подтема → `None` (нейтрально), `0` только при явном запрете
- [x] Невалидный профиль → fail early с ошибкой (вместо fallback на defaults)
- [x] Миграция профиля: `config.yaml` без `topics` больше не даёт профиль только с technology
- [x] Telegram inline keyboard обёрнут в row (`[buttons]`)
- [x] `poll_feedback`: курсор продвигается после сохранения реакций; non-dict payload отклоняется
- [x] `feedback.reaction` ограничен `VALID_REACTIONS`
- [x] Дедупликация: нормализация URL (query, fragment, trailing slash)
- [x] Пустой дайджест не пишется и не отправляется
- [x] Суммаризация: transport-ошибки (timeout/rate limit/HTTP) не роняют прогон
- [x] Web UI: `disableCategory` подключён; валидация импорта профиля; keyboard-доступные категории
- [x] 91 unit-тест

## Правки по второму раунду CodeRabbit (PR #1, после мержа в ветку)

- [x] `send_telegram.latest_digest` читает только дайджест за сегодня (пустой день больше не отправляет вчерашний)
- [x] Workflow: `git add` feedback-файлов только если они существуют; job summary не падает без файла
- [x] Экранирование title/summary для Telegram Markdown (кнопки не теряются при 400 fallback)
- [x] `rank`: feedback-скоры агрегируются один раз на категорию (O(n×r) → O(r)); `context_boost` не обнуляется округлением
- [x] `deduplicate`: word-set'ы предвычислены до цикла похожести
- [x] `profile.is_excluded` удалён (дублирует проверку в rank)
- [x] Web UI: `useState` вместо module-level refs; `@nuxt/devtools` убран (devtools отключён)

## Правки после фейла Groq 429 (TPD 100K)

- [x] `classify_batch`: `groq.RateLimitError` больше не роняет прогон — лог, оставшиеся батчи `accepted: false`, stop
- [x] `classify_batch`: прочие `groq.GroqError`/`httpx`/timeout — лог и continue со следующими батчами
- [x] Компактный payload классификации: summary обрезается до 250 символов, `link` не шлётся, `categories_json` без отступов
- [x] `max_tokens` для LLM снижен 4096 → 2048 (классификация/саммари укладываются)
- [x] Новые тесты: rate-limit stop, сохранение уже классифицированных батчей, transient error → continue
- [x] Полный прогон без ключа: 0 accepted → пустой дайджест не создаётся, прогон не падает

## Решения

- Python 3.10+, uv для управления зависимостями
- Pydantic для моделей профиля и feedback
- LangChain + Groq (llama-3.3-70b-versatile) для фильтрации и суммаризации
- feedparser для RSS
- httpx для Telegram API
- Nuxt 3 static SPA для Web UI, localStorage для persistence
- Single-user, без backend/VPS
- PostgreSQL — локальный слой персистентности (Alembic + SQLAlchemy), необязателен для пайплайна
- GitHub Actions как orchestration layer

## Что в работе

Связка Web UI с персистентностью; решение по серверному слою (FastAPI/uvicorn уже в зависимостях) относительно ограничения «без бэкенда».

## 2026-08-20: daily-commit automation (20–28.08.2026)

- Добавлен `daily_commit.py`: собирает факты репозитория (git log/status/diff, статистика файлов/тестов/TODO), отправляет Groq (модель из config.yaml), получает перезаписанные `LAST_BUILD.md` / `PROJECT_STATS.md` + commit message в JSON, валидирует ответ, коммитит и пушит только `LAST_BUILD.md`/`PROJECT_STATS.md`.
- Добавлен `.github/workflows/daily-commit.yml`: cron 05:00 UTC (после дайджеста 04:00) + workflow_dispatch, secret `GROQ_API_KEY`, permissions contents: write.
- Окно работы зашито в скрипте: только 20–28 августа 2026 (UTC), вне окна — no-op.
- Анти-фабрикация: модель только переформатирует собранные факты; запрещены выдуманные тесты/метрики и сообщения вида "chore: daily commit"; пустой diff → нет коммита.
- Проверено: ruff check/format, pytest (113 passed), сбор фактов и валидация локально без API-ключа.

### 2026-08-20: digest приостановлен до 28.08

- `daily_digest.yml`: добавлен gate-шаг — до 2026-08-28 (UTC) cron-запуски пропускают все шаги дайджеста (Groq-лимиты зарезервированы под daily-commit); с 29.08 дайджест возобновляется автоматически. `workflow_dispatch` не блокируется.
- `daily_commit.py`: добавлена загрузка `.env` (только для локальных запусков; файл в .gitignore, в git не попадает — ключи берутся из GitHub Actions secrets).

### 2026-08-20: daily-commit — live-прогон и исправления

- Groq-модель `llama-3.3-70b-versatile` больше недоступна → `config.yaml` и `daily_commit.py` переведены на `openai/gpt-oss-120b`.
- Обнаружен жёсткий лимит free-tier: 8000 TPM → промпт сокращён (журналы обрезаются до 1500 символов, log до 10 записей, max_tokens 3000).
- `main` защищён (PR обязателен) → `_publish()`: прямой push, при отказе — ветка `chore/daily-commit-<date>`, PR + auto-merge (включён `allow_auto_merge` на уровне репозитория), fallback — ожидание проверок + merge.
- В `daily-commit.yml` добавлен `GH_TOKEN` для gh CLI.
- Live-прогон: PR #11 и #13 (docs: update project state and statistics) смержены автоматически, LAST_BUILD.md/PROJECT_STATS.md обновлены агентом.
