# STATE.md — Текущее состояние

## Позиция

- **Активная задача:** Phase 9 — UX дайджеста (персонализированное summary, категория + теги, Markdown fallback)
- **Последний коммит:** feat: dynamic cutoff hours based on profile frequency (78d3dd6)
- **Следующий шаг:** Персонализированное summary в `generate_digest_summary`

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

## Решения

- Python 3.10+, uv для управления зависимостями
- Pydantic для моделей профиля и feedback
- LangChain + Groq (llama-3.3-70b-versatile) для фильтрации и суммаризации
- feedparser для RSS
- httpx для Telegram API
- Nuxt 3 static SPA для Web UI, localStorage для persistence
- Single-user, без backend/VPS/базы данных
- GitHub Actions как orchestration layer

## Что в работе

Phase 9: UX дайджеста — персонализированное summary, категория + теги в каждой новости, Markdown fallback.
