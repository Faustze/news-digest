# STATE.md — Текущее состояние

## Позиция

- **Активная задача:** Phase 8 — динамический cutoff по frequency
- **Последний коммит:** docs: update README and project documentation (dd140e1)
- **Следующий шаг:** Реализовать `cutoff_hours_for_frequency()` и интегрировать в pipeline

## Выполнено

- [x] Базовый пайплайн (RSS → LangChain+Groq → output)
- [x] Telegram-бот для отправки
- [x] GitHub Actions (cron 04:00 UTC)
- [x] Профиль пользователя (`user-profile.json`) с 12 категориями и 0-5 интересами
- [x] Pydantic-валидация профиля и feedback
- [x] Классификация, ранжирование и дедупликация новостей
- [x] Telegram inline feedback (👍👎🔕) и polling через `getUpdates`
- [x] 51 RSS фид на все 12 категорий
- [x] Nuxt Web UI: onboarding + редактор профиля
- [x] CI: lint (ruff) + test (pytest) в GitHub Actions
- [x] README + AGENTS.md + docs-структура
- [x] Локальный скилл task-workflow в `.agents/skills/`

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

Phase 8: динамический cutoff по frequency (`daily=24h`, `weekly=7d`).
