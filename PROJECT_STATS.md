# Project Statistics

Сводные метрики репозитория. Обновляется вручную вместе с `LAST_BUILD.md`.

## Основные модули

| Модуль | Назначение |
|--------|------------|
| `news/classify.py` | AI-классификация новостей Groq, rate-limit, оценка категорий |
| `news/deduplicate.py` | Дедупликация новостей |
| `news/rank.py` | Персонализированное ранжирование по интересам |
| `news/feedback.py` | Обработка Telegram-фидбека |
| `news/profile.py` | Профиль пользователя (`user-profile.json`) |
| `news/llm.py` | Фабрика LLM-провайдеров (groq/openai/anthropic/ollama) |
| `news/schedule.py` | Динамический cutoff по частоте |
| `news/console.py` | Консольные проверки БД (точка входа, переписан на репозитории) |
| `news/db.py` | SQLAlchemy engine + SessionLocal |
| `news/repositories/` | Слой доступа к данным: user/article/feedback (raw `text()`, без ORM) |
| `web-ui/` | Nuxt UI для онбординга и правки профиля |

## Метрики (18.08.2026)

| Показатель | Значение |
|------------|----------|
| Файлы исходников (py src) | 14 |
| Файлы тестов | 10 |
| Файлы web-ui (vue/ts, без node_modules/.nuxt) | 7 |
| Строки кода (py src + тесты) | 2427 |
| Строки кода (web-ui, исходники) | 1495 |
| Тестовые функции | 116 |
| TODO/FIXME в исходниках | 0 |
| Коммитов в репо | 171 |

> Примечание: в прошлых версиях в web-ui учитывался сборочный каталог `.nuxt`
> (22 файла, 1022 строки). Сейчас считаются только исходники.

## Статус проверок

- `uv run pytest`: **116 passed** (10 файлов; включая 3 интеграционных против `newsdigest_test` в Docker).
- ruff `uv run ruff check .`: **All checks passed**.
- ruff `uv run ruff format --check .`: **46 files already formatted**.
- Web UI: `pnpm build` — успешно.