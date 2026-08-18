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
| `news/console.py` | Консольные проверки БД (точка входа, переписан на репозитории) |
| `news/db.py` | SQLAlchemy engine + SessionLocal |
| `news/repositories/` | Слой доступа к данным: user/article/feedback (raw `text()`, без ORM) |
| `web-ui/` | Nuxt UI для онбординга и правки профиля |

## Метрики (18.08.2026)

| Показатель | Значение |
|------------|----------|
| Файлы исходников (py src) | 12 |
| Файлы тестов | 8 |
| Файлы web-ui (vue/ts, без node_modules/.nuxt) | 7 |
| Строки кода (py src + тесты) | 2161 |
| Строки кода (web-ui, исходники) | 1511 |
| Тестовые функции | 97 |
| TODO/FIXME в исходниках | 0 |
| Коммитов в ветке | 156 |

> Примечание: в прошлых версиях в web-ui учитывался сборочный каталог `.nuxt`
> (22 файла, 1022 строки). Сейчас считаются только исходники.

## Статус проверок

- `tests/test_feedback_repository.py`: **3 passed** (интеграционные, БД `newsdigest_test` в Docker).
- ruff `news/ tests/`: **All checks passed**.
- Полный прогон `uv run pytest`: не выполнялся в этой сессии.