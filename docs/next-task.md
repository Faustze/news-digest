# next-task — Текущее ТЗ

## Задача

**Phase 8: Динамический cutoff по frequency**

Сделать так, чтобы `cutoff_hours` вычислялся как runtime-значение на основе `user-profile.json.general.frequency`, не записывая пользовательские предпочтения обратно в `config.yaml` (технический конфиг — в `config.yaml`, предпочтения — в `user-profile.json`).

## Что сделать

1. Добавить функцию `cutoff_hours_for_frequency(profile: UserProfile) -> int` в `news/profile.py` или отдельный модуль `news/schedule.py`.

2. Поддерживаемые значения:
   - `morning` / `evening` / `daily` → 24 часа
   - `weekly` → 7 дней (168 часов)
   - `important_only` → 24 часа

3. Обновить `news_pipeline.py`:
   - Вместо `config.get("cutoff_hours", 24)` использовать значение из профиля.
   - Сохранить fallback на 24 часа, если frequency не задан.

4. Добавить тесты в `tests/test_schedule.py`:
   - daily → 24
   - weekly → 168
   - important_only → 24
   - отсутствующий frequency → 24

## Проверка

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
uv run python news_pipeline.py config.yaml
```

## Следующая задача после этого

Phase 9 — UX дайджеста (персонализация summary, категория + теги в каждой новости, Markdown fallback).

> Примечание: D-02 (деплой Web UI) будет реализован через GitHub Actions CI/CD для GitHub Pages. См. `docs/PLAN.md`.
