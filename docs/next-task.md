# next-task — Текущее ТЗ

## Задача

**Phase 9: UX дайджеста**

Сделать дайджест читабельнее и персонализированнее: summary, которое адаптируется под профиль, и структура каждой новости с категорией и тегами.

## Что сделать

1. **Персонализированное summary** — `generate_digest_summary` уже получает profile; учесть в промпте `reading_time`, `priority` и интересы, чтобы объём/акценты summary менялись под пользователя.

2. **Категория + теги в каждой новости** — в `render_telegram` добавить в каждый пункт категорию и теги статьи (есть в `item["tags"]` и `item["category"]` после классификации).

3. **Markdown fallback** — сохранить существующее поведение: если Telegram не принял Markdown-версию, отправить plain-text fallback (кнопки не теряются).

## Проверка

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
uv run python news_pipeline.py config.yaml
```

## Следующая задача после этого

D-02 — GitHub Actions CI/CD для деплоя Web UI на GitHub Pages.