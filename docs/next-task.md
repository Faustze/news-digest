# next-task — Текущее ТЗ

## Задача

**C-05 / D-01: GitHub Actions для lint + test**

Добавить в `.github/workflows/daily_digest.yml` шаги для ruff check и pytest перед запуском пайплайна.

## Что сделать

1. Добавить шаг `Lint` в workflow:
   ```yaml
   - name: Lint
     run: |
       uv run ruff check .
       uv run ruff format --check .
   ```

2. Добавить шаг `Test` в workflow:
   ```yaml
   - name: Test
     run: uv run pytest tests/ -v
   ```

3. Порядок шагов в workflow:
   - checkout
   - setup python
   - install dependencies
   - **lint** ← новый
   - **test** ← новый
   - poll feedback
   - run pipeline
   - send to telegram
   - commit

4. Если lint или test упадёт — pipeline не запускать.

## Проверка

```bash
# Локально
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```

## Следующая задача после этого

Phase 8 — динамический cutoff по frequency (daily=24h, weekly=7d).

> Примечание: D-02 (деплой Web UI) будет реализован через GitHub Actions CI/CD для GitHub Pages. См. `docs/PLAN.md`.
