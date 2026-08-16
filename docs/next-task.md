# next-task — Текущее ТЗ

## Задача

**D-02: GitHub Actions CI/CD для деплоя Web UI на GitHub Pages**

Собирать статический Nuxt-интерфейс (`web-ui/`) и публиковать на GitHub Pages. UI остаётся статичным и клиентским — без бэкенда, VPS и БД.

## Что сделать

1. **Отдельный workflow** `.github/workflows/deploy_ui.yml` (не смешивать с `daily_digest.yml`):
   - триггеры: `push` в `main` по путям `web-ui/**` + `workflow_dispatch`;
   - `contents: write` + `pages: write` + `id-token: write` (deployment);
   - `pnpm install --frozen-lockfile` + `pnpm generate` (статический output).

2. **Публикация** — деплой собранного `.output/public` на GitHub Pages:
   - настроить `nuxt.config.ts` на `ssr: false` и корректный `baseURL` (Pages-путь);
   - artifact `actions/upload-pages-artifact` + `actions/deploy-pages`.

3. **Проверки**: линт/тесты Web UI не ломают деплой; локально `pnpm generate` собирается без ошибок.

## Проверка

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
cd web-ui && pnpm generate
```

## Следующая задача после этого

B-03 — Мульти-провайдер LLM (OpenAI, Anthropic, Ollama для локального запуска).