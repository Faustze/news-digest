# AGENTS.md — News Digest Development Instructions

## Project mission

News Digest is a single-user, serverless news personalization project.

The existing runtime is:

- Python
- RSS/Feedparser
- LangChain
- Groq free tier
- Telegram Bot API
- GitHub Actions cron

A static Nuxt Web UI is being added for user-friendly profile configuration.

The project must remain runnable without a VPS, always-on backend, or database.

## Read first

Before making changes, inspect:

- `README.md`
- `config.yaml`
- `news_pipeline.py`
- `send_telegram.py`
- `.github/workflows/daily_digest.yml`
- `pyproject.toml`
- `requirements.txt`

Do not assume the architecture from this document is already implemented. Treat the repository code as the source of truth for current behavior.

## Core constraints

1. Do not introduce a backend server.
2. Do not introduce a database.
3. Do not introduce a VPS or always-on process.
4. Do not add authentication for the first single-user version.
5. Do not expose RSS/source configuration to the normal user UI.
6. Keep Groq API usage within the free-tier/rate-limit constraints configured by the project.
7. Never commit API keys, Telegram tokens, GitHub tokens, or other secrets.
8. Do not store secrets in `localStorage`.
9. Keep the Web UI static and client-side.
10. Preserve local CLI execution of the Python pipeline.
11. Preserve GitHub Actions cron and manual execution.
12. Prefer small, testable functions over a large rewrite.

## Architecture rules

### User profile

User preferences belong in `user-profile.json`.

Technical configuration belongs in `config.yaml`.

Feedback belongs in `feedback.json`.

Do not reintroduce user-specific topic lists into `config.yaml` after migration.

### Web UI

The first version uses:

- Nuxt/Vue
- client-side state
- local storage
- import/export of `user-profile.json`

Do not add GitHub OAuth or GitHub API write access unless a later roadmap item explicitly asks for it.

The UI is for one person and should prioritize usability over account systems.

### Python pipeline

Keep a clear separation between:

1. fetching
2. normalization
3. deduplication
4. LLM classification
5. personalization/scoring
6. final selection
7. summarization
8. rendering
9. Telegram delivery
10. feedback persistence

Do not put all logic back into `news_pipeline.py` if it becomes difficult to test. Extract small modules when useful.

## User profile semantics

### Interest values

Every subtopic has:

- `0` = explicitly disabled
- `1..5` = increasing interest

Never treat `0` as a neutral/default value. It is an explicit negative preference.

### Category enablement

A category has `enabled: boolean`.

Disabled categories must not contribute candidates to the digest unless an explicit future feature overrides this behavior.

### Explicit preference precedence

Use this precedence when resolving conflicts:

1. hard exclusions
2. explicit subtopic `0`
3. enabled category/subtopic weights
4. regional preferences
5. source reliability preference
6. feedback signals
7. free-text personal context
8. generic popularity/importance

Free-text personal context must not override hard exclusions or explicit zero-interest settings.

## Canonical categories

The project uses exactly these 12 user-facing categories:

1. AI
2. Technology
3. Science
4. Space
5. Gadgets
6. Games
7. Business
8. Finance
9. Running
10. Movies & Series
11. Music
12. World

Each category has exactly six canonical subtopics as defined in `ROADMAP.md`.

Do not silently rename IDs after profile files have been created. If an ID must change, add explicit migration logic.

## LLM rules

The model is an assistant inside an application, not the source of truth for configuration.

Always pass structured profile information to the pipeline.

Prefer machine-readable output from classification/ranking prompts.

Validate all LLM JSON before using it.

Handle malformed/partial model responses gracefully:

- log a concise warning
- skip the invalid result where possible
- never crash the whole digest because one candidate failed

Prompts should explicitly forbid:

- invented facts
- invented source information
- invented publication dates
- treating advertisements as editorial news
- overriding user exclusions

## News identity

Every article used for feedback must have a stable ID.

The ID should be derived from canonical article data, typically normalized URL with a title/source fallback if necessary.

Do not derive IDs from generated summaries, which can change between runs.

## Feedback rules

Supported reactions:

- `useful`
- `not_interesting`
- `hide_similar`

A reaction applies to one specific article.

Feedback should be compact. Do not persist entire article bodies unless a future feature explicitly requires it.

When aggregating feedback:

- recent feedback should be more influential than ancient feedback;
- `hide_similar` should be stronger than `not_interesting`;
- positive feedback should raise similar topics, not blindly force the same source;
- one reaction should not permanently distort the profile.

## RSS/source rules

RSS sources are application configuration, not user preferences.

Each source should have useful metadata where available:

- name
- URL
- tags/categories
- reliability tier
- optional region

Verify new feed URLs before committing them.

Do not add dozens of low-signal feeds only to increase volume. Relevance and source quality are more important than raw feed count.

## Web UI/UX rules

The UI must be understandable to someone who knows nothing about IT.

Do not expose terms like:

- RSS
- prompt
- LLM
- API
- embedding
- ranking score
- source reliability tier

inside normal user-facing copy.

Use human language such as:

- "Что тебе интересно?"
- "Насколько тебе это интересно?"
- "Что тебе точно не показывать?"
- "Сколько времени ты хочешь тратить на новости?"
- "На каком языке рассказывать новости?"

### Onboarding

Onboarding is sequential.

Only selected categories receive detailed subtopic steps.

Each selected category shows six subtopics with a 0–5 interest control.

Always make progress visible.

Allow going back without losing entered values.

Provide a skip action where the roadmap specifies one.

### Persistent profile

After onboarding, users should edit settings from a normal profile/settings page instead of repeating the entire wizard.

Support:

- export profile
- import profile
- reset profile

Do not silently overwrite an imported profile.

## Telegram rules

Keep the digest readable on mobile.

Each news item should have its own feedback controls.

Do not create one global thumbs-up/thumbs-down that ambiguously refers to the entire digest.

Respect Telegram message limits. Keep the existing chunking/fallback behavior unless there is a tested reason to change it.

Do not send empty digests just to satisfy a schedule.

## GitHub Actions rules

Keep secrets in GitHub Actions secrets.

Preserve:

- Python 3.12 unless a dependency requires a deliberate change.
- cron execution.
- `workflow_dispatch`.
- `contents: write` only when actually needed by the workflow.

When committing generated state, explicitly list which files may be written by automation.

Never blindly `git add .` in automated workflows.

## Testing rules

Every deterministic business rule must be testable without an API key.

At minimum test:

- profile schema validation
- default/missing values
- category IDs and subtopic IDs
- zero vs positive interests
- exclusions
- regional filters/signals
- source reliability behavior
- stable article IDs
- feedback aggregation
- time-budget selection
- frequency/cutoff calculation
- Telegram callback parsing
- message chunking/rendering

Use mocks/fixtures for LLM responses.

Do not make tests dependent on live RSS feeds or live Telegram/Groq calls.

## Error handling

Expected network failures must not destroy the entire run.

For RSS failures:

- log source name and URL
- continue with remaining feeds

For one failed LLM batch:

- log the batch failure
- continue with other batches

For invalid user profile:

- fail early with an actionable error
- do not silently fall back to unrelated default interests

For Telegram failure:

- surface the failure clearly in GitHub Actions
- do not claim delivery succeeded

## Code style

Prefer:

- small functions
- explicit names
- type hints where practical
- standard library first
- minimal dependencies
- simple data structures

Avoid:

- unnecessary frameworks
- service layers without a concrete need
- generic abstractions introduced before duplication exists
- hidden global mutable state
- unnecessary asynchronous complexity

Keep prompts close to the code that uses them, but extract them into dedicated files/modules when they become large enough to test/read independently.

## Dependency policy

Before adding a dependency, ask:

1. Is it actually required?
2. Can the standard library do it?
3. Does it work cleanly in GitHub Actions?
4. Does it increase cold-start/time/cost?
5. Does it create a new service/runtime requirement?

The default answer should be to keep the dependency footprint small.

## Change discipline

Do not rewrite the repository just to match the roadmap.

Implement one phase at a time.

After each major phase:

1. run tests
2. run the pipeline locally with fixtures/mocks if possible
3. inspect the generated profile/digest
4. verify GitHub Actions YAML
5. update documentation

Keep backward-compatible migrations where reasonable.

## Definition of done for an implementation task

A task is not complete merely because code compiles.

Verify:

- the intended UX exists;
- the profile format is valid;
- deterministic logic has tests;
- no secrets were introduced;
- no backend/database/VPS was added without explicit approval;
- existing local execution still works;
- GitHub Actions still has a clear path to execute the workflow;
- the relevant documentation is updated.

## Current source-of-truth documents

- `ROADMAP.md` — product/implementation roadmap and canonical requirements.
- `AGENTS.md` — coding and agent behavior rules.

When these documents conflict with vague assumptions in old code, follow the explicit roadmap, but first preserve backward compatibility where feasible.

---

## Быстрые команды

- `uv sync` — установка зависимостей
- `uv run python news_pipeline.py config.yaml` — запуск пайплайна
- `uv run python send_telegram.py` — отправка в Telegram
- `uv run pytest` — тесты
- `uv run ruff check .` — линтер
- `uv run ruff format .` — форматирование

## Структура

- `news_pipeline.py` — основной пайплайн (fetch → filter → summarize → output)
- `send_telegram.py` — отправка дайджеста в Telegram
- `config.yaml` — фиды, темы, модель, настройки
- `output/` — сгенерированные дайджесты (коммитятся в repo)
- `.github/workflows/daily_digest.yml` — ежедневный cron (04:00 UTC)
- `docs/` — рабочие документы

## Процесс

1. Новые идеи → `docs/TODO.md`
2. Архитектор (или ты сам) переносит в `docs/PLAN.md`, оформляет ТЗ в `docs/next-task.md`
3. Исполняешь задачу, проверяешь (lint + запуск), коммитишь
4. Обновляешь `docs/STATE.md` (что сделано, решения)

## Деплой

GitHub Actions: `daily_digest.yml` — cron 04:00 UTC, коммитит output, отправляет в Telegram.
Секреты: `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Конвенции

- Python 3.10+, type hints
- Pydantic для моделей
- LangChain для LLM-цепочек
- RSS: feedparser
- Конфиг: YAML
