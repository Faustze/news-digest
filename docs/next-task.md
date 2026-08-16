# next-task — Текущее ТЗ

## Задача

**B-03: Мульти-провайдер LLM**

Разрешить выбирать LLM-провайдера через `config.yaml` (Groq по умолчанию, OpenAI, Anthropic, локальный Ollama) без переписывания кода пайплайна.

## Что сделать

1. **Новый модуль `news/llm.py`** — фабрика, которая по конфигу возвращает LangChain chat-модель:
   - `build_llm(config) -> BaseChatModel`;
   - `provider`: `groq` (по умолчанию), `openai`, `anthropic`, `ollama`;
   - `model`, `temperature`, `max_tokens` из конфига;
   - ключ API из окружения: `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (Ollama — без ключа).

2. **Интеграция в `news_pipeline.py`** — заменить прямое `ChatGroq(...)` на `build_llm(config)` (передаётся в `classify_batch` и `generate_digest_summary`).

3. **Проверка конфига** — неизвестный `provider` → fail early с понятной ошибкой; отсутствие ключа для провайдера → понятная ошибка.

4. **Тесты `tests/test_llm.py`** (без живых API):
   - каждый провайдер возвращает модель ожидаемого типа;
   - неизвестный провайдер → ошибка;
   - отсутствующий ключ → ошибка (кроме ollama).

## Проверка

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```

## Следующая задача после этого

B-05 — Веб-интерфейс для просмотра архива дайджестов.