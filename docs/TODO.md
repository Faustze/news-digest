# TODO.md — бэклог задач

> Новые идеи добавлять сюда.

## Бэклог

- [ ] Добавить тему "Python/Django" в фиды (Django blog, PythonWeekly)
- [ ] Добавить тему "DevOps/Infrastructure" (Docker blog, Kubernetes blog, HashiCorp)
- [ ] Добавить тему "Mobile" (React Native, Flutter, Expo)
- [ ] Улучшить промпт фильтрации: добавить "isNew" (первый раз в дайджесте vs повтор)
- [ ] Поддержка нескольких LLM-провайдеров (OpenAI, Anthropic, Ollama для локального запуска)
- [ ] Формат дайджеста: HTML-письмо вместо plain text (для email-рассылки)
- [ ] Веб-интерфейс для просмотра архива дайджестов
- [ ] Метрики: сколько статей обработано, сколько отфильтровано, время выполнения
- [ ] Тесты: unit-тесты для fetch/filter/summarize шагов (покрыты: classify, schedule, telegram; нет: fetch, summarize)
- [ ] Сформировать mock новости без запросов через Groq, чтобы не тратить токены. Но при этом полностью тестировать работу pipeline. Можно на тестовых данных из интернета
