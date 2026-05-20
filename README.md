# Дайджест новостей

Простой инструмент для агрегирования новостей и создания дайджестов.

Проект собирает новости из нескольких источников, обрабатывает статьи и формирует краткие сводки в едином дайджесте.

## Функционал

- Агрегирует новости из разных источников
- Создает короткие читабельные дайджесты
- Простой запуск через GitHub Actions или локально
- Легко расширить за счет новых каналов или логики обобщения

## Запустить локально

### 1. Склонируй репозиторий

```bash
git clone https://github.com/Faustze/news-digest.git
cd news-digest
```

### 2. Установи зависимости

```bash
pip install -r requirements.txt
```

### 3. Настройка переменных среды

При необходимости создайте файл `.env`:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
GROQ_API_KEY=your_al_api_key
```

### 4. Запуск проекта

```bash
python main.py
```

## Структура проекта

```text
news-digest/
├── .github/workflows/daily_digest.yml    # Workflow автоматически запускает ежедневный pipeline для генерации новостного дайджеста
├── output/                               # Сгенерированные дайджесты
├── config.yaml                           # Конфиг для news_pipeline
├── news_pipeline.py                      # Канал подачи новостей — LangChain + Groq (уровень бесплатного пользования). Извлекает, фильтрует и обобщает новости из RSS-каналов.
├── send_telegram.py                      # Необязательно: отправляет последний дайджест в чат Telegram. Считывает TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID из среды.
└── requirements.txt                      # Зависимости
```

## License

MIT
