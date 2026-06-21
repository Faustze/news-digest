# News Digest

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A simple tool for aggregating news and generating digests.

The project collects news from multiple sources, processes articles, and generates concise summaries in a unified digest.

## Features

- Aggregates news from different sources
- Creates short, readable digests
- Easy to run via GitHub Actions or locally
- Easily extensible with new feeds or summarization logic

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Faustze/news-digest.git
cd news-digest
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file if needed:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the project

```bash
python news_pipeline.py
```

## Project Structure

```text
news-digest/
├── .github/workflows/daily_digest.yml    # GitHub Actions workflow that runs the daily digest pipeline
├── output/                               # Generated digests
├── config.yaml                           # Configuration for news_pipeline
├── news_pipeline.py                      # News feed pipeline — LangChain + Groq (free tier). Fetches, filters, and summarizes news from RSS feeds.
├── send_telegram.py                      # Optional: sends the latest digest to a Telegram chat. Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
├── requirements.txt                      # Python dependencies
└── pyproject.toml                       # Project metadata and dependency management (uv-compatible)
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE.md) license.

You are free to share and adapt the material, as long as you provide attribution and do not use it for commercial purposes.
