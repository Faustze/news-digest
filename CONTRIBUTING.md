# Contributing to News Digest

Thank you for your interest in contributing! This document outlines how to get involved.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/news-digest.git`
3. Install dependencies: `uv sync` (or `pip install -r requirements.txt`)
4. Create a `.env` file with your API keys (see README.md)
5. Create a branch for your changes: `git checkout -b feature/your-feature`

## How to Contribute

### Adding RSS Feeds

The easiest way to contribute is by adding new RSS feeds to `config.yaml`. Make sure the feed is:

- Relevant to the project's topics (frontend, Vue, Nuxt, CSS, AI for developers, etc.)
- Actively maintained and regularly updated
- Accessible without authentication

### Improving the Pipeline

- Better prompt engineering for summaries
- Improved deduplication of articles
- Support for additional LLM providers
- Better formatting of digests

### Bug Fixes

If you find a bug, please open an issue describing the problem and, if possible, submit a fix via pull request.

## Code Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Keep functions focused and small
- Add docstrings to public functions

## Pull Request Process

1. Update the README.md if your change affects usage or setup
2. Update `config.yaml` if you add or remove feeds
3. Ensure the pipeline runs successfully: `python news_pipeline.py`
4. Submit a pull request with a clear description of your changes

## License

By contributing to this project, you agree that your contributions will be licensed under the [CC BY-NC 4.0](LICENSE.md) license.
