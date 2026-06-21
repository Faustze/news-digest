# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.x     | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by opening a [GitHub Issue](https://github.com/Faustze/news-digest/issues) with the `security` label.

Please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if any)

I will acknowledge receipt within 48 hours and aim to provide a fix or mitigation within a reasonable timeframe.

## Security Considerations

This project handles the following sensitive data:

- **Telegram Bot Token** — stored as a GitHub Actions secret (`TELEGRAM_BOT_TOKEN`). Never commit this to the repository.
- **Telegram Chat ID** — stored as a GitHub Actions secret (`TELEGRAM_CHAT_ID`). Never commit this to the repository.
- **Groq API Key** — stored as a GitHub Actions secret (`GROQ_API_KEY`). Never commit this to the repository.

### Best Practices for Contributors

1. **Never commit secrets** — API keys, tokens, and credentials must only be stored as environment variables or GitHub Actions secrets.
2. **Keep dependencies up to date** — Run `uv sync` regularly to get security patches.
3. **Review feed sources** — RSS feeds are fetched and processed by an LLM. Malicious feed content could influence output. Only add trusted feed sources to `config.yaml`.
