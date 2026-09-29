"""
LLM provider factory: build a LangChain chat model from config.yaml.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_PROVIDERS = ("groq", "openai", "anthropic", "ollama")


class TokenCounter:
    """Sums tokens reported by the provider (LangChain `usage_metadata`)."""

    def __init__(self) -> None:
        self.total = 0

    def add(self, message: object) -> int:
        """Count one response; returns its tokens (0 if not reported)."""
        usage = getattr(message, "usage_metadata", None) or {}
        tokens = int(usage.get("total_tokens", 0) or 0)
        self.total += tokens
        return tokens


_PROVIDER_KEY = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _require_env(name: str, provider: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(
            f"Missing {name} for LLM provider '{provider}'. "
            f"Set it in the environment (GitHub Actions secret) or pick another "
            f"provider in config.yaml."
        )
    return value


def build_llm(config: dict) -> BaseChatModel:
    """
    Build a chat model for the configured LLM provider.

    Raises a clear ValueError for an unknown provider, a missing model or a
    missing API key (Ollama is local and needs no key). There are no built-in
    default models: provider model lists change too often for them to stay valid.
    """
    provider = str(config.get("provider", "groq")).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    model = str(config.get("model") or "").strip()
    if not model:
        raise ValueError(
            f"No model configured for LLM provider '{provider}'. "
            f"Set `model:` in config.yaml."
        )
    temperature = config.get("temperature", 0)
    max_tokens = config.get("max_tokens", 4096)

    if provider == "groq":
        _require_env(_PROVIDER_KEY["groq"], provider)
        from langchain_groq import ChatGroq

        extra = {}
        if config.get("reasoning_effort"):
            # Reasoning models (gpt-oss, qwen3): hidden reasoning is billed as
            # output and eats max_tokens, truncating the JSON answer.
            extra["reasoning_effort"] = config["reasoning_effort"]
        return ChatGroq(
            model=model, temperature=temperature, max_tokens=max_tokens, **extra
        )

    if provider == "openai":
        _require_env(_PROVIDER_KEY["openai"], provider)
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "anthropic":
        _require_env(_PROVIDER_KEY["anthropic"], provider)
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature, num_predict=max_tokens)
