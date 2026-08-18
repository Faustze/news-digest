"""
LLM provider factory: build a LangChain chat model from config.yaml.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_PROVIDERS = ("groq", "openai", "anthropic", "ollama")

_PROVIDER_KEY = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.2",
}


def _default_model(provider: str) -> str:
    return _DEFAULT_MODELS[provider]


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

    Raises a clear ValueError for an unknown provider or a missing API key
    (Ollama is local and needs no key).
    """
    provider = str(config.get("provider", "groq")).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    model = config.get("model") or _default_model(provider)
    temperature = config.get("temperature", 0)
    max_tokens = config.get("max_tokens", 4096)

    if provider == "groq":
        _require_env(_PROVIDER_KEY["groq"], provider)
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

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
