"""
Tests for news.llm module (no live API calls).
"""

import pytest

from news.llm import SUPPORTED_PROVIDERS, TokenCounter, build_llm

BASE_CONFIG = {"model": "test-model", "temperature": 0, "max_tokens": 512}


class TestBuildLlm:
    def test_returns_groq_model(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "key")
        from langchain_groq import ChatGroq

        assert isinstance(build_llm({**BASE_CONFIG, "provider": "groq"}), ChatGroq)

    def test_returns_openai_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        from langchain_openai import ChatOpenAI

        assert isinstance(build_llm({**BASE_CONFIG, "provider": "openai"}), ChatOpenAI)

    def test_returns_anthropic_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        from langchain_anthropic import ChatAnthropic

        assert isinstance(
            build_llm({**BASE_CONFIG, "provider": "anthropic"}), ChatAnthropic
        )

    def test_returns_ollama_model(self, monkeypatch):
        from langchain_ollama import ChatOllama

        assert isinstance(build_llm({**BASE_CONFIG, "provider": "ollama"}), ChatOllama)

    def test_provider_defaults_to_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "key")
        from langchain_groq import ChatGroq

        assert isinstance(build_llm(BASE_CONFIG), ChatGroq)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_llm({**BASE_CONFIG, "provider": "deepseek"})

    def test_missing_groq_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            build_llm({**BASE_CONFIG, "provider": "groq"})

    def test_missing_openai_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            build_llm({**BASE_CONFIG, "provider": "openai"})

    def test_missing_anthropic_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            build_llm({**BASE_CONFIG, "provider": "anthropic"})

    @pytest.mark.parametrize("provider", SUPPORTED_PROVIDERS)
    @pytest.mark.parametrize("model", [None, "", "  "])
    def test_missing_model_raises(self, monkeypatch, provider, model):
        monkeypatch.setenv("GROQ_API_KEY", "key")
        config = {"provider": provider}
        if model is not None:
            config["model"] = model
        with pytest.raises(ValueError, match="No model configured"):
            build_llm(config)

    def test_provider_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "key")
        from langchain_groq import ChatGroq

        assert isinstance(build_llm({**BASE_CONFIG, "provider": "GROQ"}), ChatGroq)

    def test_groq_reasoning_effort_passed_when_configured(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "key")
        llm = build_llm({**BASE_CONFIG, "provider": "groq", "reasoning_effort": "low"})
        assert llm.reasoning_effort == "low"

    def test_groq_reasoning_effort_absent_by_default(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "key")
        assert build_llm({**BASE_CONFIG, "provider": "groq"}).reasoning_effort is None


class TestTokenCounter:
    def test_sums_reported_usage(self):
        counter = TokenCounter()
        msg = type("Msg", (), {"usage_metadata": {"total_tokens": 40}})()
        assert counter.add(msg) == 40
        counter.add(msg)
        assert counter.total == 80

    def test_missing_usage_counts_zero(self):
        counter = TokenCounter()
        assert counter.add(object()) == 0
        assert counter.total == 0
