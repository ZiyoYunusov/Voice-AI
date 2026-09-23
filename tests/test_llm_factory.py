import pytest

from llm_factory import UnsupportedLLMProviderError, create_llm_service
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")

    service = create_llm_service("Ты голосовой ассистент")

    assert isinstance(service, GoogleLLMService)
    assert service._settings.system_instruction == "Ты голосовой ассистент"


def test_gemini_provider_uses_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-test")

    service = create_llm_service("prompt")

    assert isinstance(service, GoogleLLMService)
    assert service._settings.model == "gemini-custom-test"


def test_anthropic_provider_uses_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-custom-test")

    service = create_llm_service("prompt")

    assert isinstance(service, AnthropicLLMService)
    assert service._settings.model == "claude-custom-test"


def test_anthropic_provider_default_model_is_haiku_4_5(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    service = create_llm_service("prompt")

    assert service._settings.model == "claude-haiku-4-5-20251001"


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(UnsupportedLLMProviderError):
        create_llm_service("prompt")


def test_missing_api_key_raises_key_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(KeyError):
        create_llm_service("prompt")
