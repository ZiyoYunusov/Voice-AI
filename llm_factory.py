"""Создаёт LLM-сервис по переменной окружения LLM_PROVIDER.

LLM_PROVIDER=gemini (по умолчанию) -> GoogleLLMService, модель из GEMINI_MODEL.
LLM_PROVIDER=anthropic -> AnthropicLLMService, модель из ANTHROPIC_MODEL.

Переключение провайдера не требует правок кода — только .env.
"""

import os

from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import LLMService

SUPPORTED_PROVIDERS = ("gemini", "anthropic")


class UnsupportedLLMProviderError(ValueError):
    """LLM_PROVIDER не равен одному из поддерживаемых значений."""


def create_llm_service(system_instruction: str) -> LLMService:
    """Создаёт LLM-сервис для выбранного в .env провайдера.

    Args:
        system_instruction: Системный промпт (уже выбранный по BOT_LANGUAGE).

    Raises:
        UnsupportedLLMProviderError: Если LLM_PROVIDER не "gemini" и не "anthropic".
        KeyError: Если для выбранного провайдера не задан соответствующий API-ключ.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        return GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(
                model=model,
                system_instruction=system_instruction,
            ),
        )

    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        return AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                model=model,
                system_instruction=system_instruction,
            ),
        )

    raise UnsupportedLLMProviderError(
        f"LLM_PROVIDER={provider!r} не поддерживается. Используйте одно из: "
        f"{', '.join(SUPPORTED_PROVIDERS)}"
    )
