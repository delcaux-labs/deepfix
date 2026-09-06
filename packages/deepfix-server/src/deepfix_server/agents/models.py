"""Agno model initialization and configuration for DeepFix agents."""

from __future__ import annotations

from typing import Any, Optional

from agno.models.base import Model
from agno.models.openai import OpenAIChat
from agno.models.openai.like import OpenAILike

from ..config import LLMConfig
from ..logging import get_logger

LOGGER = get_logger(__name__)


def create_agno_model(config: LLMConfig) -> Model:
    """Create an Agno Model instance from an LLMConfig.

    Supports OpenAI models as well as LiteLLM or OpenAI-compatible endpoints
    via `OpenAILike`.

    Args:
        config: LLM configuration.

    Returns:
        Configured Agno Model instance (OpenAILike or OpenAIChat).

    Raises:
        ValueError: If config is None or api_key is missing.
    """
    assert isinstance(config, LLMConfig), f"Expected config to be an instance of LLMConfig, got {type(config)}"
    assert config.api_key, "No LLM API key configured. Please provide LLM configuration."

    model_id = config.model_name
    kwargs: dict[str, Any] = {
        "id": model_id,
        "api_key": config.api_key,
        "temperature": config.temperature if config.temperature is not None else 0.7,
        "max_tokens": config.max_tokens or 8000,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
        LOGGER.debug("Initializing Agno OpenAILike model %s (base_url=%s)", model_id, config.base_url)
        return OpenAILike(**kwargs)

    LOGGER.debug("Initializing Agno OpenAIChat model %s", model_id)
    return OpenAIChat(**kwargs)
