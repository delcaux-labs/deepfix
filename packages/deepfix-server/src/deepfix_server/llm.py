"""Pydantic AI model factory for DeepFix server agents.

Provides helpers to create Pydantic AI Model and Agent instances from the
project's existing LLMConfig, replacing dspy.LM + dspy.context(lm=...).
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import LLMConfig


def create_model(config: Optional[LLMConfig] = None) -> OpenAIChatModel:
    """Create a Pydantic AI OpenAIChatModel from an LLMConfig.

    If config is None and no env API key is set, returns None so callers
    can fall back to the Pydantic AI test model.

    Args:
        config: LLM configuration. If None, reads from environment.

    Returns:
        Configured OpenAIChatModel instance, or None if no credentials.
    """
    if config is None:
        config = LLMConfig(
            model_name=os.getenv("DEEPFIX_LLM_MODEL_NAME", "gpt-4o"),
            api_key=os.getenv("DEEPFIX_LLM_API_KEY"),
            base_url=os.getenv("DEEPFIX_LLM_BASE_URL"),
            temperature=float(os.getenv("DEEPFIX_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("DEEPFIX_LLM_MAX_TOKENS", "8000")),
        )

    if not config.api_key:
        return None

    provider_kwargs: dict = {}
    if config.api_key:
        provider_kwargs["api_key"] = config.api_key
    if config.base_url:
        provider_kwargs["base_url"] = config.base_url

    provider = OpenAIProvider(**provider_kwargs) if provider_kwargs else "openai"

    return OpenAIChatModel(
        model_name=config.model_name,
        provider=provider,
    )


def create_agent_for_analysis(
    config: Optional[LLMConfig] = None,
    system_prompt: str = "",
    output_type: Optional[type] = None,
) -> PydanticAgent:
    """Create a Pydantic AI Agent configured for artifact analysis.

    Args:
        config: LLM configuration. If None, reads from environment.
        system_prompt: System prompt for the agent.
        output_type: Optional Pydantic model for structured output (e.g.
            ArtifactAnalysisResult or CrossArtifactReasoningResult).

    Returns:
        A PydanticAgent ready for ``.run()`` calls.
    """
    if config is None:
        config = LLMConfig(
            model_name=os.getenv("DEEPFIX_LLM_MODEL_NAME", "gpt-4o"),
            api_key=os.getenv("DEEPFIX_LLM_API_KEY"),
            base_url=os.getenv("DEEPFIX_LLM_BASE_URL"),
            temperature=float(os.getenv("DEEPFIX_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("DEEPFIX_LLM_MAX_TOKENS", "8000")),
        )

    model = create_model(config)
    if model is None:
        # No API key — use Pydantic AI test model (no real LLM calls)
        return PydanticAgent("test", output_type=output_type, system_prompt=system_prompt)

    model_settings: dict = {}
    if config.temperature is not None:
        model_settings["temperature"] = config.temperature
    model_settings["max_tokens"] = config.max_tokens or 8000

    return PydanticAgent(
        model=model,
        output_type=output_type,
        system_prompt=system_prompt,
        model_settings=model_settings,
    )