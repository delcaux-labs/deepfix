"""Perplexity Sonar integration via OpenRouter using Pydantic AI for AI-powered research.

Replaces DSPy Signatures and Modules with Pydantic AI Agents.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from .base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class PerplexityError(Exception):
    """Base exception for Perplexity-related errors."""
    pass


class PerplexityConfigError(PerplexityError):
    """Raised when Perplexity client is not properly configured."""
    pass


class PerplexityAPIError(PerplexityError):
    """Raised when Perplexity API call fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.original_error = original_error


class PerplexityResponseError(PerplexityError):
    """Raised when response parsing or validation fails."""
    pass


# Perplexity Sonar models available through OpenRouter
PERPLEXITY_MODELS = {
    "sonar": "perplexity/sonar",
    "sonar-pro": "perplexity/sonar-pro",
    "sonar-reasoning": "perplexity/sonar-reasoning",
}


class PerplexityConfig(BaseModel):
    """Configuration for Perplexity Sonar client."""

    api_key: Optional[str] = Field(None, description="OpenRouter API key")
    model: Literal["sonar", "sonar-pro", "sonar-reasoning"] = Field(
        "sonar", description="Perplexity model variant"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum response tokens")


# ============================================================================
# System prompts for each research depth
# ============================================================================

RESEARCH_SYSTEM_PROMPT = (
    "You are a helpful research assistant specializing in machine learning, "
    "deep learning, data science, and AI model diagnostics. Provide accurate, "
    "well-sourced answers with citations. Focus on practical, actionable "
    "information relevant to ML practitioners."
)

BRIEF_SYSTEM_PROMPT = (
    "Provide a brief, focused answer in 2-3 paragraphs. "
    "Include only the most essential information and key citations. "
    "Focus on being concise while still being accurate and helpful."
)

DETAILED_SYSTEM_PROMPT = (
    "Provide a detailed answer covering the main aspects of the topic. "
    "Include relevant examples, best practices, and multiple citations. "
    "Cover the topic comprehensively while remaining focused."
)

COMPREHENSIVE_SYSTEM_PROMPT = (
    "Provide a comprehensive analysis covering all major aspects. "
    "Include background context, current best practices, trade-offs, "
    "practical recommendations, and extensive citations from authoritative sources."
)


# ============================================================================
# Perplexity Sonar Retriever
# ============================================================================


class PerplexitySonarRetriever(BaseRetriever):
    """Perplexity Sonar integration via OpenRouter using Pydantic AI for research.

    Features:
        - AI-powered web search with reasoning
        - Automatic citation extraction
        - Multiple model variants (sonar, sonar-pro, sonar-reasoning)
        - Configurable research depth (brief, detailed, comprehensive)

    Example:
        >>> retriever = PerplexitySonarRetriever(model="sonar-pro")
        >>> results = await retriever.retrieve(
        ...     "What are the best practices for training vision transformers?"
        ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Literal["sonar", "sonar-pro", "sonar-reasoning"] = "sonar",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize Perplexity Sonar client via OpenRouter with Pydantic AI.

        Args:
            api_key: OpenRouter API key. If not provided, loads from
                     OPENROUTER_API_KEY environment variable.
            model: Perplexity model variant to use.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in response (None for model default).
        """
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model
        self.model = PERPLEXITY_MODELS[model]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_base = "https://openrouter.ai/api/v1"

        # Pydantic AI agents are created lazily
        self._researcher_agent = None
        self._brief_agent = None
        self._detailed_agent = None
        self._comprehensive_agent = None

    def _get_model(self):
        """Lazy-create the Pydantic AI OpenAIChatModel."""
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        if not self._api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY "
                "environment variable or pass api_key parameter."
            )

        return OpenAIChatModel(
            model_name=f"openai/{self.model}",
            provider=OpenAIProvider(
                api_key=self._api_key,
                base_url=self._api_base,
            ),
        )

    def _get_agent(self, system_prompt: str):
        """Create a Pydantic AI Agent with the given system prompt."""
        from pydantic_ai import Agent as PydanticAgent

        model = self._get_model()
        model_settings = {"temperature": self.temperature}
        if self.max_tokens:
            model_settings["max_tokens"] = self.max_tokens

        return PydanticAgent(
            model=model,
            result_type=str,
            system_prompt=system_prompt,
            model_settings=model_settings,
        )

    @property
    def researcher_agent(self):
        if self._researcher_agent is None:
            self._researcher_agent = self._get_agent(RESEARCH_SYSTEM_PROMPT)
        return self._researcher_agent

    @property
    def brief_agent(self):
        if self._brief_agent is None:
            self._brief_agent = self._get_agent(BRIEF_SYSTEM_PROMPT)
        return self._brief_agent

    @property
    def detailed_agent(self):
        if self._detailed_agent is None:
            self._detailed_agent = self._get_agent(DETAILED_SYSTEM_PROMPT)
        return self._detailed_agent

    @property
    def comprehensive_agent(self):
        if self._comprehensive_agent is None:
            self._comprehensive_agent = self._get_agent(COMPREHENSIVE_SYSTEM_PROMPT)
        return self._comprehensive_agent

    @property
    def source_type(self) -> str:
        return "perplexity"

    @property
    def is_available(self) -> bool:
        return self._api_key is not None

    async def retrieve(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Query Perplexity Sonar for AI-synthesized answers with citations.

        Args:
            query: Research query or question.
            system_prompt: Optional custom system prompt.
            context: Optional additional context to include with the query.
            **kwargs: Additional parameters.

        Returns:
            List containing a single RetrievalResult with the synthesized answer.
        """
        logger.info(f"Perplexity query ({self.model_name}): '{query[:100]}...'")

        try:
            user_message = query
            if context:
                user_message = f"Context: {context}\n\nQuery: {query}"
            if system_prompt:
                user_message = f"{system_prompt}\n\n{user_message}"

            result = await self.detailed_agent.run(user_message)
            content = result.data

            citations = self._extract_citations(content)

            return [
                RetrievalResult(
                    content=content,
                    source="Perplexity Sonar",
                    source_type=self.source_type,
                    citations=citations if citations else None,
                    metadata={
                        "model": self.model,
                        "model_name": self.model_name,
                        "temperature": self.temperature,
                        "framework": "pydantic-ai",
                    },
                )
            ]

        except ValueError as e:
            logger.error(f"Perplexity configuration error: {e}")
            raise PerplexityConfigError(str(e)) from e
        except AttributeError as e:
            logger.error(f"Perplexity response error: {e}")
            raise PerplexityResponseError(f"Failed to parse response: {e}") from e
        except Exception as e:
            error_msg = f"Perplexity API call failed: {e}"
            logger.error(error_msg)
            raise PerplexityAPIError(error_msg, original_error=e) from e

    async def research(
        self,
        topic: str,
        depth: Literal["brief", "detailed", "comprehensive"] = "detailed",
        **kwargs: Any,
    ) -> RetrievalResult:
        """Conduct research on a topic with configurable depth.

        Args:
            topic: The topic to research.
            depth: How deep to go - "brief", "detailed", or "comprehensive".
            **kwargs: Additional parameters.

        Returns:
            Single RetrievalResult with the research findings.
        """
        logger.info(
            f"Perplexity research ({self.model_name}, {depth}): '{topic[:80]}...'"
        )

        try:
            if depth == "brief":
                agent = self.brief_agent
            elif depth == "comprehensive":
                agent = self.comprehensive_agent
            else:
                agent = self.detailed_agent

            result = await agent.run(topic)
            content = result.data
            citations = self._extract_citations(content)

            return RetrievalResult(
                content=content,
                source="Perplexity Sonar",
                source_type=self.source_type,
                citations=citations if citations else None,
                metadata={
                    "model": self.model,
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "depth": depth,
                    "framework": "pydantic-ai",
                },
            )

        except ValueError as e:
            logger.error(f"Perplexity configuration error: {e}")
            raise PerplexityConfigError(str(e)) from e
        except AttributeError as e:
            logger.error(f"Perplexity response error: {e}")
            raise PerplexityResponseError(f"Failed to parse response: {e}") from e
        except Exception as e:
            error_msg = f"Perplexity research failed: {e}"
            logger.error(error_msg)
            raise PerplexityAPIError(error_msg, original_error=e) from e

    def _extract_citations(self, content: str) -> List[str]:
        """Extract citation URLs from Perplexity Sonar response."""
        url_pattern = r"https?://[^\s\)\]\"\'\<\>]+"
        urls = re.findall(url_pattern, content)

        cleaned_urls = []
        for url in urls:
            url = url.rstrip(".,;:!?")
            if url:
                cleaned_urls.append(url)

        seen = set()
        unique_urls = []
        for url in cleaned_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls