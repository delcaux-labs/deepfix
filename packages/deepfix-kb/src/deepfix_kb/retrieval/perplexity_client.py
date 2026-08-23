"""Perplexity Sonar integration via OpenAI client for AI-powered research."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Literal, Optional

from openai import AsyncOpenAI

from ..config import PerplexityConfig
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
    """Perplexity Sonar integration via OpenAI client for research.

    Features:
        - AI-powered web search with reasoning
        - Automatic citation extraction
        - Multiple model variants (sonar, sonar-pro, sonar-reasoning)
        - Configurable research depth (brief, detailed, comprehensive)

    Example:
        >>> retriever = PerplexitySonarRetriever(api_key="your-key", model="sonar-pro")
        >>> results = await retriever.retrieve(
        ...     "What are the best practices for training vision transformers?"
        ... )
    """

    def __init__(
        self,
        config: Optional[PerplexityConfig] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Initialize Perplexity Sonar client.

        Args:
            config: Optional PerplexityConfig instance.
            api_key: Optional API key.
            model: Optional model identifier.
            api_base: Optional base URL endpoint.
            temperature: Optional sampling temperature.
            max_tokens: Optional max output tokens.
        """
        if config is not None:
            self.config = config
        else:
            kwargs: dict = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if model is not None:
                kwargs["model"] = model
            if api_base is not None:
                kwargs["api_base"] = api_base
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            self.config = PerplexityConfig(**kwargs)

        self._client: Optional[AsyncOpenAI] = None

    @property
    def model_name(self) -> str:
        """Return model name string."""
        model_str = str(self.config.model)
        if model_str.startswith("perplexity/"):
            return model_str[len("perplexity/") :]
        return model_str

    @property
    def _api_key(self) -> Optional[str]:
        return self.config.api_key

    @_api_key.setter
    def _api_key(self, value: Optional[str]):
        self.config.api_key = value
        self._client = None

    @property
    def source_type(self) -> str:
        return "perplexity"

    @property
    def is_available(self) -> bool:
        return bool(self.config.api_key and str(self.config.api_key).strip())

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-create the AsyncOpenAI client."""
        if not self.config.api_key:
            raise ValueError(
                "API key not provided. Set OPENROUTER_API_KEY "
                "environment variable or pass api_key parameter."
            )

        if self._client is None:
            kwargs: dict = {"api_key": self.config.api_key}
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base
            self._client = AsyncOpenAI(**kwargs)

        return self._client

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Make an async chat completion call using AsyncOpenAI."""
        client = self._get_client()
        kwargs: dict = {
            "model": str(self.config.model),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.config.temperature if self.config.temperature is not None else 0.7,
        }
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens

        response = await client.chat.completions.create(**kwargs)
        if not response.choices or not response.choices[0].message:
            raise ValueError("Empty response received from LLM completion.")

        return response.choices[0].message.content or ""

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
        logger.info("Perplexity query (%s): '%s...'", self.config.model, query[:100])

        try:
            user_message = query
            if context:
                user_message = f"Context: {context}\n\nQuery: {query}"
            if system_prompt:
                user_message = f"{system_prompt}\n\n{user_message}"

            content = await self._call_llm(
                system_prompt=DETAILED_SYSTEM_PROMPT,
                user_message=user_message,
            )

            citations = self._extract_citations(content)

            return [
                RetrievalResult(
                    content=content,
                    source="Perplexity AI",
                    source_type=self.source_type,
                    citations=citations if citations else None,
                    metadata={
                        "model": self.config.model,
                        "temperature": self.config.temperature,
                    },
                )
            ]

        except ValueError as e:
            logger.error("Perplexity configuration error: %s", e)
            raise PerplexityConfigError(str(e)) from e
        except AttributeError as e:
            logger.error("Perplexity response error: %s", e)
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
            "Perplexity research (%s, %s): '%s...'",
            self.config.model,
            depth,
            topic[:80],
        )

        try:
            if depth == "brief":
                sys_prompt = BRIEF_SYSTEM_PROMPT
            elif depth == "comprehensive":
                sys_prompt = COMPREHENSIVE_SYSTEM_PROMPT
            else:
                sys_prompt = DETAILED_SYSTEM_PROMPT

            content = await self._call_llm(
                system_prompt=sys_prompt,
                user_message=topic,
            )
            citations = self._extract_citations(content)

            return RetrievalResult(
                content=content,
                source="Perplexity AI",
                source_type=self.source_type,
                citations=citations if citations else None,
                metadata={
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "depth": depth,
                },
            )

        except ValueError as e:
            logger.error("Perplexity configuration error: %s", e)
            raise PerplexityConfigError(str(e)) from e
        except AttributeError as e:
            logger.error("Perplexity response error: %s", e)
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
