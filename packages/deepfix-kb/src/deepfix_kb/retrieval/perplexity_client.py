"""Perplexity Sonar integration via OpenRouter using DSPy for AI-powered research."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Literal, Optional

import dspy
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
    "sonar": "perplexity/sonar",  # Fast, lightweight Q&A
    "sonar-pro": "perplexity/sonar-pro",  # Enhanced accuracy
    "sonar-reasoning": "perplexity/sonar-reasoning",  # Deep reasoning
}


class PerplexityConfig(BaseModel):
    """Configuration for Perplexity Sonar client.

    Attributes:
        api_key: OpenRouter API key (loaded from env if not provided).
        model: Perplexity model variant to use.
        temperature: Sampling temperature for responses.
        max_tokens: Maximum tokens in response.
    """

    api_key: Optional[str] = Field(None, description="OpenRouter API key")
    model: Literal["sonar", "sonar-pro", "sonar-reasoning"] = Field(
        "sonar", description="Perplexity model variant"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum response tokens")


# ============================================================================
# DSPy Signatures for Perplexity Research
# ============================================================================


class ResearchQuery(dspy.Signature):
    """Research a topic and provide well-sourced answers with citations.

    You are a helpful research assistant specializing in machine learning,
    deep learning, data science, and AI model diagnostics. Provide accurate,
    well-sourced answers with citations. Focus on practical, actionable
    information relevant to ML practitioners.
    """

    query: str = dspy.InputField(desc="The research query or question to answer")
    context: str = dspy.InputField(
        default="", desc="Optional additional context to consider when answering"
    )

    answer: str = dspy.OutputField(
        desc="A comprehensive answer with inline citations as URLs where applicable"
    )


class BriefResearchQuery(dspy.Signature):
    """Provide a brief, focused answer in 2-3 paragraphs.

    Include only the most essential information and key citations.
    Focus on being concise while still being accurate and helpful.
    """

    topic: str = dspy.InputField(desc="The topic to research")

    answer: str = dspy.OutputField(
        desc="A brief 2-3 paragraph answer with key citations"
    )


class DetailedResearchQuery(dspy.Signature):
    """Provide a detailed answer covering the main aspects of the topic.

    Include relevant examples, best practices, and multiple citations.
    Cover the topic comprehensively while remaining focused.
    """

    topic: str = dspy.InputField(desc="The topic to research in detail")

    answer: str = dspy.OutputField(
        desc="A detailed answer with examples, best practices, and citations"
    )


class ComprehensiveResearchQuery(dspy.Signature):
    """Provide a comprehensive analysis covering all major aspects.

    Include background context, current best practices, trade-offs,
    practical recommendations, and extensive citations from authoritative sources.
    """

    topic: str = dspy.InputField(desc="The topic for comprehensive analysis")

    answer: str = dspy.OutputField(
        desc="A comprehensive analysis with background, best practices, trade-offs, and extensive citations"
    )


# ============================================================================
# DSPy Module for Perplexity Research
# ============================================================================


class PerplexityResearchModule(dspy.Module):
    """DSPy Module for AI-powered research via Perplexity Sonar.

    This module provides multiple research strategies based on depth:
    - brief: Quick, focused answers
    - detailed: Comprehensive coverage with examples
    - comprehensive: Full analysis with trade-offs and recommendations
    """

    def __init__(self):
        super().__init__()
        self.researcher = dspy.ChainOfThought(ResearchQuery)
        self.brief_researcher = dspy.ChainOfThought(BriefResearchQuery)
        self.detailed_researcher = dspy.ChainOfThought(DetailedResearchQuery)
        self.comprehensive_researcher = dspy.ChainOfThought(ComprehensiveResearchQuery)

    def forward(
        self,
        query: str,
        context: str = "",
        depth: Literal["brief", "detailed", "comprehensive"] = "detailed",
    ) -> dspy.Prediction:
        """Execute research with the specified depth.

        Args:
            query: The research query or question.
            context: Optional additional context.
            depth: Research depth - brief, detailed, or comprehensive.

        Returns:
            DSPy Prediction containing the answer.
        """
        if depth == "brief":
            return self.brief_researcher(topic=query)
        elif depth == "comprehensive":
            return self.comprehensive_researcher(topic=query)
        else:  # detailed (default)
            if context:
                return self.researcher(query=query, context=context)
            return self.detailed_researcher(topic=query)

    async def aforward(
        self,
        query: str,
        context: str = "",
        depth: Literal["brief", "detailed", "comprehensive"] = "detailed",
    ) -> dspy.Prediction:
        if depth == "brief":
            return await self.brief_researcher.acall(topic=query)
        elif depth == "comprehensive":
            return await self.comprehensive_researcher.acall(topic=query)
        else:  # detailed (default)
            if context:
                return await self.researcher.acall(query=query, context=context)
            return await self.detailed_researcher.acall(topic=query)


# ============================================================================
# Perplexity Sonar Retriever
# ============================================================================


class PerplexitySonarRetriever(BaseRetriever):
    """Perplexity Sonar integration via OpenRouter using DSPy for AI-powered research.

    Perplexity Sonar provides AI-powered web search with reasoning and
    citation capabilities. This implementation uses DSPy modules for
    more structured and composable research queries.

    Features:
        - AI-powered web search with reasoning
        - Automatic citation extraction
        - Multiple model variants (sonar, sonar-pro, sonar-reasoning)
        - DSPy-based signatures for composable research
        - Configurable research depth (brief, detailed, comprehensive)

    Model Variants:
        - sonar: Fast, lightweight - good for quick Q&A
        - sonar-pro: Enhanced accuracy - better for detailed research
        - sonar-reasoning: Deep reasoning - best for complex analysis

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
        """Initialize Perplexity Sonar client via OpenRouter with DSPy.

        Args:
            api_key: OpenRouter API key. If not provided, loads from
                     OPENROUTER_API_KEY environment variable.
            model: Perplexity model variant to use.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in response (None for model default).
        """
        import os

        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model
        self.model = PERPLEXITY_MODELS[model]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._lm = None
        self._research_module = None
        self._api_base = "https://openrouter.ai/api/v1"

    @property
    def lm(self) -> dspy.LM:
        """Lazy initialization of DSPy language model for OpenRouter."""
        if self._lm is None:
            if not self._api_key:
                raise ValueError(
                    "OpenRouter API key not provided. Set OPENROUTER_API_KEY "
                    "environment variable or pass api_key parameter."
                )

            self._lm = dspy.LM(
                model=f"openai/{self.model}",
                api_key=self._api_key,
                api_base=self._api_base,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._lm

    @property
    def research_module(self) -> PerplexityResearchModule:
        """Lazy initialization of the research module."""
        if self._research_module is None:
            self._research_module = PerplexityResearchModule()
        return self._research_module

    @property
    def source_type(self) -> str:
        """Return the type identifier for this source."""
        return "perplexity"

    @property
    def is_available(self) -> bool:
        """Check if Perplexity is properly configured."""
        return self._api_key is not None

    async def retrieve(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Query Perplexity Sonar for AI-synthesized answers with citations.

        Note: Perplexity returns a single synthesized response rather than
        multiple discrete results. The max_results parameter is not directly
        used but kept for interface compatibility.

        Args:
            query: Research query or question.
            max_results: Not directly used (Sonar returns synthesized answer).
            system_prompt: Optional custom system prompt (used as context hint).
            context: Optional additional context to include with the query.
            **kwargs: Additional parameters.

        Returns:
            List containing a single RetrievalResult with the synthesized
            answer and extracted citations.

        Raises:
            PerplexityConfigError: If API key is not configured.
            PerplexityAPIError: If the API call fails.
            PerplexityResponseError: If response parsing fails.
        """
        logger.info(f"Perplexity query ({self.model_name}): '{query[:100]}...'")

        try:
            full_context = ""
            if context:
                full_context = context
            if system_prompt:
                full_context = (
                    f"{system_prompt}\n\n{full_context}".strip()
                    if full_context
                    else system_prompt
                )

            with dspy.context(lm=self.lm):
                prediction = await self.research_module.acall(
                    query=query,
                    context=full_context,
                    depth="detailed",
                )

            content = prediction.answer

            # Extract citations from the response
            citations = self._extract_citations(content)

            result = RetrievalResult(
                content=content,
                source="Perplexity Sonar",
                source_type=self.source_type,
                citations=citations if citations else None,
                metadata={
                    "model": self.model,
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "dspy_module": "PerplexityResearchModule",
                },
            )

            logger.info(f"Perplexity returned response with {len(citations)} citations")
            return [result]

        except ValueError as e:
            # Configuration errors (e.g., missing API key)
            logger.error(f"Perplexity configuration error: {e}")
            raise PerplexityConfigError(str(e)) from e
        except AttributeError as e:
            # Response parsing errors (e.g., missing 'answer' field)
            logger.error(f"Perplexity response error: {e}")
            raise PerplexityResponseError(f"Failed to parse response: {e}") from e
        except Exception as e:
            # API or network errors
            error_msg = f"Perplexity API call failed: {e}"
            logger.error(error_msg)
            raise PerplexityAPIError(error_msg, original_error=e) from e

    def _extract_citations(self, content: str) -> List[str]:
        """Extract citation URLs from Perplexity Sonar response.

        Perplexity often includes citations as markdown links or plain URLs.
        This method extracts all unique URLs from the response.

        Args:
            content: The response content from Perplexity.

        Returns:
            List of unique URLs found in the content.
        """
        # Match URLs - handles both markdown links and plain URLs
        url_pattern = r"https?://[^\s\)\]\"\'\<\>]+"
        urls = re.findall(url_pattern, content)

        # Clean up URLs (remove trailing punctuation that might be captured)
        cleaned_urls = []
        for url in urls:
            # Remove common trailing punctuation
            url = url.rstrip(".,;:!?")
            if url:
                cleaned_urls.append(url)

        # Return unique URLs while preserving order
        seen = set()
        unique_urls = []
        for url in cleaned_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    async def research(
        self,
        topic: str,
        depth: Literal["brief", "detailed", "comprehensive"] = "detailed",
        **kwargs: Any,
    ) -> RetrievalResult:
        """Conduct research on a topic with configurable depth using DSPy.

        This method leverages DSPy's structured signatures to perform
        research at varying levels of depth.

        Args:
            topic: The topic to research.
            depth: How deep to go - "brief", "detailed", or "comprehensive".
            **kwargs: Additional parameters passed to retrieve().

        Returns:
            Single RetrievalResult with the research findings.
        """
        logger.info(
            f"Perplexity research ({self.model_name}, {depth}): '{topic[:80]}...'"
        )

        try:
            # Use DSPy's context manager to set the LM for this call
            with dspy.context(lm=self.lm):
                prediction = await self.research_module.aforward(
                    query=topic,
                    depth=depth,
                )

            content = prediction.answer
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
                    "dspy_module": "PerplexityResearchModule",
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
