"""Configuration models for DeepFix Knowledge Base."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KnowledgeDomain(StrEnum):
    """Knowledge domains for categorizing documents."""

    GLOBAL = "global"
    TRAINING = "training"
    DATA_QUALITY = "data_quality"
    MODEL_OPTIMIZATION = "model_optimization"
    DEBUGGING = "debugging"

class RetrievalStrategy(StrEnum):
    """Retrieval strategies for hybrid retrieval.

    Defines how multiple retrieval sources are combined.
    """

    PARALLEL = "parallel"  # Query all sources simultaneously
    CASCADING = "cascading"  # Try sources in order until enough results
    WEB_FIRST = "web_first"  # Prioritize web search results
    AI_FIRST = "ai_first"  # Prioritize Perplexity AI results
    LOCAL_FIRST = "local_first"  # Prioritize local knowledge base

class PerplexityModels(StrEnum):
    """Perplexity models available through OpenRouter using Pydantic AI for AI-powered research.
    """
    SONAR = "perplexity/sonar"
    SONAR_PRO = "perplexity/sonar-pro"
    SONAR_REASONING = "perplexity/sonar-reasoning"

class KnowledgeDocument(BaseModel):
    """Knowledge document model for local knowledge base.

    Attributes:
        id: Unique document identifier.
        title: Document title.
        content: Main document content.
        domain: Knowledge domain category.
        knowledge_type: Type of knowledge (concept, pattern, solution, etc.).
        source: Source reference for the document.
        confidence_level: Confidence in the information (0-1).
        tags: List of tags for categorization.
        examples: Optional example use cases.
        ml_frameworks: Applicable ML frameworks.
        model_types: Applicable model types.
    """

    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Main document content")
    domain: KnowledgeDomain = Field(
        KnowledgeDomain.GLOBAL, description="Knowledge domain"
    )
    knowledge_type: str = Field("general", description="Type of knowledge")
    source: Optional[str] = Field(None, description="Source reference")
    confidence_level: float = Field(0.8, ge=0.0, le=1.0, description="Confidence level")
    tags: List[str] = Field(default_factory=list, description="Tags")
    examples: Optional[List[str]] = Field(None, description="Example use cases")
    ml_frameworks: List[str] = Field(default_factory=list, description="ML frameworks")
    model_types: List[str] = Field(default_factory=list, description="Model types")

class TavilyConfig(BaseModel):
    """Configuration for Tavily Search client.

    Attributes:
        api_key: Tavily API key.
        search_depth: Search depth ("basic" or "advanced").
        default_topic: Default topic filter.
        include_answer: Include AI-generated answer.
    """

    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"),
        description="Tavily API key",
    )
    search_depth: Literal["basic", "advanced"] = Field(
        "basic", description="Search depth"
    )
    default_topic: str = Field(
        "general", description="Default topic"
    )
    include_answer: bool = Field(False, description="Include AI answer")

class PerplexityConfig(BaseModel):
    """Configuration for Perplexity Sonar client.

    Attributes:
        api_key: OpenRouter API key.
        model: Perplexity model variant.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
    """

    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY"),
        description="OpenRouter API key",
    )
    api_base: str = Field(
        "https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    model: PerplexityModels = Field(
        PerplexityModels.SONAR, description="Perplexity model variant"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: Optional[int] = Field(None, description="Max tokens")

class HybridRetrieverConfig(BaseModel):
    """Configuration for hybrid retriever.

    Attributes:
        default_strategy: Default strategy for combining sources.
        max_results_per_source: Maximum results from each individual source.
        max_total_results: Maximum total results after merging.
        enable_deduplication: Whether to deduplicate similar results.
        similarity_threshold: Threshold for considering results as duplicates.
    """

    default_strategy: RetrievalStrategy = Field(
        RetrievalStrategy.PARALLEL, description="Default retrieval strategy"
    )
    max_results_per_source: int = Field(
        3, ge=1, le=10, description="Max results per source"
    )
    max_total_results: int = Field(10, ge=1, le=50, description="Max total results")
    enable_deduplication: bool = Field(True, description="Enable result deduplication")
    similarity_threshold: float = Field(
        0.85, ge=0.0, le=1.0, description="Similarity threshold for deduplication"
    )

class KnowledgeBridgeConfig(BaseSettings):
    """Configuration for KnowledgeBridge.

    This configuration model can be used to create a KnowledgeBridge
    instance with all settings.

    Attributes:
        tavily: Tavily configuration (built on demand via property).
        perplexity: Perplexity configuration (built on demand via property).
        default_strategy: Default retrieval strategy.
        max_results_per_source: Max results from each source.
        max_total_results: Max total results.
        enable_local_kb: Enable local knowledge base.
        enable_synthesis: Enable result synthesis by default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tavily_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("TAVILY_API_KEY"),
        description="Tavily API key",
    )
    tavily_search_depth: Literal["basic", "advanced"] = Field(
        "basic", validation_alias=AliasChoices("TAVILY_SEARCH_DEPTH"),
        description="Tavily search depth"
    )
    tavily_default_topic: str = Field(
        "general", validation_alias=AliasChoices("TAVILY_DEFAULT_TOPIC"),
        description="Tavily default topic"
    )
    tavily_include_answer: bool = Field(
        False, validation_alias=AliasChoices("TAVILY_INCLUDE_ANSWER"),
        description="Include AI answer"
    )

    openrouter_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY"),
        description="OpenRouter API key",
    )
    perplexity_model: PerplexityModels = Field(
        PerplexityModels.SONAR,
        validation_alias=AliasChoices("PERPLEXITY_MODEL"),
        description="Perplexity model variant",
    )
    perplexity_temperature: float = Field(
        0.7, validation_alias=AliasChoices("PERPLEXITY_TEMPERATURE"),
        description="Perplexity temperature"
    )
    perplexity_max_tokens: Optional[int] = Field(
        None, validation_alias=AliasChoices("PERPLEXITY_MAX_TOKENS"),
        description="Perplexity max tokens"
    )

    default_strategy: RetrievalStrategy = Field(
        RetrievalStrategy.PARALLEL,
        validation_alias=AliasChoices("KNOWLEDGE_BRIDGE_STRATEGY"),
        description="Default retrieval strategy",
    )

    max_results_per_source: int = Field(
        3, ge=1, le=10, description="Max results per source"
    )
    max_total_results: int = Field(10, ge=1, le=50, description="Max total results")

    enable_local_kb: bool = Field(
        False,
        validation_alias=AliasChoices("KNOWLEDGE_BRIDGE_ENABLE_LOCAL_KB"),
        description="Enable local KB",
    )
    enable_synthesis: bool = Field(True, description="Enable synthesis by default")

    # Private caches for lazily-built sub-configs
    _tavily: Optional[TavilyConfig] = None
    _perplexity: Optional[PerplexityConfig] = None
    _hybrid_retriever: Optional[HybridRetrieverConfig] = None

    @property
    def tavily(self) -> TavilyConfig:
        """Build and cache TavilyConfig on first access."""
        if self._tavily is None:
            self._tavily = TavilyConfig(
                api_key=self.tavily_api_key,
                search_depth=self.tavily_search_depth,
                default_topic=self.tavily_default_topic,
                include_answer=self.tavily_include_answer,
            )
        return self._tavily

    @property
    def perplexity(self) -> PerplexityConfig:
        """Build and cache PerplexityConfig on first access."""
        if self._perplexity is None:
            self._perplexity = PerplexityConfig(
                api_key=self.openrouter_api_key,
                model=self.perplexity_model,
                temperature=self.perplexity_temperature,
                max_tokens=self.perplexity_max_tokens,
            )
        return self._perplexity
    
    @property
    def hybrid_retriever(self):
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetrieverConfig(
                default_strategy=self.default_strategy,
                max_results_per_source=self.max_results_per_source,
                max_total_results=self.max_total_results,
                enable_deduplication=self.enable_deduplication,
                similarity_threshold=self.similarity_threshold,
            )
        return self._hybrid_retriever

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        data = self.model_dump()
        data["tavily"] = self.tavily.model_dump()
        data["perplexity"] = self.perplexity.model_dump()
        return data


# Global configuration instance
config = KnowledgeBridgeConfig()
