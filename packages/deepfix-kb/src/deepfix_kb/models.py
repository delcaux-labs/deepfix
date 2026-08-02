"""Models for the KnowledgeBridge knowledge retrieval system."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Standardized result from any retrieval source.

    This model provides a unified format for results coming from different
    retrieval sources (web search, AI research, local knowledge base).

    Attributes:
        content: The main content/text retrieved.
        source: The source name or title.
        source_type: Type identifier ("web", "perplexity", "local_kb").
        url: Optional URL for web sources.
        relevance_score: Optional relevance/similarity score.
        citations: Optional list of citation URLs.
        metadata: Optional additional metadata.
    """

    content: str = Field(..., description="The main content/text retrieved")
    source: str = Field(..., description="The source name or title")
    source_type: str = Field(
        ..., description="Type identifier: 'web', 'perplexity', 'local_kb'"
    )
    url: Optional[str] = Field(None, description="URL for web sources")
    relevance_score: Optional[float] = Field(
        None, description="Relevance/similarity score", ge=0.0, le=1.0
    )
    citations: Optional[List[str]] = Field(None, description="List of citation URLs")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class KnowledgeResponse(BaseModel):
    """Response from KnowledgeBridge.query().

    Contains retrieved results, optional synthesis, and citation tracking.

    Attributes:
        query: The original query string.
        results: List of retrieval results from various sources.
        synthesis: Optional AI-synthesized summary of results.
        sources_used: List of sources that contributed results.
        total_citations: All unique citation URLs across results.
        metadata: Additional response metadata (strategy, result count, etc.).
    """

    query: str = Field(..., description="The original query")
    results: List[RetrievalResult] = Field(
        default_factory=list, description="List of retrieval results"
    )
    synthesis: Optional[str] = Field(
        None, description="AI-synthesized summary of results"
    )
    sources_used: List[str] = Field(
        default_factory=list, description="Sources that contributed results"
    )
    total_citations: List[str] = Field(
        default_factory=list, description="All unique citation URLs"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
