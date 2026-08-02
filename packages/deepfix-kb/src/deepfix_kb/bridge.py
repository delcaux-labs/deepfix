"""KnowledgeBridge - Main interface for knowledge retrieval system."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

from deepfix_core.models import Analysis

from .config import KnowledgeBridgeConfig, RetrievalStrategy
from .config import config as default_config
from .models import KnowledgeResponse, RetrievalResult
from .retrieval import (
    HybridRetriever,
    PerplexitySonarRetriever,
    TavilySearchRetriever,
)

logger = logging.getLogger(__name__)


class KnowledgeBridge:
    """Main interface for the knowledge retrieval system.

    KnowledgeBridge provides a unified API for retrieving knowledge from
    multiple sources, designed to be consumed as a tool by DeepFix agents.

    Features:
        - Unified query interface for all knowledge sources
        - Multiple retrieval strategies (parallel, cascading, priority)
        - Optional synthesis of results using AI
        - Citation aggregation and tracking
        - Context-aware query enhancement

    Data Sources:
        - Web Search (Tavily): Real-time web information
        - AI Research (Perplexity Sonar): Synthesized analysis with citations
        - Local KB: Domain-specific ML/DL knowledge

    Example:
        >>> bridge = KnowledgeBridge()
        >>>
        >>> # Simple query
        >>> response = await bridge.query(
        ...     "How to fix overfitting in transformers?"
        ... )
        >>> print(response.synthesis)
        >>>
        >>> # Query with specific sources
        >>> response = await bridge.query(
        ...     "Latest papers on gradient accumulation",
        ...     sources=["web", "perplexity"]
        ... )
    """

    def __init__(
        self,
        config: Optional[KnowledgeBridgeConfig] = None
    ):
        """Initialize KnowledgeBridge with configured retrieval sources.

        Args:
            config: KnowledgeBridgeConfig instance. If not provided, loads from default settings.
        """
        logger.info("Initializing KnowledgeBridge")

        # Fall back to default config if none provided
        active_config = config or default_config

        self._tavily = TavilySearchRetriever(config=active_config.tavily)
        self._perplexity = PerplexitySonarRetriever(config=active_config.perplexity)

        # Local KB is optional and requires additional setup
        local_kb = None
        if active_config.enable_local_kb:
            raise ValueError("Local KB not implemented yet")

        # Initialize hybrid retriever
        self._hybrid = HybridRetriever(
            tavily=self._tavily,
            perplexity=self._perplexity,
            local_kb=local_kb,
            config=active_config.hybrid_retriever
        )
        self.retriever = self._hybrid
        self.default_strategy = active_config.default_strategy

        # Log available sources
        logger.info(f"Available sources: {self.retriever.available_sources}")
    
    @property
    def available_sources(self,) -> list[str]:
        return self.retriever.available_sources
    
    @property
    def has_available_sources(self,) -> bool:
        """Check if there are any available sources."""
        return bool(self.available_sources)

    @property
    def perplexity(self) -> PerplexitySonarRetriever:
        """Get Perplexity retriever."""
        return self._perplexity

    @property
    def tavily(self) -> TavilySearchRetriever:
        """Get Tavily retriever."""
        return self._tavily

    async def query(
        self,
        query: Union[str, Analysis],
        context: Optional[str] = None,
        sources: Optional[List[str]] = None,
        max_results: int = 5,
        strategy: Optional[RetrievalStrategy] = None,
        synthesize: bool = True,
        **kwargs: Any,
    ) -> KnowledgeResponse:
        """Query the knowledge system and return grounded results.

        Args:
            query: The question or topic to research.
            context: Additional context (e.g., from agent's current analysis).
            sources: Limit to specific sources ["web", "perplexity", "local_kb"].
            max_results: Maximum number of results.
            strategy: Retrieval strategy to use.
            synthesize: Whether to create an AI synthesis of results.
            **kwargs: Additional parameters passed to retrievers.

        Returns:
            KnowledgeResponse with results, synthesis, and citations.
        """
        if isinstance(query, Analysis):
            # Extract the main query from findings description
            logger.info(f"Knowledge query: '{str(query.findings.description)[:80]}...'")
        else:
            logger.info(f"Knowledge query: '{query[:80]}...'")

        # Enhance query with context if provided
        enhanced_query = self._enhance_query(query, context)

        # Retrieve from sources
        results = await self.retriever.retrieve(
            query=enhanced_query,
            strategy=strategy or self.default_strategy,
            sources=sources,
            max_total_results=max_results,
            **kwargs,
        )

        # Optionally synthesize results
        synthesis = None
        if synthesize and results:
            synthesis = await self._synthesize_results(query, results)

        # Collect all citations
        all_citations = self._collect_citations(results)

        # Identify sources used
        sources_used = list(set(r.source_type for r in results))

        return KnowledgeResponse(
            query=str(query),
            results=results,
            synthesis=synthesis,
            sources_used=sources_used,
            total_citations=all_citations,
            metadata={
                "strategy": (strategy or self.default_strategy).value,
                "enhanced_query": enhanced_query if context else None,
                "result_count": len(results),
            },
        )

    async def search(
        self,
        query: str,
        max_results: int = 5,
        sources: List[str] = ["web"],
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Perform a web search using Tavily.

        A convenience method for quick web searches.

        Args:
            query: Search query.
            max_results: Maximum results.
            sources: Limit to specific sources ["web", "perplexity", "local_kb"].
            **kwargs: Additional Tavily parameters.

        Returns:
            List of web search results.
        """
        return await self.retriever.retrieve(
            query=query,
            sources=sources,
            max_total_results=max_results,
            **kwargs,
        )

    async def research(
        self,
        topic: str,
        depth: str = "detailed",
        **kwargs: Any,
    ) -> RetrievalResult:
        """Research a topic using Perplexity Sonar.

        A convenience method for AI-powered research.

        Args:
            topic: Topic to research.
            depth: Research depth ("brief", "detailed", "comprehensive").
            **kwargs: Additional parameters.

        Returns:
            Single RetrievalResult with research findings.
        """
        if hasattr(self.perplexity, "research"):
            return await self.perplexity.research(topic, depth=depth, **kwargs)

        # Fallback to regular retrieve
        results = await self.retriever.retrieve(
            query=topic,
            sources=["perplexity"],
            max_total_results=1,
            **kwargs,
        )
        return results[0] if results else None

    def _enhance_query(
        self,
        query: Union[str, Analysis],
        context: Optional[str],
    ) -> str:
        """Enhance query with additional context.

        When agents provide context about their current analysis,
        this context can help retrieve more relevant information.

        For Analysis objects:
        - Query is extracted from findings.description
        - Context is built from evidence, severity, and recommendations

        Args:
            query: Original query string or Analysis object.
            context: Additional context from the agent (used for string queries).

        Returns:
            Enhanced query string optimized for knowledge retrieval.
        """
        if isinstance(query, Analysis):
            # Extract the main query from findings description
            main_query = query.findings.description

            # Build context from analysis details
            context_parts = []

            # Add evidence as context
            if query.findings.evidence:
                context_parts.append(f"Evidence: {query.findings.evidence}")

            # Add severity for prioritization
            context_parts.append(f"Severity: {query.findings.severity.value}")

            # Add recommendation context
            if query.recommendations.action:
                context_parts.append(f"Proposed action: {query.recommendations.action}")
            if query.recommendations.rationale:
                context_parts.append(f"Rationale: {query.recommendations.rationale}")

            analysis_context = "\n".join(context_parts)

            return (
                f"ML Training Issue: {main_query}\n\n"
                f"Context:\n{analysis_context}\n\n"
                f"Find best practices, research, and solutions for this issue."
            )

        # String query handling
        if not context:
            return query

        # Prepend context to query
        return f"Given the following context: {context}\n\nAnswer: {query}"

    async def _synthesize_results(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> Optional[str]:
        """Create a synthesized summary of retrieval results.

        Uses Perplexity to synthesize information from multiple sources
        into a coherent answer.

        Args:
            query: Original query.
            results: Retrieved results to synthesize.

        Returns:
            Synthesized summary string, or None if synthesis fails.
        """
        if not self.perplexity.is_available:
            logger.warning("Synthesis unavailable: Perplexity not configured")
            return None

        # Build context from results
        context_parts = []
        for i, r in enumerate(results[:5], 1):  # Limit to top 5 for synthesis
            source_info = f"[{r.source}]" if r.source else ""
            url_info = f" ({r.url})" if r.url else ""
            context_parts.append(f"{i}. {source_info}{url_info}\n{r.content[:500]}")

        context = "\n\n".join(context_parts)

        synthesis_prompt = (
            f"Based on the following search results, provide a concise and "
            f"accurate synthesis answering: {query}\n\n"
            f"Search Results:\n{context}\n\n"
            f"Synthesize the key information into a coherent answer. "
            f"Include specific details and cite the sources where relevant."
        )

        try:
            response = await self.perplexity.retrieve(
                synthesis_prompt,
                max_results=1,
            )
            if response:
                return response[0].content
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

        return None

    def _collect_citations(
        self,
        results: List[RetrievalResult],
    ) -> List[str]:
        """Collect all unique citations from results.

        Args:
            results: List of retrieval results.

        Returns:
            List of unique citation URLs.
        """
        all_citations = []

        for result in results:
            # Add explicit citations
            if result.citations:
                all_citations.extend(result.citations)

            # Add result URL as citation
            if result.url:
                all_citations.append(result.url)

        # Return unique citations preserving order
        seen = set()
        unique = []
        for citation in all_citations:
            if citation not in seen:
                seen.add(citation)
                unique.append(citation)

        return unique
