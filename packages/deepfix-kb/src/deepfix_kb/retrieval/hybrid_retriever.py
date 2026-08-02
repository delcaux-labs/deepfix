"""Hybrid retriever that orchestrates multiple retrieval sources."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models import RetrievalResult
from .base import BaseRetriever
from ..config import HybridRetrieverConfig, RetrievalStrategy

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates multiple retrieval sources with intelligent routing.

    The HybridRetriever combines results from multiple sources (web search,
    AI research, local knowledge base) using configurable strategies.

    Features:
        - Multiple retrieval strategies (parallel, cascading, priority-based)
        - Source weighting and prioritization
        - Result deduplication and ranking
        - Fallback handling for failed sources

    Example:
        >>> from deepfix_kb.retrieval import (
        ...     TavilySearchRetriever,
        ...     PerplexitySonarRetriever,
        ...     HybridRetriever,
        ...     RetrievalStrategy
        ... )
        >>>
        >>> hybrid = HybridRetriever(
        ...     tavily=TavilySearchRetriever(),
        ...     perplexity=PerplexitySonarRetriever(),
        ... )
        >>>
        >>> results = await hybrid.retrieve(
        ...     "How to implement gradient checkpointing?",
        ...     strategy=RetrievalStrategy.PARALLEL
        ... )
    """

    # Source priority order for priority-based strategies
    PRIORITY_ORDER = {
        RetrievalStrategy.WEB_FIRST: ["web", "perplexity", "local_kb"],
        RetrievalStrategy.AI_FIRST: ["perplexity", "web", "local_kb"],
        RetrievalStrategy.LOCAL_FIRST: ["local_kb", "web", "perplexity"],
    }

    def __init__(
        self,
        config: HybridRetrieverConfig,
        tavily: Optional[BaseRetriever] = None,
        perplexity: Optional[BaseRetriever] = None,
        local_kb: Optional[BaseRetriever] = None,
    ):
        """Initialize hybrid retriever with multiple sources.

        Args:
            tavily: Tavily web search retriever instance.
            perplexity: Perplexity Sonar retriever instance.
            local_kb: Local knowledge base retriever instance.
            config: Configuration for the hybrid retriever.
        """
        self.config = config
        self.sources: Dict[str, BaseRetriever] = {}

        if tavily and tavily.is_available:
            self.sources["web"] = tavily
            logger.info("Tavily web search source registered")

        if perplexity and perplexity.is_available:
            self.sources["perplexity"] = perplexity
            logger.info("Perplexity Sonar source registered")

        if local_kb and local_kb.is_available:
            self.sources["local_kb"] = local_kb
            logger.info("Local knowledge base source registered")

        if not self.sources:
            logger.warning("No retrieval sources registered")

    @property
    def available_sources(self) -> List[str]:
        """Return list of available source names."""
        return list(self.sources.keys())

    async def retrieve(
        self,
        query: str,
        strategy: Optional[RetrievalStrategy] = None,
        sources: Optional[List[str]] = None,
        max_results_per_source: Optional[int] = None,
        max_total_results: Optional[int] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Retrieve from multiple sources based on strategy.

        Args:
            query: Search/research query.
            strategy: How to combine sources. Uses default if not specified.
            sources: Which sources to use (None = all available).
            max_results_per_source: Override config max per source.
            max_total_results: Override config max total results.
            **kwargs: Additional parameters passed to individual retrievers.

        Returns:
            List of RetrievalResult objects from combined sources.
        """
        strategy = strategy or self.config.default_strategy
        per_source = max_results_per_source or self.config.max_results_per_source
        total = max_total_results or self.config.max_total_results

        # Filter to requested sources
        active_sources = self._filter_sources(sources)

        if not active_sources:
            logger.warning("No active sources available for retrieval")
            return []

        logger.info(
            f"Hybrid retrieval: strategy={strategy}, "
            f"sources={list(active_sources.keys())}"
        )

        # Execute based on strategy
        if strategy == RetrievalStrategy.PARALLEL:
            results = await self._parallel_retrieve(
                query, active_sources, per_source, **kwargs
            )
        elif strategy == RetrievalStrategy.CASCADING:
            results = await self._cascading_retrieve(
                query, active_sources, total, **kwargs
            )
        else:
            # Priority-based strategies
            results = await self._priority_retrieve(
                query, active_sources, strategy, per_source, **kwargs
            )

        # Post-process results
        if self.config.enable_deduplication:
            results = self._deduplicate_results(results)

        # Rank and limit results
        results = self._rank_results(results)[:total]

        logger.info(f"Hybrid retrieval returned {len(results)} results")
        return results

    def _filter_sources(
        self, requested: Optional[List[str]]
    ) -> Dict[str, BaseRetriever]:
        """Filter sources based on request."""
        if requested is None:
            return self.sources.copy()

        return {
            name: retriever
            for name, retriever in self.sources.items()
            if name in requested
        }

    async def _parallel_retrieve(
        self,
        query: str,
        sources: Dict[str, BaseRetriever],
        per_source: int,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Query all sources in parallel and merge results."""
        tasks = []
        source_names = []

        for name, retriever in sources.items():
            tasks.append(self._safe_retrieve(retriever, query, per_source, **kwargs))
            source_names.append(name)

        all_results = await asyncio.gather(*tasks)

        # Merge results from all sources
        merged = []
        for name, results in zip(source_names, all_results):
            if results:
                logger.debug(f"Source '{name}' returned {len(results)} results")
                merged.extend(results)
            else:
                logger.debug(f"Source '{name}' returned no results")

        return merged

    async def _cascading_retrieve(
        self,
        query: str,
        sources: Dict[str, BaseRetriever],
        target_count: int,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Try sources sequentially until sufficient results."""
        results = []

        for name, retriever in sources.items():
            if len(results) >= target_count:
                break

            remaining = target_count - len(results)
            source_results = await self._safe_retrieve(
                retriever, query, remaining, **kwargs
            )

            if source_results:
                results.extend(source_results)
                logger.debug(
                    f"Cascading: '{name}' added {len(source_results)} results "
                    f"(total: {len(results)})"
                )

        return results

    async def _priority_retrieve(
        self,
        query: str,
        sources: Dict[str, BaseRetriever],
        strategy: RetrievalStrategy,
        per_source: int,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Retrieve with priority ordering based on strategy."""
        priority_order = self.PRIORITY_ORDER.get(strategy, [])

        # Sort sources by priority
        ordered_sources = {}
        for source_name in priority_order:
            if source_name in sources:
                ordered_sources[source_name] = sources[source_name]

        # Add any remaining sources not in priority list
        for name, retriever in sources.items():
            if name not in ordered_sources:
                ordered_sources[name] = retriever

        # Use parallel retrieval but preserve priority in results
        return await self._parallel_retrieve(
            query, ordered_sources, per_source, **kwargs
        )

    async def _safe_retrieve(
        self,
        retriever: BaseRetriever,
        query: str,
        max_results: int,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Safely retrieve from a source, handling errors gracefully."""
        try:
            return await retriever.retrieve(query, max_results=max_results, **kwargs)
        except Exception as e:
            logger.warning(f"Retrieval from {retriever.source_type} failed: {e}")
            return []

    def _deduplicate_results(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Remove duplicate or highly similar results.

        Currently uses URL-based deduplication for web sources.
        TODO: Add content-based similarity for more sophisticated deduplication.
        """
        seen_urls = set()
        deduplicated = []

        for result in results:
            # For web sources, deduplicate by URL
            if result.url:
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)

            deduplicated.append(result)

        if len(results) != len(deduplicated):
            logger.debug(
                f"Deduplication: {len(results)} -> {len(deduplicated)} results"
            )

        return deduplicated

    def _rank_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rank results by relevance score.

        Results with scores are sorted by score (descending).
        Results without scores are placed at the end.
        """
        # Separate scored and unscored results
        scored = [
            (r, r.relevance_score) for r in results if r.relevance_score is not None
        ]
        unscored = [r for r in results if r.relevance_score is None]

        # Sort scored results by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Combine: scored first, then unscored
        return [r for r, _ in scored] + unscored
