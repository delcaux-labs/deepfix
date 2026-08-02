"""Tavily Search API integration for real-time web retrieval."""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional

from tavily import AsyncTavilyClient

from ..config import TavilyConfig
from .base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)



class TavilySearchRetriever(BaseRetriever):
    """Tavily Search API integration for real-time web retrieval.

    Tavily is a search engine specifically designed for AI agents and LLMs,
    providing real-time, accurate, and concise search results.

    Features:
        - Search depth control (basic, advanced)
        - Topic filtering (general, news, finance, code)
        - Domain inclusion/exclusion
        - Time range filtering
        - AI-generated answer summaries

    Example:
        >>> retriever = TavilySearchRetriever()
        >>> results = await retriever.retrieve(
        ...     "How to fix gradient vanishing in transformers?",
        ...     max_results=5,
        ...     search_depth="advanced"
        ... )
    """

    def __init__(
        self,
        config: TavilyConfig,
    ):
        """Initialize Tavily Search client.

        Args:
            config: TavilySearchConfig instance.
        """
        self._config = config
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Tavily client."""
        if self._client is None:
            if not self._config.api_key:
                raise ValueError(
                    "Tavily API key not provided. Set TAVILY_API_KEY environment "
                    "variable or pass api_key parameter."
                )
            self._client = AsyncTavilyClient(api_key=self._config.api_key)
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    @client.deleter
    def client(self):
        self._client = None

    @property
    def source_type(self) -> str:
        """Return the type identifier for this source."""
        return "web"

    @property
    def is_available(self) -> bool:
        """Check if Tavily is properly configured."""
        return self._config.api_key is not None

    async def retrieve(
        self,
        query: str,
        max_results: int = 5,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        topic: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        days: Optional[int] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Execute Tavily search and return standardized results.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            search_depth: "basic" (fast) or "advanced" (comprehensive).
            topic: Filter by topic type.
            include_domains: Whitelist specific domains.
            exclude_domains: Blacklist specific domains.
            days: Limit results to past N days.
            **kwargs: Additional parameters passed to Tavily.

        Returns:
            List of RetrievalResult objects with web search results.

        Raises:
            ValueError: If API key is not configured.
            Exception: If Tavily API call fails.
        """
        logger.info(f"Tavily search: '{query}' (max_results={max_results})")

        try:
            # Prepare search parameters
            search_params = {
                "query": query,
                "search_depth": search_depth or self._config.search_depth,
                "topic": topic or self._config.default_topic,
                "max_results": max_results,
                "include_answer": self._config.include_answer,
                "include_raw_content": self._config.include_raw_content,
            }

            # Add optional filters
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            if days:
                search_params["days"] = days

            # Execute search
            response = await self.client.search(**search_params)

            # Parse results
            results = []
            answer = response.get("answer")

            for item in response.get("results", []):
                result = RetrievalResult(
                    content=item.get("content", ""),
                    source=item.get("title", "Unknown"),
                    source_type=self.source_type,
                    url=item.get("url"),
                    relevance_score=item.get("score"),
                    metadata={
                        "published_date": item.get("published_date"),
                        "query_answer": answer,
                        "search_depth": search_params["search_depth"],
                    },
                )
                results.append(result)

            logger.info(f"Tavily returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise

    async def search_with_answer(
        self,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> tuple[str, List[RetrievalResult]]:
        """Search and return both the AI answer and detailed results.

        This is useful when you want both a quick answer and the
        supporting sources.

        Args:
            query: Search query string.
            max_results: Maximum number of results.
            **kwargs: Additional search parameters.

        Returns:
            Tuple of (answer_string, list_of_results).
        """
        results = await self.retrieve(query, max_results=max_results, **kwargs)

        # Extract answer from first result's metadata
        answer = ""
        if results and results[0].metadata:
            answer = results[0].metadata.get("query_answer", "")

        return answer, results
