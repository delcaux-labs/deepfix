"""Base interfaces and models for knowledge retrieval."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List

from ..models import RetrievalResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for all retrieval implementations.

    All retrieval sources (Tavily, Perplexity, Local KB) must implement
    this interface to ensure consistent behavior across the system.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Retrieve information based on query.

        Args:
            query: The search/research query.
            max_results: Maximum number of results to return.
            **kwargs: Additional retriever-specific parameters.

        Returns:
            List of RetrievalResult objects.
        """
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the type identifier for this source.

        Returns:
            String identifier for this retrieval source.
        """
        pass

    @property
    def is_available(self) -> bool:
        """Check if this retriever is properly configured and available.

        Returns:
            True if the retriever can be used, False otherwise.
        """
        return True

    async def health_check(self) -> bool:
        """Perform a health check on the retrieval source.

        Returns:
            True if the source is healthy and responding.
        """
        try:
            # Simple health check - try a minimal query
            await self.retrieve("test", max_results=1)
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.source_type}: {e}")
            return False
