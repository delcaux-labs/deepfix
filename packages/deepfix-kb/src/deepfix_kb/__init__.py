"""DeepFix Knowledge Base - Knowledge retrieval system for AI agents.

This package provides a unified knowledge retrieval system that combines:
- Web Search (Tavily): Real-time web information
- AI Research (Perplexity Sonar): Synthesized analysis with citations
- Local KB (LlamaIndex): Domain-specific ML/DL knowledge

Example:
    >>> from deepfix_kb import KnowledgeBridge
    >>>
    >>> # Initialize the bridge
    >>> bridge = KnowledgeBridge()
    >>>
    >>> # Direct query
    >>> response = await bridge.query("How to fix gradient vanishing?")
    >>> print(response.synthesis)
"""

from .bridge import KnowledgeBridge
from .config import RetrievalStrategy
from .models import KnowledgeResponse, RetrievalResult
from .retrieval import (
    BaseRetriever,
    HybridRetriever,
    PerplexitySonarRetriever,
    TavilySearchRetriever,
)

__all__ = [
    # Main interface
    "KnowledgeBridge",
    "KnowledgeResponse",
    # Models
    "RetrievalResult",
    "RetrievalStrategy",
    # Retrievers
    "BaseRetriever",
    "TavilySearchRetriever",
    "PerplexitySonarRetriever",
    "HybridRetriever",
]

__version__ = "0.1.0"
