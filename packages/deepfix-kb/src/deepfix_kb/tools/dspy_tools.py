"""Knowledge retrieval tools for agent consumption.

This module provides callable tool classes that can be used by agents
via Pydantic AI's tool registration or manual tool handling.

Example:
    >>> from deepfix_kb import KnowledgeBridge
    >>> from deepfix_kb.tools import create_knowledge_tools
    >>>
    >>> bridge = KnowledgeBridge()
    >>> tools = create_knowledge_tools(bridge)
    >>>
    >>> # Use with a Pydantic AI Agent
    >>> agent = Agent('openai:gpt-4o', tools=tools)
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..bridge import KnowledgeBridge

logger = logging.getLogger(__name__)


def _run_sync(coro):
    """Run async coroutine in sync context.

    DSPy tools are typically called synchronously, so we need to
    handle the async retrieval methods.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


class WebSearchTool:
    """Web search tool for retrieving real-time information.

    Use this tool when you need:
    - Current information that may not be in training data
    - Specific technical documentation or papers
    - Recent developments in ML/DL field
    - Error messages or troubleshooting guides

    Attributes:
        name: Tool identifier used by agents.
        desc: Description for agent understanding.
    """

    name = "web_search"
    desc = (
        "Search the web for current information about machine learning, "
        "deep learning, and data science topics. Returns relevant web pages "
        "with content and URLs. Best for: recent papers, documentation, "
        "specific error messages, library updates, framework guides."
    )

    def __init__(self, bridge: "KnowledgeBridge"):
        """Initialize with a KnowledgeBridge instance.

        Args:
            bridge: Configured KnowledgeBridge for retrieval.
        """
        self.bridge = bridge

    def __call__(self, query: str) -> str:
        """Execute web search and format results for agent.

        Args:
            query: The search query.

        Returns:
            Formatted string with search results.
        """
        return self._search(query)

    def _search(self, query: str) -> str:
        """Execute web search and format results."""
        logger.info(f"WebSearchTool: {query[:80]}...")

        try:
            response = _run_sync(
                self.bridge.query(
                    query=query,
                    sources=["web"],
                    max_results=5,
                    synthesize=False,
                )
            )

            if not response.results:
                return "No web search results found."

            formatted = []
            for i, result in enumerate(response.results, 1):
                url_part = f" ({result.url})" if result.url else ""
                formatted.append(
                    f"{i}. **{result.source}**{url_part}\n{result.content}"
                )

            return "\n\n".join(formatted)

        except Exception as e:
            logger.error(f"WebSearchTool failed: {e}")
            return f"Web search failed: {str(e)}"


class ResearchTool:
    """AI-powered research tool for deep analysis and synthesis.

    Use this tool when you need:
    - Synthesized analysis of complex topics
    - Explanations with reasoning
    - Comparison of approaches or techniques
    - Understanding of concepts with citations

    Attributes:
        name: Tool identifier used by agents.
        desc: Description for agent understanding.
    """

    name = "research"
    desc = (
        "Use AI to research and synthesize information about complex ML/DL topics. "
        "Returns a comprehensive answer with citations. Best for: understanding "
        "concepts, comparing approaches, troubleshooting complex issues, "
        "learning best practices, exploring trade-offs."
    )

    def __init__(self, bridge: "KnowledgeBridge"):
        """Initialize with a KnowledgeBridge instance.

        Args:
            bridge: Configured KnowledgeBridge for retrieval.
        """
        self.bridge = bridge

    def __call__(self, query: str) -> str:
        """Execute research query and format results for agent.

        Args:
            query: The research question.

        Returns:
            Formatted string with research findings.
        """
        return self._research(query)

    def _research(self, query: str) -> str:
        """Execute research query and format results."""
        logger.info(f"ResearchTool: {query[:80]}...")

        try:
            response = _run_sync(
                self.bridge.query(
                    query=query,
                    sources=["perplexity"],
                    max_results=1,
                    synthesize=False,  # Perplexity already synthesizes
                )
            )

            if not response.results:
                return "No research results found."

            result = response.results[0]
            output = result.content

            # Add citations if available
            if result.citations:
                output += "\n\n**Citations:**\n"
                for citation in result.citations[:5]:
                    output += f"- {citation}\n"

            return output

        except Exception as e:
            logger.error(f"ResearchTool failed: {e}")
            return f"Research failed: {str(e)}"


class KnowledgeLookupTool:
    """Local knowledge base lookup for domain-specific ML expertise.

    Use this tool when you need:
    - DeepFix-specific best practices
    - Common ML debugging patterns
    - Established solutions to known problems
    - Domain-specific optimization strategies

    Note: This tool requires local knowledge base to be enabled
    and populated with domain knowledge.

    Attributes:
        name: Tool identifier used by agents.
        desc: Description for agent understanding.
    """

    name = "knowledge_lookup"
    desc = (
        "Search the local knowledge base for established ML/DL best practices, "
        "debugging patterns, and optimization strategies. Best for: known issues, "
        "standard approaches, DeepFix-specific guidance, common patterns."
    )

    def __init__(self, bridge: "KnowledgeBridge"):
        """Initialize with a KnowledgeBridge instance.

        Args:
            bridge: Configured KnowledgeBridge for retrieval.
        """
        self.bridge = bridge

    def __call__(self, query: str) -> str:
        """Execute knowledge lookup and format results for agent.

        Args:
            query: The knowledge query.

        Returns:
            Formatted string with knowledge base results.
        """
        return self._lookup(query)

    def _lookup(self, query: str) -> str:
        """Execute knowledge lookup and format results."""
        logger.info(f"KnowledgeLookupTool: {query[:80]}...")

        try:
            response = _run_sync(
                self.bridge.query(
                    query=query,
                    sources=["local_kb"],
                    max_results=5,
                    synthesize=False,
                )
            )

            if not response.results:
                return "No matching knowledge found in local knowledge base."

            formatted = []
            for result in response.results:
                score = (
                    f" (relevance: {result.relevance_score:.2f})"
                    if result.relevance_score
                    else ""
                )
                formatted.append(f"**{result.source}**{score}\n{result.content}")

            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            logger.error(f"KnowledgeLookupTool failed: {e}")
            return f"Knowledge lookup failed: {str(e)}"


class HybridSearchTool:
    """Combined search tool that queries all available sources.

    Use this tool when you need:
    - Comprehensive information from multiple sources
    - Both current web info and established knowledge
    - Cross-validated information

    Attributes:
        name: Tool identifier used by agents.
        desc: Description for agent understanding.
    """

    name = "hybrid_search"
    desc = (
        "Search across all knowledge sources (web, AI research, local KB) "
        "for comprehensive information. Returns combined and synthesized "
        "results from multiple sources. Best for: comprehensive research, "
        "cross-validation, complex questions requiring multiple perspectives."
    )

    def __init__(self, bridge: "KnowledgeBridge"):
        """Initialize with a KnowledgeBridge instance.

        Args:
            bridge: Configured KnowledgeBridge for retrieval.
        """
        self.bridge = bridge

    def __call__(self, query: str) -> str:
        """Execute hybrid search and format results for agent.

        Args:
            query: The search query.

        Returns:
            Formatted string with combined results and synthesis.
        """
        return self._search(query)

    def _search(self, query: str) -> str:
        """Execute hybrid search and format results."""
        logger.info(f"HybridSearchTool: {query[:80]}...")

        try:
            response = _run_sync(
                self.bridge.query(
                    query=query,
                    sources=None,  # All sources
                    max_results=5,
                    synthesize=True,  # Get synthesis
                )
            )

            output_parts = []

            # Add synthesis if available
            if response.synthesis:
                output_parts.append(f"**Summary:**\n{response.synthesis}")

            # Add individual results
            if response.results:
                output_parts.append("\n**Sources:**")
                for i, result in enumerate(response.results, 1):
                    source_type = result.source_type.upper()
                    url_part = f" ({result.url})" if result.url else ""
                    output_parts.append(
                        f"{i}. [{source_type}] {result.source}{url_part}"
                    )

            # Add citations
            if response.total_citations:
                output_parts.append("\n**Citations:**")
                for citation in response.total_citations[:5]:
                    output_parts.append(f"- {citation}")

            return "\n".join(output_parts) if output_parts else "No results found."

        except Exception as e:
            logger.error(f"HybridSearchTool failed: {e}")
            return f"Hybrid search failed: {str(e)}"


def create_knowledge_tools(
    bridge: "KnowledgeBridge",
    include_hybrid: bool = False,
) -> List:
    """Create all knowledge tools for agent consumption.

    Returns plain callable tool instances compatible with Pydantic AI's
    ``tools`` parameter.

    Args:
        bridge: Configured KnowledgeBridge instance.
        include_hybrid: Whether to include the hybrid search tool.

    Returns:
        List of callable tool instances.
    """
    tools = [
        WebSearchTool(bridge),
        ResearchTool(bridge),
        KnowledgeLookupTool(bridge),
    ]

    if include_hybrid:
        tools.append(HybridSearchTool(bridge))

    return tools


def create_tool_descriptions() -> str:
    """Generate documentation string for all available tools.

    Returns:
        Formatted string describing all tools for prompt engineering.
    """
    tools_info = [
        (WebSearchTool.name, WebSearchTool.desc),
        (ResearchTool.name, ResearchTool.desc),
        (KnowledgeLookupTool.name, KnowledgeLookupTool.desc),
        (HybridSearchTool.name, HybridSearchTool.desc),
    ]

    lines = ["Available Knowledge Tools:"]
    for name, desc in tools_info:
        lines.append(f"\n**{name}**: {desc}")

    return "\n".join(lines)
