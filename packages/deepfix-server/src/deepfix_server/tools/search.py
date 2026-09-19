"""Search tool factory for Agno agents in DeepFix server."""

from __future__ import annotations

from typing import List, Literal, Optional

from agno.tools import Toolkit
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools

from ..config import Settings, settings
from ..logging import get_logger

LOGGER = get_logger(__name__)


def get_search_tools(
    search_provider: Optional[Literal["duckduckgo", "tavily", "none"]] = None,
    tavily_api_key: Optional[str] = None,
    settings_override: Optional[Settings] = None,
) -> List[Toolkit]:
    """Resolve and instantiate web search tools based on server settings or explicit arguments.

    Loads TavilyTools if a Tavily API key is set, otherwise defaults to DuckDuckGoTools.
    Search tools can also be explicitly disabled by setting search_provider='none'.

    Args:
        search_provider: Optional search provider override ('duckduckgo', 'tavily', or 'none').
            If set to 'none', search tools are disabled.
        tavily_api_key: Optional Tavily API key.
            If not provided, uses `settings.tavily_api_key`.
        settings_override: Optional custom Settings instance.

    Returns:
        List of Agno Toolkit instances configured for search.
    """
    cfg = settings_override or settings
    provider = search_provider or cfg.search_provider
    api_key = tavily_api_key or cfg.tavily_api_key

    if provider == "none":
        LOGGER.debug("Search tools disabled (search_provider='none')")
        return []

    if provider == "duckduckgo":
        LOGGER.info("Configuring DuckDuckGoTools for web search")
        return [DuckDuckGoTools()]

    if provider == "tavily":
        if api_key is None:
            LOGGER.error("Tavily search provider requested but TAVILY_API_KEY is not set.")
            raise ValueError("Tavily API key is not set.")
        LOGGER.info("Configuring TavilyTools for web search")
        return [TavilyTools(api_key=api_key,
            enable_search=True,
            enable_extract=True,
            extract_depth="basic",
            extract_format="text",)]

    LOGGER.info("Configuring DuckDuckGoTools for web search")
    return [DuckDuckGoTools()]
