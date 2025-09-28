"""
Intelligence client and provider interfaces for executing prompts against LLMs and agents.
"""

from .client import IntelligenceClient
from .models import (
    IntelligenceResponse,
    IntelligenceProviderError,
    IntelligenceProviders,
    IntelligenceConfig,
    CursorConfig,
    LLMConfig,
)
from .providers.llm.dspy import DspyLLMProvider
from .providers.coding_agent.cursor import CursorAgentProvider

__all__ = [
    "IntelligenceClient",
    "IntelligenceResponse",
    "ProviderType",
    "IntelligenceProviderError",
    "DspyLLMProvider",
    "CursorAgentProvider",
    "IntelligenceProviders",
    "IntelligenceConfig",
    "CursorConfig",
    "LLMConfig",
]
