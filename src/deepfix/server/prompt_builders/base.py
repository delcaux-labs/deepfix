"""
Base prompt builder for PromptBuilder.

This module provides the abstract base class for all prompt builders.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ...shared.models import Artifacts

class BasePromptBuilder(ABC):
    """Abstract base class for prompt builders."""

    @abstractmethod
    def can_build(self, artifact: Artifacts) -> bool:
        """Check if this builder can handle the artifact type.

        Args:
            artifact: The artifact to build a prompt for

        Returns:
            True if this builder can handle the artifact, False otherwise
        """
        pass

    @abstractmethod
    def build_prompt(
        self,
        artifact: Artifacts,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt from artifact.

        Args:
            artifact: The artifact to build a prompt for
            context: Additional context for the prompt

        Returns:
            A structured prompt string ready for LLM completion
        """
        pass

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary into a readable string.

        Args:
            context: Context dictionary to format

        Returns:
            Formatted context string
        """
        assert isinstance(context, dict), "Context must be a dictionary"
        context_parts = []
        for key, value in context.items():
            context_parts.append(f"- {key}: {value}")

        return "\n".join(context_parts)
