"""
Main PromptBuilder class for orchestrating prompt creation from existing Pydantic models.
"""

import traceback
from typing import Any, Dict, List, Optional

from deepfix_core.models import Artifacts

from ..config import PromptConfig
from ..logging import get_logger
from .base import BasePromptBuilder
from .checkpoint_prompt import CheckpointPromptBuilder
from .dataset_prompt import DatasetPromptBuilder
from .deepchecks_prompt import DeepchecksPromptBuilder
from .training_prompt import TrainingPromptBuilder


class PromptBuilderError(Exception):
    """Base exception for PromptBuilder errors."""

    pass


class PromptBuilder:
    """Main class for orchestrating prompt creation from existing Artifacts Pydantic models."""

    def __init__(self, config: Optional[PromptConfig] = None):
        """Initialize the PromptBuilder.

        Args:
            config: Optional configuration for the PromptBuilder
            config_path: Optional path to configuration file
        """
        
        self.prompt_builders = self._initialize_prompt_builders()
        self.config: Optional[PromptConfig] = config or PromptConfig()
        #self.logger = get_logger(self.__class__.__name__)
        #self.logger.debug("PromptBuilder initialized successfully")

    def build_prompt(
        self,
        artifacts: List[Artifacts],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt based on artifact types."""
        try:
            assert len(artifacts) > 0, "No artifacts provided for prompt generation"
            prompt_parts = []

            for artifact in artifacts:
                builder = self._get_prompt_builder(artifact)
                if builder:
                    try:
                        prompt = builder.build_prompt(artifact, context)
                        prompt_parts.append(prompt)
                    except Exception:
                        raise PromptBuilderError(
                            artifact.__class__.__name__,
                            f"Failed to build prompt: {traceback.format_exc()}",
                        )
                else:
                    raise PromptBuilderError(
                        artifact.__class__.__name__,
                        "No prompt builder found for artifact type",
                    )

            # Combine all prompts
            full_prompt = "\n\n".join(prompt_parts)

            return full_prompt

        except Exception:
            raise PromptBuilderError(
                "unknown",
                f"Unexpected error during prompt building: {traceback.format_exc()}",
            )

    def _get_prompt_builder(self, artifact: Artifacts) -> Optional[BasePromptBuilder]:
        """Get appropriate prompt builder for the given artifact type."""
        for builder in self.prompt_builders:
            if builder.can_build(artifact):
                return builder
        return None

    def _initialize_prompt_builders(
        self,
    ) -> List[BasePromptBuilder]:
        """Initialize prompt builders based on configuration."""
        builders = [
            DeepchecksPromptBuilder(),
            TrainingPromptBuilder(),
            DatasetPromptBuilder(),
            CheckpointPromptBuilder(),
        ]
        return builders
