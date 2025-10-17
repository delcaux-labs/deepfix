from typing import Optional, Dict, Any, List
import json

from .base import BasePromptBuilder
from ..artifacts import DatasetArtifacts, Artifacts


class DatasetPromptBuilder(BasePromptBuilder):
    """Builds prompts for training artifact analysis."""

    def can_build(self, artifact: Artifacts) -> bool:
        """Check if this builder can handle TrainingArtifacts."""
        return isinstance(artifact, DatasetArtifacts)

    def build_prompt(
        self,
        artifact: DatasetArtifacts,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt from TrainingArtifacts."""
        prompt_parts = []
        prompt_parts.append(f"\nDataset name: {artifact.dataset_name}")
        prompt_parts.append(f"\nDataset statistics:")
        prompt_parts.append(f"- {json.dumps(artifact.statistics, indent=2)}")

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            if context_str:
                prompt_parts.append(f"\nAdditional context:\n{context_str}")

        # Combine and truncate if necessary
        full_prompt = "\n".join(prompt_parts)
        return full_prompt
