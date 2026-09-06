import json
from typing import Any, Dict, Optional

from deepfix_core.models import (
    Artifacts,
    DatasetArtifacts,
)

from .base import BasePromptBuilder


class DatasetPromptBuilder(BasePromptBuilder):
    """Builds prompts for dataset artifact analysis."""

    def can_build(self, artifact: Any) -> bool:
        """Check if this builder can handle DatasetArtifacts."""
        if isinstance(artifact, DatasetArtifacts):
            return True
        if isinstance(artifact, dict) and ("train_statistics" in artifact or "task_type" in artifact):
            return True
        return False

    def build_prompt(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt from DatasetArtifacts."""
        if isinstance(artifact, dict):
            artifact = DatasetArtifacts.from_dict(artifact)
        prompt_parts = []
        prompt_parts.append(f"\nDataset name: {artifact.dataset_name}")
        if artifact.train_statistics:
            prompt_parts.append("\nDataset statistics:")
            prompt_parts.append(
                f"- {json.dumps(artifact.train_statistics.to_dict(), indent=2)}"
            )
        if artifact.test_statistics:
            prompt_parts.append("\nTest dataset statistics:")
            prompt_parts.append(
                f"- {json.dumps(artifact.test_statistics.to_dict(), indent=2)}"
            )

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            if context_str:
                prompt_parts.append(f"\nAdditional context:\n{context_str}")

        # Combine and truncate if necessary
        full_prompt = "\n".join(prompt_parts)
        return full_prompt
