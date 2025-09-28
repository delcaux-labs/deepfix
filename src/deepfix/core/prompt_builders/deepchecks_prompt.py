"""
Deepchecks prompt builder for PromptBuilder.

This module provides the DeepchecksPromptBuilder for creating prompts
from DeepchecksArtifacts instances.
"""

from typing import Optional, Dict, Any, Union
from .base import BasePromptBuilder
from ..artifacts import (
    DeepchecksArtifacts,
    DeepchecksParsedResult,
    Artifacts,
)
from pydantic import BaseModel
import re
from omegaconf import OmegaConf


class ConditionResult(BaseModel):
    status: str
    condition: str
    more_info: str


class Result(BaseModel):
    check: str
    params: dict
    summary: str
    value: Union[dict, list, str]
    conditions_results: list[ConditionResult]
    link_in_summary: Optional[str] = None
    display_text: Optional[str] = None

    def to_dict(
        self, exclude: list[str] = ["value", "params", "link_in_summary"]
    ) -> dict:
        dumped = self.model_dump()
        keys_to_remove = set(exclude + [k for k, v in dumped.items() if v is None])
        for key in keys_to_remove:
            dumped.pop(key)
        return dumped


class Extractor:
    def extract_urls_regex(self, text: str) -> list[str]:
        """
        Extract URLs using regex pattern matching.
        """
        # Comprehensive URL regex pattern
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        # Find all matches
        urls = re.findall(url_pattern, text)
        return urls

    # Remove HTML anchor tags with empty href attributes
    def remove_anchor_tags(self, text: str) -> str:
        """
        Remove any HTML anchor tags (all <a> tags).
        Handles both regular and self-closing anchor tags.
        """
        # Pattern to match any anchor tag with content
        pattern = r"<a\b[^>]*>.*?</a>"
        # Remove all anchor tags
        cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        return cleaned_text

    def extract_values(self, result: DeepchecksParsedResult) -> Result:
        json_result = result.json_result
        check = json_result.get("check", {})
        check_name = check.get("name", "")
        params = check.get("params", {})
        summary = check.get("summary", "")
        value = json_result.get("value", {})
        conditions_results = []

        for cr in json_result.get("conditions_results", []):
            status = cr.get("Status", {})
            condition = cr.get("Condition", {})
            more_info = cr.get("More Info", {})
            conditions_results.append(
                ConditionResult(status=status, condition=condition, more_info=more_info)
            )

        links = self.extract_urls_regex(summary)
        if len(links) > 0:
            link_in_summary = " ".join(links)
            summary = self.remove_anchor_tags(summary).strip()
        else:
            link_in_summary = None

        return Result(
            check=check_name,
            params=params,
            summary=summary,
            value=value,
            conditions_results=conditions_results,
            link_in_summary=link_in_summary,
            display_text=json_result.get("display_text", None),
        )


class DeepchecksPromptBuilder(BasePromptBuilder):
    """Builds prompts for Deepchecks artifact analysis."""

    def can_build(self, artifact: Artifacts) -> bool:
        """Check if this builder can handle DeepchecksArtifacts."""
        return isinstance(artifact, DeepchecksArtifacts)

    def build_prompt(
        self,
        artifact: DeepchecksArtifacts,
        context: Optional[Dict[str, Any]] = None,
        exclude_fields_from_result: list[str] = ["value", "params", "link_in_summary"],
        exclude_empty_conditions_results: bool = True,
    ) -> str:
        extractor = Extractor()

        """Build structured prompt from DeepchecksArtifacts."""
        prompt_parts = [
            f"Dataset: {artifact.dataset_name}",
            f"Validation checks performed: {sum(len(results) for results in artifact.results.values())}",
            f"Check types: {', '.join(artifact.results.keys())}",
        ]

        # Add configuration context if available
        if artifact.config:
            prompt_parts.append(f"\nValidation configuration:")
            prompt_parts.append(
                f"- Train-test validation: {artifact.config.train_test_validation}"
            )
            prompt_parts.append(
                f"- Data integrity checks: {artifact.config.data_integrity}"
            )
            prompt_parts.append(
                f"- Model evaluation: {artifact.config.model_evaluation}"
            )
            if artifact.config.max_samples:
                prompt_parts.append(f"- Max samples: {artifact.config.max_samples}")

        # Add results
        for k, v in artifact.results.items():
            prompt_parts.append(f"\n{k}:")
            for result in v:
                extracted_result = extractor.extract_values(result).to_dict(
                    exclude_fields_from_result
                )
                if (
                    len(extracted_result.get("conditions_results", [])) == 0
                    and exclude_empty_conditions_results
                ):
                    continue
                else:
                    prompt_parts.append(
                        f"- {OmegaConf.to_yaml(extracted_result, resolve=True)}"
                    )

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            if context_str:
                prompt_parts.append(f"\nAdditional context:\n{context_str}")

        # Combine and truncate if necessary
        full_prompt = "\n".join(prompt_parts)
        return full_prompt
