"""
Configuration management for PromptBuilder.
"""

from typing import Dict, Any
from omegaconf import OmegaConf
from pydantic import BaseModel, Field


class PromptBuilderConfig(BaseModel):
    """Configuration for PromptBuilder."""

    # Prompt building configuration
    prompt_builders: Dict[str, Any] = Field(default_factory=dict)
    detail_level: str = Field(default="comprehensive")

    # File loading configuration
    file_loading: Dict[str, Any] = Field(default_factory=dict)

    # Logging configuration
    logging: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PromptBuilderConfig":
        """Create a PromptBuilderConfig from a dictionary."""
        return cls(**config)

    @classmethod
    def from_file(cls, file_path: str) -> "PromptBuilderConfig":
        """Create a PromptBuilderConfig from a file."""
        return cls.from_dict(OmegaConf.load(file_path))
