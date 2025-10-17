"""
Configuration management for DeepSight Advisor.

This module provides comprehensive configuration classes for the advisor,
including YAML loading, validation, and default value management.
"""

from typing import Optional
import os
from pydantic import BaseModel, Field, field_validator


class MLflowConfig(BaseModel):
    """Configuration for MLflow integration."""

    tracking_uri: str = Field(
        default=DefaultPaths.MLFLOW_TRACKING_URI.value,
        description="MLflow tracking server URI",
    )
    run_id: Optional[str] = Field(default=None, description="MLflow run ID to analyze")
    download_dir: str = Field(
        default=DefaultPaths.MLFLOW_DOWNLOADS.value,
        description="Local directory for downloading artifacts",
    )
    experiment_name: Optional[str] = Field(
        default=None, description="MLflow experiment name (optional)"
    )
    create_run_if_not_exists: bool = Field(
        default=False,
        description="Whether to create the run if it doesn't exist",
    )
    run_name: Optional[str] = Field(default=None, description="MLflow run name")

    dataset_experiment_name: str = Field(
        default=DefaultPaths.DATASETS_EXPERIMENT_NAME.value,
        description="MLflow experiment name for datasets",
    )

    trace_dspy: bool = Field(
        default=True,
        description="Whether to trace dspy requests",
    )

    @field_validator("tracking_uri")
    @classmethod
    def validate_tracking_uri(cls, v):
        if not v.startswith(
            (
                "http://",
                "https://",
                "file://",
            )
        ):
            raise ValueError(
                "tracking_uri must start with http://, https://, or file://"
            )
        return v


class ArtifactConfig(BaseModel):
    """Configuration for artifact management."""

    dataset_name: Optional[str] = Field(
        default=None, description="Name of the dataset to load"
    )

    load_training: bool = Field(
        default=True, description="Whether to load training artifacts"
    )
    load_checks: bool = Field(
        default=True, description="Whether to load Deepchecks artifacts"
    )
    load_dataset_metadata: bool = Field(
        default=True, description="Whether to load dataset metadata"
    )
    load_model_checkpoint: bool = Field(
        default=False, description="Whether to load model checkpoint"
    )
    download_if_missing: bool = Field(
        default=True, description="Whether to download artifacts if not locally cached"
    )
    cache_enabled: bool = Field(
        default=True, description="Whether to enable local caching"
    )
    sqlite_path: str = Field(
        default=DefaultPaths.ARTIFACTS_SQLITE_PATH.value,
        description="Path to SQLite database for artifact caching",
    )


class PromptConfig(BaseModel):
    """Configuration for query generation."""

    custom_instructions: Optional[str] = Field(
        default=None, description="Custom instructions to append to created prompts"
    )
    dataset_analysis: bool = Field(
        default=True, description="Whether to analyze the dataset"
    )
    training_results_analysis: bool = Field(
        default=False, description="Whether to analyze the training"
    )


class LLMConfig(BaseModel):
    api_key: Optional[str] = Field(
        default=None, description="API key for the LLM provider"
    )
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the LLM API"
    )
    model_name: str = Field(default=None, description="Model name to use for the LLM")
    temperature: float = Field(
        default=0.7, description="Sampling temperature for text generation"
    )
    max_tokens: int = Field(
        default=8000, description="Maximum tokens to generate in the response"
    )
    cache: bool = Field(default=True, description="Cache request")
    track_usage: bool = Field(default=True, description="Track usage")
    
    @classmethod
    def load_from_env(cls, env_file: Optional[str] = None):
        if env_file is not None:
            load_dotenv(env_file)
        api_key = os.getenv("DEEPFIX_LLM_API_KEY")
        base_url = os.getenv("DEEPFIX_LLM_BASE_URL")
        model_name = os.getenv("DEEPFIX_LLM_MODEL_NAME")
        temperature = float(os.getenv("DEEPFIX_LLM_TEMPERATURE"))
        max_tokens = int(os.getenv("DEEPFIX_LLM_MAX_TOKENS"))
        cache = bool(os.getenv("DEEPFIX_LLM_CACHE"))
        track_usage = bool(os.getenv("DEEPFIX_LLM_TRACK_USAGE"))
        return cls(api_key=api_key, base_url=base_url, model_name=model_name, 
        temperature=temperature, max_tokens=max_tokens, 
        cache=cache, track_usage=track_usage)


class OutputConfig(BaseModel):
    """Configuration for output management."""

    save_prompt: bool = Field(
        default=False, description="Whether to save generated prompts"
    )
    save_response: bool = Field(
        default=True, description="Whether to save AI responses"
    )
    output_dir: str = Field(
        default=DefaultPaths.ADVISOR_OUTPUT_DIR.value,
        description="Directory to save outputs",
    )
    format: str = Field(default="txt", description="Output format (txt, json, yaml)")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        allowed_formats = ["txt", "json", "yaml"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"format must be one of {allowed_formats}")
        return v.lower()

