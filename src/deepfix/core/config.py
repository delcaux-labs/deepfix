"""
Configuration management for DeepSight Advisor.

This module provides comprehensive configuration classes for the advisor,
including YAML loading, validation, and default value management.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from omegaconf import DictConfig, OmegaConf
from enum import StrEnum


class DefaultPaths(StrEnum):
    MLFLOW_TRACKING_URI = "file:./deepfix_mlflow"
    MLFLOW_DOWNLOADS = "mlflow_downloads"
    MLFLOW_RUN_NAME = "default"

    DATASETS_EXPERIMENT_NAME = "deepfix_datasets"

    ARTIFACTS_SQLITE_PATH = "tmp/artifacts.db"

    ADVISOR_OUTPUT_DIR = "advisor_output"


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
    create_if_not_exists: bool = Field(
        default=False,
        description="Whether to create the experiment if it doesn't exist",
    )
    run_name: Optional[str] = Field(default=None, description="MLflow run name")

    dataset_experiment_name: str = Field(
        default=DefaultPaths.DATASETS_EXPERIMENT_NAME.value,
        description="MLflow experiment name for datasets",
    )

    @field_validator("tracking_uri")
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
    def validate_format(cls, v):
        allowed_formats = ["txt", "json", "yaml"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"format must be one of {allowed_formats}")
        return v.lower()


class DeepchecksConfig(BaseModel):
    train_test_validation: bool = Field(
        default=True, description="Whether to run the train_test_validation suite"
    )
    data_integrity: bool = Field(
        default=True, description="Whether to run the data_integrity suite"
    )
    model_evaluation: bool = Field(
        default=False, description="Whether to run the model_evaluation suite"
    )
    max_samples: Optional[int] = Field(
        default=None, description="Maximum number of samples to run the suites on"
    )
    random_state: int = Field(
        default=42, description="Random seed to use for the suites"
    )
    save_results: bool = Field(default=False, description="Whether to save the results")
    output_dir: Optional[str] = Field(
        default=None, description="Output directory to save the results"
    )
    batch_size: int = Field(default=16, description="Batch size to use for the suites")

    @classmethod
    def from_dict(cls, config: Union[Dict[str, Any], DictConfig]) -> "DeepchecksConfig":
        return cls(**config)

    @classmethod
    def from_file(cls, file_path: str) -> "DeepchecksConfig":
        return cls.from_dict(OmegaConf.load(file_path))
