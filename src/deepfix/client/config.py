"""
Configuration management for DeepSight Advisor.

This module provides comprehensive configuration classes for the advisor,
including YAML loading, validation, and default value management.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from ..shared.models import DefaultPaths

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