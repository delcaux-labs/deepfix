"""
Configuration management for DeepSight Advisor.

This module provides comprehensive configuration classes for the advisor,
including YAML loading, validation, and default value management.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import os
from pydantic import BaseModel, Field, field_validator
from omegaconf import DictConfig, OmegaConf
from enum import StrEnum
from platformdirs import (
                user_data_dir,
                user_cache_dir,
                user_log_dir,
            )

def _get_base_dirs() -> Dict[str, Path]:
    """Resolve base directories with precedence:
    1) DEEPFIX_HOME env var
    2) platform-appropriate user dirs (via platformdirs)
    3) fallback to ~/.deepfix

    Ensures directories exist.
    """
    env_home = os.environ.get("DEEPFIX_HOME")
    if env_home:
        base = Path(env_home).expanduser()
        data_dir = base / "data"
        cache_dir = base / "cache"
        log_dir = base / "logs"
    else:
        try:
            data_dir = Path(user_data_dir("deepfix", "deepfix"))
            cache_dir = Path(user_cache_dir("deepfix", "deepfix"))
            log_dir = Path(user_log_dir("deepfix", "deepfix"))
        except:
            base = Path("~/.deepfix").expanduser()
            data_dir = base / "data"
            cache_dir = base / "cache"
            log_dir = base / "logs"

    for d in (data_dir, cache_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data": data_dir,
        "cache": cache_dir,
        "log": log_dir,
    }

def _default_mlflow_tracking_uri(data_dir: Path) -> str:
    mlruns_dir = data_dir / "deepfix_mlflow"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    # Use an OS-correct file:// URI (e.g., file:///C:/... on Windows)
    return mlruns_dir.resolve().as_uri()

def _default_mlflow_downloads_dir(data_dir: Path) -> str:
    downloads = data_dir / "mlflow_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    return str(downloads)

def _default_mlflow_artifact_root(data_dir: Path) -> str:
    artifact_root = data_dir / "mlflow_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return str(artifact_root)
def _default_sqlite_path(data_dir: Path) -> str:
    sqlite_path = data_dir / "tmp" / "artifacts.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return str(sqlite_path)

def _default_output_dir(data_dir: Path) -> str:
    out = data_dir / "advisor_output"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)

_BASE_DIRS = _get_base_dirs()


class DefaultPaths(StrEnum):
    MLFLOW_TRACKING_URI = _default_mlflow_tracking_uri(_BASE_DIRS["data"])
    MLFLOW_DOWNLOADS = _default_mlflow_downloads_dir(_BASE_DIRS["data"])
    MLFLOW_RUN_NAME = "default"
    MLFLOW_DEFAULT_ARTIFACT_ROOT = _default_mlflow_artifact_root(_BASE_DIRS["data"])

    DATASETS_EXPERIMENT_NAME = "deepfix_datasets"

    ARTIFACTS_SQLITE_PATH = _default_sqlite_path(_BASE_DIRS["data"])

    ADVISOR_OUTPUT_DIR = _default_output_dir(_BASE_DIRS["data"])


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
