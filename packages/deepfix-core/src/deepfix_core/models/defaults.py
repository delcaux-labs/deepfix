from __future__ import annotations

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf
from platformdirs import user_data_dir
from pydantic import BaseModel, Field

# Defaults
logger = logging.getLogger(__name__)


def get_workdir():
    """Get the working directory for DeepFix.

    Tries multiple candidate directories in order:
    1. ~/.deepfix
    2. /content/.deepfix (for Google Colab)
    3. <tempdir>/.deepfix

    Returns:
        Path to the first writable directory found.

    Raises:
        RuntimeError: If no writable directory is found.
    """
    candidates = [
        Path(user_data_dir("deepfix")),
        Path.cwd() / ".deepfix",
        # Path("/content/.deepfix"),  # for Google Colab
        # Path(tempfile.gettempdir()) / ".deepfix",
    ]

    for path in candidates:
        parent = path.parent
        # Check if parent exists and is writable
        if parent.exists():
            path.mkdir(parents=True, exist_ok=True)
            return path

    raise RuntimeError("No writable directory found")


def _get_base_dirs() -> Dict[str, Path]:
    """Get base directory paths for data, cache, and logs.

    Uses DEEPFIX_HOME environment variable if set, otherwise uses get_workdir().

    Returns:
        Dictionary mapping 'data', 'cache', and 'log' to their respective Path objects.
    """
    env_home_str = os.environ.get("DEEPFIX_HOME")
    env_home = Path(env_home_str) if env_home_str else get_workdir()
    return {
        "data": env_home / "data",
        "cache": env_home / "cache",
        "log": env_home / "logs",
    }


def _default_mlflow_tracking_uri(data_dir: Path) -> str:
    """Get default MLflow tracking URI.

    Args:
        data_dir: Base data directory.

    Returns:
        File URI string pointing to the MLflow tracking directory.
    """
    mlruns_dir = data_dir / "deepfix_mlflow"
    mlruns_dir.parent.mkdir(parents=True, exist_ok=True)
    return mlruns_dir.resolve().as_uri()  # .replace("file://", "sqlite://")


def _default_mlflow_downloads_dir(data_dir: Path) -> str:
    """Get default MLflow downloads directory.

    Args:
        data_dir: Base data directory.

    Returns:
        String path to the downloads directory.
    """
    downloads = data_dir / "mlflow_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    return str(downloads)


def _default_mlflow_artifact_root(data_dir: Path) -> str:
    """Get default MLflow artifact root directory.

    Args:
        data_dir: Base data directory.

    Returns:
        String path to the artifact root directory.
    """
    artifact_root = data_dir / "mlflow_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return str(artifact_root)


def _default_sqlite_path(data_dir: Path) -> str:
    """Get default SQLite database path for artifacts.

    Args:
        data_dir: Base data directory.

    Returns:
        String path to the SQLite database file.
    """
    sqlite_path = data_dir / "artifacts.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return str(sqlite_path)


def _default_output_dir(data_dir: Path) -> str:
    """Get default output directory for advisor results.

    Args:
        data_dir: Base data directory.

    Returns:
        String path to the output directory.
    """
    out = data_dir / "advisor_output"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _default_knowledge_base_dir(data_dir: Path) -> str:
    """Get default knowledge base directory.

    Args:
        data_dir: Base data directory.

    Returns:
        String path to the knowledge base directory.
    """
    knowledge_base_dir = data_dir / "knowledge_base"
    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    return str(knowledge_base_dir)


def _default_knowledge_base_indices_dir(data_dir: Path) -> str:
    """Get default knowledge base indices directory.

    Args:
        data_dir: Base data directory.

    Returns:
        String path to the knowledge base indices directory.
    """
    p = _default_knowledge_base_dir(data_dir)
    knowledge_base_indices_dir = Path(p) / "indices"
    knowledge_base_indices_dir.mkdir(parents=True, exist_ok=True)
    return str(knowledge_base_indices_dir)


def _default_knowledge_base_documents_dir(data_dir: Path) -> str:
    """Get default knowledge base documents directory.

    Args:
        data_dir: Base data directory.

    Returns:
        String path to the knowledge base documents directory.
    """
    p = _default_knowledge_base_dir(data_dir)
    knowledge_base_documents_dir = Path(p) / "documents"
    knowledge_base_documents_dir.mkdir(parents=True, exist_ok=True)
    return str(knowledge_base_documents_dir)


_BASE_DIRS = _get_base_dirs()


class DataType(StrEnum):
    """Types of data supported by the system."""

    VISION = "vision"
    TABULAR = "tabular"
    NLP = "nlp"
    IR = "ir"


class TaskType(StrEnum):
    """Types of machine learning tasks supported by the system."""

    # tabular tasks
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    # vision tasks
    IMAGE_SEGMENTATION = "image_segmentation"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    # NLP tasks
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_TOKEN_CLASSIFICATION = "text_token_classification"


class DeepchecksConfig(BaseModel):
    """Configuration for Deepchecks validation suites.

    Attributes:
        train_test_validation: Whether to run the train-test validation suite.
        data_integrity: Whether to run the data integrity suite.
        model_evaluation: Whether to run the model evaluation suite.
        max_samples: Optional maximum number of samples to use for validation.
        random_state: Random seed for reproducibility. Defaults to 42.
        save_results: Whether to save validation results to disk.
        output_dir: Optional directory to save results.
        batch_size: Batch size for processing. Defaults to 16.
        data_type: Type of data (vision, tabular, nlp). Defaults to VISION.
    """

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
    data_type: DataType = Field(
        default=DataType.VISION, description="Type of data to run the suites on"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary with data_type converted to its string value.
        """
        dumped_dict = self.model_dump()
        dumped_dict["data_type"] = self.data_type.value
        return dumped_dict

    @classmethod
    def from_dict(cls, config: Union[Dict[str, Any], DictConfig]) -> "DeepchecksConfig":
        """Create DeepchecksConfig from a dictionary.

        Args:
            config: Dictionary or DictConfig containing configuration.

        Returns:
            DeepchecksConfig instance.
        """
        return cls(**config)

    @classmethod
    def from_file(cls, file_path: str) -> "DeepchecksConfig":
        """Load DeepchecksConfig from a file.

        Args:
            file_path: Path to the configuration file (YAML/OmegaConf format).

        Returns:
            DeepchecksConfig instance loaded from file.

        Raises:
            FileNotFoundError: If the file does not exist.
            OmegaConfException: If the file contains invalid configuration.
        """
        return cls.from_dict(OmegaConf.load(file_path))


class DefaultPaths(StrEnum):
    """Default paths and names used throughout the system.

    These are computed at module import time based on the base directories.
    All paths are relative to the DeepFix working directory unless overridden
    by environment variables.
    """

    MLFLOW_TRACKING_URI = _default_mlflow_tracking_uri(_BASE_DIRS["data"])
    MLFLOW_DOWNLOADS = _default_mlflow_downloads_dir(_BASE_DIRS["data"])
    MLFLOW_RUN_NAME = "default"
    MLFLOW_DEFAULT_ARTIFACT_ROOT = _default_mlflow_artifact_root(_BASE_DIRS["data"])

    DATASETS_EXPERIMENT_NAME = "deepfix_datasets"
    EXPERIMENT_NAME = "deepfix"
    TRAINING_EXPERIMENT_NAME = "deepfix_training"

    ARTIFACTS_SQLITE_PATH = _default_sqlite_path(_BASE_DIRS["data"])

    ADVISOR_OUTPUT_DIR = _default_output_dir(_BASE_DIRS["data"])

    # KNOWLEDGE_BASE_DIR = _default_knowledge_base_dir(_BASE_DIRS["data"])
    # KNOWLEDGE_BASE_INDICES_DIR = _default_knowledge_base_indices_dir(_BASE_DIRS["data"])
    # KNOWLEDGE_BASE_DOCUMENTS_DIR = _default_knowledge_base_documents_dir(
    #    _BASE_DIRS["data"]
    # )
