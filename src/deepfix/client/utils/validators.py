"""
Input validation and sanitization for DeepSight MLOps Copilot.

This module provides validation functions for user inputs, configurations,
and external data to ensure data integrity and security.
"""

from typing import Dict, Any, List, Optional, Union
import re
from pathlib import Path
from urllib.parse import urlparse
import os


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_model_uri(model_uri: str) -> bool:
    """
    Validate MLflow model URI format.

    Args:
        model_uri: MLflow model URI to validate

    Returns:
        True if URI is valid

    Raises:
        ValidationError: If URI format is invalid
    """
    if not model_uri or not isinstance(model_uri, str):
        raise ValidationError("Model URI must be a non-empty string")

    # MLflow URI patterns
    patterns = [
        r"^runs:/[a-f0-9]{32}/.*$",  # runs:/run_id/path
        r"^models:/.*$",  # models:/model_name/version
        r"^file://.*$",  # file://path
        r"^s3://.*$",  # s3://bucket/path
        r"^gs://.*$",  # gs://bucket/path
        r"^azure://.*$",  # azure://container/path
    ]

    for pattern in patterns:
        if re.match(pattern, model_uri):
            return True

    raise ValidationError(f"Invalid MLflow model URI format: {model_uri}")


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and normalize file path.

    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")

    path = Path(file_path)

    # Check for path traversal attempts
    if ".." in str(path):
        raise ValidationError("Path traversal not allowed")

    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")

    return path.resolve()


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary structure and values.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Validated configuration dictionary

    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")

    # Validate MLflow configuration
    if "mlflow" in config:
        _validate_mlflow_config(config["mlflow"])

    # Validate detection configuration
    if "detection" in config:
        _validate_detection_config(config["detection"])

    # Validate research configuration
    if "research" in config:
        _validate_research_config(config["research"])

    # Validate output configuration
    if "output" in config:
        _validate_output_config(config["output"])

    return config


def _validate_mlflow_config(mlflow_config: Dict[str, Any]) -> None:
    """Validate MLflow configuration section."""
    if "tracking_uri" in mlflow_config:
        uri = mlflow_config["tracking_uri"]
        if uri and not _is_valid_uri(uri):
            raise ValidationError(f"Invalid MLflow tracking URI: {uri}")


def _validate_detection_config(detection_config: Dict[str, Any]) -> None:
    """Validate detection configuration section."""
    if "thresholds" in detection_config:
        thresholds = detection_config["thresholds"]

        for key, value in thresholds.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Threshold '{key}' must be numeric")

            if value < 0 or value > 1:
                raise ValidationError(f"Threshold '{key}' must be between 0 and 1")

    if "metrics" in detection_config:
        metrics = detection_config["metrics"]
        if not isinstance(metrics, list):
            raise ValidationError("Metrics must be a list")

        valid_metrics = {
            "accuracy",
            "loss",
            "f1_score",
            "precision",
            "recall",
            "auc",
            "mse",
            "mae",
            "rmse",
        }

        for metric in metrics:
            if metric not in valid_metrics:
                raise ValidationError(f"Unknown metric: {metric}")


def _validate_research_config(research_config: Dict[str, Any]) -> None:
    """Validate research configuration section."""
    if "sources" in research_config:
        sources = research_config["sources"]
        if not isinstance(sources, list):
            raise ValidationError("Research sources must be a list")

        valid_sources = {"arxiv", "semantic_scholar"}
        for source in sources:
            if source not in valid_sources:
                raise ValidationError(f"Unknown research source: {source}")

    if "max_papers" in research_config:
        max_papers = research_config["max_papers"]
        if not isinstance(max_papers, int) or max_papers <= 0:
            raise ValidationError("max_papers must be a positive integer")


def _validate_output_config(output_config: Dict[str, Any]) -> None:
    """Validate output configuration section."""
    if "format" in output_config:
        formats = output_config["format"]
        if not isinstance(formats, list):
            raise ValidationError("Output format must be a list")

        valid_formats = {"html", "pdf", "json"}
        for fmt in formats:
            if fmt not in valid_formats:
                raise ValidationError(f"Unknown output format: {fmt}")


def _is_valid_uri(uri: str) -> bool:
    """Check if URI has valid format."""
    try:
        result = urlparse(uri)
        return all([result.scheme, result.netloc]) or result.scheme == "file"
    except Exception:
        return False


def validate_experiment_name(name: str) -> str:
    """
    Validate MLflow experiment name.

    Args:
        name: Experiment name to validate

    Returns:
        Validated experiment name

    Raises:
        ValidationError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValidationError("Experiment name must be a non-empty string")

    # MLflow experiment name restrictions
    if len(name) > 256:
        raise ValidationError("Experiment name too long (max 256 characters)")

    # Check for invalid characters
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in invalid_chars:
        if char in name:
            raise ValidationError(f"Experiment name contains invalid character: {char}")

    return name.strip()


def validate_run_id(run_id: str) -> str:
    """
    Validate MLflow run ID format.

    Args:
        run_id: Run ID to validate

    Returns:
        Validated run ID

    Raises:
        ValidationError: If run ID is invalid
    """
    if not run_id or not isinstance(run_id, str):
        raise ValidationError("Run ID must be a non-empty string")

    # MLflow run IDs are 32-character hexadecimal strings
    if not re.match(r"^[a-f0-9]{32}$", run_id):
        raise ValidationError("Invalid run ID format")

    return run_id


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem usage.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")

    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    sanitized = re.sub(r"\.+", ".", sanitized)  # Replace multiple dots
    sanitized = sanitized.strip(". ")  # Remove leading/trailing dots and spaces

    # Ensure reasonable length
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: max_length - len(ext)] + ext

    return sanitized


def validate_dataset_format(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate dataset information format.

    Args:
        dataset_info: Dataset information dictionary

    Returns:
        Validated dataset information

    Raises:
        ValidationError: If dataset format is invalid
    """
    required_fields = ["path", "format"]

    for field in required_fields:
        if field not in dataset_info:
            raise ValidationError(f"Required dataset field missing: {field}")

    # Validate supported formats
    supported_formats = {"csv", "json", "parquet", "image", "text"}
    if dataset_info["format"] not in supported_formats:
        raise ValidationError(f"Unsupported dataset format: {dataset_info['format']}")

    return dataset_info
