"""
Error hierarchy for DeepSight Advisor.

This module defines a comprehensive error hierarchy for the advisor,
providing specific error types for different failure scenarios.
"""

from typing import Optional, Dict, Any


class AdvisorError(Exception):
    """Base exception for all advisor-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.details:
            details_str = ", ".join([f"{k}={v}" for k, v in self.details.items()])
            base_msg = f"{base_msg} (Details: {details_str})"
        return base_msg


class ConfigurationError(AdvisorError):
    """Raised when there are configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = config_value

        super().__init__(message=message, error_code="CONFIG_ERROR", details=details)


class ArtifactError(AdvisorError):
    """Raised when there are artifact-related errors."""

    def __init__(
        self,
        message: str,
        run_id: Optional[str] = None,
        artifact_key: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if run_id:
            details["run_id"] = run_id
        if artifact_key:
            details["artifact_key"] = artifact_key

        super().__init__(message=message, error_code="ARTIFACT_ERROR", details=details)


class QueryError(AdvisorError):
    """Raised when there are query generation or execution errors."""

    def __init__(
        self,
        message: str,
        query_type: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if query_type:
            details["query_type"] = query_type
        if provider:
            details["provider"] = provider

        super().__init__(message=message, error_code="QUERY_ERROR", details=details)


class OutputError(AdvisorError):
    """Raised when there are output-related errors."""

    def __init__(
        self,
        message: str,
        output_path: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if output_path:
            details["output_path"] = output_path
        if output_format:
            details["output_format"] = output_format

        super().__init__(message=message, error_code="OUTPUT_ERROR", details=details)


class MLflowError(AdvisorError):
    """Raised when there are MLflow-related errors."""

    def __init__(
        self,
        message: str,
        tracking_uri: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if tracking_uri:
            details["tracking_uri"] = tracking_uri
        if run_id:
            details["run_id"] = run_id

        super().__init__(message=message, error_code="MLFLOW_ERROR", details=details)


class IntelligenceError(AdvisorError):
    """Raised when there are intelligence client errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        provider_type: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
        if provider_type:
            details["provider_type"] = provider_type

        super().__init__(
            message=message, error_code="INTELLIGENCE_ERROR", details=details
        )


class ValidationError(AdvisorError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value

        super().__init__(
            message=message, error_code="VALIDATION_ERROR", details=details
        )


class TimeoutError(AdvisorError):
    """Raised when operations timeout."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation

        super().__init__(message=message, error_code="TIMEOUT_ERROR", details=details)


# Convenience functions for common error scenarios
def raise_config_error(message: str, **kwargs) -> None:
    """Raise a configuration error with the given message."""
    raise ConfigurationError(message, **kwargs)


def raise_artifact_error(message: str, **kwargs) -> None:
    """Raise an artifact error with the given message."""
    raise ArtifactError(message, **kwargs)


def raise_query_error(message: str, **kwargs) -> None:
    """Raise a query error with the given message."""
    raise QueryError(message, **kwargs)


def raise_output_error(message: str, **kwargs) -> None:
    """Raise an output error with the given message."""
    raise OutputError(message, **kwargs)


def raise_mlflow_error(message: str, **kwargs) -> None:
    """Raise an MLflow error with the given message."""
    raise MLflowError(message, **kwargs)


def raise_intelligence_error(message: str, **kwargs) -> None:
    """Raise an intelligence error with the given message."""
    raise IntelligenceError(message, **kwargs)
