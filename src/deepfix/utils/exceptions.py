"""
Custom exceptions for PromptBuilder module.
"""


class PromptBuilderError(Exception):
    """Base exception for PromptBuilder errors."""

    pass


class ArtifactLoadError(PromptBuilderError):
    """Raised when artifact loading fails."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to load artifact from {file_path}: {reason}")


class PromptBuilderError(PromptBuilderError):
    """Raised when prompt building fails."""

    def __init__(self, artifact_type: str, reason: str):
        self.artifact_type = artifact_type
        self.reason = reason
        super().__init__(f"Failed to build prompt for {artifact_type}: {reason}")


class LLMClientError(PromptBuilderError):
    """Raised when LLM client operations fail."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"LLM client {operation} failed: {reason}")


class ConfigurationError(PromptBuilderError):
    """Raised when configuration is invalid."""

    def __init__(self, config_section: str, reason: str):
        self.config_section = config_section
        self.reason = reason
        super().__init__(f"Configuration error in {config_section}: {reason}")


class ValidationError(PromptBuilderError):
    """Raised when input validation fails."""

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Validation error for {field}: {reason}")
