import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import dspy

class PromptConfig(BaseModel):
    """Configuration for prompt generation.

    Attributes:
        custom_instructions: Optional custom instructions to append to prompts.
        dataset_analysis: Whether to include dataset analysis in prompts.
            Defaults to True.
        training_results_analysis: Whether to include training results analysis
            in prompts. Defaults to False.
    """

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
    """Configuration for LLM provider settings.

    Attributes:
        api_key: Optional API key for the LLM provider.
        base_url: Optional base URL for the LLM API endpoint.
        model_name: Name of the LLM model to use.
        temperature: Sampling temperature for text generation. Defaults to 0.7.
        max_tokens: Maximum number of tokens to generate. Defaults to 8000.
        cache: Whether to cache LLM requests. Defaults to True.
        track_usage: Whether to track LLM usage. Defaults to True.
    """

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


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM Settings
    llm_api_key: Optional[str] = Field(default=None, alias="DEEPFIX_LLM_API_KEY")
    llm_base_url: Optional[str] = Field(default=None, alias="DEEPFIX_LLM_BASE_URL")
    llm_model_name: str = Field(default="gpt-4o", alias="DEEPFIX_LLM_MODEL_NAME")
    llm_temperature: float = Field(default=0.7, alias="DEEPFIX_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=8000, alias="DEEPFIX_LLM_MAX_TOKENS")
    llm_cache: bool = Field(default=True, alias="DEEPFIX_LLM_CACHE")
    llm_track_usage: bool = Field(default=True, alias="DEEPFIX_LLM_TRACK_USAGE")

    # Database Settings
    database_url: str = Field(
        default="sqlite:///./deepfix_server.db", alias="DEEPFIX_SERVER_DATABASE_URL"
    )
    database_echo: bool = Field(default=False, alias="DEEPFIX_SERVER_DATABASE_ECHO")
    job_ttl_hours: int = Field(default=3, alias="DEEPFIX_JOB_TTL_HOURS")

    def get_llm_config(self) -> LLMConfig:
        """Create an LLMConfig instance from current settings."""
        return LLMConfig(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            model_name=self.llm_model_name,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            cache=self.llm_cache,
            track_usage=self.llm_track_usage,
        )


# Global settings instance
settings = Settings()

dspy.configure_cache(enable_disk_cache=settings.llm_cache)

class TrainingDynamicsConfig(BaseModel):
    """Configuration for training dynamics analysis.

    Attributes:
        enabled_analyzers: List of analyzer names to enable. Defaults to:
            ["overfitting_detection", "training_stability", "gradient_analysis",
            "performance_trends"].
        overfitting_thresholds: Dictionary of thresholds for overfitting detection.
            Keys: train_val_divergence, val_loss_plateau_epochs, early_stopping_patience.
        stability_thresholds: Dictionary of thresholds for stability analysis.
            Keys: loss_variance_threshold, metric_volatility_window,
            gradient_norm_std_threshold.
        gradient_thresholds: Dictionary of thresholds for gradient analysis.
            Keys: exploding_gradient_threshold, vanishing_gradient_threshold,
            gradient_clip_threshold.
        lightweight_mode: Enable lightweight mode with <10% overhead. Defaults to True.
        max_analysis_time: Maximum analysis time in seconds. Defaults to 30.0.
        small_model_optimized: Optimize for models <100M parameters. Defaults to True.
    """

    # Analysis Configuration
    enabled_analyzers: List[str] = [
        "overfitting_detection",
        "training_stability",
        "gradient_analysis",
        "performance_trends",
    ]

    # Detection Thresholds
    overfitting_thresholds: Dict[str, float] = {
        "train_val_divergence": 0.1,  # Relative divergence threshold
        "val_loss_plateau_epochs": 5,  # Epochs for plateau detection
        "early_stopping_patience": 10,  # Patience for early stopping recommendation
    }

    stability_thresholds: Dict[str, float] = {
        "loss_variance_threshold": 0.05,  # Coefficient of variation threshold
        "metric_volatility_window": 10,  # Window size for volatility analysis
        "gradient_norm_std_threshold": 2.0,  # Standard deviation threshold for gradient norms
    }

    gradient_thresholds: Dict[str, float] = {
        "exploding_gradient_threshold": 10.0,  # Gradient norm threshold
        "vanishing_gradient_threshold": 1e-6,  # Minimum gradient norm
        "gradient_clip_threshold": 1.0,  # Recommended gradient clipping value
    }

    # Performance Configuration
    lightweight_mode: bool = True  # <10% overhead constraint
    max_analysis_time: float = 30.0  # Maximum analysis time in seconds
    small_model_optimized: bool = True  # Optimized for <100M parameters
