from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field, field_validator
import os
from dotenv import load_dotenv

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
        default="deepfix_output",
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

class TrainingDynamicsConfig(BaseModel):
    # Analysis Configuration
    enabled_analyzers: List[str] = [
        "overfitting_detection",
        "training_stability",
        "gradient_analysis",
        "performance_trends"
    ]
    
    # Detection Thresholds
    overfitting_thresholds: Dict[str, float] = {
        "train_val_divergence": 0.1,        # Relative divergence threshold
        "val_loss_plateau_epochs": 5,        # Epochs for plateau detection
        "early_stopping_patience": 10       # Patience for early stopping recommendation
    }
    
    stability_thresholds: Dict[str, float] = {
        "loss_variance_threshold": 0.05,     # Coefficient of variation threshold
        "metric_volatility_window": 10,     # Window size for volatility analysis
        "gradient_norm_std_threshold": 2.0  # Standard deviation threshold for gradient norms
    }
    
    gradient_thresholds: Dict[str, float] = {
        "exploding_gradient_threshold": 10.0,   # Gradient norm threshold
        "vanishing_gradient_threshold": 1e-6,   # Minimum gradient norm
        "gradient_clip_threshold": 1.0          # Recommended gradient clipping value
    }
    
    # Performance Configuration
    lightweight_mode: bool = True           # <10% overhead constraint
    max_analysis_time: float = 30.0        # Maximum analysis time in seconds
    small_model_optimized: bool = True     # Optimized for <100M parameters