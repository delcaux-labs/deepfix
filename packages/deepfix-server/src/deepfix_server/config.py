from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class EmbedderConfig(BaseModel):
    """Configuration for OpenAI-compatible embedder provider settings.

    Attributes:
        api_key: Optional API key for the embedder provider.
        base_url: Optional base URL for the OpenAI-compatible embedding API endpoint.
        model_name: Name of the embedding model to use. Defaults to 'text-embedding-3-small'.
        dimensions: Optional embedding dimensions. Defaults to 1536.
    """

    api_key: Optional[str] = Field(
        default=None, description="API key for the embedder provider"
    )
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the OpenAI-compatible embedding API"
    )
    model_name: str = Field(
        default="text-embedding-3-small",
        description="Model name to use for the embedder",
    )
    dimensions: Optional[int] = Field(
        default=1536,
        description="Dimensions of the embeddings",
    )


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM Settings
    llm_api_key: Optional[str] = Field(default=None, alias="DEEPFIX_LLM_API_KEY")
    llm_base_url: Optional[str] = Field(
        default="https://openrouter.ai/api/v1", alias="DEEPFIX_LLM_BASE_URL"
    )
    llm_model_name: str = Field(
        default="qwen/qwen3.8-flash", alias="DEEPFIX_LLM_MODEL_NAME"
    )
    llm_temperature: float = Field(default=0.7, alias="DEEPFIX_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=8000, alias="DEEPFIX_LLM_MAX_TOKENS")
    llm_cache: bool = Field(default=True, alias="DEEPFIX_LLM_CACHE")
    llm_track_usage: bool = Field(default=True, alias="DEEPFIX_LLM_TRACK_USAGE")

    num_reasoning_chains: int = Field(default=1,alias="NUM_REASONING_CHAINS")

    # Search Settings
    search_provider: Optional[Literal["duckduckgo", "tavily", "none"]] = Field(
        default=None,
        validation_alias=AliasChoices("DEEPFIX_SEARCH_PROVIDER", "search_provider"),
    )
    tavily_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "TAVILY_API_KEY", "DEEPFIX_TAVILY_API_KEY", "tavily_api_key"
        ),
    )

    # Database Settings
    database_url: str = Field(
        default="sqlite:///./deepfix_server.db", alias="DEEPFIX_SERVER_DATABASE_URL"
    )
    database_echo: bool = Field(default=False, alias="DEEPFIX_SERVER_DATABASE_ECHO")
    job_ttl_hours: int = Field(default=3, alias="DEEPFIX_JOB_TTL_HOURS")

    # Mlflow
    mlflow_exp_name: str = Field(default="deepfix-server", alias="MLFLOW_EXP_NAME")
    mlflow_tracking_uri: Optional[str] = Field(
        default=None, alias="MLFLOW_TRACKING_URI"
    )

    # S3 Settings
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(
        default=None, alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_default_region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION")
    aws_endpoint_url: Optional[str] = Field(default=None, alias="AWS_ENDPOINT_URL")
    s3_bucket: Optional[str] = Field(default=None, alias="DEEPFIX_S3_BUCKET")

    
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
