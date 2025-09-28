from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class IntelligenceProviders(str, Enum):
    CURSOR = "cursor"
    LLM = "llm"


class Capabilities(Enum):
    TEXT_GENERATION = "text_generation"
    CODE_ANALYSIS = "code_analysis"
    REASONING = "reasoning"
    ML_INSIGHTS = "ml_insights"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    IMAGE_UNDERSTANDING = "image_understanding"


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
        default=4000, description="Maximum tokens to generate in the response"
    )
    cache: bool = Field(default=True, description="Cache request")
    track_usage: bool = Field(default=True, description="Track usage")


class CursorConfig(BaseModel):
    """Configuration for Cursor CLI integration."""

    model: str = Field(
        default="auto", description="Model to use for Cursor CLI integration"
    )
    output_format: str = Field(
        default="text",
        description="Output format to use for Cursor CLI integration",
        examples=["text", "json", "stream-json"],
    )
    timeout: int = Field(
        default=300, description="Timeout to use for Cursor CLI integration"
    )
    cli_path: str = Field(
        default="cursor-agent", description="Path to the Cursor CLI executable"
    )
    working_directory: Optional[str] = Field(
        default=None, description="Working directory to use for Cursor CLI integration"
    )

    def to_cli_args(self) -> list[str]:
        """Convert configuration to CLI arguments."""
        args = [
            self.cli_path,
            "-p",  # non-interactive mode
        ]

        # Add model if specified
        if self.model:
            args.extend(["--model", self.model])

        # Add output format if specified
        if self.output_format:
            args.extend(["--output-format", self.output_format])

        return args


class IntelligenceConfig(BaseModel):
    """Configuration for intelligence client."""

    provider_name: IntelligenceProviders = Field(
        default=..., description="Specific provider name: llm, cursor, etc."
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context for query execution"
    )
    timeout: Optional[int] = Field(
        default=300, description="Timeout in seconds for query execution"
    )
    cursor_config: CursorConfig = Field(
        default=CursorConfig(), description="Configuration for Cursor provider"
    )
    llm_config: LLMConfig = Field(
        default=LLMConfig(), description="Configuration for LLM provider"
    )


class IntelligenceResponse(BaseModel):
    content: str
    provider: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[int] = None


class IntelligenceProviderError(RuntimeError):
    pass
