import os
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
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


class AutonomousFixConfig(BaseModel):
    """Configuration for OpenHands autonomous fix execution and OTEL observability.

    Attributes:
        openhands_llm_api_key: Optional API key for OpenHands LLM provider.
        openhands_llm_model: LLM model for OpenHands.
        openhands_llm_base_url: Optional base URL for OpenHands LLM.
        openhands_docker_image: Docker image for OpenHands agent server workspace.
        openhands_sandbox_port: Base host port for sandbox.
        openhands_server_url: URL of the local OpenHands agent server.
        openhands_container_network: Optional Docker network for the sandbox container.
        deepfix_server_webhook_url: URL of the webhook endpoint to receive fix reports.
        aws_access_key_id: Optional AWS access key ID for S3 persistence.
        aws_secret_access_key: Optional AWS secret access key for S3 persistence.
        aws_default_region: AWS default region.
        aws_endpoint_url: Optional S3/MinIO endpoint URL.
        s3_bucket: Optional target S3 bucket name.
        otel_exporter_otlp_endpoint: OTEL OTLP exporter endpoint.
        otel_exporter_otlp_headers: OTEL OTLP exporter headers.
        otel_exporter_otlp_traces_protocol: OTEL OTLP traces protocol.
    """

    openhands_llm_api_key: Optional[str] = Field(
        default=None, description="API key for OpenHands LLM provider"
    )
    openhands_llm_model: str = Field(
        default="openai/deepseek/deepseek-v4-flash-0731",
        description="Model name for OpenHands LLM",
    )
    openhands_llm_base_url: Optional[str] = Field(
        default="https://api.tensorix.ai/v1", description="Base URL for OpenHands LLM provider"
    )
    openhands_docker_image: str = Field(
        default="ghcr.io/openhands/agent-server:latest-python",
        description="Docker image for OpenHands sandbox",
    )
    openhands_sandbox_port: int = Field(
        default=8010, description="Base host port for sandbox container"
    )
    openhands_use_local_server: bool = Field(
        default=True,
        description="Whether to use a local OpenHands agent server instead of DockerWorkspace",
    )
    openhands_server_url: str = Field(
        default="http://localhost:60000",
        description="URL of the local OpenHands agent server",
    )
    openhands_container_network: Optional[str] = Field(
        default=None,
        description="Docker network name to connect the sandbox container to",
    )
    deepfix_server_webhook_url: str = Field(
        default="http://localhost:8844/webhook/completion",
        description="Webhook URL on DeepFix Server for completion notifications",
    )

    # S3 Settings
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key ID for S3 operations"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret access key for S3 operations"
    )
    aws_default_region: str = Field(
        default="us-east-1", description="AWS default region"
    )
    aws_endpoint_url: Optional[str] = Field(
        default=None, description="Endpoint URL for S3-compatible stores (e.g. MinIO/RustFS)"
    )
    s3_bucket: Optional[str] = Field(
        default=None, description="Target S3 bucket for model weights"
    )

    # persistence
    persistence_dir: str = Field(
        default=".openhands-conversations", description="Path to persistence directory"
    )
    load_memory: bool = Field(
        default=True, description="Whether to load memory from persistence"
    )

    # OTEL Settings
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default=None,
        description="OTEL OTLP exporter endpoint",
    )
    otel_exporter_otlp_headers: Optional[str] = Field(
        default=None,
        description="OTEL OTLP exporter headers",
    )
    otel_exporter_otlp_traces_protocol: str = Field(
        default="http/protobuf",
        description="OTEL OTLP traces protocol",
    )

    def get_sandbox_environment(
        self,
        job_id: str,
        mlflow_experiment_id: Union[str, int] = "0",
        s3_bucket: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build dictionary of environment variables to inject into the sandbox workspace.

        Args:
            job_id: Unique identifier for the fix job.
            mlflow_experiment_id: MLflow experiment ID.
            s3_bucket: Target S3 bucket name override.
            mlflow_tracking_uri: MLflow tracking server URI.

        Returns:
            Dict[str, str]: Dictionary of environment variables.
        """
        env_vars: Dict[str, str] = {
            "DEEPFIX_JOB_ID": str(job_id),
            "DEEPFIX_WEBHOOK_URL": self.deepfix_server_webhook_url,
            "AWS_DEFAULT_REGION": self.aws_default_region,
            "AWS_REGION": self.aws_default_region,
        }
        if self.aws_access_key_id:
            env_vars["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            env_vars["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.aws_endpoint_url:
            env_vars["AWS_ENDPOINT_URL"] = self.aws_endpoint_url

        effective_bucket = s3_bucket or self.s3_bucket
        if effective_bucket:
            env_vars["AWS_S3_BUCKET"] = effective_bucket
            env_vars["DEEPFIX_S3_BUCKET"] = effective_bucket

        if mlflow_tracking_uri:
            env_vars["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

        env_vars["MLFLOW_EXPERIMENT_ID"] = str(mlflow_experiment_id)

        if self.otel_exporter_otlp_endpoint:
            env_vars["OTEL_EXPORTER_OTLP_ENDPOINT"] = self.otel_exporter_otlp_endpoint
            env_vars["OTEL_EXPORTER_OTLP_HEADERS"] = f"x-mlflow-experiment-id={mlflow_experiment_id}"
            env_vars["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = self.otel_exporter_otlp_traces_protocol
            env_vars["OTEL_METRICS_EXPORTER"] = "none"
            env_vars["OTEL_LOGS_EXPORTER"] = "none"

        return env_vars

    def setup_otel_environment(self, experiment_id: Union[str, int]) -> Dict[str, str]:
        """Configure OpenTelemetry environment variables for MLflow tracing.

        Args:
            experiment_id: MLflow experiment ID or name.

        Returns:
            Dict[str, str]: Dictionary of set environment variables.
        """
        if not self.otel_exporter_otlp_endpoint:
            return {}
        env_vars = {
            "OTEL_EXPORTER_OTLP_ENDPOINT": self.otel_exporter_otlp_endpoint,
            "OTEL_EXPORTER_OTLP_HEADERS": f"x-mlflow-experiment-id={experiment_id}",
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": self.otel_exporter_otlp_traces_protocol,
            "OTEL_METRICS_EXPORTER": "none",
            "OTEL_LOGS_EXPORTER": "none",
        }
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        return env_vars


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM Settings
    llm_api_key: Optional[str] = Field(default=None, alias="DEEPFIX_LLM_API_KEY")
    llm_base_url: Optional[str] = Field(default="https://api.tensorix.ai/v1", alias="DEEPFIX_LLM_BASE_URL")
    llm_model_name: str = Field(default="openai/deepseek/deepseek-v4-flash-0731", alias="DEEPFIX_LLM_MODEL_NAME")
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

    # Mlflow
    mlflow_exp_name: str = Field(default="deepfix-server", alias="MLFLOW_EXP_NAME")
    mlflow_tracking_uri: Optional[str] = Field(default=None, alias="MLFLOW_TRACKING_URI")

    # S3 Settings
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION")
    aws_endpoint_url: Optional[str] = Field(default=None, alias="AWS_ENDPOINT_URL")
    s3_bucket: Optional[str] = Field(default=None, alias="DEEPFIX_S3_BUCKET")

    # Autonomous Fix System Settings
    openhands_llm_api_key: Optional[str] = Field(default=None, alias="OPENHANDS_LLM_API_KEY")
    openhands_llm_model: str = Field(default="openai/deepseek/deepseek-v4-flash-0731", alias="OPENHANDS_LLM_MODEL")
    openhands_llm_base_url: Optional[str] = Field(default="https://api.tensorix.ai/v1", alias="OPENHANDS_LLM_BASE_URL")
    openhands_docker_image: str = Field(default="ghcr.io/openhands/agent-server:latest-python", alias="OPENHANDS_DOCKER_IMAGE")
    openhands_sandbox_port: int = Field(default=8010, alias="OPENHANDS_SANDBOX_PORT")
    openhands_use_local_server: bool = Field(default=True, alias="OPENHANDS_USE_LOCAL_SERVER")
    openhands_server_url: str = Field(default="http://localhost:60000", alias="OPENHANDS_SERVER_URL")
    openhands_container_network: Optional[str] = Field(default=None, alias="OPENHANDS_CONTAINER_NETWORK")
    deepfix_server_webhook_url: str = Field(
        default="http://host.docker.internal:4141/webhook/completion",
        alias="DEEPFIX_SERVER_WEBHOOK_URL",
    )
    max_fix_iterations: int = Field(default=5, alias="MAX_FIX_ITERATIONS")
    fix_execution_timeout: int = Field(default=300, alias="FIX_EXECUTION_TIMEOUT")
    target_metric_name: str = Field(default="accuracy", alias="TARGET_METRIC_NAME")
    target_metric_value: float = Field(default=0.90, alias="TARGET_METRIC_VALUE")
    plateau_threshold: float = Field(default=0.01, alias="PLATEAU_THRESHOLD")
    plateau_window: int = Field(default=2, alias="PLATEAU_WINDOW")

    # OTEL Settings
    otel_exporter_otlp_endpoint: Optional[str] = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_exporter_otlp_headers: Optional[str] = Field(default=None, alias="OTEL_EXPORTER_OTLP_HEADERS")
    otel_exporter_otlp_traces_protocol: str = Field(default="http/protobuf", alias="OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")

    def get_autonomous_fix_config(self) -> AutonomousFixConfig:
        """Create an AutonomousFixConfig instance from current settings."""
        return AutonomousFixConfig(
            openhands_llm_api_key=self.openhands_llm_api_key,
            openhands_llm_model=self.openhands_llm_model,
            openhands_llm_base_url=self.openhands_llm_base_url,
            openhands_docker_image=self.openhands_docker_image,
            openhands_sandbox_port=self.openhands_sandbox_port,
            openhands_use_local_server=self.openhands_use_local_server,
            openhands_server_url=self.openhands_server_url,
            openhands_container_network=self.openhands_container_network,
            deepfix_server_webhook_url=self.deepfix_server_webhook_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_default_region=self.aws_default_region,
            aws_endpoint_url=self.aws_endpoint_url,
            s3_bucket=self.s3_bucket,
            max_fix_iterations=self.max_fix_iterations,
            fix_execution_timeout=self.fix_execution_timeout,
            target_metric_name=self.target_metric_name,
            target_metric_value=self.target_metric_value,
            plateau_threshold=self.plateau_threshold,
            plateau_window=self.plateau_window,
            otel_exporter_otlp_endpoint=self.otel_exporter_otlp_endpoint,
            otel_exporter_otlp_headers=self.otel_exporter_otlp_headers,
            otel_exporter_otlp_traces_protocol=self.otel_exporter_otlp_traces_protocol,
        )

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
