"""
Main orchestrator class for DeepSight Advisor.

This module provides the DeepSightAdvisor class that coordinates the complete
ML analysis pipeline from artifact loading to intelligent query execution.
"""

import time
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import traceback
from pydantic import BaseModel, Field
import yaml
from copy import deepcopy
from ...integrations.mlflow import MLflowManager

from ..query import IntelligenceClient, IntelligenceConfig
from ...utils.logging import get_logger

from ..config import PromptConfig, OutputConfig, MLflowConfig, ArtifactConfig
from ..pipelines import ArtifactLoadingPipeline, Pipeline, Query
from ..pipelines.prompts import BuildPrompt
from .result import AdvisorResult
from .errors import (
    ConfigurationError,
    OutputError,
    IntelligenceError,
)


class AdvisorConfig(BaseModel):
    """Main configuration class for DeepSight Advisor."""

    mlflow: MLflowConfig = Field(description="MLflow configuration")
    artifacts: ArtifactConfig = Field(
        default_factory=ArtifactConfig, description="Artifact management configuration"
    )
    prompt: PromptConfig = Field(
        default_factory=PromptConfig, description="Query generation configuration"
    )
    intelligence: IntelligenceConfig = Field(
        default_factory=IntelligenceConfig,
        description="Intelligence client configuration",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
    )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "AdvisorConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            return cls(**config_data)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AdvisorConfig":
        """Create configuration from dictionary."""
        try:
            return cls(**config_dict)
        except Exception as e:
            raise ValueError(f"Error creating configuration from dict: {e}")

    def to_file(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w") as f:
                yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")

    def merge(self, other: "AdvisorConfig") -> "AdvisorConfig":
        """Merge another configuration into this one."""
        # Convert to dict, merge, and create new instance
        self_dict = self.model_dump()
        other_dict = other.model_dump()

        # Deep merge dictionaries
        merged_dict = self._deep_merge(self_dict, other_dict)

        return self.__class__(**merged_dict)

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def validate(self) -> None:
        """Validate the complete configuration."""
        # Create output directory if it doesn't exist
        output_dir = Path(self.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)


def load_config(
    config_source: Union[str, Path, Dict[str, Any], AdvisorConfig],
) -> AdvisorConfig:
    """Load configuration from various sources."""
    if isinstance(config_source, AdvisorConfig):
        return config_source
    elif isinstance(config_source, dict):
        return AdvisorConfig.from_dict(config_source)
    elif isinstance(config_source, (str, Path)):
        return AdvisorConfig.from_file(config_source)
    else:
        raise ValueError(f"Unsupported config source type: {type(config_source)}")


class DeepSightAdvisor:
    """
    Global orchestrator for DeepSight ML analysis pipeline.

    This class coordinates the complete workflow:
    1. Artifact loading and management
    2. Prompt building from artifacts
    3. Intelligent analysis execution
    4. Result formatting and output
    """

    def __init__(self, config: Union[AdvisorConfig, str, Path, Dict[str, Any]]):
        """
        Initialize the advisor with configuration.

        Args:
            config: Configuration object, file path, or dictionary
        """
        self.logger = get_logger(self.__class__.__name__)

        # Load and validate configuration
        try:
            self.config = load_config(config)
            self.config.validate()
            self.logger.info("Configuration loaded and validated successfully")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def initialize_processing_pipeline(self, run_id: str) -> Pipeline:
        if not isinstance(run_id, str):
            raise ConfigurationError(
                f"run_id provided is not of type str, type: {type(run_id)}"
            )

        try:
            # step 1: load artifacts
            mlflow_config = deepcopy(self.config.mlflow)
            mlflow_config.run_id = run_id
            pipe = ArtifactLoadingPipeline.from_config(
                mlflow_config=mlflow_config, artifact_config=self.config.artifacts
            )
            self.logger.info("Artifacts loaders initialized")

            # step 2-3: build prompt and execute query
            pipe.append_steps(
                [BuildPrompt(self.config.prompt), Query(self.config.intelligence)]
            )
            self.logger.info("Query generator initialized")

        except Exception:
            raise ConfigurationError(
                f"Failed to initialize preprocessing pipeline: {traceback.format_exc()}"
            )
        return pipe

    def run_analysis(self, run_id: str) -> Union[AdvisorResult, None]:
        """
        Run complete analysis pipeline for given run_id.

        Args:
            run_id: mlflow run ID

        Returns:
            AdvisorResult containing all analysis information
        """
        # Initialize components
        pipe = self.initialize_processing_pipeline(run_id=run_id)

        start_time = time.time()
        try:
            self.logger.info(f"Starting analysis for run_id: {run_id}")

            # run pipeline
            context = pipe.run()
            if context.get("advisor_result") is None:
                self.logger.error("Advisor failed. No result found.")
                return
            result: AdvisorResult = context["advisor_result"]

            self.logger.info("Saving Advisor results...")
            result.execution_time = time.time() - start_time
            self.save_results(result)

            self.logger.info(
                f"Analysis completed successfully in {result.execution_time:.2f} seconds"
            )
            return result

        except Exception as e:
            self.logger.error(f"Analysis failed: {traceback.format_exc()}")
            return None

    def save_results(self, result: AdvisorResult) -> None:
        """
        Save analysis results to configured output directory.

        Args:
            result: Result object to save
        """
        try:
            output_dir = Path(self.config.output.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp-based filename prefix
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            base_filename = f"advisor_result_{result.run_id}_{timestamp_str}"

            # Save prompt if requested
            if self.config.output.save_prompt and result.prompt_generated:
                prompt_path = output_dir / f"{base_filename}_prompt.txt"
                with open(prompt_path, "w") as f:
                    f.write(result.prompt_generated)
                self.logger.info(f"Prompt saved to: {prompt_path}")

            # Save response if requested
            if self.config.output.save_response and result.response_content:
                response_path = output_dir / f"{base_filename}_response.txt"
                with open(response_path, "w") as f:
                    f.write(result.response_content)
                self.logger.info(f"Response saved to: {response_path}")

            # Save result in configured format
            if self.config.output.format == "json":
                result_path = output_dir / f"{base_filename}.json"
                result.to_json(result_path, include_content=True)
            elif self.config.output.format == "yaml":
                result_path = output_dir / f"{base_filename}.yaml"
                result.to_yaml(result_path, include_content=True)
            elif self.config.output.format == "txt":
                result_path = output_dir / f"{base_filename}.txt"
                result.to_text(result_path)

            self.logger.info(f"Results saved to: {result_path}")

        except Exception:
            raise OutputError(f"Failed to save results: {traceback.format_exc()}")

    def validate_configuration(self) -> None:
        """Validate the current configuration."""
        try:
            self.config.validate()
            self.logger.info("Configuration validation passed")
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        try:
            # Create new config with updates
            current_dict = self.config.dict()
            current_dict.update(updates)

            # Reload configuration
            self.config = AdvisorConfig(**current_dict)
            self.config.validate()

            # Reinitialize components if needed
            self._initialize_components()

            self.logger.info("Configuration updated successfully")
        except Exception as e:
            raise ConfigurationError(f"Failed to update configuration: {e}")


# Convenience function for quick analysis
def run_analysis(
    run_id: str,
    tracking_uri: str = "http://localhost:5000",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> AdvisorResult:
    """
    Quick function to run analysis with minimal configuration.

    Args:
        run_id: MLflow run ID to analyze
        tracking_uri: MLflow tracking server URI
        config_overrides: Optional configuration overrides

    Returns:
        AdvisorResult containing analysis information
    """
    # Create basic config
    config = {"mlflow": {"tracking_uri": tracking_uri, "run_id": run_id}}

    # Apply overrides if provided
    if config_overrides:
        config.update(config_overrides)

    # Create advisor and run analysis
    advisor = DeepSightAdvisor(config)
    return advisor.run_analysis()


def create_default_config(
    run_id: str, tracking_uri: str = "http://localhost:5000"
) -> AdvisorConfig:
    """Create a default configuration with minimal required parameters."""
    return AdvisorConfig(mlflow=MLflowConfig(tracking_uri=tracking_uri, run_id=run_id))
