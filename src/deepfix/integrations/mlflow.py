"""
MLflow integration for model registry and experiment tracking.

This module provides comprehensive MLflow integration including:
- Model registry access and management
- Experiment tracking and comparison
- Artifact management and retrieval
- Metric aggregation and analysis
"""

from typing import Dict, List, Optional, Any, Union
import mlflow
import traceback
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
import os
import time

from ..core.artifacts.datamodel import (
    DeepchecksArtifacts,
    ArtifactPath,
    TrainingArtifacts,
    DatasetArtifacts,
)
from .deepchecks import DeepchecksConfig
from ..core.config import DefaultPaths, MLflowConfig
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class MLflowManager:
    """
    MLflow integration client for model, experiment and artifact management.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_id: Optional[str] = None,
        dwnd_dir: Optional[str] = None,
        create_if_not_exists: bool = False,
        run_name: Optional[str] = None,
    ):
        """
        Initialize MLflow client with tracking configuration.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name for operations
        """
        self.tracking_uri = tracking_uri or DefaultPaths.MLFLOW_TRACKING_URI.value
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.current_experiment: Experiment = None
        self.current_run: Run = None
        self.dwnd_dir = dwnd_dir or DefaultPaths.MLFLOW_DOWNLOADS.value
        self.create_if_not_exists = create_if_not_exists

        if experiment_name is not None:
            self.set_experiment(experiment_name)

        if run_id is not None:
            assert run_name is None, "run_name must be None if run_id is provided"
            self.set_run(run_id)
            self.dwnd_dir = str(Path(self.dwnd_dir) / self.run_id)
        elif create_if_not_exists:
            assert isinstance(self.experiment_name, str), (
                f"experiment_name must be a string, got {type(self.experiment_name)}"
            )
            self.create_run(run_name=run_name or DefaultPaths.MLFLOW_RUN_NAME.value)

        # create directory if it doesn't exist
        Path(self.dwnd_dir).mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)

    @classmethod
    def from_config(cls, config: MLflowConfig) -> "MLflowManager":
        return cls(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            run_id=config.run_id,
            dwnd_dir=config.download_dir,
            create_if_not_exists=config.create_if_not_exists,
            run_name=config.run_name,
        )

    def set_experiment(self, experiment_name: str) -> None:
        """
        Set the current experiment for operations.

        Args:
            experiment_name: Name of the experiment to use
        """
        try:
            self.current_experiment = self.client.get_experiment_by_name(
                experiment_name
            )
            if self.current_experiment is None:
                if not self.create_if_not_exists:
                    LOGGER.warning(
                        f"Experiment {experiment_name} not found and create_if_not_exists is False"
                    )

                experiment_id = self.client.create_experiment(
                    experiment_name,
                )
                # Wait a moment for the experiment to be fully created and available

                time.sleep(0.1)
                # Try to get the experiment, with fallback to get_experiment_by_name
                try:
                    self.current_experiment = self.client.get_experiment(experiment_id)
                except Exception:
                    # Fallback: get by name in case the ID lookup fails
                    self.current_experiment = self.client.get_experiment_by_name(
                        experiment_name
                    )
                    if self.current_experiment is None:
                        # If still None, wait a bit more and try again
                        time.sleep(0.5)
                        self.current_experiment = self.client.get_experiment_by_name(
                            experiment_name
                        )
        except Exception as e:
            raise ValueError(f"Experiment {experiment_name} not found: {str(e)}")

    def set_run(self, run_id: str) -> None:
        """
        Set the current run for operations.
        """
        self.current_run = self.client.get_run(run_id)
        self.run_id = run_id

    def create_run(self, run_name: str) -> None:
        """
        Create a new MLflow run.

        Args:
            run_name: Name of the new run
        Returns:
            The ID of the created run
        """
        self.current_run = self.client.create_run(
            self.current_experiment.experiment_id,
            run_name=run_name or DefaultPaths.MLFLOW_RUN_NAME.value,
        )
        self.run_id = self.current_run.info.run_id

    def get_run_info(self) -> Dict[str, Any]:
        if self.current_run is None:
            raise ValueError("Run not set")

        info = dict(
            id=self.current_run.info.run_id,
            experiment_id=self.current_run.info.experiment_id,
            user_id=self.current_run.info.user_id,
            status=self.current_run.info.status,
            duration=(self.current_run.info.end_time - self.current_run.info.start_time)
            / 1000
            / 60,  # in minutes
        )
        return info

    def get_run_metrics(self) -> Dict[str, Any]:
        if self.current_run is None:
            raise ValueError("Run not set")
        return self.current_run.data.metrics

    def _get_run_metric_history(self, metric_name: str) -> pd.DataFrame:
        metric_history = self.client.get_metric_history(self.run_id, metric_name)
        metric_history = pd.DataFrame([m.to_dictionary() for m in metric_history])
        metric_history = metric_history.sort_values(by="step", ascending=True)
        return metric_history

    def get_run_metric_histories(
        self,
        metric_names: List[str],
    ) -> pd.DataFrame:
        assert isinstance(metric_names, list), "Metric names must be a list"
        assert len(metric_names) > 0, "Metric names must be a non-empty list"
        assert all(isinstance(metric_name, str) for metric_name in metric_names), (
            "Metric names must be a list of strings"
        )
        df = pd.concat(
            [self._get_run_metric_history(metric_name) for metric_name in metric_names]
        ).reset_index(drop=True)

        return df

    def get_run_tags(self) -> Dict[str, Any]:
        if self.current_run is None:
            raise ValueError("Run not set")
        return self.current_run.data.tags

    def get_model_checkpoint(self) -> str:
        assert self.run_id is not None
        LOGGER.info(f"Downloading model checkpoint for run {self.run_id}")
        best_checkpoint = self.client.download_artifacts(
            self.run_id, ArtifactPath.MODEL_CHECKPOINT.value, dst_path=self.dwnd_dir
        )
        artifacts = list(Path(best_checkpoint).iterdir())
        assert len(artifacts) == 1, (
            "There should be only one artifact in the best checkpoint"
        )
        assert artifacts[0].is_file(), "The artifact should be a file"
        return str(artifacts[0])

    def get_run_parameters(
        self,
    ) -> Dict[str, Any]:
        if self.current_run is None:
            raise ValueError("Run not set")
        params = self.current_run.data.params
        return params

    def get_deepchecks_artifacts(self) -> DeepchecksArtifacts:
        LOGGER.info(f"Downloading deepchecks artifacts for run {self.run_id}")
        deepchecks = self.client.download_artifacts(
            self.run_id, ArtifactPath.DEEPCHECKS.value, dst_path=self.dwnd_dir
        )
        artifacts = os.path.join(deepchecks, ArtifactPath.DEEPCHECKS_ARTIFACTS.value)
        artifacts = DeepchecksArtifacts.from_file(artifacts)

        if artifacts.config is None:
            config = os.path.join(deepchecks, ArtifactPath.DEEPCHECKS_CONFIG.value)
            config = DeepchecksConfig.from_file(config)
            artifacts.config = config

        return artifacts

    def get_dataset_artifacts(self) -> DatasetArtifacts:
        LOGGER.info(f"Downloading dataset artifacts for run {self.run_id}")
        dataset = self.client.download_artifacts(
            self.run_id, ArtifactPath.DATASET.value, dst_path=self.dwnd_dir
        )
        path = os.path.join(dataset, ArtifactPath.DATASET_METADATA.value)
        artifacts = DatasetArtifacts.from_file(path)
        return artifacts

    def get_local_path(
        self,
        artifact_key: Union[str, ArtifactPath],
        run_id: Optional[str] = None,
        download_if_missing: bool = True,
    ) -> str:
        artifact_key = (
            ArtifactPath(artifact_key)
            if isinstance(artifact_key, str)
            else artifact_key
        )
        run_id = run_id or self.run_id
        if download_if_missing:
            return self.client.download_artifacts(
                run_id, artifact_key.value, dst_path=self.dwnd_dir
            )
        else:
            path = os.path.join(self.dwnd_dir, artifact_key.value)
            return path

    def get_training_artifacts(self) -> TrainingArtifacts:
        LOGGER.info(f"Downloading training artifacts for run {self.run_id}")
        training = self.client.download_artifacts(
            self.run_id, ArtifactPath.TRAINING.value, dst_path=self.dwnd_dir
        )
        metrics = os.path.join(training, ArtifactPath.TRAINING_METRICS.value)
        # get params and save them as an artifact
        params = self.get_run_parameters()
        params_path = os.path.join(training, ArtifactPath.TRAINING_PARAMS.value)
        OmegaConf.save(params, params_path)
        self.add_artifact(ArtifactPath.TRAINING.value, params_path)
        # read metrics
        return TrainingArtifacts(
            metrics_path=metrics,
            metrics_values=pd.read_csv(metrics),
            params=params,
        )

    def add_artifact(self, artifact_key: str, local_path: str) -> None:
        LOGGER.info(f"Adding artifact {artifact_key} for run {self.run_id}")
        self.client.log_artifact(
            run_id=self.run_id, artifact_path=artifact_key, local_path=local_path
        )

    def log_artifact(
        self,
        artifact_key: str,
        local_path: str,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        if run_id is None:
            return self.add_artifact(artifact_key=artifact_key, local_path=local_path)

        with mlflow.start_run(run_id=run_id, run_name=run_name):
            try:
                mlflow.log_artifact(str(local_path), artifact_key)
            except Exception:
                LOGGER.error(
                    f"Error logging model checkpoint: {traceback.format_exc()}"
                )
