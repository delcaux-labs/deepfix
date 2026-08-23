from typing import Any, List, Optional

from deepfix_core.models import (
    ArtifactPath,
    DatasetArtifacts,
    DataType,
    DeepchecksArtifacts,
    TrainingArtifacts,
)

from ..artifacts import ArtifactsManager
from ..data import BaseDataset, get_data_statistics
from ..logging import get_logger
from ..utils import get_model_metadata
from .base import Step

LOGGER = get_logger(__name__)


class LogArtifact(Step):
    def __init__(
        self,
        artifact_key: ArtifactPath,
        artifact_mgr: ArtifactsManager,
        run_name: str,
    ):
        self.artifact_key = artifact_key
        self.artifact_mgr = artifact_mgr
        self.run_name = run_name


class LogTrainingArtifact(LogArtifact):
    def __init__(self, artifact_mgr: ArtifactsManager, run_name: str):
        super().__init__(
            artifact_key=ArtifactPath.TRAINING,
            artifact_mgr=artifact_mgr,
            run_name=run_name,
        )
        self.mlflow_manager = self.artifact_mgr.mlflow_manager

    def run(
        self,
        context: dict,
        metric_names: List[str] = None,
    ) -> dict:
        metric_names = metric_names or context.get("metric_names", None)
        if metric_names is None:
            LOGGER.warning("metric_names not provided, will not log metric histories")
            return context

        # get training artifacts
        params = self.mlflow_manager.get_run_parameters()
        df = self.mlflow_manager.get_run_metric_histories(metric_names=metric_names)
        training_artifacts = TrainingArtifacts(params=params, metrics_values=df)
        self.artifact_mgr.register_artifact(
            run_id=self.run_name,
            artifact_key=ArtifactPath.TRAINING,
            artifacts=training_artifacts,
            add_to_mlflow=True,
        )
        return context


class LogChecksArtifacts(LogArtifact):
    def __init__(self, artifact_mgr: ArtifactsManager, run_name: str):
        super().__init__(
            artifact_key=ArtifactPath.DEEPCHECKS,
            artifact_mgr=artifact_mgr,
            run_name=run_name,
        )

    def run(
        self,
        context: dict,
        checks_artifacts: Optional[DeepchecksArtifacts] = None,
    ) -> dict:
        checks_artifacts = checks_artifacts or context.get("checks_artifacts")
        assert checks_artifacts is not None, (
            "checks_artifacts must be provided in context or as an argument"
        )
        self.artifact_mgr.register_artifact(
            run_id=self.run_name,
            artifact_key=ArtifactPath.DEEPCHECKS,
            artifacts=checks_artifacts,
            add_to_mlflow=True,
        )
        return context


class LogDatasetMetadata(LogArtifact):
    def __init__(
        self,
        artifact_mgr: ArtifactsManager,
        run_name: str,
        data_type: DataType,
    ):
        super().__init__(
            artifact_key=ArtifactPath.DATASET,
            artifact_mgr=artifact_mgr,
            run_name=run_name,
        )
        self.data_type = data_type

    def run(
        self,
        context: dict,
        train_data: Optional[BaseDataset] = None,
        test_data: Optional[BaseDataset] = None,
    ) -> dict:
        train_data = train_data or context.get("train_data")
        test_data = test_data or context.get("test_data")
        dataset_name = self.run_name or context.get("run_name")

        assert isinstance(dataset_name, str), (
            f"run_name must be a string, got {type(dataset_name)}"
        )

        data_statistics = get_data_statistics(
            data_type=self.data_type, train_data=train_data, test_data=test_data
        )
        dataset_artifacts = DatasetArtifacts(
            dataset_name=dataset_name,
            train_statistics=data_statistics["train"],
            test_statistics=data_statistics.get("test"),
            task_type=data_statistics["task_type"],
        )
        self.artifact_mgr.register_artifact(
            run_id=dataset_name,
            artifact_key=ArtifactPath.DATASET,
            add_to_mlflow=True,
            artifacts=dataset_artifacts,
        )
        return context


class LogModelCheckpoint(LogArtifact):
    def __init__(
        self,
        run_name: str,
        artifact_mgr: ArtifactsManager,
    ):
        super().__init__(
            artifact_key=ArtifactPath.MODEL_CHECKPOINT,
            artifact_mgr=artifact_mgr,
            run_name=run_name,
        )

    def run(
        self,
        context: dict,
        model: Optional[Any] = None,
    ) -> dict:
        model = model or context.get("model")

        if model is None:
            LOGGER.warning("model not provided, will not log model metadata")
            return context

        # Extract model metadata
        try:
            artifacts = get_model_metadata(model)
        except Exception as e:
            LOGGER.error(f"Failed to extract model metadata: {str(e)}")
            raise e

        # Create model checkpoint artifacts with metadata
        artifacts.path = context.get("model_checkpoint_path", None)
        artifacts.model_type = model.__class__.__name__
        artifacts.config = context.get("model_config", None)
        self.artifact_mgr.register_artifact(
            run_id=self.run_name,
            artifact_key=ArtifactPath.MODEL_CHECKPOINT,
            add_to_mlflow=True,
            artifacts=artifacts,
        )
        return context
