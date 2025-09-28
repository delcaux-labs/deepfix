from typing import Optional, List
from torch.utils.data import Dataset

from .base import Step
from ...utils.logging import get_logger
from ..artifacts import (
    ArtifactsManager,
    ArtifactPath,
    DeepchecksArtifacts,
    DatasetArtifacts,
    TrainingArtifacts,
)
from ..data.utils import DataStatistics
from ...integrations import MLflowManager

LOGGER = get_logger(__name__)


class LogArtifact(Step):
    def __init__(
        self,
        artifact_key: ArtifactPath,
        sqlite_path: str,
        mlflow_manager: MLflowManager,
    ):
        self.artifact_key = artifact_key
        self.mlflow_manager = mlflow_manager
        self.artifact_mgr = ArtifactsManager(
            sqlite_path=sqlite_path, mlflow_manager=self.mlflow_manager
        )


class LogTrainingArtifact(LogArtifact):
    def __init__(self, sqlite_path: str, mlflow_manager: MLflowManager):
        super().__init__(
            artifact_key=ArtifactPath.TRAINING,
            sqlite_path=sqlite_path,
            mlflow_manager=mlflow_manager,
        )

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
            run_id=self.mlflow_manager.run_id,
            artifact_key=ArtifactPath.TRAINING,
            artifacts=training_artifacts,
            add_to_mlflow=True,
        )
        return context


class LogChecksArtifacts(LogArtifact):
    def __init__(self, sqlite_path: str, mlflow_manager: MLflowManager):
        super().__init__(
            artifact_key=ArtifactPath.DEEPCHECKS,
            sqlite_path=sqlite_path,
            mlflow_manager=mlflow_manager,
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
            run_id=self.mlflow_manager.run_id,
            artifact_key=ArtifactPath.DEEPCHECKS,
            artifacts=checks_artifacts,
            add_to_mlflow=True,
        )
        return context


class LogModelCheckpoint(LogArtifact):
    def __init__(self, mlflow_manager: MLflowManager, sqlite_path: str):
        super().__init__(
            artifact_key=ArtifactPath.MODEL_CHECKPOINT,
            sqlite_path=sqlite_path,
            mlflow_manager=mlflow_manager,
        )

    def run(
        self, context: dict, checkpoint_artifact_path: Optional[str] = None
    ) -> dict:
        local_path = checkpoint_artifact_path or context.get("checkpoint_artifact_path")
        if local_path is None:
            LOGGER.warning(
                "checkpoint_artifact_path not provided, will not log model checkpoint"
            )
            return context
        self.artifact_mgr.register_artifact(
            run_id=self.mlflow_manager.run_id,
            artifact_key=ArtifactPath.MODEL_CHECKPOINT,
            local_path=local_path,
            add_to_mlflow=True,
        )
        return context


class LogDatasetMetadata(LogArtifact):
    def __init__(
        self, sqlite_path: str, mlflow_manager: MLflowManager, dataset_name: str
    ):
        super().__init__(
            artifact_key=ArtifactPath.DATASET,
            sqlite_path=sqlite_path,
            mlflow_manager=mlflow_manager,
        )
        self.dataset_name = dataset_name

    def run(
        self,
        context: dict,
        train_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
    ) -> dict:
        train_data = train_data or context.get("train_data")
        test_data = test_data or context.get("test_data")
        dataset_name = self.dataset_name or context.get("dataset_name")

        assert isinstance(dataset_name, str), (
            f"dataset_name must be a string, got {type(dataset_name)}"
        )
        assert isinstance(train_data, Dataset), (
            f"train_data must be a PyTorch Dataset, got {type(train_data)}"
        )
        assert isinstance(test_data, Dataset), (
            f"test_data must be a PyTorch Dataset, got {type(test_data)}"
        )
        data_statistics = DataStatistics(train_data=train_data, test_data=test_data)
        dataset_artifacts = DatasetArtifacts(
            dataset_name=dataset_name, statistics=data_statistics.get_statistics()
        )
        self.artifact_mgr.register_artifact(
            run_id=dataset_name,
            artifact_key=ArtifactPath.DATASET,
            add_to_mlflow=True,
            artifacts=dataset_artifacts,
        )
        return context
