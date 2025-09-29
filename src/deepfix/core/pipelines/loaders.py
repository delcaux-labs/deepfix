from typing import Optional
from .base import Step
from ...utils.logging import get_logger
from ..artifacts import (
    ArtifactsManager,
    ArtifactPath,
    DatasetArtifacts,
    DeepchecksArtifacts,
)
from ...integrations import MLflowManager


class LoadArtifact(Step):
    def __init__(
        self,
        artifact_key: ArtifactPath,
        mlflow_manager: MLflowManager,
        artifact_sqlite_path: str,
        run_id: Optional[str] = None,
    ):
        self.mlflow_manager = mlflow_manager
        self.artifact_key = artifact_key
        self.artifact_mgr = ArtifactsManager(
            sqlite_path=artifact_sqlite_path, mlflow_manager=self.mlflow_manager
        )
        self.run_id = run_id or self.mlflow_manager.run_id
        self.logger = get_logger(self.__class__.__name__)

    def run(self, context: dict) -> dict:
        assert self.run_id is not None, "run_id must be set in MLflowManager"
        self.logger.info(
            f"Loading artifact: {self.artifact_key} for run_id: {self.run_id}"
        )
        artifact = self.artifact_mgr.load_artifact(
            run_id=self.run_id, artifact_key=self.artifact_key, download_if_missing=True
        )
        if "artifacts" in context.keys():
            context["artifacts"].append(artifact)
        else:
            context["artifacts"] = [artifact]

        return context


class LoadTrainingArtifact(LoadArtifact):
    def __init__(self, mlflow_manager: MLflowManager, artifact_sqlite_path: str):
        super().__init__(
            artifact_key=ArtifactPath.TRAINING,
            mlflow_manager=mlflow_manager,
            artifact_sqlite_path=artifact_sqlite_path,
        )


class LoadDeepchecksArtifacts(LoadArtifact):
    def __init__(self, mlflow_manager: MLflowManager, artifact_sqlite_path: str):
        super().__init__(
            artifact_key=ArtifactPath.DEEPCHECKS,
            mlflow_manager=mlflow_manager,
            artifact_sqlite_path=artifact_sqlite_path,
        )


class LoadModelCheckpoint(LoadArtifact):
    def __init__(self, mlflow_manager: MLflowManager, artifact_sqlite_path: str):
        super().__init__(
            artifact_key=ArtifactPath.MODEL_CHECKPOINT,
            mlflow_manager=mlflow_manager,
            artifact_sqlite_path=artifact_sqlite_path,
        )


class LoadDatasetArtifact(LoadArtifact):
    def __init__(
        self,
        dataset_name: str,
        mlflow_manager: MLflowManager,
        artifact_sqlite_path: str,
    ):
        super().__init__(
            artifact_key=ArtifactPath.DATASET,
            mlflow_manager=mlflow_manager,
            artifact_sqlite_path=artifact_sqlite_path,
            run_id=dataset_name,
        )

    def run(self, context: dict) -> dict:
        assert self.run_id is not None, "run_id must be set in MLflowManager"
        self.logger.info(
            f"Loading artifact: {self.artifact_key} for run_id: {self.run_id}"
        )
        # get artifacts
        metadata_artifact = self._load_dataset_metadata()
        deepchecks_artifact = self._load_deepchecks_artifacts()
        arts = [metadata_artifact, deepchecks_artifact]
        if "artifacts" in context.keys():
            context["artifacts"].extend(arts)
        else:
            context["artifacts"] = arts
        return context

    def _load_dataset_metadata(self) -> DatasetArtifacts:
        # Dataset metadata
        artifact = self.artifact_mgr.load_artifact(
            run_id=self.run_id,
            artifact_key=ArtifactPath.DATASET,
            download_if_missing=True,
        )
        return artifact

    def _load_deepchecks_artifacts(self) -> DeepchecksArtifacts:
        mlflow_run_id = self.artifact_mgr.get_mlflow_run_id(
            self.run_id, ArtifactPath.DATASET
        )
        if mlflow_run_id is None:
            raise ValueError(f"MLflow run ID not found for dataset {self.run_id}")
        artifact = self.artifact_mgr.load_artifact(
            run_id=mlflow_run_id,
            artifact_key=ArtifactPath.DEEPCHECKS,
            download_if_missing=True,
        )
        return artifact
