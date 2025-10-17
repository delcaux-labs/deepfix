from typing import Optional, List, Union
from .base import Step
from ..utils.logging import get_logger
from ...shared.models import (
    ArtifactPath,
    DatasetArtifacts,
    DeepchecksArtifacts,
    Artifacts,
    TrainingArtifacts,
    ModelCheckpointArtifacts,
)
from ..integrations import MLflowManager
from ..artifacts import ArtifactsManager


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

    def run(self) -> Union[Artifacts, dict[str, Artifacts]]:
        if self.run_id is None:
            raise ValueError(f"run_id must be set in MLflowManager for artifact: {self.artifact_key}")
        self.logger.info(
            f"Loading artifact: {self.artifact_key} for run_id: {self.run_id}"
        )
        artifact = self.artifact_mgr.load_artifact(
            run_id=self.run_id, artifact_key=self.artifact_key, download_if_missing=True
        )
        return artifact
    
    @property
    def name(self) -> str:
        return self.get_name()
    
    def get_name(self) -> str:
        return self.artifact_key.value


class LoadTrainingArtifact(LoadArtifact):
    def __init__(self, mlflow_manager: MLflowManager, artifact_sqlite_path: str):
        super().__init__(
            artifact_key=ArtifactPath.TRAINING,
            mlflow_manager=mlflow_manager,
            artifact_sqlite_path=artifact_sqlite_path,
        )

    def run(self) -> TrainingArtifacts:
        return super().run()



class LoadDeepchecksArtifacts(LoadArtifact):
    def __init__(self, mlflow_manager: MLflowManager, artifact_sqlite_path: str):
        super().__init__(
            artifact_key=ArtifactPath.DEEPCHECKS,
            mlflow_manager=mlflow_manager,
            artifact_sqlite_path=artifact_sqlite_path,
        )
       
    def run(self) -> DeepchecksArtifacts:
        return super().run()


class LoadModelCheckpoint(LoadArtifact):
    def __init__(self, mlflow_manager: MLflowManager, artifact_sqlite_path: str):
        super().__init__(
            artifact_key=ArtifactPath.MODEL_CHECKPOINT,
            mlflow_manager=mlflow_manager,
            artifact_sqlite_path=artifact_sqlite_path,
        )

    def run(self) -> ModelCheckpointArtifacts:
        return super().run()


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
        self.dataset_name = dataset_name

    def run(self) -> dict[str, Artifacts]:
        """
        Returns a dict with ArtifactPath.DATASET and optionally ArtifactPath.DEEPCHECKS keys.
        This allows loading both artifacts while keeping them separated.
        """
        assert self.run_id is not None, "run_id must be set in MLflowManager"
        self.logger.info(
            f"Loading artifact: {self.artifact_key} for run_id: {self.run_id}"
        )
        
        result = {}
        
        # Load dataset metadata
        dataset_artifact = self._load_dataset_metadata()
        if dataset_artifact is not None:
            result[ArtifactPath.DATASET.value] = dataset_artifact
        
        deepchecks_artifact = self._load_deepchecks_artifacts()
        if deepchecks_artifact is not None:
            result[ArtifactPath.DEEPCHECKS.value] = deepchecks_artifact
        
        return result

    def _load_dataset_metadata(self) -> Optional[DatasetArtifacts]:
        # Dataset metadata
        artifact = self.artifact_mgr.load_artifact(
            run_id=self.run_id,
            artifact_key=ArtifactPath.DATASET,
            download_if_missing=True,
        )
        return artifact

    def _load_deepchecks_artifacts(self) -> Optional[DeepchecksArtifacts]:
        mlflow_run_id = self.artifact_mgr.get_mlflow_run_id(
            self.run_id, ArtifactPath.DATASET
        )
        if mlflow_run_id is None:
            self.logger.warning(f"MLflow run ID not found for dataset {self.run_id}")
            return None
        artifact = self.artifact_mgr.load_artifact(
            run_id=mlflow_run_id,
            artifact_key=ArtifactPath.DEEPCHECKS,
            download_if_missing=True,
        )
        return artifact
