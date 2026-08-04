from deepfix_core.models import (
    ArtifactPath,
    Artifacts,
    DatasetArtifacts,
    DeepchecksArtifacts,
    ModelCheckpointArtifacts,
    TrainingArtifacts,
)

from ..artifacts import ArtifactsManager
from ..logging import get_logger
from .base import Step


class LoadArtifact(Step):
    def __init__(
        self, artifact_key: ArtifactPath, artifact_mgr: ArtifactsManager, run_id: str
    ):
        self.artifact_key = artifact_key
        self.artifact_mgr = artifact_mgr
        self.run_id = run_id
        self.logger = get_logger(self.__class__.__name__)

    def run(self, context: dict, **kwargs) -> Artifacts:
        self.logger.info(
            f"Loading artifact: {self.artifact_key} for run_id: {self.run_id}"
        )
        artifact = self.artifact_mgr.load_artifact(
            run_id=self.run_id, artifact_key=self.artifact_key, download_if_missing=True
        )
        context[self.artifact_key.value] = artifact
        return artifact

    def get_name(self) -> str:
        return self.artifact_key.value


class LoadTrainingArtifact(LoadArtifact):
    def __init__(self, artifact_mgr: ArtifactsManager, run_name: str):
        super().__init__(
            artifact_key=ArtifactPath.TRAINING,
            artifact_mgr=artifact_mgr,
            run_id=run_name,
        )

    def run(self, context: dict, **kwargs) -> TrainingArtifacts:
        return super().run(context=context)


class LoadDeepchecksArtifacts(LoadArtifact):
    def __init__(self, artifact_mgr: ArtifactsManager, run_name: str):
        super().__init__(
            artifact_key=ArtifactPath.DEEPCHECKS,
            artifact_mgr=artifact_mgr,
            run_id=run_name,
        )

    def run(self, context: dict, **kwargs) -> DeepchecksArtifacts:
        return super().run(context=context)


class LoadModelCheckpoint(LoadArtifact):
    def __init__(self, artifact_mgr: ArtifactsManager, run_name: str):
        super().__init__(
            artifact_key=ArtifactPath.MODEL_CHECKPOINT,
            artifact_mgr=artifact_mgr,
            run_id=run_name,
        )

    def run(self, context: dict, **kwargs) -> ModelCheckpointArtifacts:
        return super().run(context=context)


class LoadDatasetArtifact(LoadArtifact):
    def __init__(
        self,
        run_name: str,
        artifact_mgr: ArtifactsManager,
    ):
        super().__init__(
            artifact_key=ArtifactPath.DATASET,
            artifact_mgr=artifact_mgr,
            run_id=run_name,
        )

    def run(self, context: dict, **kwargs) -> DatasetArtifacts:
        return super().run(context=context)
