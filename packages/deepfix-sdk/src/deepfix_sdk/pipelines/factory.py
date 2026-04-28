from typing import Any, Callable, List, Optional, Union

import torch
from deepfix_core.models import ArtifactPath, DataType

from ..artifacts import ArtifactRepository, ArtifactsManager
from ..config import (
    ArtifactConfig,
    DeepchecksConfig,
    DefaultPaths,
    IngestionPipelineConfig,
    MLflowConfig,
)
from ..data import BaseDataset
from ..integrations import MLflowManager
from ..utils.logging import get_logger
from .base import Pipeline, Step
from .checks import Checks
from .data_ingestion import DataIngestor
from .loaders import (
    LoadDatasetArtifact,
    LoadDeepchecksArtifacts,
    LoadModelCheckpoint,
    LoadTrainingArtifact,
)
from .loggers import (
    LogChecksArtifacts,
    LogDatasetMetadata,
    LogModelCheckpoint,
    LogTrainingArtifact,
)

LOGGER = get_logger(__name__)


def create_run_name(dataset_name: str, model_name: Optional[str] = None) -> str:
    if model_name:
        return f"{dataset_name}___{model_name}"
    return dataset_name


class TrainLoggingPipeline(Pipeline):
    def __init__(
        self,
        dataset_name: str,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        batch_size: int = 8,
        model_evaluation_checks: bool = True,
        model: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.model_evaluation_checks = model_evaluation_checks

        mlflow_manager = MLflowManager(
            tracking_uri=mlflow_tracking_uri,
            create_run_if_not_exists=True,
            experiment_name=experiment_name,
            run_id=run_id,
            run_name=run_name,
        )

        steps_evaluation = []
        sqlite_path = sqlite_path or DefaultPaths.ARTIFACTS_SQLITE_PATH
        cfg = dict(mlflow_manager=mlflow_manager, sqlite_path=sqlite_path)
        if self.model_evaluation_checks:
            deepchecks_config = DeepchecksConfig(
                model_evaluation=model_evaluation_checks,
                train_test_validation=False,
                data_integrity=False,
                batch_size=batch_size,
            )
            steps_evaluation.append(
                DataIngestor(batch_size=deepchecks_config.batch_size, model=model),
                Checks(deepchecks_config=deepchecks_config, dataset_name=dataset_name),
                LogChecksArtifacts(**cfg),
            )
        steps = [
            LogModelCheckpoint(**cfg),
            LogTrainingArtifact(**cfg),
            *steps_evaluation,
        ]
        super().__init__(steps=steps)

    def run(
        self,
        metric_names: List[str],
        checkpoint_artifact_path: str,
        train_data: Optional[BaseDataset] = None,
        test_data: Optional[BaseDataset] = None,
    ) -> dict:
        self.context = {}
        self.context["checkpoint_artifact_path"] = checkpoint_artifact_path
        self.context["metric_names"] = metric_names
        if self.model_evaluation_checks:
            assert train_data is not None, (
                "train_data must be provided if model_evaluation_checks is True"
            )
            self.context["train_data"] = train_data
            self.context["test_data"] = test_data
        return super().run(**self.context)


class ChecksPipeline(Pipeline):
    def __init__(
        self,
        dataset_name: str,
        train_test_validation: bool = True,
        data_integrity: bool = True,
        model_evaluation: bool = False,
        model: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mlflow_tracking_uri: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        batch_size: int = 16,
        max_samples: Optional[int] = None,
        random_state: int = 42,
        save_results: bool = False,
        output_dir: Optional[str] = None,
        log_artifacts: bool = True,
        model_name: Optional[str] = None,
    ):
        deepchecks_config = DeepchecksConfig(
            train_test_validation=train_test_validation,
            data_integrity=data_integrity,
            model_evaluation=model_evaluation,
            batch_size=batch_size,
            max_samples=max_samples,
            random_state=random_state,
            save_results=save_results,
            output_dir=output_dir,
        )
        sqlite_path = sqlite_path or DefaultPaths.ARTIFACTS_SQLITE_PATH
        steps = [
            DataIngestor(batch_size=deepchecks_config.batch_size, model=model),
            Checks(
                deepchecks_config=deepchecks_config,
                dataset_name=dataset_name,
                model_name=model_name,
            ),
        ]
        run_name = create_run_name(dataset_name, model_name=model_name)
        if log_artifacts:
            mlflow_manager = MLflowManager(
                tracking_uri=mlflow_tracking_uri
                or DefaultPaths.MLFLOW_TRACKING_URI.value,
                create_run_if_not_exists=True,
                experiment_name=DefaultPaths.DATASETS_EXPERIMENT_NAME.value,
                run_name=run_name,
            )
            steps.append(
                LogChecksArtifacts(
                    mlflow_manager=mlflow_manager,
                    sqlite_path=sqlite_path,
                    run_name=run_name,
                )
            )
        super().__init__(steps=steps)

    def run(
        self,
        train_data: BaseDataset,
        test_data: Optional[BaseDataset] = None,
    ) -> dict:
        self.context = {}
        self.context["test_data"] = test_data
        self.context["train_data"] = train_data
        return super().run(**self.context)


class IngestionPipeline(Pipeline):
    def __init__(
        self,
        dataset_name: str,
        data_type: Union[str, DataType],
        batch_size: int = 16,
        mlflow_tracking_uri: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        train_test_validation: bool = True,
        data_integrity: bool = True,
        model_evaluation: bool = False,
        model_name: Optional[str] = None,
        max_samples: Optional[int] = None,
        random_state: int = 42,
        save_results: bool = False,
        output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        overwrite: bool = False,
    ):
        self.sqlite_path = sqlite_path or DefaultPaths.ARTIFACTS_SQLITE_PATH
        self.dataset_name = dataset_name
        self.model_name = model_name

        if not isinstance(data_type, DataType):
            data_type = DataType(data_type)

        if model_evaluation:
            assert model_name is not None, (
                "model_name must be provided if model_evaluation is True"
            )

        self.run_name = create_run_name(dataset_name, model_name=model_name)

        self.deepchecks_config = DeepchecksConfig(
            model_evaluation=model_evaluation,
            train_test_validation=train_test_validation,
            data_integrity=data_integrity,
            batch_size=batch_size,
            data_type=data_type,
            max_samples=max_samples,
            random_state=random_state,
            save_results=save_results,
            output_dir=output_dir,
        )
        self.mlflow_manager = MLflowManager(
            tracking_uri=mlflow_tracking_uri or DefaultPaths.MLFLOW_TRACKING_URI.value,
            create_run_if_not_exists=True,
            experiment_name=experiment_name or DefaultPaths.EXPERIMENT_NAME.value,
            run_name=self.run_name,
        )
        self.do_checks = train_test_validation or data_integrity
        self.artifact_mgr = ArtifactsManager(
            sqlite_path=self.sqlite_path, mlflow_manager=self.mlflow_manager
        )
        if self.check_if_exists(self.run_name, self.sqlite_path):
            if overwrite:
                self.delete_artifact(self.run_name)
            else:
                raise ValueError(
                    f"Run `{self.run_name}` already exists in the database. Use overwrite=True to overwrite it."
                )

        cfg = dict(artifact_mgr=self.artifact_mgr, run_name=self.run_name)
        steps = [
            LogDatasetMetadata(data_type=data_type, **cfg),
        ]
        if train_test_validation or data_integrity:
            steps.extend(
                [
                    DataIngestor(batch_size=batch_size, model=None),
                    Checks(
                        deepchecks_config=self.deepchecks_config,
                        dataset_name=self.dataset_name,
                        model_name=self.model_name,
                    ),
                    LogChecksArtifacts(**cfg),
                ]
            )
        if model_evaluation:
            steps.append(LogModelCheckpoint(**cfg))

        super().__init__(steps=steps)

    @classmethod
    def from_config(cls, config: IngestionPipelineConfig) -> "IngestionPipeline":
        return cls(**config.to_pipeline_kwargs())

    def delete_artifact(
        self,
        run_name: str,
    ) -> None:
        for artifact_key in ArtifactPath.__members__.values():
            self.artifact_mgr.delete_artifact(run_name, artifact_key)
            mlflow_run_id = self.artifact_mgr.get_mlflow_run_id(
                run_id=run_name, artifact_key=artifact_key
            )
        if mlflow_run_id:
            self.mlflow_manager.delete_run(mlflow_run_id)
            LOGGER.info(f"Deleted MLflow run `{mlflow_run_id}`")

        return None

    def check_if_exists(self, run_name: str, sqlite_path: str) -> bool:
        repo = ArtifactRepository(sqlite_path)
        return repo.get(run_name, ArtifactPath.DATASET.value) is not None

    def run(
        self,
        train_data: BaseDataset,
        test_data: Optional[BaseDataset] = None,
        model: Optional[Any] = None,
    ) -> dict:
        self.context = {}
        self.context["test_data"] = test_data
        self.context["train_data"] = train_data
        self.context["model"] = model
        try:
            for step in self.steps:
                step.run(context=self.context)
        except Exception as e:
            self.delete_artifact(self.run_name)
            raise e

        return self.context


class ArtifactLoadingPipeline(Pipeline):
    def __init__(
        self,
        dataset_name: str,
        model_name: Optional[str] = None,
        mlflow_config: Optional[MLflowConfig] = None,
        artifact_config: Optional[ArtifactConfig] = None,
    ):
        self.run_name = create_run_name(dataset_name, model_name=model_name)
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.artifact_config = artifact_config or ArtifactConfig()
        mlflow_manager = MLflowManager.from_config(
            self.mlflow_config, run_name=self.run_name
        )
        self.artifact_mgr = ArtifactsManager(
            mlflow_manager=mlflow_manager, sqlite_path=self.artifact_config.sqlite_path
        )

        super().__init__(steps=self._load_steps())

    def _load_steps(self) -> list[Step]:
        """Build steps based on configuration. Supports loading multiple artifact types."""
        steps = []
        cfg = dict(
            artifact_mgr=self.artifact_mgr,
            run_name=self.run_name,
        )

        # Load dataset metadata if configured
        if self.artifact_config.load_dataset_metadata:
            LOGGER.info("Loading dataset metadata")
            steps.append(LoadDatasetArtifact(**cfg))

        # Load deepchecks artifacts if configured
        if self.artifact_config.load_checks:
            LOGGER.info("Loading deepchecks artifacts")
            steps.append(LoadDeepchecksArtifacts(**cfg))

        # Load model checkpoint if configured
        if self.artifact_config.load_model_checkpoint:
            LOGGER.info("Loading model checkpoint")
            steps.append(LoadModelCheckpoint(**cfg))

        # Load training artifacts if configured
        if self.artifact_config.load_training:
            LOGGER.info("Loading training artifacts")
            steps.append(LoadTrainingArtifact(**cfg))

        # Ensure at least one artifact type is configured to load
        if not steps:
            raise ValueError(
                "No artifacts to load. Please enable at least one of: "
                "load_dataset_metadata, load_checks, load_model_checkpoint, load_training"
            )

        return steps
