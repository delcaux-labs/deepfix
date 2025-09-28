from lightning.pytorch.callbacks import Callback
import lightning as L
from torch.utils.data import Dataset
from typing import Optional, List, Union

from ..utils.logging import get_logger
from .mlflow import MLflowManager
from ..core.pipelines.factory import TrainLoggingPipeline

LOGGER = get_logger(__name__)


class DeepSightCallback(Callback):
    def __init__(
        self,
        dataset_name: str,
        train_dataset: Dataset,
        metric_names: Union[List[str], None],
        val_dataset: Optional[Dataset] = None,
        model_evaluation_checks: bool = True,
        batch_size: int = 16,
    ):
        super().__init__()
        self.mlflow_run_id = None
        self.mlflow_experiment_id = None
        self.best_model_path: Optional[str] = None
        self.best_model_score: Optional[float] = None
        self.dataset_name: str = dataset_name

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mlflow_manager = None
        self.model_evaluation_checks = model_evaluation_checks
        self.batch_size = batch_size
        self.metric_names = metric_names

        if model_evaluation_checks:
            assert isinstance(metric_names, list), "metric_names must be a list"

    def setup(self, trainer, pl_module, stage):
        LOGGER.info(f"Setup callback for {stage} stage")

    @property
    def state(self):
        return dict(
            mlflow_run_id=self.mlflow_run_id,
            mlflow_experiment_id=self.mlflow_experiment_id,
            best_model_path=self.best_model_path,
            best_model_score=self.best_model_score,
        )

    def load_state_dict(self, state_dict):
        self.mlflow_run_id = state_dict.get("mlflow_run_id", None)
        self.mlflow_experiment_id = state_dict.get("mlflow_experiment_id", None)
        self.best_model_path = state_dict.get("best_model_path", None)
        self.best_model_score = state_dict.get("best_model_score", None)

    def state_dict(self):
        return self.state.copy()

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.mlflow_run_id = getattr(pl_module.logger, "run_id", None)
        self.mlflow_experiment_id = getattr(pl_module.logger, "experiment_id", None)
        tracking_uri = getattr(pl_module.logger, "_tracking_uri", None)
        if self.mlflow_run_id is not None:
            LOGGER.info(f"MLflow run_id: {self.mlflow_run_id}")
            LOGGER.info(f"MLflow experiment_id: {self.mlflow_experiment_id}")
            self.mlflow_manager = MLflowManager(
                run_id=self.mlflow_run_id, tracking_uri=tracking_uri
            )
        else:
            LOGGER.warning("No mlflow logger found")

    def run(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # get best model path and score from trainer
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.best_model_score = trainer.checkpoint_callback.best_model_score

        pipeline = TrainLoggingPipeline(
            mlflow_manager=self.mlflow_manager,
            dataset_name=self.dataset_name,
            model=pl_module.predict_step,
            model_evaluation_checks=self.model_evaluation_checks,
            batch_size=self.batch_size,
        )
        pipeline.run(
            metric_names=self.metric_names,
            checkpoint_artifact_path=self.best_model_path,
        )
        return None

    # TODO: make sure that on_fit_end pl_module is the best model, automatically loaded by trainer
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.run(trainer=trainer, pl_module=pl_module)
