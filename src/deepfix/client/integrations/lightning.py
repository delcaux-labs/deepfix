from lightning.pytorch.callbacks import Callback
import lightning as L
from typing import Optional, List

from ..utils.logging import get_logger
from ..pipelines.factory import TrainLoggingPipeline

LOGGER = get_logger(__name__)


class DeepSightCallback(Callback):
    def __init__(
        self,
        dataset_name: str,
        metric_names: List[str],
        batch_size: int = 16,
    ):
        super().__init__()
        self.mlflow_run_id = None
        self.mlflow_experiment_id = None
        self.best_model_path: Optional[str] = None
        self.best_model_score: Optional[float] = None
        self.dataset_name: str = dataset_name
        self.batch_size = batch_size
        self.metric_names = metric_names


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
        if self.mlflow_run_id is not None:
            LOGGER.info(f"MLflow run_id: {self.mlflow_run_id}")
            LOGGER.info(f"MLflow experiment_id: {self.mlflow_experiment_id}")
        else:
            LOGGER.warning("No mlflow logger found")

    #TODO: make it work with any logger?
    def run(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # get best model path and score from trainer
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.best_model_score = trainer.checkpoint_callback.best_model_score

        pipeline = TrainLoggingPipeline(
            dataset_name=self.dataset_name,
            run_id=self.mlflow_run_id,
            model=pl_module.predict_step,
            model_evaluation_checks=True,
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
