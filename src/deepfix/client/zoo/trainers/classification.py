import traceback
import os
from typing import Any, Tuple, Dict, Literal, Optional
import mlflow
from omegaconf import OmegaConf
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path
import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, DataLoader

from ...utils.logging import get_logger
from ...integrations.lightning import DeepSightCallback

LOGGER = get_logger(__name__)


class ClassificationTrainerConfig(BaseModel):
    # classes
    num_classes: int = Field(description="Number of classes")
    label_to_class_map: Dict[int, str] = Field(
        default={}, description="Label to class map"
    )

    # Dataloading
    batch_size: int = Field(default=16, description="Batch size")
    num_workers: int = Field(default=4, description="Number of workers")
    pin_memory: bool = Field(default=False, description="Pin memory")

    # Accelerator
    accelerator: str = Field(default="auto", description="Accelerator")
    precision: str = Field(default="16-mixed", description="Precision")

    # Val check interval
    val_check_interval: int = Field(default=1, description="Val check interval")

    # training
    epochs: int = Field(default=20, description="Number of epochs to train for")
    label_smoothing: float = Field(default=0.0, description="Label smoothing")
    lr: float = Field(default=1e-3, description="Learning rate")
    lrf: float = Field(default=1e-2, description="Learning rate factor")
    weight_decay: float = Field(default=5e-3, description="Weight decay")
    model_metadata: Dict[str, Any] = Field(default={}, description="Model metadata")
    reweight_classes: bool = Field(default=False, description="Reweight classes")

    # Monitoring
    monitor: str = Field(default="val_f1score", description="Monitor")
    patience: int = Field(default=10, description="Patience")
    min_delta: float = Field(default=1e-3, description="Minimum delta")
    mode: Literal["min", "max"] = Field(default="max", description="Mode")

    # Mlflow
    experiment_name: str = Field(
        default="classification", description="Experiment name"
    )
    run_name: str = Field(default="default", description="Run name")
    log_best_model: bool = Field(default=True, description="Log best model")
    tracking_uri: str = Field(
        default="http://localhost:5000", description="Tracking URI"
    )

    # Checkpoint
    dirpath: str = Field(default="checkpoints", description="Directory path")
    filename: str = Field(default="best-{epoch:02d}", description="Filename")
    save_weights_only: bool = Field(default=True, description="Save weights only")


class ClassifierModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        config: ClassificationTrainerConfig,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters(config.model_dump(), ignore=["model"])

        self.model = model
        self.num_classes = self.hparams.num_classes
        self.config = config

        # metrics
        cfg = {"task": "multiclass", "num_classes": self.num_classes, "average": None}
        self.accuracy = Accuracy(**cfg)
        self.precision = Precision(**cfg)
        self.recall = Recall(**cfg)
        self.f1score = F1Score(**cfg)
        self.ap = AUROC(**cfg)

        self.metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1score": self.f1score,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        logits = self(batch)
        return logits.softmax(dim=1)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        weight = None
        if self.hparams.reweight_classes:
            classes = y.cpu().flatten().tolist()
            weight = [
                len(classes) / torch.log2(torch.tensor(classes.count(i) + 2)).item()
                for i in range(self.num_classes)
            ]
            weight = torch.Tensor(weight).float().to(y.device)

        logits = self(x)
        loss = F.cross_entropy(
            logits,
            y.long().flatten(),
            label_smoothing=self.hparams.label_smoothing,
            weight=weight,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y = y.long().flatten()
        logits = self(x)

        loss = F.cross_entropy(logits, y, label_smoothing=self.hparams.label_smoothing)

        for _, metric in self.metrics.items():
            metric.update(logits.softmax(dim=1), y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        for name, metric in self.metrics.items():
            score = metric.compute().cpu()
            self.log(f"val_{name}", score.mean())
            # for i, score in enumerate(score):
            #    cls_name = self.config.label_to_class_map.get(i, i)
            #    self.log(f"val_{name}_class_{cls_name}", score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.epochs,
            T_mult=1,
            eta_min=self.hparams.lr * self.hparams.lrf,
        )
        return [optimizer], [lr_scheduler]


class ClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        pass


class ClassificationTrainer(object):
    """
    Trainer class for image classification models.

    This class handles the training and evaluation of classification models
    using PyTorch Lightning and MLflow for experiment tracking.
    """

    def __init__(self, config: ClassificationTrainerConfig):
        self.config = config
        self.best_model_path = None
        self.best_model_score = None

    def get_callbacks(self) -> tuple[list[Any], MLFlowLogger]:
        """
        Get the callbacks for the trainer.
        """

        mlflow.set_tracking_uri(self.config.tracking_uri)
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.monitor,
            save_last=True,
            mode=self.config.mode,
            dirpath=os.path.join(self.config.dirpath, self.config.run_name),
            filename=self.config.filename,
            save_weights_only=self.config.save_weights_only,
            save_on_train_epoch_end=False,
        )
        lr_callback = LearningRateMonitor(logging_interval="epoch")
        early_stopping = EarlyStopping(
            monitor=self.config.monitor,
            patience=self.config.patience,
            mode=self.config.mode,
            min_delta=self.config.min_delta,
        )
        mlflow_logger = MLFlowLogger(
            experiment_name=self.config.experiment_name,
            run_name=self.config.run_name,
            log_model=False,
            tracking_uri=self.config.tracking_uri,
        )

        # Create base callbacks list
        callbacks = [checkpoint_callback, early_stopping, lr_callback]

        return callbacks, mlflow_logger

    def run(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        deepsight_callback: Optional[DeepSightCallback] = None,
        debug: bool = False,
    ) -> None:
        """
        Run image classification training or evaluation based on config.
        """
        # Set float32 matmul precision to medium
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")

        # DataModule
        datamodule = ClassificationDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        example_input, _ = next(iter(datamodule.train_dataloader()))

        # Model
        classifier = ClassifierModule(
            model=model,
            config=self.config,
        )
        classifier.example_input_array = example_input

        # Callbacks
        callbacks, mlflow_logger = self.get_callbacks()
        if isinstance(deepsight_callback, DeepSightCallback):
            callbacks.append(deepsight_callback)
        trainer = Trainer(
            max_epochs=self.config.epochs if not debug else 1,
            accelerator=self.config.accelerator,
            precision=self.config.precision,
            logger=mlflow_logger,
            limit_train_batches=3 if debug else None,
            check_val_every_n_epoch=self.config.val_check_interval,
            limit_val_batches=3 if debug else None,
            callbacks=callbacks,
            default_root_dir=self.config.dirpath,
        )

        trainer.fit(classifier, datamodule=datamodule)

        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.best_model_score = trainer.checkpoint_callback.best_model_score
