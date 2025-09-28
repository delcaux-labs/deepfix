from __future__ import annotations
from abc import abstractmethod, ABC
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import StrEnum
import os
from sqlmodel import SQLModel, Field as SQLField
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Enum as SAEnum,
    JSON,
    Index,
    UniqueConstraint,
)
from datetime import datetime
from omegaconf import DictConfig
import pandas as pd
import yaml

from ..config import DeepchecksConfig


class ArtifactPath(StrEnum):
    # training artifacts
    TRAINING = "training_artifacts"
    TRAINING_METRICS = "metrics.csv"
    MODEL_CHECKPOINT = "best_checkpoint"
    TRAINING_PARAMS = "params.yaml"
    # deepchecks artifacts
    DEEPCHECKS = "deepchecks"
    # dataset artifacts
    DATASET = "dataset"


## Deepchecks
class DeepchecksResultHeaders(StrEnum):
    # Train-Test Validation
    LabelDrift = "Label Drift"
    ImageDatasetDrift = "Image Dataset Drift"
    ImagePropertyDrift = "Image Property Drift"
    PropertyLabelCorrelationChange = "Property Label Correlation Change"
    HeatmapComparison = "Heatmap Comparison"
    NewLabels = "New Labels"
    # Data Integrity
    ImagePropertyOutliers = "Image Property Outliers"
    PropertyLabelCorrelation = "Property Label Correlation"
    LabelPropertyOutliers = "Label Property Outliers"
    ClassPerformance = "Class Performance"


class DeepchecksParsedResult(BaseModel):
    header: str = Field(description="Header of the result")
    json_result: Dict[str, Any] = Field(description="JSON result of the result")
    display_images: Optional[List[str]] = Field(
        default=None,
        description="Display images of the result as base64 encoded strings",
    )
    display_txt: Optional[str] = Field(
        default=None, description="Display text of the result"
    )

    def to_dict(self, exclude_images: bool = False) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        dumped_dict["header"] = dumped_dict["header"]
        dumped_dict.pop("display_txt")
        if exclude_images:
            dumped_dict.pop("display_images")
        return dumped_dict

    @classmethod
    def from_dict(
        self, d: Union[Dict[str, Any], DictConfig]
    ) -> "DeepchecksParsedResult":
        return DeepchecksParsedResult(
            header=d["header"],
            json_result=d["json_result"],
            display_images=d.get("display_images", None),
            display_txt=d.get("display_txt", None),
        )


class Artifacts(BaseModel,ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


class DeepchecksArtifacts(Artifacts):
    dataset_name: str = Field(description="Name of the dataset")
    results: Dict[str, List[DeepchecksParsedResult]] = Field(
        description="Results of the artifact"
    )
    config: Optional[DeepchecksConfig] = Field(
        default=None, description="Config of the artifact"
    )

    def to_dict(self) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        dumped_dict["results"] = {
            k: [r.to_dict() for r in v] for k, v in self.results.items()
        }
        dumped_dict["config"] = self.config.model_dump() if self.config else None
        return dumped_dict

    @classmethod
    def from_dict(self, d: Union[Dict[str, Any], DictConfig]) -> "DeepchecksArtifacts":
        results = {
            k: [DeepchecksParsedResult.from_dict(r) for r in v]
            for k, v in d["results"].items()
        }
        config = None
        if d.get("config"):
            config = DeepchecksConfig.from_dict(d["config"])
        return DeepchecksArtifacts(
            dataset_name=d["dataset_name"], results=results, config=config
        )

    @classmethod
    def from_file(
        cls, file_path: Optional[str] = None, dir_path: Optional[str] = None
    ) -> "DeepchecksArtifacts":
        assert (file_path is None) ^ (dir_path is None), (
            "Either file_path or dir_path must be provided"
        )

        file_path = (
            file_path
            if (file_path is not None)
            else os.path.join(dir_path, ArtifactPath.DEEPCHECKS_ARTIFACTS.value)
        )
        with open(file_path, "r") as f:
            d = yaml.safe_load(f)

        artifacts = cls.from_dict(d)
        if dir_path is not None:
            artifacts.config = DeepchecksConfig.from_file(
                os.path.join(dir_path, ArtifactPath.DEEPCHECKS_CONFIG.value)
            )

        return artifacts


class ModelCheckpointArtifacts(Artifacts):
    path: str = Field(description="Path to the model checkpoint")
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Config of the model"
    )

    def to_dict(self) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict):
        return cls(model_path=d["model_path"], model_config=d.get("config"))

    @classmethod
    def from_file(cls, path: str) -> "ModelCheckpointArtifacts":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


# Training Artifacts
class TrainingArtifacts(Artifacts):
    model_config = {"arbitrary_types_allowed": True}

    metrics_path: Optional[str] = Field(
        default=None, description="Path to the metrics file"
    )
    metrics_values: Optional[pd.DataFrame] = Field(
        default=None, description="Metrics of the artifact"
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters of the training routine"
    )

    def to_dict(self) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        if self.metrics_values is not None:
            dumped_dict["metrics_values"] = self.metrics_values.to_dict(orient="list")
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict):
        if d.get("metrics_values"):
            metrics_values = pd.DataFrame.from_dict(d.get("metrics_values"))
        elif d.get("metrics_path"):
            metrics_values = pd.read_csv(d.get("metrics_path"))
        else:
            metrics_values = None
        return cls(
            metrics_path=d.get("metrics_path"),
            metrics_values=metrics_values,
            params=d.get("params"),
        )

    @classmethod
    def from_file(cls, metrics_path: str) -> "TrainingArtifacts":
        return cls(metrics_path=metrics_path, metrics_values=pd.read_csv(metrics_path))


# Dataset
class DatasetArtifacts(Artifacts):
    dataset_name: str = Field(..., description="Name of the dataset")
    statistics: Optional[Dict[str, Any]] = Field(
        default=None, description="Statistics of the dataset"
    )

    def to_dict(self) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        return dumped_dict

    @classmethod
    def from_file(cls, path: str) -> "DatasetArtifacts":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls(dataset_name=d["dataset_name"], statistics=d["statistics"])


# SQLModel
class ArtifactStatus(StrEnum):
    REGISTERED = "REGISTERED"
    DOWNLOADED = "DOWNLOADED"
    MISSING = "MISSING"
    ERROR = "ERROR"


class ArtifactRecord(SQLModel, table=True):
    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("run_id", "artifact_key", name="uq_run_id_artifact_key"),
        Index("idx_artifacts_run_id", "run_id"),
        Index("idx_artifacts_status", "status"),
        Index("idx_artifacts_mlflow_run_id", "mlflow_run_id"),
    )

    id: Optional[int] = SQLField(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=True),
    )
    run_id: str = SQLField(sa_column=Column(String, nullable=False))
    mlflow_run_id: Optional[str] = SQLField(
        default=None, sa_column=Column(String, nullable=True)
    )
    artifact_key: str = SQLField(sa_column=Column(String, nullable=False))
    source_uri: Optional[str] = SQLField(
        default=None, sa_column=Column(String, nullable=True)
    )
    local_path: Optional[str] = SQLField(
        default=None, sa_column=Column(String, nullable=True)
    )
    size_bytes: Optional[int] = SQLField(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    checksum_sha256: Optional[str] = SQLField(
        default=None, sa_column=Column(String, nullable=True)
    )
    status: ArtifactStatus = SQLField(
        default=ArtifactStatus.REGISTERED,
        sa_column=Column(SAEnum(ArtifactStatus), nullable=False),
    )
    metadata_json: Optional[Dict[str, Any]] = SQLField(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    tags_json: Optional[Dict[str, Any]] = SQLField(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    downloaded_at: Optional[datetime] = SQLField(
        default=None, sa_column=Column(DateTime(timezone=False), nullable=True)
    )
    last_accessed_at: Optional[datetime] = SQLField(
        default=None, sa_column=Column(DateTime(timezone=False), nullable=True)
    )
    created_at: datetime = SQLField(
        default_factory=datetime.now,
        sa_column=Column(
            DateTime(timezone=False), nullable=False, default=datetime.now
        ),
    )
    updated_at: datetime = SQLField(
        default_factory=datetime.now,
        sa_column=Column(
            DateTime(timezone=False),
            nullable=False,
            default=datetime.now,
            onupdate=datetime.now,
        ),
    )
