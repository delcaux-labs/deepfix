from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml
from omegaconf import OmegaConf

from .repository import ArtifactRepository, ArtifactRecord, ArtifactStatus
from .services import ChecksumService
from ...shared.models import (
    ArtifactPath,
    DeepchecksArtifacts,
    TrainingArtifacts,
    DatasetArtifacts,
    Artifacts,
    ModelCheckpointArtifacts,
)
from ..utils.logging import get_logger
from ..config import MLflowConfig

LOGGER = get_logger(__name__)


class ArtifactsManager:
    def __init__(
        self,
        mlflow_manager,
        sqlite_path: str,
    ) -> None:
        from ..integrations import MLflowManager

        self.repo = ArtifactRepository(sqlite_path)
        self.checksum = ChecksumService()
        self.mlflow: MLflowManager = mlflow_manager
    
    @classmethod
    def from_config(cls, mlflow_config: MLflowConfig, sqlite_path: str) -> 'ArtifactsManager':
        from ..integrations import MLflowManager
        mlflow_manager = MLflowManager.from_config(mlflow_config)
        return cls(mlflow_manager=mlflow_manager, sqlite_path=sqlite_path)

    def register_artifact(
        self,
        run_id: str,
        artifact_key: Union[str, ArtifactPath],
        local_path: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
        add_to_mlflow: bool = False,
        artifacts: Optional[Artifacts] = None,
    ) -> ArtifactRecord:
        artifact_key = (
            ArtifactPath(artifact_key)
            if isinstance(artifact_key, str)
            else artifact_key
        )
        # check if artifact already registered
        rec = self.repo.get(run_id, artifact_key.value)
        if isinstance(rec, ArtifactRecord):
            raise ValueError(
                f"Artifact {artifact_key.value} already registered for run {run_id}"
            )
        # create record
        record = ArtifactRecord(
            run_id=run_id,
            mlflow_run_id=self.mlflow.run_id,
            artifact_key=artifact_key.value,
            source_uri=source_uri or self.mlflow.tracking_uri,
            local_path=local_path,
            status=ArtifactStatus.REGISTERED,
            metadata_json=metadata,
            tags_json=tags,
        )
        if add_to_mlflow:
            if artifacts:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(
                        tmp, Path(artifact_key.value).with_suffix(".yaml")
                    )
                    OmegaConf.save(artifacts.to_dict(), path)
                    self.mlflow.add_artifact(artifact_key.value, path)
            else:
                assert (local_path is not None) and os.path.exists(local_path), (
                    "local_path must be provided if artifacts is not provided"
                )
                self.mlflow.add_artifact(artifact_key.value, local_path)
        return self.repo.upsert(record)

    def ensure_downloaded(
        self, run_id: str, artifact_key: str, mlflow_run_id: str
    ) -> Path:
        local_path = self.mlflow.get_local_path(
            artifact_key, run_id=mlflow_run_id, download_if_missing=False
        )

        rec = self.repo.get(run_id, artifact_key)
        if rec and rec.local_path and Path(rec.local_path).exists():
            self.repo.touch_access(run_id, artifact_key)
            return Path(rec.local_path)

        downloaded_dir = self.mlflow.get_local_path(
            artifact_key, download_if_missing=True, run_id=mlflow_run_id
        )
        candidate = Path(downloaded_dir)
        final_path = candidate if candidate.is_file() else local_path
        if candidate.is_dir():
            final_path = candidate

        checksum = None
        size_bytes = None
        if final_path.is_file():
            checksum = self.checksum.compute_sha256(str(final_path))
            size_bytes = final_path.stat().st_size

        # update or create record
        rec = self.repo.update_local_path(
            run_id=run_id,
            artifact_key=artifact_key,
            local_path=str(final_path),
            status=ArtifactStatus.DOWNLOADED,
        )
        if rec is None:
            self.register_artifact(
                run_id=run_id,
                artifact_key=artifact_key,
                local_path=str(final_path),
                source_uri=self.mlflow.tracking_uri,
            )

        # update checksum/size if record exists
        existing = self.repo.get(run_id, artifact_key)
        if existing is not None:
            existing.checksum_sha256 = checksum
            existing.size_bytes = size_bytes
            existing.updated_at = datetime.now()
            self.repo.upsert(existing)

        return final_path

    def get_mlflow_run_id(
        self, run_id: str, artifact_key: ArtifactPath
    ) -> Optional[str]:
        rec = self.repo.get(run_id, artifact_key.value)
        if rec:
            return rec.mlflow_run_id
        return None

    def get_local_path(
        self,
        run_id: str,
        artifact_key: Union[str, ArtifactPath],
        download_if_missing: bool = True,
    ) -> Optional[Path]:
        artifact_key = (
            ArtifactPath(artifact_key)
            if isinstance(artifact_key, str)
            else artifact_key
        )
        artifact_key = artifact_key.value
        rec = self.repo.get(run_id, artifact_key)
        if rec and rec.local_path and Path(rec.local_path).exists():
            self.repo.touch_access(run_id, artifact_key)
            return Path(rec.local_path)
        if download_if_missing and rec:
            return self.ensure_downloaded(
                run_id=run_id,
                mlflow_run_id=rec.mlflow_run_id,
                artifact_key=artifact_key,
            )
        return None

    def load_artifact(
        self,
        run_id: str,
        artifact_key: Union[str, ArtifactPath],
        download_if_missing: bool = True,
    ) -> Optional[Artifacts]:
        artifact_key = (
            ArtifactPath(artifact_key)
            if isinstance(artifact_key, str)
            else artifact_key
        )
        path = self.get_local_path(run_id, artifact_key.value, download_if_missing)
        if path is None:
            LOGGER.warning(f"Artifact {artifact_key} not found for run {run_id}")
            return None
        if artifact_key == ArtifactPath.DEEPCHECKS:
            return self._load_deepchecks_artifacts(path)
        elif artifact_key == ArtifactPath.TRAINING:
            return self._load_training_artifacts(path)
        elif artifact_key == ArtifactPath.MODEL_CHECKPOINT:
            return self._load_model_checkpoint(path)
        elif artifact_key == ArtifactPath.DATASET:
            return self._load_dataset_artifacts(path)
        else:
            raise ValueError(f"Artifact key {artifact_key} not supported")

    def _load_training_artifacts(self, local_path: str) -> TrainingArtifacts:
        metrics = os.path.join(local_path, ArtifactPath.TRAINING_METRICS.value)
        params = os.path.join(local_path, ArtifactPath.TRAINING_PARAMS.value)
        if not os.path.exists(params):
            return self.mlflow.get_training_artifacts()
        with open(params, "r") as f:
            params = yaml.safe_load(f)
        return TrainingArtifacts(
            metrics_path=metrics,
            metrics_values=pd.read_csv(metrics),
            params=params,
        )

    def _load_deepchecks_artifacts(self, local_path: str) -> DeepchecksArtifacts:
        artifacts = os.path.join(
            local_path, Path(ArtifactPath.DEEPCHECKS.value).with_suffix(".yaml")
        )
        artifacts = DeepchecksArtifacts.from_file(artifacts)
        return artifacts

    def _load_model_checkpoint(self, local_path: str) -> ModelCheckpointArtifacts:
        best_checkpoint = os.path.join(local_path, ArtifactPath.MODEL_CHECKPOINT.value)
        artifacts = list(Path(best_checkpoint).iterdir())
        assert len(artifacts) == 1, (
            "There should be only one artifact in the best checkpoint"
        )
        assert artifacts[0].is_file(), (
            "The artifact should be a file, but got a directory."
        )
        return ModelCheckpointArtifacts(model_path=str(artifacts[0]), model_config=None)

    def _load_dataset_artifacts(self, local_path: str) -> DatasetArtifacts:
        artifacts = os.path.join(
            local_path, Path(ArtifactPath.DATASET.value).with_suffix(".yaml")
        )
        artifacts = DatasetArtifacts.from_file(artifacts)
        return artifacts

    def list_artifacts(
        self,
        run_id: str,
        prefix: Optional[str] = None,
        status: Optional[ArtifactStatus] = None,
    ) -> List[ArtifactRecord]:
        return self.repo.list_by_run(run_id, prefix=prefix, status=status)
    
    def _delete_dataset_artifact(self, dataset_name: str, checks:bool=False) -> Optional[bool]:        
        # delete dataset artifact
        record = self.repo.get(dataset_name, ArtifactPath.DATASET.value)
        if record is None:
            return False            
        mlflow_run_id = "" + record.mlflow_run_id
        success = self.repo.delete(run_id=dataset_name, artifact_key=ArtifactPath.DATASET.value)
        if record.local_path:
            self.remove_local_artifact(record.local_path)
        # delete deepchecks artifact
        if checks:            
            record_checks = self.repo.get(mlflow_run_id, ArtifactPath.DEEPCHECKS.value)
            success = success or self.repo.delete(run_id=mlflow_run_id, artifact_key=ArtifactPath.DEEPCHECKS.value)
            if record_checks.local_path:
                self.remove_local_artifact(record_checks.local_path)
        # delete run
        self.mlflow.delete_run(mlflow_run_id)

        return success
    
    def remove_local_artifact(self, local_path:str) -> Optional[bool]:
        p = Path(local_path)
        if p.exists():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)
            return True
        else:
            return False
        
    def delete_artifact(
        self, run_id: str, artifact_key: Union[str, ArtifactPath]
    ) -> Optional[bool]:
        artifact_key = (
            ArtifactPath(artifact_key)
            if isinstance(artifact_key, str)
            else artifact_key
        )
        if artifact_key == ArtifactPath.DATASET:
            return self._delete_dataset_artifact(run_id, checks=True)
        rec = self.repo.get(run_id, artifact_key.value)
        if rec is None:
            LOGGER.warning(
                f"Artifact {artifact_key.value} not found for for run_id: {run_id}"
            )
            return None
        if rec.local_path:
            p = Path(rec.local_path)
            if p.exists():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
        return self.repo.delete(run_id, artifact_key.value)
