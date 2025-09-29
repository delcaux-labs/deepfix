from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Optional
from datetime import datetime
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine, select, delete
from sqlalchemy.exc import IntegrityError

from .datamodel import ArtifactRecord, ArtifactStatus


class ArtifactRepository:
    def __init__(self, sqlite_path: str):
        self.sqlite_path = sqlite_path
        self.engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        SQLModel.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterable[Session]:
        with Session(self.engine) as s:
            yield s

    def upsert(self, record: ArtifactRecord) -> ArtifactRecord:
        with self.session() as s:
            existing = s.exec(
                select(ArtifactRecord).where(
                    ArtifactRecord.run_id == record.run_id,
                    ArtifactRecord.artifact_key == record.artifact_key,
                )
            ).one_or_none()
            if existing is None:
                s.add(record)
                s.commit()
                s.refresh(record)
                return record
            # update fields
            for field in (
                "mlflow_run_id",
                "source_uri",
                "local_path",
                "size_bytes",
                "checksum_sha256",
                "status",
                "metadata_json",
                "tags_json",
                "downloaded_at",
                "last_accessed_at",
            ):
                setattr(existing, field, getattr(record, field))
            existing.updated_at = datetime.now()
            s.add(existing)
            s.commit()
            s.refresh(existing)
            return existing

    def get(self, run_id: str, artifact_key: str) -> Optional[ArtifactRecord]:
        with self.session() as s:
            return s.exec(
                select(ArtifactRecord).where(
                    ArtifactRecord.run_id == run_id,
                    ArtifactRecord.artifact_key == artifact_key,
                )
            ).one_or_none()

    def delete(self, run_id: str, artifact_key: str) -> bool:
        assert isinstance(run_id, str) and isinstance(artifact_key, str), f"run_id and artifact_key must be strings. Received {type(run_id)} and {type(artifact_key)}"
        with self.session() as s:
            rec = s.exec(
                select(ArtifactRecord).where(
                    ArtifactRecord.run_id == run_id,
                    ArtifactRecord.artifact_key == artifact_key,
                )
            ).one_or_none()
            if rec is None:
                return False
            s.delete(rec)
            s.commit()
            return True

    def list_by_run(
        self,
        run_id: str,
        prefix: Optional[str] = None,
        status: Optional[ArtifactStatus] = None,
    ) -> List[ArtifactRecord]:
        with self.session() as s:
            stmt = select(ArtifactRecord).where(ArtifactRecord.run_id == run_id)
            if prefix:
                stmt = stmt.where(ArtifactRecord.artifact_key.like(f"{prefix}%"))
            if status:
                stmt = stmt.where(ArtifactRecord.status == status)
            return list(s.exec(stmt).all())

    def update_local_path(
        self,
        run_id: str,
        artifact_key: str,
        local_path: Optional[str],
        status: ArtifactStatus,
    ) -> Optional[ArtifactRecord]:
        with self.session() as s:
            rec = s.exec(
                select(ArtifactRecord).where(
                    ArtifactRecord.run_id == run_id,
                    ArtifactRecord.artifact_key == artifact_key,
                )
            ).one_or_none()
            if rec is None:
                return None
            rec.local_path = local_path
            rec.status = status
            if local_path:
                rec.downloaded_at = datetime.now()
            rec.updated_at = datetime.now()
            s.add(rec)
            s.commit()
            s.refresh(rec)
            return rec

    def touch_access(self, run_id: str, artifact_key: str) -> None:
        with self.session() as s:
            rec = s.exec(
                select(ArtifactRecord).where(
                    ArtifactRecord.run_id == run_id,
                    ArtifactRecord.artifact_key == artifact_key,
                )
            ).one_or_none()
            if rec is None:
                return
            rec.last_accessed_at = datetime.now()
            rec.updated_at = datetime.now()
            s.add(rec)
            s.commit()
