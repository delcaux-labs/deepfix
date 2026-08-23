from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class FixJobStatus(StrEnum):
    """Lifecycle status of an autonomous fix job."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FixJob(BaseModel):
    """Tracks the overall status of an autonomous fix run.

    Attributes:
        job_id: Unique identifier for the fix job.
        status: Current status of the job.
        started_at: Timestamp when the job was started.
        baseline_metrics: Baseline metrics before the fix was applied.
    """

    job_id: str = Field(
        default=..., description="Unique identifier for the fix job"
    )
    status: FixJobStatus = Field(
        default=FixJobStatus.PENDING, description="Current status of the job"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the job was started",
    )
    baseline_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Baseline metrics before the fix was applied",
    )


class FinalFixReport(BaseModel):
    """Payload submitted by OpenHands to the webhook upon completion.

    Attributes:
        success: Whether the fix session was ultimately successful.
        final_metrics: Metrics achieved by the final iteration.
        applied_fixes: List of fixes that were applied.
        run_id: Final MLflow run ID containing the fixed model and metrics.
    """

    success: bool = Field(
        default=..., description="Whether the fix session was successful"
    )
    final_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics achieved by the final iteration",
    )
    applied_fixes: List[str] = Field(
        default_factory=list, description="List of fixes that were applied"
    )
    run_id: str = Field(
        default=..., description="Final MLflow run ID containing the fixed model"
    )
