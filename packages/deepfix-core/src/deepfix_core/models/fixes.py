from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FixJobStatus(StrEnum):
    """Lifecycle status of an autonomous fix job."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FinalFixReport(BaseModel):
    """Payload submitted by OpenHands to the webhook upon completion.

    Attributes:
        success: Whether the fix session was ultimately successful.
        final_metrics: Metrics achieved by the final iteration.
        applied_fixes: List of fixes that were applied.
        run_id: Final MLflow run ID containing the fixed model and metrics.
        s3_weights_uri: S3 URI where the fixed model weights are saved.
        summary: Optional summary notes of the fix session.
    """

    success: bool = Field(
        default=True, description="Whether the fix session was successful"
    )
    final_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics achieved by the final iteration",
    )
    applied_fixes: List[str] = Field(
        default_factory=list, description="List of fixes that were applied"
    )
    run_id: Optional[str] = Field(
        default=None, description="Final MLflow run ID containing the fixed model"
    )
    s3_weights_uri: Optional[str] = Field(
        default=None,
        description="S3 URI where the fixed model weights are saved",
    )
    summary: Optional[str] = Field(
        default=None, description="Summary description of the fixes"
    )


class FixJob(BaseModel):
    """Tracks the overall status of an autonomous fix run.

    Attributes:
        job_id: Unique identifier for the fix job.
        status: Current status of the job.
        dataset_name: Name of the dataset being fixed.
        model_name: Optional name of the model being fixed.
        target_metric: Metric key targeted for optimization.
        target_value: Target metric threshold.
        max_iterations: Maximum refinement iterations.
        s3_bucket: Target S3 bucket for saving model weights.
        iteration: Current iteration count.
        started_at: Timestamp when the job was started.
        updated_at: Timestamp when the job was last updated.
        baseline_metrics: Baseline metrics before the fix was applied.
        result: Optional final fix report if completed.
        error: Optional error message if failed.
    """

    job_id: str = Field(
        description="Unique identifier for the fix job"
    )
    status: FixJobStatus = Field(
        default=FixJobStatus.PENDING, description="Current status of the job"
    )
    dataset_name: Optional[str] = Field(
        default=None, description="Name of the dataset"
    )
    model_name: Optional[str] = Field(
        default=None, description="Name of the model"
    )
    target_metric: Optional[str] = Field(
        default="accuracy", description="Target metric key to optimize"
    )
    target_value: Optional[float] = Field(
        default=0.90, description="Target metric threshold value"
    )
    max_iterations: Optional[int] = Field(
        default=5, description="Maximum iterations"
    )
    s3_bucket: Optional[str] = Field(
        default=None, description="Target S3 bucket"
    )
    dataset_uri: Optional[str] = Field(
        default=None, description="URI of dataset in S3 or local"
    )
    model_uri: Optional[str] = Field(
        default=None, description="URI of baseline model in S3 or local"
    )
    iteration: int = Field(
        default=0, description="Current iteration count"
    )
    started_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the job was started",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the job was last updated",
    )
    baseline_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Baseline metrics before the fix was applied",
    )
    phase: Optional[str] = Field(
        default=None,
        description="Current operational phase of the fix agent",
    )
    events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Real-time activity log events and phase transitions",
    )
    intermediate_metrics: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Intermediate evaluation metrics per iteration",
    )
    result: Optional[FinalFixReport] = Field(
        default=None, description="Final fix report payload"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if job failed"
    )


class FixJobRequest(BaseModel):
    """Request model for submitting an autonomous fix job.

    Attributes:
        dataset_name: Name of the dataset registered in MLflow / DeepFix.
        model_name: Name or URI of the baseline model artifact.
        target_metric: Target metric key to optimize.
        target_value: Threshold value to consider the fix successful.
        max_iterations: Maximum autonomous refinement loops.
        s3_bucket: Target S3 bucket for model weights.
        baseline_run_id: Optional MLflow run ID of baseline model.
        model_class: Model class name or path.
        dataset_load_code: Python code to load dataset.
        experiment_name: MLflow experiment name.
        mlflow_experiment_id: MLflow experiment ID.
        hf_dataset_dir: Directory path for Hugging Face dataset.
        hf_dataset_name: Name of Hugging Face dataset.
        dataset_digest: Digest of dataset.
        dataset_uri: URI of dataset.
        model_uri: URI of baseline model.
        label_column: Name of label column.
    """

    dataset_name: str = Field(
        description="Name of the dataset registered in MLflow / DeepFix"
    )
    model_name: Optional[str] = Field(
        default=None, description="Name or URI of baseline model"
    )
    target_metric: Optional[str] = Field(
        default="accuracy", description="Target metric key to optimize"
    )
    target_value: Optional[float] = Field(
        default=0.90, description="Target metric threshold value"
    )
    max_iterations: Optional[int] = Field(
        default=5, description="Maximum autonomous refinement loops"
    )
    s3_bucket: Optional[str] = Field(
        default=None, description="Target S3 bucket for model weights"
    )
    baseline_run_id: Optional[str] = Field(
        default=None, description="MLflow run ID of baseline model"
    )
    model_class: Optional[str] = Field(
        default=None, description="Model class name or path"
    )
    dataset_load_code: Optional[str] = Field(
        default=None, description="Python code to load the dataset"
    )
    experiment_name: Optional[str] = Field(
        default="deepfix-autonomous", description="MLflow experiment name"
    )
    mlflow_experiment_id: Optional[str] = Field(
        default="0", description="MLflow experiment ID"
    )
    hf_dataset_dir: Optional[str] = Field(
        default=None, description="Directory path for Hugging Face dataset"
    )
    hf_dataset_name: Optional[str] = Field(
        default=None, description="Name of Hugging Face dataset"
    )
    dataset_digest: Optional[str] = Field(
        default=None, description="Digest of dataset"
    )
    dataset_uri: Optional[str] = Field(
        default=None, description="URI of dataset"
    )
    model_uri: Optional[str] = Field(
        default=None, description="URI of baseline model"
    )
    label_column: Optional[str] = Field(
        default=None, description="Name of label column"
    )
    dataset_artifacts: Optional[Dict[str, Any]] = Field(
        default=None, description="Dataset artifacts / summary statistics"
    )
    training_artifacts: Optional[Any] = Field(
        default=None, description="Training artifacts"
    )
    deepchecks_artifacts: Optional[Any] = Field(
        default=None, description="Deepchecks artifacts"
    )
    model_checkpoint_artifacts: Optional[Any] = Field(
        default=None, description="Model checkpoint artifacts"
    )
    diagnosis: Optional[str] = Field(
        default=None, description="Pre-computed diagnostic findings"
    )
    language: str = Field(
        default="english", description="Language of the analysis and prompt"
    )

