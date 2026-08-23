import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import uvicorn
from deepfix_core.models import (
    AnalysisJobStatus,
    APIJobResponse,
    APIRequest,
    APIResponse,
    DatasetArtifacts,
    FinalFixReport,
    FixJob,
    FixJobRequest,
    FixJobStatus,
)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import settings
from .database import Base, get_db, get_engine, init_database
from .engine import DiagnosticSystem
from .logging import get_logger, setup_mlflow_tracing
from .models import AgentContext, AnalysisJob, FixJobRecord
from .openhands_executor import OpenHandsExecutor

LOGGER = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the FastAPI application."""
    # Startup logic
    if settings.mlflow_exp_name and settings.mlflow_tracking_uri:
        setup_mlflow_tracing(
            experiment_name=settings.mlflow_exp_name,
            tracking_uri=settings.mlflow_tracking_uri,
        )
    else:
        LOGGER.warning(
            "No MLflow tracking configured, LLMs traces will not be sent to MLflow."
        )

    # Initialize database
    init_database(settings.database_url, settings.database_echo)

    # Create tables if they don't exist
    engine = get_engine()
    if engine:
        Base.metadata.create_all(bind=engine)
        LOGGER.info("Database tables initialized.")

    # Start periodic cleanup task (every hour)
    cleanup_task = asyncio.create_task(run_periodic_cleanup())

    yield

    # Shutdown logic
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        LOGGER.info("Periodic cleanup task stopped.")


app = FastAPI(
    title="DeepFix Analysis API",
    description="API for analyzing ML artifacts and returning diagnostic results.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
    }


def get_coordinator() -> DiagnosticSystem:
    """Dependency that provides a DiagnosticSystem instance."""
    llm_config = settings.get_llm_config()
    return DiagnosticSystem(config=llm_config)


async def decode_agent_context(request: APIRequest) -> AgentContext:
    """Helper to convert APIRequest to AgentContext."""
    try:
        dataset_artifacts = request.dataset_artifacts
        if isinstance(dataset_artifacts, dict):
            dataset_artifacts = DatasetArtifacts.from_dict(dataset_artifacts)
        elif dataset_artifacts is not None and not isinstance(
            dataset_artifacts, DatasetArtifacts
        ):
            raise ValueError("Dataset artifacts must be a DatasetArtifacts object")

        return AgentContext(
            dataset_artifacts=dataset_artifacts,
            training_artifacts=request.training_artifacts,
            deepchecks_artifacts=request.deepchecks_artifacts,
            model_checkpoint_artifacts=request.model_checkpoint_artifacts,
            dataset_name=request.dataset_name,
            language=request.language,
        )
    except Exception as exc:
        LOGGER.error(f"Error decoding request: {exc}")
        raise HTTPException(
            status_code=400,
            detail=f"Error decoding request: {str(exc)}",
        ) from exc


async def run_diagnosis(request: APIRequest) -> APIResponse:
    coordinator = get_coordinator()
    request_ctx = await decode_agent_context(request)
    results = await coordinator.arun(request_ctx)

    response = APIResponse(
        agent_results=results.get_agent_results(),
        summary=results.summary,
        additional_outputs=results.additional_outputs,
        error_messages=results.get_error_messages(),
        dataset_name=request_ctx.dataset_name,
    )
    return response


def cleanup_old_jobs(db: Session):
    """Delete jobs older than the configured TTL."""
    ttl_hours = settings.job_ttl_hours
    cutoff_date = datetime.utcnow() - timedelta(hours=ttl_hours)

    try:
        deleted_analysis = (
            db.query(AnalysisJob).filter(AnalysisJob.created_at < cutoff_date).delete()
        )
        deleted_fix = (
            db.query(FixJobRecord).filter(FixJobRecord.created_at < cutoff_date).delete()
        )
        total_deleted = deleted_analysis + deleted_fix
        if total_deleted > 0:
            db.commit()
            LOGGER.info(
                f"Cleaned up {total_deleted} old jobs (older than {ttl_hours} hours)."
            )
    except Exception as exc:
        db.rollback()
        LOGGER.error(f"Error during job cleanup: {exc}")


async def run_periodic_cleanup():
    """Run job cleanup periodically every hour."""
    from .database import get_session

    while True:
        try:
            with get_session() as db:
                cleanup_old_jobs(db)
        except Exception as e:
            LOGGER.error(f"Periodic cleanup iteration failed: {e}")

        # Wait for 1 hour before next cleanup
        await asyncio.sleep(3600)


async def process_analysis_job(job_id: str, request: APIRequest, db: Session):
    """Background task to process an analysis job."""

    # Update job status to PROCESSING
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        LOGGER.error(f"Job {job_id} not found in background task.")
        return

    job.status = AnalysisJobStatus.PROCESSING
    db.commit()

    try:
        response = await run_diagnosis(request)

        job.result_data = response.model_dump_json()
        job.status = AnalysisJobStatus.COMPLETED
    except Exception as exc:
        LOGGER.error(f"Analysis failed for job {job_id}: {traceback.format_exc()}")
        job.error = str(exc)
        job.status = AnalysisJobStatus.FAILED
    finally:
        db.commit()


@app.post("/v1/analyse", response_model=APIJobResponse)
async def analyse_artifacts(
    request: APIRequest,
):
    """Run artifact analysis synchronously and return results."""

    job_id = (
        f"sync_{request.dataset_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    try:
        response = await run_diagnosis(request)
        response.job_id = job_id
        now = datetime.now().isoformat()
        return APIJobResponse(
            job_id=job_id,
            status=AnalysisJobStatus.COMPLETED,
            result=response,
            created_at=now,
            updated_at=now,
        )
    except Exception as exc:
        LOGGER.error(f"Analysis failed for job {job_id}: {traceback.format_exc()}")
        now = datetime.now().isoformat()
        return APIJobResponse(
            job_id=job_id,
            status=AnalysisJobStatus.FAILED,
            error=str(exc),
            created_at=now,
            updated_at=now,
        )


@app.post("/v2/analyse", status_code=202, response_model=APIJobResponse)
async def analyse_artifacts_async(
    request: APIRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Submit artifact analysis job and return job_id immediately."""
    # Create new job entry
    job = AnalysisJob(
        request_data=request.model_dump_json(),
        status=AnalysisJobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Schedule background task
    background_tasks.add_task(process_analysis_job, job.id, request, db)

    return APIJobResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
        result=None,
        error=None,
    )


def add_fix_job_event(
    job: FixJobRecord, phase: str, message: str, db: Session
) -> None:
    """Helper to append timestamped activity events to a FixJobRecord."""
    events = []
    if job.events_data:
        try:
            events = json.loads(job.events_data)
        except Exception:
            events = []
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "phase": phase,
        "message": message,
    }
    events.append(event)
    job.events_data = json.dumps(events)
    job.phase = phase
    db.commit()


def _fix_job_from_record(job: FixJobRecord) -> FixJob:
    """Helper to convert a FixJobRecord database model to a FixJob Pydantic schema."""
    result = None
    if job.result_data:
        try:
            report_dict = json.loads(job.result_data)
            if "fix_report" in report_dict and report_dict["fix_report"]:
                result = FinalFixReport.model_validate(report_dict["fix_report"])
            else:
                result = FinalFixReport.model_validate(report_dict)
        except Exception as exc:
            LOGGER.error(f"Error decoding fix report for job {job.id}: {exc}")

    events = []
    if job.events_data:
        try:
            events = json.loads(job.events_data)
        except Exception:
            events = []

    intermediate_metrics = []
    if job.intermediate_metrics_data:
        try:
            intermediate_metrics = json.loads(job.intermediate_metrics_data)
        except Exception:
            intermediate_metrics = []

    return FixJob(
        job_id=job.id,
        status=job.status,
        dataset_name=job.dataset_name,
        model_name=job.model_name,
        target_metric=job.target_metric,
        target_value=job.target_value,
        max_iterations=job.max_iterations,
        s3_bucket=job.s3_bucket,
        dataset_uri=job.dataset_uri,
        model_uri=job.model_uri,
        iteration=job.iteration,
        phase=job.phase or "Pending",
        events=events,
        intermediate_metrics=intermediate_metrics,
        started_at=job.created_at,
        updated_at=job.updated_at,
        result=result,
        error=job.error,
    )


async def process_fix_job(job_id: str, request: FixJobRequest, db: Session):
    """Background task to process an autonomous diagnostic analysis and fix job."""

    job = db.query(FixJobRecord).filter(FixJobRecord.id == job_id).first()
    if not job:
        LOGGER.error(f"Fix job {job_id} not found in background task.")
        return

    job.status = FixJobStatus.IN_PROGRESS
    job.iteration = max(job.iteration, 1)
    job.phase = "Initializing"
    add_fix_job_event(job, "Initializing", f"Autonomous fix job {job_id} initialized.", db)
    db.commit()

    current_task = asyncio.current_task()
    if current_task is not None:
        OpenHandsExecutor.register_task(job_id, current_task)

    try:
        diagnosis_text = getattr(request, "diagnosis", None) or ""
        if not diagnosis_text and hasattr(request, "dataset_artifacts") and getattr(request, "dataset_artifacts"):
            try:
                add_fix_job_event(job, "Diagnosing", "Analyzing dataset integrity and pre-computing diagnostic findings...", db)
                # Build APIRequest for pre-diagnosis
                diag_req = APIRequest(
                    dataset_artifacts=request.dataset_artifacts,
                    training_artifacts=getattr(request, "training_artifacts", None),
                    deepchecks_artifacts=getattr(request, "deepchecks_artifacts", None),
                    model_checkpoint_artifacts=getattr(request, "model_checkpoint_artifacts", None),
                    dataset_name=request.dataset_name,
                    model_name=request.model_name,
                    language=getattr(request, "language", "english"),
                )
                response = await run_diagnosis(diag_req)
                response.job_id = job_id
                diagnosis_text = response.get_results_as_text()
                add_fix_job_event(job, "Diagnosing", "Diagnostic analysis complete. Identified structural issues.", db)
            except Exception as d_exc:
                LOGGER.warning(
                    f"Could not run pre-diagnosis on request for {job_id}: {d_exc}"
                )

        # Instantiate OpenHandsExecutor
        fix_config = settings.get_autonomous_fix_config()
        executor = OpenHandsExecutor(config=fix_config)

        add_fix_job_event(job, "Synthesizing Fix", "Launching OpenHands agent in sandbox Docker workspace...", db)

        # Launch agent (asynchronously)
        await executor.launch_autonomous_fix(
            job_id=job_id,
            diagnosis=diagnosis_text,
            mlflow_experiment_id=request.mlflow_experiment_id or "0",
            s3_bucket=request.s3_bucket,
            mlflow_tracking_uri=settings.mlflow_tracking_uri,
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            target_metric=request.target_metric,
            target_value=request.target_value,
            max_iterations=request.max_iterations,
            dataset_uri=request.dataset_uri,
            model_uri=request.model_uri,
            is_dataset_only=(request.model_name is None),
        )

    except asyncio.CancelledError:
        LOGGER.info(f"Fix job {job_id} background task was cancelled.")
        job.status = FixJobStatus.CANCELLED
        job.phase = "Cancelled"
        job.error = "Job cancelled by user request"
        add_fix_job_event(job, "Cancelled", "Fix job execution was halted and cancelled.", db)
        db.commit()
    except Exception as exc:
        LOGGER.error(
            f"Fix job failed to start for job {job_id}: {traceback.format_exc()}"
        )
        job.error = str(exc)
        job.status = FixJobStatus.FAILED
        job.phase = "Failed"
        add_fix_job_event(job, "Failed", f"Fix execution failed with error: {exc}", db)
        db.commit()
    finally:
        OpenHandsExecutor.unregister_task(job_id)


@app.post("/v2/fix", status_code=202, response_model=FixJob)
async def fix(
    request: FixJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Submit autonomous diagnostic analysis and fix job and return unique job_id."""
    # Create new FixJobRecord
    initial_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "phase": "Pending",
        "message": f"Fix job registered for dataset '{request.dataset_name}'.",
    }
    job = FixJobRecord(
        dataset_name=request.dataset_name,
        model_name=request.model_name,
        target_metric=request.target_metric,
        target_value=request.target_value,
        max_iterations=request.max_iterations,
        s3_bucket=request.s3_bucket,
        dataset_uri=request.dataset_uri,
        model_uri=request.model_uri,
        status=FixJobStatus.PENDING,
        phase="Pending",
        iteration=0,
        events_data=json.dumps([initial_event]),
        intermediate_metrics_data=json.dumps([]),
        request_data=request.model_dump_json(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Schedule background task
    background_tasks.add_task(process_fix_job, job.id, request, db)

    return _fix_job_from_record(job)


@app.post("/v2/fix/{job_id}/cancel", response_model=FixJob)
async def cancel_fix_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel an active or pending autonomous fix job."""
    job = db.query(FixJobRecord).filter(FixJobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Fix job '{job_id}' not found")

    if job.status not in (FixJobStatus.COMPLETED, FixJobStatus.FAILED, FixJobStatus.CANCELLED):
        OpenHandsExecutor.cancel_task(job_id)
        job.status = FixJobStatus.CANCELLED
        job.phase = "Cancelled"
        job.error = "Job cancelled by user request"
        add_fix_job_event(job, "Cancelled", "Fix job cancelled by user request.", db)
        db.commit()
        db.refresh(job)

    return _fix_job_from_record(job)


@app.get("/v2/fix/{job_id}", response_model=FixJob)
async def get_fix_job_status(job_id: str, db: Session = Depends(get_db)):
    """Retrieve the status, iteration count, and result of a fix job."""
    job = db.query(FixJobRecord).filter(FixJobRecord.id == job_id).first()
    if not job:
        # Fallback check in AnalysisJob for backward compatibility
        analysis_job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not analysis_job:
            raise HTTPException(
                status_code=404, detail=f"Fix job '{job_id}' not found"
            )

        status_mapping = {
            AnalysisJobStatus.PENDING: FixJobStatus.PENDING,
            AnalysisJobStatus.PROCESSING: FixJobStatus.IN_PROGRESS,
            AnalysisJobStatus.COMPLETED: FixJobStatus.COMPLETED,
            AnalysisJobStatus.FAILED: FixJobStatus.FAILED,
            AnalysisJobStatus.CANCELLED: FixJobStatus.CANCELLED,
        }
        fix_status = status_mapping.get(analysis_job.status, FixJobStatus.PENDING)
        result = None
        if analysis_job.result_data:
            try:
                report_dict = json.loads(analysis_job.result_data)
                if "fix_report" in report_dict and report_dict["fix_report"]:
                    result = FinalFixReport.model_validate(report_dict["fix_report"])
            except Exception:
                LOGGER.warning(f"Could not parse analysis_job result_data for {job_id}")

        return FixJob(
            job_id=analysis_job.id,
            status=fix_status,
            phase="Completed" if fix_status == FixJobStatus.COMPLETED else "In Progress",
            started_at=analysis_job.created_at,
            updated_at=analysis_job.updated_at,
            result=result,
            error=analysis_job.error,
        )

    return _fix_job_from_record(job)


@app.get("/v2/jobs/{job_id}", response_model=APIJobResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Retrieve the status and results of a background analysis job."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = APIJobResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
        result=None,
        error=None,
    )

    if job.status == AnalysisJobStatus.COMPLETED:
        try:
            response.result = APIResponse.model_validate(json.loads(job.result_data))
        except Exception as exc:
            LOGGER.error(
                f"Error decoding job result for job {job_id}: {traceback.format_exc()}"
            )
            response.error = f"Error decoding job result: {str(exc)}"
            response.status = AnalysisJobStatus.FAILED

    elif job.status == AnalysisJobStatus.FAILED:
        response.error = job.error

    return response


class WebhookPayload(BaseModel):
    job_id: str
    success: bool = True
    status: Optional[str] = None
    final_metrics: Dict[str, Any] = {}
    applied_fixes: List[str] = []
    run_id: Optional[str] = None
    s3_weights_uri: Optional[str] = None
    summary: Optional[str] = None
    iteration: Optional[int] = None
    phase: Optional[str] = None
    intermediate_metrics: Optional[List[Dict[str, Any]]] = None


class FixJobStepPayload(BaseModel):
    phase: Optional[str] = None
    iteration: Optional[int] = None
    message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@app.post("/v2/fix/{job_id}/step", status_code=200)
async def report_fix_job_step(
    job_id: str,
    payload: FixJobStepPayload,
    db: Session = Depends(get_db),
):
    """Update ongoing fix job progress, phase, or intermediate metrics."""
    job = db.query(FixJobRecord).filter(FixJobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Fix job '{job_id}' not found")

    if payload.iteration is not None:
        job.iteration = payload.iteration
    if payload.phase is not None:
        job.phase = payload.phase
    if payload.message:
        add_fix_job_event(job, job.phase or "In Progress", payload.message, db)
    if payload.metrics:
        existing_metrics = []
        if job.intermediate_metrics_data:
            try:
                existing_metrics = json.loads(job.intermediate_metrics_data)
            except Exception:
                existing_metrics = []
        step_metric = {"iteration": job.iteration, **payload.metrics}
        existing_metrics.append(step_metric)
        job.intermediate_metrics_data = json.dumps(existing_metrics)
        db.commit()

    return {"status": "ok", "job_id": job_id, "phase": job.phase, "iteration": job.iteration}


@app.post("/webhook/completion", status_code=200)
async def webhook_completion(
    payload: WebhookPayload,
    db: Session = Depends(get_db),
):
    """Webhook for OpenHands agent to report completion."""
    fix_job = db.query(FixJobRecord).filter(FixJobRecord.id == payload.job_id).first()
    analysis_job = (
        db.query(AnalysisJob).filter(AnalysisJob.id == payload.job_id).first()
    )

    if not fix_job and not analysis_job:
        raise HTTPException(status_code=404, detail="Job not found")

    if payload.status is not None:
        is_success = payload.status.upper() == "COMPLETED"
    else:
        is_success = payload.success

    report = FinalFixReport(
        success=is_success,
        final_metrics=payload.final_metrics,
        applied_fixes=payload.applied_fixes,
        run_id=payload.run_id,
        s3_weights_uri=payload.s3_weights_uri,
        summary=payload.summary,
    )

    if fix_job:
        fix_job.status = (
            FixJobStatus.COMPLETED if is_success else FixJobStatus.FAILED
        )
        if payload.iteration is not None:
            fix_job.iteration = payload.iteration
        if payload.phase is not None:
            fix_job.phase = payload.phase
        else:
            fix_job.phase = "Completed" if is_success else "Failed"
        if payload.intermediate_metrics:
            fix_job.intermediate_metrics_data = json.dumps(payload.intermediate_metrics)
        elif payload.final_metrics:
            # If intermediate metrics weren't populated yet, add the final metrics as the last run
            existing = []
            if fix_job.intermediate_metrics_data:
                try:
                    existing = json.loads(fix_job.intermediate_metrics_data)
                except Exception:
                    existing = []
            if not existing:
                existing.append({"iteration": fix_job.iteration, **payload.final_metrics})
                fix_job.intermediate_metrics_data = json.dumps(existing)

        add_fix_job_event(
            fix_job,
            fix_job.phase,
            f"Fix job concluded with status {fix_job.status.value}.",
            db,
        )
        fix_job.result_data = report.model_dump_json()

    if analysis_job:
        try:
            if analysis_job.result_data:
                response = APIResponse.model_validate(
                    json.loads(analysis_job.result_data)
                )
            else:
                response = APIResponse()

            response.fix_report = report
            analysis_job.result_data = response.model_dump_json()
            analysis_job.status = (
                AnalysisJobStatus.COMPLETED
                if is_success
                else AnalysisJobStatus.FAILED
            )
        except Exception as exc:
            LOGGER.error(f"Error processing analysis job in webhook: {exc}")

    db.commit()
    return {
        "status": "ok",
        "job_id": payload.job_id,
        "job_status": fix_job.status.value if fix_job else "unknown",
    }


def run_analyse_artifacts_api(
    port: int = 4141,
    host: str = "0.0.0.0",
    workers: int = 1,
    reload: bool = False,
    reload_dirs: list[str] | None = None,
    reload_excludes: list[str] = ["server_data*", "*.venv*", ".git*"],
    **kwargs,
):
    """Run the artifact analysis API server using uvicorn.

    Args:
        port: Port number to listen on. Defaults to 4141.
        host: Host address to bind to. Defaults to "0.0.0.0".
        workers: Number of worker processes. Defaults to 1.
        reload: Enable auto-reload. Defaults to False.
        reload_dirs: List of directories to watch for reload.
        reload_excludes: List of glob patterns to exclude from reload watching.
    """

    uvicorn.run(
        "deepfix_server.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        reload_dirs=reload_dirs,
        reload_excludes=reload_excludes,
        log_level="info",
    )
