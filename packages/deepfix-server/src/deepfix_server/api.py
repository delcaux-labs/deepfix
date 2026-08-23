import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List

import uvicorn
from deepfix_core.models import (
    AnalysisJobStatus,
    APIJobResponse,
    APIRequest,
    APIResponse,
    AutonomousFixRequest,
    DatasetArtifacts,
)
from deepfix_core.models.fixes import FinalFixReport
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import settings
from .coordinators import ArtifactAnalysisCoordinator
from .database import Base, get_db, get_engine, init_database
from .logging import get_logger, setup_mlflow_tracing
from .models import AgentContext, AnalysisJob
from .openhands_executor import OpenHandsExecutor

LOGGER = get_logger(__name__)


def setup_llm_tracing():
    """Setup logging for LLM traces."""
    if settings.mlflow_exp_name and settings.mlflow_tracking_uri:
        setup_mlflow_tracing(
            experiment_name=settings.mlflow_exp_name,
            tracking_uri=settings.mlflow_tracking_uri,
        )
    else:
        LOGGER.warning(
            "No MLflow tracking configured, LLMs traces will not be sent to MLflow."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the FastAPI application."""
    # Startup logic
    setup_llm_tracing()

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


def get_coordinator() -> ArtifactAnalysisCoordinator:
    """Dependency that provides an ArtifactAnalysisCoordinator instance."""
    llm_config = settings.get_llm_config()
    return ArtifactAnalysisCoordinator(config=llm_config)

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
        deleted_count = (
            db.query(AnalysisJob).filter(AnalysisJob.created_at < cutoff_date).delete()
        )
        if deleted_count > 0:
            db.commit()
            LOGGER.info(
                f"Cleaned up {deleted_count} old analysis jobs (older than {ttl_hours} hours)."
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


async def process_fix_job(job_id: str, request: AutonomousFixRequest, db: Session):
    """Background task to process an autonomous diagnostic analysis and fix job."""

    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        LOGGER.error(f"Job {job_id} not found in background task.")
        return

    job.status = AnalysisJobStatus.PROCESSING
    db.commit()

    try:
        # a. Performs diagnostic analysis (existing coordinator flow)
        response = await run_diagnosis(request)
        response.job_id = job_id
        job.result_data = response.model_dump_json()
        db.commit()

        # b. Instantiate OpenHandsExecutor and prepare response
        fix_config = settings.get_autonomous_fix_config()
        executor = OpenHandsExecutor(config=fix_config)

        # Launch agent (asynchronously)
        await executor.launch_autonomous_fix(job_id=job_id, diagnosis=response.get_results_as_text())

    except Exception as exc:
        LOGGER.error(f"Fix job failed to start for job {job_id}: {traceback.format_exc()}")
        job.error = str(exc)
        job.status = AnalysisJobStatus.FAILED
        db.commit()


@app.post("/v2/fix", status_code=202, response_model=APIJobResponse)
async def fix(
    request: AutonomousFixRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Submit autonomous diagnostic analysis and fix job and return job_id immediately."""
    # Create new job entry
    job = AnalysisJob(
        request_data=request.model_dump_json(),
        status=AnalysisJobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Schedule background task
    background_tasks.add_task(process_fix_job, job.id, request, db)

    return APIJobResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
        result=None,
        error=None,
    )


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
    success: bool
    final_metrics: Dict[str, Any] = {}
    applied_fixes: List[str] = []
    run_id: str


@app.post("/webhook/completion", status_code=200)
async def webhook_completion(
    payload: WebhookPayload,
    db: Session = Depends(get_db),
):
    """Webhook for OpenHands agent to report completion."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == payload.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        if job.result_data:
            response = APIResponse.model_validate(json.loads(job.result_data))
        else:
            response = APIResponse()

        response.fix_report = FinalFixReport(
            success=payload.success,
            final_metrics=payload.final_metrics,
            applied_fixes=payload.applied_fixes,
            run_id=payload.run_id,
        )

        job.result_data = response.model_dump_json()
        job.status = AnalysisJobStatus.COMPLETED
        db.commit()
        return {"status": "ok"}
    except Exception as exc:
        LOGGER.error(f"Error processing webhook: {exc}")
        job.status = AnalysisJobStatus.FAILED
        job.error = str(exc)
        db.commit()
        raise HTTPException(status_code=500, detail="Internal server error")


def run_analyse_artifacts_api(
    port: int = 4141,
    host: str = "0.0.0.0",
    workers: int = 1,
    reload: bool = False,
    **kwargs,
):
    """Run the artifact analysis API server using uvicorn.

    Args:
        port: Port number to listen on. Defaults to 4141.
        host: Host address to bind to. Defaults to "0.0.0.0".
        workers: Number of worker processes. Defaults to 1.
        reload: Enable auto-reload. Defaults to False.
    """
    uvicorn.run(
        "deepfix_server.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )
