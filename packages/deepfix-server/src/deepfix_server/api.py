import os
import traceback
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
import json
import asyncio
from deepfix_core.models import (
    APIRequest,
    APIResponse,
    DatasetArtifacts,
    AnalysisJobStatus,
    APIJobResponse,
)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from .config import settings, LLMConfig
from .coordinators import ArtifactAnalysisCoordinator
from .database import get_db, get_engine, init_database, Base
from .db_models import AnalysisJob
from .logging import get_logger, setup_dspy_logging
from .models import AgentContext

LOGGER = get_logger(__name__)


def setup_llm_logging():
    """Setup logging for LLM traces."""
    if os.getenv("MLFLOW_EXP_NAME") and os.getenv("MLFLOW_TRACKING_URI"):
        setup_dspy_logging(
            experiment_name=os.getenv("MLFLOW_EXP_NAME"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )
    else:
        LOGGER.warning(
            "No MLflow tracking configured, LLMs traces will not be sent to MLflow."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the FastAPI application."""
    # Startup logic
    setup_llm_logging()

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
    coordinator = get_coordinator()

    # Update job status to PROCESSING
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        LOGGER.error(f"Job {job_id} not found in background task.")
        return

    job.status = AnalysisJobStatus.PROCESSING
    db.commit()

    try:
        request_ctx = await decode_agent_context(request)
        results = await coordinator.arun(request_ctx)

        response = APIResponse(
            agent_results=results.get_agent_results(),
            summary=results.summary,
            additional_outputs=results.additional_outputs,
            error_messages=results.get_error_messages(),
            dataset_name=request_ctx.dataset_name,
        )

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
    coordinator: ArtifactAnalysisCoordinator = Depends(get_coordinator),
):
    """Run artifact analysis synchronously and return results."""

    request_ctx = await decode_agent_context(request)

    job_id = (
        f"sync_{request_ctx.dataset_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    try:
        results = await coordinator.arun(request_ctx)
        response = APIResponse(
            agent_results=results.get_agent_results(),
            summary=results.summary,
            additional_outputs=results.additional_outputs,
            error_messages=results.get_error_messages(),
            dataset_name=request_ctx.dataset_name,
        )
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
