import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import uvicorn
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from deepfix_core.models import (
    AnalysisJobStatus,
    APIJobResponse,
    APIRequest,
    APIResponse,
    DatasetArtifacts,
)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .agents.workflow import AnalysisWorkflow
from .config import LLMConfig, settings
from .database import Base, get_db, get_engine, init_database
from .engine import DiagnosticSystem
from .logging import get_logger, setup_mlflow_tracing
from .models import AgentContext, AnalysisJob


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


def create_agent_os(
    llm_config: Optional[LLMConfig] = None,
    knowledge_bridge: Optional[Any] = None,
    base_app: Optional[FastAPI] = None,
) -> AgentOS:
    """Create and configure an AgentOS instance with registered agents and workflows."""
   
    config = llm_config or settings.get_llm_config()
    db = SqliteDb(db_url=settings.database_url)

    analysis_workflow = AnalysisWorkflow(
        llm_config=config, knowledge_bridge=knowledge_bridge
    )

    workflows = [analysis_workflow]

    agent_os = AgentOS(
        base_app=base_app,
        on_route_conflict="preserve_base_app",
        db=db,
        workflows=workflows,
        telemetry=False,
    )
    return agent_os


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
        total_deleted = deleted_analysis
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


def run_analyse_artifacts_api(
    port: int = 8844,
    host: str = "0.0.0.0",
    workers: int = 1,
    reload: bool = False,
    reload_dirs: list[str] | None = None,
    reload_excludes: list[str] = ["server_data*", "*.venv*", ".git*"],
    **kwargs,
):
    """Run the artifact analysis API server using uvicorn.

    Args:
        port: Port number to listen on. Defaults to 8844.
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

# Initialize and mount Agno AgentOS on the base FastAPI application
agent_os = create_agent_os(base_app=app)
app = agent_os.get_app()