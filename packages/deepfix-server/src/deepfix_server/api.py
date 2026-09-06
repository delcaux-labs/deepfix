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
    AgentContext
)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .agents.workflow import AnalysisWorkflow
from .config import LLMConfig, settings
from .database import Base, get_db, get_engine, init_database
from .engine import DiagnosticSystem
from .logging import get_logger, setup_mlflow_tracing
from .models import AnalysisJob


LOGGER = get_logger(__name__)


app = FastAPI(
    title="DeepFix Analysis API",
    description="API for analyzing ML artifacts and returning diagnostic results.",
    version="0.1.0",
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