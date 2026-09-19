from datetime import datetime
from typing import List, Optional

import uvicorn
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from agno.tools import Toolkit
from fastapi import FastAPI

from .agents.workflow import AnalysisWorkflow
from .config import LLMConfig, settings
from .logging import get_logger

from starlette.formparsers import FormParser, MultiPartParser
from starlette.requests import Request

LOGGER = get_logger(__name__)

# Configure Starlette form parsers to support large artifact payloads (up to 100MB)
DEFAULT_MAX_PART_SIZE = 100 * 1024 * 1024
if hasattr(Request.form, "__kwdefaults__") and Request.form.__kwdefaults__:
    Request.form.__kwdefaults__["max_part_size"] = DEFAULT_MAX_PART_SIZE
if hasattr(Request._get_form, "__kwdefaults__") and Request._get_form.__kwdefaults__:
    Request._get_form.__kwdefaults__["max_part_size"] = DEFAULT_MAX_PART_SIZE
if hasattr(FormParser.__init__, "__kwdefaults__") and FormParser.__init__.__kwdefaults__:
    FormParser.__init__.__kwdefaults__["max_part_size"] = DEFAULT_MAX_PART_SIZE
if hasattr(MultiPartParser.__init__, "__kwdefaults__") and MultiPartParser.__init__.__kwdefaults__:
    MultiPartParser.__init__.__kwdefaults__["max_part_size"] = DEFAULT_MAX_PART_SIZE
MultiPartParser.max_part_size = DEFAULT_MAX_PART_SIZE


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
    tools: Optional[List[Toolkit]] = None,
    base_app: Optional[FastAPI] = None,
) -> AgentOS:
    """Create and configure an AgentOS instance with registered agents and workflows."""

    config = llm_config or settings.get_llm_config()
    db = SqliteDb(db_url=settings.database_url)

    analysis_workflow = AnalysisWorkflow(llm_config=config, tools=tools, db=db)

    workflows = [analysis_workflow]

    agent_os = AgentOS(
        base_app=base_app,
        on_route_conflict="preserve_base_app",
        db=db,
        workflows=workflows,
        telemetry=False,
        tracing=True,
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
