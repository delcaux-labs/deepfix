import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from deepfix_core.models import AgentResult, AnalysisJobStatus, FixJobStatus
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Enum, Float, Integer, String, Text

from .agents.schemas import AgentContext
from .database import Base


## Database
class AnalysisJob(Base):
    """Model to track background analysis jobs."""

    __tablename__ = "analysis_jobs"

    id = Column(
        String,
        primary_key=True,
        default=lambda: (
            f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        ),
    )
    status = Column(
        Enum(AnalysisJobStatus), nullable=False, default=AnalysisJobStatus.PENDING
    )
    request_data = Column(Text, nullable=True)
    result_data = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FixJobRecord(Base):
    """Model to track background autonomous fix jobs in SQLite."""

    __tablename__ = "fix_jobs"

    id = Column(
        String,
        primary_key=True,
        default=lambda: (
            f"fix_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        ),
    )
    dataset_name = Column(String, nullable=False)
    model_name = Column(String, nullable=True)
    target_metric = Column(String, nullable=True, default="accuracy")
    target_value = Column(Float, nullable=True, default=0.90)
    max_iterations = Column(Integer, nullable=True, default=5)
    s3_bucket = Column(String, nullable=True)
    dataset_uri = Column(String, nullable=True)
    model_uri = Column(String, nullable=True)
    status = Column(
        Enum(FixJobStatus), nullable=False, default=FixJobStatus.PENDING
    )
    iteration = Column(Integer, nullable=False, default=0)
    phase = Column(String, nullable=True, default="Pending")
    events_data = Column(Text, nullable=True)
    intermediate_metrics_data = Column(Text, nullable=True)
    request_data = Column(Text, nullable=True)
    result_data = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Result(BaseModel):
    """Result from artifact analysis.

    Attributes:
        context: Agent context containing all artifacts and results.
        summary: Optional overall summary of the analysis.
        analysis: List of analysis results.
        additional_outputs: Dictionary of additional outputs from agents.
    """

    context: AgentContext = Field(default=..., description="Context of the analysis")
    summary: Optional[str] = Field(default=..., description="Summary of the analysis")
    additional_outputs: Dict[str, Any] = Field(
        default={}, description="Additional outputs from the agent"
    )

    def get_agent_results(self) -> Dict[str, AgentResult]:
        """Get all agent results from the context.

        Returns:
            Dictionary mapping agent names to their results.
        """
        return self.context.agent_results

    def get_error_messages(self) -> Dict[str, str]:
        """Get error messages from all agents that failed.

        Returns:
            Dictionary mapping agent names to their error messages.
            Only includes agents that have error messages.
        """
        return {
            agent_name: agent_result.error_message
            for agent_name, agent_result in self.context.agent_results.items()
        }
