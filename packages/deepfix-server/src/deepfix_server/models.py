import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from deepfix_core.models import (
    AgentContext,
    AgentResult,
    AnalysisJobStatus,
    APIResponse,
)
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Enum, String, Text

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


class Result(BaseModel):
    """Result from artifact analysis.

    Attributes:
        context: Agent context containing all artifacts and results.
        summary: Optional overall summary of the analysis.
        analysis: List of analysis results.
        additional_outputs: Dictionary of additional outputs from agents.
    """

    context: AgentContext = Field(default=..., description="Context of the analysis")
    summary: Optional[str] = Field(default=None, description="Summary of the analysis")
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

    def to_api_response(
        self,
    ):
        return APIResponse(
            agent_results=self.get_agent_results(),
            summary=self.summary,
            additional_outputs=self.additional_outputs,
            error_messages=self.get_error_messages(),
            dataset_name=self.context.dataset_name,
        )
