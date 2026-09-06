from typing import Any, Dict, List, Optional

from deepfix_core.models import (
    AgentResult,
    Analysis,
    Artifacts,
    DatasetArtifacts,
    DeepchecksArtifacts,
    ModelCheckpointArtifacts,
    TrainingArtifacts,
)
from pydantic import BaseModel, Field


class ArtifactAnalysisResult(BaseModel):
    """Result from artifact analysis.

    Attributes:
        summary: Optional overall summary of the analysis.
        analysis: List of analysis results.
    """

    summary: Optional[str] = Field(default=None, description="Summary of the analysis")
    analysis: List[Analysis] = Field(
        default=[], description="List of Analysis elements"
    )

class ReasoningWorkflowInput(BaseModel):
    """Input for the reasoning workflow"""
    previous_analyses: Dict[str, AgentResult] = Field(default_factory=dict)
    output_language: str = Field("english", description="Output language for the analysis")


class CrossArtifactReasoningInput(BaseModel):
    """Structured output for the cross-artifact reasoning agent."""

    artifact_analysis_results: List[AgentResult] = Field(
        description="List of previous results from agents that analyzed artifacts",
        default_factory=[],
    )
    retrieved_knowledge: Optional[List[str]] = Field(
        default=None, description="External knowledge relevant to the analysis"
    )
    output_language: str = Field(default="english", description="Language of the analysis")


class CrossArtifactReasoningResult(BaseModel):
    """Structured output for the cross-artifact reasoning agent."""

    analysis: List[Analysis] = Field(
        description="Consolidated analysis with cross-artifact insights and recommendations",
        default_factory=list,
    )
    summary: str = Field(
        description="Summary of the cross-artifact reasoning and analysis",
        default="",
    )


class SynthesisJudgeInput(BaseModel):
    """Structured input for the synthesis judge agent."""
    runs: List[CrossArtifactReasoningResult] = Field(
        description="Runs from cross-artifact reasoning agent",
        default_factory=list,
    )
    output_language: str = Field(default="english", description="Language of the analysis")
