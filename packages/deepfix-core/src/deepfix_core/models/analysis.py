from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Analysis data Models
# ============================================================================
class Severity(StrEnum):
    """Severity levels for findings.

    Used to categorize the importance and urgency of issues found during analysis.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalyzerTypes(StrEnum):
    """Types of artifact analyzers available in the system."""

    TRAINING = "training"
    DEEPCHECKS = "deepchecks"
    DATASET = "dataset"
    MODEL_CHECKPOINT = "model_checkpoint"


class Finding(BaseModel):
    """A finding discovered during analysis.

    Attributes:
        description: Short description of the finding.
        evidence: Evidence supporting the finding.
        severity: Severity level of the finding (low, medium, high).
        confidence: Confidence score between 0.0 and 1.0 in the finding and severity.
    """

    description: str = Field(
        default=..., description="Short Description of the finding"
    )
    evidence: str = Field(default=..., description="Evidence of the finding")
    severity: Severity = Field(default=..., description="Severity of the finding")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the finding and severity"
    )


class Recommendation(BaseModel):
    """A recommendation to address a finding.

    Attributes:
        action: Action to take to address the finding.
        rationale: Rationale explaining why this recommendation is suggested.
        confidence: Confidence score between 0.0 and 1.0 in the recommendation.
    """

    action: str = Field(
        default=..., description="Action to take to address the finding"
    )
    rationale: str = Field(default=..., description="Rationale for the recommendation")
    # priority: Severity = Field(default=...,description="Priority of the recommendation")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the recommendation"
    )


class Analysis(BaseModel):
    """A complete analysis combining a finding with its recommendation.

    Attributes:
        findings: The finding discovered during analysis.
        recommendations: The recommendation to address the finding.
    """

    findings: Finding = Field(default=..., description="Finding of the analysis")
    recommendations: Recommendation = Field(
        default=..., description="Recommendation based on the findings"
    )

    @field_validator("findings", mode="before")
    @classmethod
    def _coerce_finding(cls, v: Any) -> Any:
        if isinstance(v, list):
            if len(v) > 0:
                return v[0]
            raise ValueError("findings list cannot be empty")
        return v

    @field_validator("recommendations", mode="before")
    @classmethod
    def _coerce_recommendation(cls, v: Any) -> Any:
        if isinstance(v, list):
            if len(v) > 0:
                return v[0]
            raise ValueError("recommendations list cannot be empty")
        return v


class AgentResult(BaseModel):
    """Result from an analysis agent.

    Attributes:
        agent_name: Name of the agent that performed the analysis.
        analysis: List of analysis results (findings and recommendations).
        analyzed_artifacts: Optional list of artifact types that were analyzed.
        retrieved_knowledge: Optional list of external knowledge sources used.
        additional_outputs: Dictionary of additional outputs from the agent.
        error_message: Optional error message if the agent failed.
    """

    agent_name: str
    analysis: List[Analysis] = Field(
        default=[], description="List of Analysis elements"
    )
    analyzed_artifacts: Optional[List[str]] = Field(
        default=None, description="List of artifacts analyzed by the agent"
    )
    retrieved_knowledge: Optional[List[str]] = Field(
        default=None, description="External knowledge relevant to the analysis"
    )
    additional_outputs: Dict[str, Any] = Field(
        default={}, description="Additional outputs from the agent"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if the agent failed"
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert agent results to a pandas DataFrame.

        Each row represents one analysis (finding + recommendation) with
        all associated metadata.

        Returns:
            DataFrame with columns: agent_name, analyzed_artifacts,
            retrieved_knowledge, summary, finding_description, finding_evidence,
            error_message, finding_severity, finding_confidence,
            recommendation_action, recommendation_rationale,
            recommendation_confidence.
        """
        rows = []
        for analysis in self.analysis:
            # Extract findings and recommendations from the Analysis object
            findings = analysis.findings
            recommendations = analysis.recommendations

            # Create a row combining findings and recommendations
            row = {
                "agent_name": self.agent_name,
                "analyzed_artifacts": ", ".join(self.analyzed_artifacts)
                if self.analyzed_artifacts
                else "",
                "retrieved_knowledge": ", ".join(self.retrieved_knowledge)
                if self.retrieved_knowledge
                else "",
                "summary": self.additional_outputs.get("summary", ""),
                "finding_description": findings.description,
                "finding_evidence": findings.evidence,
                "error_message": self.error_message,
                "finding_severity": findings.severity.value,
                "finding_confidence": findings.confidence,
                "recommendation_action": recommendations.action,
                "recommendation_rationale": recommendations.rationale,
                #'recommendation_priority': recommendations.priority.value,
                "recommendation_confidence": recommendations.confidence,
            }
            rows.append(row)

        return pd.DataFrame(rows)
