"""Pydantic AI result models replacing DSPy Signatures.

These Pydantic BaseModel classes are used as ``result_type`` on Pydantic AI
Agent instances, replacing dspy.Signature OutputFields.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from deepfix_core.models import Analysis


class ArtifactAnalysisResult(BaseModel):
    """Structured output for a single artifact analyzer agent.

    Replaces ``ArtifactAnalysisSignature`` from dspy signatures.
    """

    analysis: List[Analysis] = Field(
        description="Findings and recommendations based on the analyzed artifacts",
        default_factory=list,
    )


class CrossArtifactReasoningResult(BaseModel):
    """Structured output for the cross-artifact reasoning agent.

    Replaces ``CrossArtifactReasoningSignature`` from dspy signatures.
    """

    analysis: List[Analysis] = Field(
        description="Consolidated analysis with cross-artifact insights and recommendations",
        default_factory=list,
    )
    summary: str = Field(
        description="Summary of the cross-artifact reasoning and analysis",
        default="",
    )