from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import StrEnum


from ..artifacts import Artifacts


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalyzerTypes(StrEnum):
    TRAINING = "training"
    DEEPCHECKS = "deepchecks"
    DATASET = "dataset"
    MODEL_CHECKPOINT = "model_checkpoint"


class Finding(BaseModel):
    description: str
    evidence: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)


class Recommendation(BaseModel):
    action: str
    rationale: str
    priority: Severity
    confidence: float = Field(ge=0.0, le=1.0)


class Analysis(BaseModel):
    findings: Finding
    recommendations: Recommendation


class AgentResult(BaseModel):
    agent_name: str
    analysis: Union[List[Analysis], str] = None
    analyzed_artifacts: Optional[List[str]] = None


class AgentContext(BaseModel):
    artifacts: List[Artifacts]

    run_id: Optional[str] = None
    # Agent coordination
    agent_results: Dict[str, AgentResult] = Field(default={})
    knowledge_cache: Dict[str, Any] = Field(default={})


class ArtifactAnalysisConfig(BaseModel):
    enabled_analyzers: List[AnalyzerTypes] = [
        AnalyzerTypes.TRAINING,
        AnalyzerTypes.DEEPCHECKS,
        AnalyzerTypes.DATASET,
        AnalyzerTypes.MODEL_CHECKPOINT,
    ]
