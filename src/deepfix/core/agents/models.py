from pydantic import BaseModel,Field
from typing import List, Dict, Any, Optional
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

class AgentResult(BaseModel):
    agent_name: str    
    # Core outputs
    findings: List[Finding]
    recommendations: List[Recommendation]     
    # Metadata
    execution_time: float
    artifacts_analyzed: List[str]
    knowledge_refs: List[str] = []

class AgentContext(BaseModel):
    run_id: str
    
    # Typed artifact accessors
    artifacts: List[Artifacts]    
    # Agent coordination
    agent_results: Dict[str, AgentResult] = Field(default={})
    knowledge_cache: Dict[str, Any] = Field(default={})

class ArtifactAnalysisConfig(BaseModel):
    enabled_analyzers: List[AnalyzerTypes] = [AnalyzerTypes.TRAINING, AnalyzerTypes.DEEPCHECKS, AnalyzerTypes.DATASET, AnalyzerTypes.MODEL_CHECKPOINT]