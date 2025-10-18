from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import StrEnum
from datetime import datetime

from ..shared.models import Artifacts, AgentResult, DatasetArtifacts, TrainingArtifacts, DeepchecksArtifacts, ModelCheckpointArtifacts


## Agent Context
class AgentContext(BaseModel):
    dataset_artifacts: Optional[DatasetArtifacts] = None
    training_artifacts: Optional[TrainingArtifacts] = None
    deepchecks_artifacts: Optional[DeepchecksArtifacts] = None
    model_checkpoint_artifacts: Optional[ModelCheckpointArtifacts] = None
    dataset_name: Optional[str] = None
    agent_results: Dict[str, AgentResult] = Field(default={},description="Results of the agents")
    knowledge_cache: Dict[str, Any] = Field(default={})

    @property
    def artifacts(self,) -> List[Artifacts]:
        artifacts = [self.dataset_artifacts, self.training_artifacts, self.deepchecks_artifacts, self.model_checkpoint_artifacts]
        return [a for a in artifacts if a is not None]
    
    def insert_artifact(self, artifact: Artifacts):
        if isinstance(artifact, DatasetArtifacts):
            self.dataset_artifacts = artifact
        elif isinstance(artifact, TrainingArtifacts):
            self.training_artifacts = artifact
        elif isinstance(artifact, DeepchecksArtifacts):
            self.deepchecks_artifacts = artifact
        elif isinstance(artifact, ModelCheckpointArtifacts):
            self.model_checkpoint_artifacts = artifact
        else:
            raise ValueError(f"Invalid artifact type: {type(artifact)}")
    
class ArtifactAnalysisResult(BaseModel):
    context: AgentContext = Field(default=...,description="Context of the analysis")
    summary: Optional[str] = Field(default=...,description="Summary of the analysis")
    additional_outputs: Dict[str, Any] = Field(default={},description="Additional outputs from the agent")
    
    def get_error_messages(self) -> Dict[str, str]:
        return {agent_name: agent_result.error_message for agent_name, agent_result in self.context.agent_results.items()}
# ============================================================================
# KnowledgeBridge Models
# ============================================================================

class KnowledgeDomain(StrEnum):
    """Knowledge domains for retrieval"""
    TRAINING = "training"
    DATA_QUALITY = "data_quality"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    GLOBAL = "global"

class QueryType(StrEnum):
    """Types of knowledge queries"""
    BEST_PRACTICE = "best_practice"
    DIAGNOSTIC = "diagnostic"
    SOLUTION = "solution"
    VALIDATION = "validation"

class KnowledgeDocument(BaseModel):
    """Structured knowledge document for indexing"""
    # Core Content
    title: str
    content: str
    domain: KnowledgeDomain
    knowledge_type: QueryType
    
    # Metadata for Retrieval
    tags: List[str] = Field(default=[])
    ml_frameworks: List[str] = Field(default=[], description="e.g., pytorch, lightning, tensorflow")
    model_types: List[str] = Field(default=[], description="e.g., cnn, transformer, mlp")
    problem_types: List[str] = Field(default=[], description="e.g., classification, regression")
    
    # Validation
    confidence_level: float = Field(ge=0.0, le=1.0, default=0.8)
    source: str = Field(description="Research paper, documentation, case study")
    last_updated: Optional[datetime] = Field(default=None)
    
    # Application Context
    prerequisites: List[str] = Field(default=[], description="When is this knowledge applicable?")
    contraindications: List[str] = Field(default=[], description="When should this NOT be used?")
    examples: List[str] = Field(default=[])

class KnowledgeItem(BaseModel):
    """Single piece of retrieved knowledge"""
    content: str
    source: str
    confidence: Optional[float] = Field(default=None,description="Confidence score on relevance of evidence to the question, between 0.0 and 1.0")
    relevance_score: Optional[float] = Field(default=None,description="Relevance score of the evidence to the question, between 0.0 and 1.0")
    metadata: Dict[str, Any] = Field(default={},description="Metadata of the evidence")

class AgentKnowledgeRequest(BaseModel):
    """Standard knowledge request from agents"""
    requesting_agent: str
    domain: Optional[KnowledgeDomain] = Field(default=None,description="Knowledge domain for the retrieval. If None, all domains will be searched.")
    query_type: QueryType
    
    # Context from agent analysis
    agent_result: AgentResult
    
    # Retrieval preferences
    max_results: int = Field(default=5, ge=1, le=20)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

class QueryGenerationResult(BaseModel):
    domain: KnowledgeDomain = Field(
        description="Knowledge domain for the retrieval"
    )
    retrieval_queries: List[str] = Field(
        description="List of optimized queries for multi-aspect retrieval"
    )
    search_strategy: str = Field(
        description="Retrieval strategy (semantic, hybrid, keyword-based)"
    )
    rationale: str = Field(
        description="Concise rationale behind the query formulation"
    )

class EvidenceValidationResult(BaseModel):
    """Evidence validation result. It contains the confidence, relevance, and actionable status for an evidence."""
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score on relevance of evidence to the question, between 0.0 and 1.0")
    relevance: str = Field(description="Explanation of why this evidence is or isn't relevant")
    is_actionable: bool = Field(description="Whether the evidence provides actionable insights True/False")
    is_contradictory: bool = Field(description="Whether the evidence contradicts the question True/False")
    
class KnowledgeResponse(BaseModel):
    """Structured knowledge response"""
    query: str
    retrieved_knowledge: List[KnowledgeItem]
    validation_results: List[EvidenceValidationResult]


