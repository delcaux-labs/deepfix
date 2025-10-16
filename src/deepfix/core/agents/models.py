from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import StrEnum
from datetime import datetime
import pandas as pd

from ..artifacts import Artifacts

# ============================================================================
# Analysis data Models
# ============================================================================
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
    description: str = Field(default=...,description="Short Description of the finding")
    evidence: str = Field(default=...,description="Evidence of the finding")
    severity: Severity = Field(default=...,description="Severity of the finding")
    confidence: float = Field(ge=0.0, le=1.0,description="Confidence in the finding and severity")

class Recommendation(BaseModel):
    action: str =  Field(default=...,description="Action to take to address the finding")
    rationale: str =  Field(default=...,description="Rationale for the recommendation")
    priority: Severity = Field(default=...,description="Priority of the recommendation")
    confidence: float = Field(ge=0.0, le=1.0,description="Confidence in the recommendation")

class Analysis(BaseModel):
    findings: Finding = Field(default=...,description="Finding of the analysis")
    recommendations: Recommendation = Field(default=...,description="Recommendation based on the findings")

class AgentResult(BaseModel):
    agent_name: str
    analysis: List[Analysis] = Field(default=[],description="List of Analysis elements")
    analyzed_artifacts: Optional[List[str]] = Field(default=None,description="List of artifacts analyzed by the agent")
    retrieved_knowledge: Optional[List[str]] = Field(default=None,description="External knowledge relevant to the analysis")
    additional_outputs: Dict[str, Any] = Field(default={},description="Additional outputs from the agent")
    error_message: Optional[str] = Field(default=None,description="Error message if the agent failed")

class AgentContext(BaseModel):
    artifacts: List[Artifacts]
    run_id: Optional[str] = None
    dataset_name: Optional[str] = None
    agent_results: Dict[str, AgentResult] = Field(default={})
    knowledge_cache: Dict[str, Any] = Field(default={})

    def to_dataframe(self) -> pd.DataFrame:
        """
        Transform ArtifactAnalysisResult.context into a pandas DataFrame.
        """
        rows = []
        
        for agent_name, agent_result in self.agent_results.items():
            for analysis in agent_result.analysis:
                # Extract findings and recommendations from the Analysis object
                findings = analysis.findings
                recommendations = analysis.recommendations
                
                # Create a row combining findings and recommendations
                row = {
                    'agent_name': agent_name,
                    'analyzed_artifacts': ', '.join(agent_result.analyzed_artifacts) if agent_result.analyzed_artifacts else '',
                    'retrieved_knowledge': ', '.join(agent_result.retrieved_knowledge) if agent_result.retrieved_knowledge else '',
                    'summary': agent_result.additional_outputs.get('summary', ''),
                    'finding_description': findings.description,
                    'finding_evidence': findings.evidence,
                    'error_message': agent_result.error_message,
                    'finding_severity': findings.severity.value,
                    'finding_confidence': findings.confidence,
                    'recommendation_action': recommendations.action,
                    'recommendation_rationale': recommendations.rationale,
                    'recommendation_priority': recommendations.priority.value,
                    'recommendation_confidence': recommendations.confidence
                }
                rows.append(row)
        
        return pd.DataFrame(rows)

    def to_text(self) -> str:
        df = self.to_dataframe()
        summary = "="*80
        summary += "\nSUMMARY STATISTICS"
        summary += ("\n" + "="*80)

        summary += (f"\nTotal findings: {len(df)}")
        summary += (f"\nAgents involved: {df['agent_name'].unique().tolist()}")
        summary += ("\nSeverity distribution:")
        summary += f"\n{df['finding_severity'].value_counts().to_dict()}"

        summary += (f"\nPriority distribution:")
        summary += f"\n{df['recommendation_priority'].value_counts().to_dict()}"
        
        for severity in df['finding_severity'].unique():
            summary += ("\n" + "="*80)
            summary += (f"\n{severity.upper()} SEVERITY ISSUES")
            summary += ("\n" + "="*80)

            df_severity = df[df['finding_severity'] == severity]
            for i,(idx, row) in enumerate(df_severity.iterrows()):
                summary += (f"\n{i+1}. [{row['agent_name']}] {row['finding_description']}")
                summary += (f"\n   Evidence: {row['finding_evidence']}")
                summary += (f"\n   Action: {row['recommendation_action']}")
                summary += (f"\n   Rationale: {row['recommendation_rationale']}")

        summary += ("\n" + "="*80)
        summary += ("\nAGENT-SPECIFIC ANALYSIS")
        summary += ("\n" + "="*80)

        for agent in df['agent_name'].unique():
            agent_df = df[df['agent_name'] == agent]
            summary += (f"\n{agent}:")
            summary += (f"\n  - Findings: {len(agent_df)}")
            summary += (f"\n  - Artifacts analyzed: {agent_df['analyzed_artifacts'].iloc[0] if not agent_df.empty else 'None'}")
            if agent_df['summary'].iloc[0]:
                summary += (f"\n  - Summary: {agent_df['summary'].iloc[0][:100]}...")

        return summary

class ArtifactAnalysisResult(BaseModel):
    context: AgentContext = Field(default=...,description="Context of the analysis")
    summary: Optional[str] = Field(default=...,description="Summary of the analysis")
    additional_outputs: Dict[str, Any] = Field(default={},description="Additional outputs from the agent")

    def to_text(self) -> str:
        summary = "\n" + "="*80
        summary += "\nDEEPFIX ANALYSIS RESULT\n"
        summary += f"\nDataset: {self.context.dataset_name}"
        summary += f"\nRun ID: {self.context.run_id}"
        if self.additional_outputs.get('optimization_areas'):
            summary += f"\nOptimization areas: {self.additional_outputs['optimization_areas']}"
        if self.additional_outputs.get('constraints'):
            summary += f"\nConstraints: {self.additional_outputs['constraints']}"
        summary += "\n" + "="*80
        summary += f"\nSummary of the analysis:\n{self.summary}"
        summary += "\n" + "="*80
        summary += self.context.to_text()        
        return summary
    
class TrainingDynamicsConfig(BaseModel):
    # Analysis Configuration
    enabled_analyzers: List[str] = [
        "overfitting_detection",
        "training_stability",
        "gradient_analysis",
        "performance_trends"
    ]
    
    # Detection Thresholds
    overfitting_thresholds: Dict[str, float] = {
        "train_val_divergence": 0.1,        # Relative divergence threshold
        "val_loss_plateau_epochs": 5,        # Epochs for plateau detection
        "early_stopping_patience": 10       # Patience for early stopping recommendation
    }
    
    stability_thresholds: Dict[str, float] = {
        "loss_variance_threshold": 0.05,     # Coefficient of variation threshold
        "metric_volatility_window": 10,     # Window size for volatility analysis
        "gradient_norm_std_threshold": 2.0  # Standard deviation threshold for gradient norms
    }
    
    gradient_thresholds: Dict[str, float] = {
        "exploding_gradient_threshold": 10.0,   # Gradient norm threshold
        "vanishing_gradient_threshold": 1e-6,   # Minimum gradient norm
        "gradient_clip_threshold": 1.0          # Recommended gradient clipping value
    }
    
    # Performance Configuration
    lightweight_mode: bool = True           # <10% overhead constraint
    max_analysis_time: float = 30.0        # Maximum analysis time in seconds
    small_model_optimized: bool = True     # Optimized for <100M parameters

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


