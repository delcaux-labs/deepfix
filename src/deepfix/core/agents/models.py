from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import StrEnum
import pandas as pd

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
    analysis: List[Analysis] = Field(default=[],description="List of Analysis elements")
    analyzed_artifacts: Optional[List[str]] = Field(default=None,description="List of artifacts analyzed by the agent")
    refined_analysis: Optional[str] = Field(default=None,description="Refined analysis of the artifacts using all the available context")

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
                    'refined_analysis': agent_result.refined_analysis or '',
                    'finding_description': findings.description,
                    'finding_evidence': findings.evidence,
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
            if agent_df['refined_analysis'].iloc[0]:
                summary += (f"\n  - Refined analysis: {agent_df['refined_analysis'].iloc[0][:100]}...")

        return summary

class ArtifactAnalysisResult(BaseModel):
    context: AgentContext = Field(default=...,description="Context of the analysis")
    summary: Optional[str] = Field(default=...,description="Summary of the analysis")

    def to_text(self) -> str:
        summary = "\n" + "="*80
        summary += "\nDEEPFIX ANALYSIS RESULT"
        summary += "\n" + "="*80
        summary += f"\nDataset: {self.context.dataset_name}"
        summary += f"\nRun ID: {self.context.run_id}"
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