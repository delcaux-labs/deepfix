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
    dataset_name: Optional[str] = None
    # Agent coordination
    agent_results: Dict[str, AgentResult] = Field(default={})
    knowledge_cache: Dict[str, Any] = Field(default={})


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