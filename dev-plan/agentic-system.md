# DeepFix Multi-Agent System Plan

## Architecture Foundation


### Data Flow
```
Raw Artifacts
     ↓
┌─────────────────┬─────────────────┐
│ ArtifactAnalysis│ TrainingDynamics│ (Parallel execution)
│    Coordinator  │     Agent       │
└─────────────────┴─────────────────┘
     ↓                    ↓
     └──────────┬─────────┘
                ↓
    CrossArtifactIntegrationAgent  (Sequential execution)
                ↓
     OptimizationAdvisorAgent      (Uses integration insights)
```
```
ArtifactAnalysisCoordinator (Coordinator)
├── TrainingArtifactsAnalyzerAgent (Training Expert)
├── DeepchecksArtifactsAnalyzerAgent (Data Quality Expert)  
├── DatasetArtifactsAnalyzerAgent (Dataset Expert)
└── ModelCheckpointArtifactsAnalyzerAgent (Model Expert)
```

## Concrete Agent Architecture

### Core Agents (Phase 1)

**1. ArtifactAnalysisAgent**
- **Purpose**: Validate artifacts, extract metadata, surface data quality signals
- **Inputs**: All available artifacts
- **Outputs**: Data quality findings, artifact summaries, missing data warnings
- **Implementation**: Direct analysis of artifact data structures

**2. TrainingDynamicsAgent**
- **Purpose**: Analyze training metrics for anomalies and overfitting
- **Inputs**: `TrainingArtifacts` (metrics.csv, params.yaml)
- **Outputs**: Gradient anomalies, overfitting detection, training stability assessment
- **Implementation**: Statistical analysis of training curves, threshold-based detection

**3. OptimizationAdvisorAgent**
- **Purpose**: Suggest model and data improvements based on training context
- **Inputs**: Training artifacts + Deepchecks results + dataset metadata
- **Outputs**: Hyperparameter recommendations, augmentation strategies, architecture suggestions
- **Implementation**: Rule-based recommendations + knowledge retrieval

## Knowledge Integration

**KnowledgeBridge**: Extends existing `IntelligenceClient` pattern
```python
class KnowledgeBridge:
    def __init__(self, intelligence_config: IntelligenceConfig):
        self.client = IntelligenceClient(intelligence_config)
        self.cache = {}
    
    def retrieve_best_practices(self, domain: str, context: Dict) -> List[str]:
        # Query LLM for domain-specific best practices
        
    def validate_recommendation(self, recommendation: str, evidence: Dict) -> float:
        # Get confidence score for recommendation
```

## Configuration Integration

**Extend AdvisorConfig**:
```python
class AgentConfig(BaseModel):
    enabled_agents: List[str] = ["artifact_analysis", "training_dynamics", "optimization_advisor"]
    agent_timeouts: Dict[str, float] = {"default": 30.0}
    confidence_thresholds: Dict[str, float] = {"default": 0.7}
    fallback_to_legacy: bool = True  # Backward compatibility

class AdvisorConfig(BaseModel):
    # ... existing fields ...
    agents: AgentConfig = Field(default_factory=AgentConfig)
```
