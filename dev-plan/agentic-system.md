# DeepFix Multi-Agent System Plan

## Architecture Foundation

### Agent Integration Strategy

**Agents as Enhanced Steps**: Each agent implements the `Step` interface but with richer contracts:
- Input: Typed `AgentContext` (extends pipeline `dict` context)  
- Output: Structured `AgentResult` with confidence, citations, recommendations
- Interface: `Agent.run(context: AgentContext) → AgentResult`

**Pipeline Integration**: 
1. `AgentCoordinator` runs applicable agents sequentially based on available artifacts
2. `NarrativeSynthesizer` consolidates agent outputs into final `AdvisorResult`

### Data Flow
```
Raw Artifacts
     ↓
┌─────────────────┬─────────────────┐
│ ArtifactAnalysis│ TrainingDynamics│ (Parallel execution)
│     Agent       │     Agent       │
└─────────────────┴─────────────────┘
     ↓                    ↓
     └──────────┬─────────┘
                ↓
    CrossArtifactIntegrationAgent  (Sequential execution)
                ↓
     OptimizationAdvisorAgent      (Uses integration insights)
                ↓
         NarrativeSynthesizer      (Final synthesis)
```
```
ArtifactAnalysisCoordinator (Coordinator)
├── TrainingArtifactsAnalyzerAgent (Training Expert)
├── DeepchecksArtifactsAnalyzerAgent (Data Quality Expert)  
├── DatasetArtifactsAnalyzerAgent (Dataset Expert)
└── ModelCheckpointArtifactsAnalyzerAgent (Model Expert)
```

## Concrete Agent Architecture

### Agent Context Schema
```python
class AgentContext(BaseModel):
    run_id: str
    artifacts: List[Artifacts]  # From existing artifact loading
    mlflow_metadata: Dict[str, Any]
    
    # Typed artifact accessors
    training_artifacts: Optional[TrainingArtifacts] = None
    deepchecks_artifacts: Optional[DeepchecksArtifacts] = None  
    dataset_artifacts: Optional[DatasetArtifacts] = None
    
    # Agent coordination
    completed_agents: List[str] = []
    agent_results: Dict[str, AgentResult] = {}
    knowledge_cache: Dict[str, Any] = {}
```

### Agent Result Schema
```python
class AgentResult(BaseModel):
    agent_name: str
    status: Literal["success", "partial", "failed"]
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Core outputs
    findings: List[Finding]
    recommendations: List[Recommendation] 
    risks: List[Risk]
    
    # Metadata
    execution_time: float
    artifacts_analyzed: List[str]
    knowledge_refs: List[str] = []
    
class Finding(BaseModel):
    type: str  # "anomaly", "pattern", "metric"
    description: str
    evidence: Dict[str, Any]
    severity: Literal["low", "medium", "high"]

class Recommendation(BaseModel):
    action: str
    rationale: str
    priority: Literal["low", "medium", "high"]
    implementation: str
    expected_impact: str
```

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

### Agent Coordination

**AgentCoordinator Step**:
```python
class AgentCoordinator(Step):
    def __init__(self, intelligence_config: IntelligenceConfig):
        self.agents = self._load_agents()
        self.knowledge_bridge = KnowledgeBridge(intelligence_config)
    
    def run(self, context: dict) -> dict:
        agent_context = AgentContext.from_pipeline_context(context)
        
        # Run applicable agents sequentially
        for agent in self._get_applicable_agents(agent_context):
            try:
                result = agent.run(agent_context)
                agent_context.agent_results[agent.name] = result
                agent_context.completed_agents.append(agent.name)
            except Exception as e:
                self._handle_agent_failure(agent, e, agent_context)
        
        # Synthesize results
        synthesizer = NarrativeSynthesizer(self.knowledge_bridge)
        advisor_result = synthesizer.synthesize(agent_context)
        
        context["advisor_result"] = advisor_result
        return context
```

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
