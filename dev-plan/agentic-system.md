# DeepSight Multi-Agent System Plan (Revised)

## Overview
- **Goal**: Transform DeepSight into a modular multi-agent system while preserving existing `Pipeline`/`Step` architecture
- **Context**: Extend current `DeepSightAdvisor` orchestration with specialized agents as enhanced pipeline steps
- **Principles**: Incremental enhancement, backwards compatibility, concrete interfaces, measurable improvements

## Architecture Foundation

### Current Pipeline Flow Analysis
```
DeepSightAdvisor.run_analysis(run_id) → 
  ArtifactLoadingPipeline → [LoadTrainingArtifact, LoadDeepchecksArtifacts, LoadDatasetArtifact] → 
  BuildPrompt → 
  Query → 
  AdvisorResult
```

**Key Insight**: Current system loads artifacts → builds single prompt → executes one query. Agents will enhance this by analyzing artifacts independently and contributing specialized insights.

### Agent Integration Strategy

**Agents as Enhanced Steps**: Each agent implements the `Step` interface but with richer contracts:
- Input: Typed `AgentContext` (extends pipeline `dict` context)  
- Output: Structured `AgentResult` with confidence, citations, recommendations
- Interface: `Agent.run(context: AgentContext) → AgentResult`

**Pipeline Integration**: 
1. Replace `BuildPrompt + Query` with `AgentCoordinator` step
2. `AgentCoordinator` runs applicable agents sequentially based on available artifacts
3. `NarrativeSynthesizer` consolidates agent outputs into final `AdvisorResult`

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

**1. ArtifactAnalysisAgent** (replaces Artifact Curator)
- **Purpose**: Validate artifacts, extract metadata, surface data quality signals
- **Inputs**: All available artifacts
- **Outputs**: Data quality findings, artifact summaries, missing data warnings
- **Implementation**: Direct analysis of artifact data structures

**2. TrainingDynamicsAgent** (replaces Training Dynamics Monitor)  
- **Purpose**: Analyze training metrics for anomalies and overfitting
- **Inputs**: `TrainingArtifacts` (metrics.csv, params.yaml)
- **Outputs**: Gradient anomalies, overfitting detection, training stability assessment
- **Implementation**: Statistical analysis of training curves, threshold-based detection

**3. OptimizationAdvisorAgent** (merges Dataset Strategist + Model Tuner)
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

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Create `AgentContext` and `AgentResult` models
- [ ] Implement `AgentCoordinator` step with stub agents
- [ ] Modify `DeepSightAdvisor.initialize_processing_pipeline()` to use `AgentCoordinator` instead of `BuildPrompt + Query`
- [ ] Ensure backward compatibility with existing `AdvisorResult` output

### Phase 2: Core Agents (Week 3-4)
- [ ] Implement `ArtifactAnalysisAgent` with basic validation logic
- [ ] Implement `TrainingDynamicsAgent` with overfitting detection
- [ ] Create `NarrativeSynthesizer` to convert agent results to natural language
- [ ] Add configuration options to enable/disable agent system

### Phase 3: Intelligence Integration (Week 5-6)
- [ ] Implement `KnowledgeBridge` extending `IntelligenceClient`
- [ ] Add `OptimizationAdvisorAgent` with knowledge retrieval
- [ ] Implement recommendation validation and confidence scoring
- [ ] Add agent execution metrics and logging

### Phase 4: Enhancement & Testing (Week 7-8)
- [ ] Add comprehensive agent unit tests
- [ ] Implement A/B testing framework for agent vs. legacy recommendations
- [ ] Add user feedback collection for recommendation quality
- [ ] Performance optimization and caching strategies

## Validation Strategy

### Technical Validation
- **Unit Tests**: Each agent tested independently with mock artifacts
- **Integration Tests**: Full pipeline tests comparing agent vs. legacy outputs
- **Performance Tests**: Ensure agent system doesn't exceed current execution time budgets

### User Validation  
- **A/B Testing**: Side-by-side comparison of agent vs. current recommendations
- **Feedback Collection**: Built-in rating system for recommendation usefulness
- **Success Metrics**: Track adoption rate of agent recommendations

### Fallback Strategy
- **Configuration-based**: Users can disable agent system via `fallback_to_legacy: true`
- **Graceful Degradation**: Individual agent failures don't break pipeline
- **Performance Guardrails**: Automatic fallback if agent execution exceeds time limits

## Benefits Over Current Approach

1. **Modularity**: Each agent handles specific domain expertise
2. **Extensibility**: New agents can be added without changing core pipeline
3. **Testability**: Individual agents can be unit tested and validated
4. **Traceability**: Clear provenance of recommendations with confidence scores
5. **Incremental Enhancement**: Gradual rollout with fallback to current system
6. **Performance**: Parallel agent development and optimization opportunities

## Risk Mitigation

- **Backward Compatibility**: Existing `DeepSightAdvisor` API unchanged
- **Feature Flags**: Agent system can be disabled per user/environment
- **Graceful Failures**: Pipeline continues even if individual agents fail
- **Performance Monitoring**: Automatic fallback if execution time exceeds thresholds
- **Quality Assurance**: A/B testing ensures agent recommendations meet quality bar