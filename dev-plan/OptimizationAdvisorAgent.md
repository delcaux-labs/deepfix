# OptimizationAdvisorAgent Specification

## Overview

The `OptimizationAdvisorAgent` is a **knowledge-driven recommendation agent** that synthesizes findings from previous agents (ArtifactAnalysisAgent, TrainingDynamicsAgent, CrossArtifactIntegrationAgent) to provide actionable, evidence-based optimization recommendations. It acts as an "ML optimization expert" that understands the current state of the ML experiment and suggests specific improvements for data, training, and model architecture.

**Core Philosophy**: This agent transforms diagnostic insights into prescriptive actions, leveraging both rule-based heuristics and LLM-powered knowledge retrieval to provide contextual, implementable recommendations that address root causes rather than symptoms.

## Implementation Timeline

### Phase 1: Core Recommendation Framework (Week 1)
- [X] Implement base `OptimizationAdvisorAgent` class with state analysis
- [X] Design core optimization system prompt

### Phase 2: Knowledge Integration (Week 2)
- [ ] Integrate `KnowledgeBridge` for evidence-based recommendations
- [ ] Implement best practice retrieval system
- [ ] Add recommendation enhancement with research citations
- [ ] Create knowledge confidence scoring

### Phase 3: Advanced Optimization (Week 3)
- [ ] Implement multi-criteria recommendation prioritization
- [ ] Add impact estimation and risk assessment
- [ ] Create recommendation validation and testing framework
- [ ] Add implementation example generation

### Phase 4: Validation & Refinement (Week 4)
- [ ] Comprehensive testing with known optimization scenarios
- [ ] Validate recommendation accuracy against expert knowledge
- [ ] Performance optimization and caching strategies
- [ ] Documentation and usage examples



## Testing Strategy

### DsPy Integration Testing
- **Signature Validation**: Test that the OptimizationRecommendationSignature works correctly with various LLM providers
- **Response Parsing**: Validate that recommendation and finding parsing handles various LLM output formats
- **Context Preparation**: Ensure optimization context is properly formatted for LLM consumption
- **Error Handling**: Test graceful degradation when LLM calls fail or return malformed responses

### Recommendation Quality Testing
- **Known Optimization Scenarios**: Test with curated ML problems with known solutions
- **Implementation Feasibility**: Ensure all recommendations are technically implementable
- **Response Consistency**: Test that similar inputs produce consistent recommendation quality
- **Format Compliance**: Validate that LLM outputs follow the expected recommendation format

### Integration Testing
- **Multi-Agent Coordination**: Test integration with upstream diagnostic agents (ArtifactAnalysisAgent, TrainingDynamicsAgent, CrossArtifactIntegrationAgent)
- **Context Flow**: Validate that agent results are properly passed and processed
- **Configuration Flexibility**: Test different optimization focuses and constraints
- **Edge Cases**: Test with minimal diagnostic data, conflicting findings, and various artifact types

### LLM Provider Testing
- **Provider Compatibility**: Test with different LLM providers (OpenAI, Anthropic, local models)
- **Temperature Sensitivity**: Validate recommendation quality across different temperature settings
- **Token Limits**: Test behavior when context exceeds token limits
- **Rate Limiting**: Test graceful handling of API rate limits and timeouts



This specification creates an intelligent optimization advisor that can provide actionable, evidence-based recommendations to improve ML experiments based on comprehensive analysis of diagnostic findings from other agents.
