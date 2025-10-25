# Service Specification

## Overview

This document defines the responsibilities, constraints, and boundaries for the DeepFix client and server components. It serves as the contract for component ownership and interaction rules.

## Architecture Principles

1. **Separation of Concerns**: Client handles computation; server handles analysis
2. **Fail Independently**: Components can fail without cascading failures
3. **Stateless Server**: No session state; enables horizontal scaling
4. **Idempotent Operations**: Same request produces same result
5. **Graceful Degradation**: System provides partial functionality when components unavailable

## Server Specification

### Server Identity

**Name**: DeepFix Analysis Server  
**Type**: REST API Service  
**Technology**: FastAPI (Python 3.11+)  
**Deployment**: Local-first, containerized (Docker)

### Server Responsibilities

The server is responsible for:

#### 1. Artifact Retrieval
- Connect to MLflow tracking server using provided URI
- Fetch artifacts for specified run_id
- Download and cache artifacts locally (temporary storage)
- Handle missing or incomplete artifacts gracefully
- **NOT responsible for**: Artifact computation or storage

**Implementation Notes**:
- Reuse existing `ArtifactsManager` from `src/deepfix/core/artifacts/`
- Use MLflow Python client for artifact downloads
- Implement parallel downloads for multiple artifact types
- Store downloaded artifacts in ephemeral temp directory (clean up after analysis)

#### 2. AI-Powered Analysis
- Execute multi-agent analysis pipeline
- Coordinate agent execution (parallel where possible)
- Aggregate agent results into unified output
- Generate natural language summaries
- **NOT responsible for**: Training models or computing metrics

**Implementation Notes**:
- Reuse existing agent system from `src/deepfix/core/agents/`
- Maintain agent execution order: 
  - Parallel: TrainingAnalyzer, DatasetAnalyzer, DeepchecksAnalyzer
  - Sequential: CrossArtifactIntegration → OptimizationAdvisor
- Implement timeout handling per agent
- Support partial results when some agents fail

#### 3. Knowledge Retrieval
- Query knowledge base using KnowledgeBridge
- Cache knowledge retrieval results
- Validate retrieved knowledge against agent context
- Provide knowledge citations in responses
- **NOT responsible for**: Knowledge base updates or curation

**Implementation Notes**:
- Reuse `KnowledgeBridge` from `src/deepfix/core/agents/knowledge_bridge.py`
- Implement in-memory LRU cache for query results
- Cache key: hash(query + domain + query_type)
- TTL: 24 hours

#### 4. Result Formatting
- Transform agent results into API response format
- Generate natural language summaries
- Prioritize findings by severity and confidence
- Format recommendations with actionable steps
- **NOT responsible for**: Result persistence or visualization

#### 5. Error Handling
- Validate incoming requests against schema
- Handle MLflow connection failures
- Manage agent execution errors
- Provide detailed error messages with recovery suggestions
- **NOT responsible for**: Client-side error recovery

### Server Constraints

#### Performance Constraints
| Metric | Constraint | Rationale |
|--------|-----------|-----------|
| Response Time | <60s (typical) | User expectation for interactive analysis |
| Timeout | 300s (max) | Hard limit to prevent resource exhaustion |
| Concurrent Requests | 10 simultaneous | Local-first deployment assumption |
| Memory Usage | <4GB per analysis | Supports deployment on standard machines |
| Knowledge Query Time | <2s | Interactive query experience |

#### Resource Constraints
- **Cache Size**: Max 1GB or 1000 entries (LRU eviction)
- **Artifact Storage**: Temporary only, cleaned after analysis
- **Database**: No persistent state (optional SQLite for artifact tracking)
- **Network**: Must handle MLflow server on different host

#### Operational Constraints
- **Stateless Design**: No session state between requests
- **No User Authentication**: Relies on network security (add later)
- **No Artifact Storage**: Server doesn't persist artifacts beyond analysis
- **Single Tenant**: No multi-tenancy support in v1

#### Compatibility Constraints
- **Python Version**: 3.11+
- **MLflow Version**: Compatible with 2.0+
- **Artifact Formats**: Must handle legacy and current formats
- **API Version**: Semantic versioning (v1 = stable)

### Server Boundaries

**What Server DOES**:
- ✅ Fetch artifacts from MLflow
- ✅ Run AI analysis on artifacts
- ✅ Query and cache knowledge
- ✅ Return structured results
- ✅ Health monitoring

**What Server DOES NOT**:
- ❌ Compute or generate artifacts
- ❌ Store artifacts permanently
- ❌ Log to MLflow
- ❌ Train models
- ❌ Manage user sessions
- ❌ Persist analysis history (v1)
- ❌ Update knowledge base (v1)

### Server Configuration

```yaml
# server/config/default.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 300
  
cache:
  max_size_mb: 1024
  max_entries: 1000
  ttl_hours: 24
  eviction_policy: "lru"

analysis:
  max_concurrent: 10
  default_timeout: 60
  enable_parallel_agents: true
  
artifacts:
  temp_dir: "/tmp/deepfix_artifacts"
  cleanup_after_analysis: true
  download_timeout: 120

logging:
  level: "INFO"
  format: "json"
  output: "stdout"
```

---

## Client Specification

### Client Identity

**Name**: DeepFix Client SDK  
**Type**: Python Library + Lightning Callback  
**Technology**: Python 3.11+  
**Distribution**: PyPI package (`pip install deepfix[client]`)

### Client Responsibilities

#### 1. Artifact Computation
- Compute training artifacts during/after training
- Compute dataset artifacts (statistics, metadata)
- Run Deepchecks analysis (if enabled)
- Validate artifact completeness before logging
- **NOT responsible for**: AI analysis or knowledge retrieval

**Implementation Notes**:
- Keep existing `DatasetIngestionPipeline` for dataset artifacts
- Lightning callback collects training metrics during fit
- Deepchecks runs client-side before logging to MLflow

#### 2. Artifact Recording
- Log artifacts to MLflow tracking server
- Organize artifacts by type and run_id
- Ensure artifact availability before requesting analysis
- **NOT responsible for**: Artifact storage infrastructure

**Implementation Notes**:
- Use MLflow Python client for logging
- Follow existing artifact structure (metrics.csv, params.yaml, etc.)
- Verify artifacts logged successfully before analysis request

#### 3. Server Communication
- Send analysis requests to server
- Handle synchronous and asynchronous workflows
- Poll for long-running analysis status
- Retrieve and display results
- **NOT responsible for**: Server availability or performance

**Implementation Notes**:
- Implement Python SDK wrapping REST API
- Support both blocking (`analyze()`) and async (`analyze_async()`) modes
- Use `requests` or `httpx` for HTTP communication

#### 4. Result Consumption
- Receive and parse analysis results
- Display findings and recommendations
- Log results back to MLflow (optional)
- **NOT responsible for**: Result interpretation or visualization (v1)

#### 5. Error Recovery
- Implement retry logic with exponential backoff
- Handle server unavailability gracefully
- Provide offline mode (skip analysis if server down)
- Log errors for debugging
- **NOT responsible for**: Server error recovery

**Implementation Notes**:
- Max 3 retries with exponential backoff (1s, 2s, 4s)
- Timeout per request: 60s (configurable)
- Fallback to offline mode with user warning

### Client Constraints

#### Performance Constraints
| Metric | Constraint | Rationale |
|--------|-----------|-----------|
| Artifact Computation | <10% training overhead | Minimal impact on training workflow |
| Network Timeout | 60s default | Balance between patience and responsiveness |
| Retry Attempts | 3 max | Avoid indefinite waiting |
| Backoff Max | 10s | Cap retry delay |

#### Resource Constraints
- **Memory**: Share with training process (no dedicated allocation)
- **Disk**: Temporary artifact storage before MLflow upload
- **Network**: Must handle remote MLflow and server

#### Operational Constraints
- **Offline Support**: Must work when server unavailable
- **Backward Compatible**: Support users without server
- **Non-Blocking**: Analysis doesn't block training completion
- **Logging**: Use Python logging (no custom logger required)

### Client Boundaries

**What Client DOES**:
- ✅ Compute artifacts during/after training
- ✅ Log artifacts to MLflow
- ✅ Send analysis requests to server
- ✅ Handle server responses
- ✅ Retry on transient failures
- ✅ Operate offline when server down

**What Client DOES NOT**:
- ❌ Perform AI analysis
- ❌ Query knowledge base directly (v1)
- ❌ Store analysis results permanently
- ❌ Manage server lifecycle
- ❌ Validate server responses deeply (trusts server)

### Client Configuration

```python
# Example: Lightning callback configuration
from deepfix.integrations import DeepSightCallback

callback = DeepSightCallback(
    dataset_name="food_waste_v2",
    train_dataset=train_ds,
    val_dataset=val_ds,
    metric_names=["train_loss", "val_loss", "accuracy"],
    batch_size=16,
    # Server configuration
    server_url="http://localhost:8000",
    enable_analysis=True,
    analysis_mode="async",  # or "sync"
    offline_mode="fallback",  # "strict" or "fallback"
    max_retries=3,
    timeout=60
)
```

---

## Component Interaction Rules

### 1. Request/Response Flow

```
Client                          Server
  |                               |
  |-- POST /api/v1/analyze ------>|
  |   (run_id, mlflow_uri)        |
  |                               |
  |<----- 202 Accepted ----------|
  |   (request_id)                |
  |                               |
  |-- GET /status/{request_id} ->|
  |                               |
  |<----- 200 OK ----------------|
  |   (status: running)           |
  |                               |
  |-- GET /result/{request_id} ->|
  |                               |
  |<----- 200 OK ----------------|
  |   (full AnalysisResult)       |
```

### 2. Error Handling Responsibilities

| Error Type | Client Responsibility | Server Responsibility |
|------------|----------------------|----------------------|
| Network failure | Retry with backoff | N/A |
| MLflow unavailable | Log warning, continue | Return 503 with details |
| Missing artifacts | Ensure logged before request | Partial analysis with warnings |
| Invalid request | Validate before sending | Return 400 with validation errors |
| Analysis timeout | Handle partial results | Return partial results with warnings |
| Server crash | Fallback to offline mode | N/A (restart via orchestrator) |

### 3. Data Ownership

| Data Type | Owner | Storage | Lifetime |
|-----------|-------|---------|----------|
| Raw artifacts | Client | MLflow | Persistent |
| Analysis results | Server | Memory (return to client) | Request duration |
| Knowledge cache | Server | Memory (LRU) | 24 hours |
| Temporary artifacts | Server | Temp filesystem | Analysis duration |

### 4. Concurrency Model

**Client**:
- Sequential artifact computation (part of training pipeline)
- Concurrent analysis requests allowed (multiple runs)

**Server**:
- Concurrent request handling (up to 10 simultaneous)
- Parallel agent execution within single analysis
- Thread-safe caching

---

## Migration Considerations

### Backward Compatibility

**Phase 1 (Current State)**:
- Monolithic pipeline: Client runs everything locally

**Phase 2 (Transitional)**:
- Client SDK available but optional
- Legacy `DeepSightAdvisor` class wraps client (compatibility layer)
- Users can opt-in to server-based analysis

**Phase 3 (Target State)**:
- Server-based analysis is default
- Legacy support deprecated but functional
- Migration guide provided

### Compatibility Layer

```python
# src/deepfix/compat/legacy_advisor.py
class DeepSightAdvisor:
    """Backward compatibility wrapper using client SDK"""
    def __init__(self, config):
        self.config = config
        self.client = DeepFixClient(server_url=config.get("server_url"))
    
    def run_analysis(self, run_id):
        # Wraps client.analyze() to match old API
        return self.client.analyze(
            run_id=run_id,
            mlflow_uri=self.config.mlflow.tracking_uri
        )
```

---

## Security Considerations (Future)

While not required for v1 (local deployment), future versions should consider:

- **Authentication**: API keys or OAuth for server access
- **Authorization**: Role-based access control
- **Encryption**: TLS for client-server communication
- **Rate Limiting**: Prevent abuse of public servers
- **Audit Logging**: Track analysis requests and results

---

## Success Metrics

### Server Performance
- ✅ 95th percentile response time <60s
- ✅ Availability >99% (local deployment)
- ✅ Cache hit rate >70% for knowledge queries
- ✅ Concurrent request handling: 10 simultaneous

### Client Reliability
- ✅ Successful analysis request rate >95%
- ✅ Offline mode graceful degradation: 100%
- ✅ Artifact computation overhead <10% training time
- ✅ Retry success rate >80% on transient failures

### Integration Quality
- ✅ API contract compliance: 100%
- ✅ Backward compatibility: 2 major versions
- ✅ Migration success rate: >90% of users

---

**Document Version**: 1.0  
**Last Updated**: October 15, 2025  
**Status**: Specification Complete

