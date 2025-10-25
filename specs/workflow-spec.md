# Workflow Specification

## Overview

This document defines the interaction patterns, workflows, and error handling strategies for the DeepFix client-server architecture. It provides sequence diagrams and detailed flow descriptions for all major use cases.

---

## Primary Workflows

### Workflow 1: Training Analysis (Synchronous)

**Use Case**: User trains a model with PyTorch Lightning and requests immediate analysis feedback.

**Actors**: User, Lightning Callback (Client), DeepFix Server, MLflow

**Preconditions**:
- MLflow tracking server is running
- DeepFix server is running and accessible
- Training has completed

**Flow**:

```
┌─────┐      ┌──────────┐      ┌─────────┐      ┌────────┐
│User │      │Lightning │      │ Server  │      │MLflow  │
│     │      │Callback  │      │         │      │        │
└──┬──┘      └────┬─────┘      └────┬────┘      └───┬────┘
   │              │                 │               │
   │ fit(model)   │                 │               │
   ├─────────────>│                 │               │
   │              │                 │               │
   │              │ Training...     │               │
   │              │                 │               │
   │              │ on_fit_end()    │               │
   │              ├─┐               │               │
   │              │ │ Compute       │               │
   │              │ │ Artifacts     │               │
   │              │<┘               │               │
   │              │                 │               │
   │              │ Log Artifacts   │               │
   │              ├────────────────────────────────>│
   │              │                 │               │
   │              │                 │   Acknowledge │
   │              │<────────────────────────────────┤
   │              │                 │               │
   │              │ POST /api/v1/analyze           │
   │              │ {run_id, mlflow_uri}           │
   │              ├────────────────>│               │
   │              │                 │               │
   │              │                 │ Fetch Artifacts
   │              │                 ├──────────────>│
   │              │                 │               │
   │              │                 │ Artifacts     │
   │              │                 │<──────────────┤
   │              │                 │               │
   │              │                 ├─┐             │
   │              │                 │ │ Execute     │
   │              │                 │ │ Agents      │
   │              │                 │ │ (30-60s)    │
   │              │                 │<┘             │
   │              │                 │               │
   │              │ 200 OK          │               │
   │              │ {AnalysisResult}│               │
   │              │<────────────────┤               │
   │              │                 │               │
   │              ├─┐               │               │
   │              │ │ Display       │               │
   │              │ │ Results       │               │
   │              │<┘               │               │
   │              │                 │               │
   │ Complete     │                 │               │
   │<─────────────┤                 │               │
   │              │                 │               │
```

**Steps**:

1. **Training Execution** (User → Lightning)
   - User calls `trainer.fit(model)`
   - Training proceeds normally
   
2. **Artifact Computation** (Callback)
   - On `on_fit_end()` hook
   - Compute training metrics summary
   - Compute dataset statistics (if not exists)
   - Run Deepchecks (if enabled)
   - Duration: <5% of training time

3. **Artifact Logging** (Callback → MLflow)
   - Log `training_artifacts/metrics.csv`
   - Log `training_artifacts/params.yaml`
   - Log `deepchecks/` (if computed)
   - Log `dataset/` metadata (if computed)
   - Verify successful upload

4. **Analysis Request** (Callback → Server)
   - POST `/api/v1/analyze`
   - Payload: `{run_id, mlflow_tracking_uri, analysis_options}`
   - Timeout: 60s
   
5. **Artifact Retrieval** (Server → MLflow)
   - Server fetches artifacts using MLflow client
   - Downloads to temporary directory
   - Validates artifact completeness
   - Duration: 5-10s

6. **Analysis Execution** (Server)
   - Parallel execution:
     - TrainingArtifactsAnalyzerAgent
     - DatasetArtifactsAnalyzerAgent  
     - DeepchecksArtifactsAnalyzerAgent
   - Sequential execution:
     - CrossArtifactIntegrationAgent
     - OptimizationAdvisorAgent
   - Duration: 30-60s

7. **Result Return** (Server → Callback)
   - 200 OK with `AnalysisResult`
   - Cleanup temporary artifacts

8. **Result Display** (Callback → User)
   - Print summary to console
   - Optionally log to MLflow
   - Training completes

**Postconditions**:
- User sees analysis results
- Artifacts stored in MLflow
- Server has cleaned up temporary files

**Timing**:
- Total overhead: <70s (artifact logging + analysis)
- User-perceived delay: Minimal (happens after training)

---

### Workflow 2: Training Analysis (Asynchronous)

**Use Case**: Long-running analysis that doesn't block training completion.

**Flow**:

```
┌──────────┐      ┌─────────┐      ┌────────┐
│Lightning │      │ Server  │      │MLflow  │
│Callback  │      │         │      │        │
└────┬─────┘      └────┬────┘      └───┬────┘
     │                 │               │
     │ POST /analyze   │               │
     ├────────────────>│               │
     │                 │               │
     │ 202 Accepted    │               │
     │ {request_id}    │               │
     │<────────────────┤               │
     │                 │               │
     ├─┐               │               │
     │ │ Continue      │               │
     │ │ Training      │               │
     │<┘               │               │
     │                 │               │
     │                 ├─┐ Fetch +     │
     │                 │ │ Analyze     │
     │                 │ │ (async)     │
     │                 │<┘             │
     │                 │               │
     │ GET /status     │               │
     ├────────────────>│               │
     │                 │               │
     │ {status: running}               │
     │<────────────────┤               │
     │                 │               │
     │ (poll every 5s) │               │
     │                 │               │
     │ GET /result     │               │
     ├────────────────>│               │
     │                 │               │
     │ 200 OK          │               │
     │ {AnalysisResult}│               │
     │<────────────────┤               │
     │                 │               │
```

**Configuration**:

```python
callback = DeepSightCallback(
    ...,
    analysis_mode="async",  # Enable async mode
    poll_interval=5,        # Poll every 5 seconds
    max_wait_time=300       # Give up after 5 minutes
)
```

**Benefits**:
- Training completes immediately
- Long-running analysis doesn't block
- User can check results later

---

### Workflow 3: Standalone Analysis Request

**Use Case**: Analyze a past run without retraining.

**Actor**: Data Scientist (via Python script or CLI)

**Flow**:

```python
from deepfix.client import DeepFixClient

client = DeepFixClient(server_url="http://localhost:8000")

# Analyze a specific run
result = client.analyze(
    run_id="abc123def456",
    mlflow_uri="http://localhost:5000",
    analysis_options={
        "enable_training_analysis": True,
        "optimization_focus": ["overfitting"]
    }
)

print(result.summary)
for agent_result in result.agent_results:
    for analysis in agent_result.analysis:
        print(f"Finding: {analysis.findings.description}")
        print(f"Action: {analysis.recommendations.action}")
```

**Use Cases**:
- Re-analyze old runs with updated knowledge base
- Compare multiple runs
- Generate reports for stakeholders
- Debug training issues post-mortem

---

### Workflow 4: Knowledge Base Query

**Use Case**: Directly query knowledge base without running full analysis.

**Flow**:

```
┌─────┐      ┌─────────┐      ┌─────────────┐
│User │      │ Client  │      │   Server    │
│     │      │         │      │             │
└──┬──┘      └────┬────┘      └──────┬──────┘
   │              │                  │
   │ query_knowledge()               │
   ├─────────────>│                  │
   │              │                  │
   │              │ POST /knowledge/query
   │              ├─────────────────>│
   │              │                  │
   │              │                  ├─┐
   │              │                  │ │ Search KB
   │              │                  │ │ + Cache
   │              │                  │<┘
   │              │                  │
   │              │ KnowledgeResponse│
   │              │<─────────────────┤
   │              │                  │
   │ results      │                  │
   │<─────────────┤                  │
   │              │                  │
```

**Example**:

```python
client = DeepFixClient(server_url="http://localhost:8000")

results = client.query_knowledge(
    query="How to prevent overfitting in CNNs?",
    domain="training",
    query_type="best_practice",
    max_results=5
)

for item in results:
    print(f"{item.content} (confidence: {item.confidence})")
```

---

## Error Scenarios

### Error 1: MLflow Server Unavailable

**Scenario**: Server cannot connect to MLflow to fetch artifacts.

**Flow**:

```
┌──────────┐      ┌─────────┐      ┌────────┐
│ Client   │      │ Server  │      │MLflow  │
└────┬─────┘      └────┬────┘      └───┬────┘
     │                 │               │
     │ POST /analyze   │               │
     ├────────────────>│               │
     │                 │               │
     │                 │ Fetch         │
     │                 ├──────────────>│
     │                 │               │
     │                 │ ❌ Timeout    │
     │                 │<──────────────┤
     │                 │               │
     │ 503 Service     │               │
     │ Unavailable     │               │
     │ Retry-After: 30 │               │
     │<────────────────┤               │
     │                 │               │
     ├─┐               │               │
     │ │ Wait 30s      │               │
     │<┘               │               │
     │                 │               │
     │ POST /analyze   │               │
     │ (retry)         │               │
     ├────────────────>│               │
     │                 │               │
```

**Server Response**:

```json
{
  "error": "MLFLOW_CONNECTION_ERROR",
  "message": "Unable to connect to MLflow tracking server at http://localhost:5000",
  "details": {
    "tracking_uri": "http://localhost:5000",
    "timeout": 120,
    "suggestion": "Check MLflow server is running and accessible"
  },
  "timestamp": "2025-10-15T10:30:00Z"
}
```

**Client Behavior**:
1. Receive 503 with `Retry-After` header
2. Wait specified time (or use exponential backoff)
3. Retry up to 3 times
4. If all retries fail: log warning, continue without analysis

---

### Error 2: Missing Artifacts

**Scenario**: Run exists but some artifacts are missing (e.g., no deepchecks).

**Flow**:

```
Server fetches artifacts:
✅ training_artifacts/metrics.csv  
✅ training_artifacts/params.yaml  
❌ deepchecks/  (not found)
❌ dataset/  (not found)

Server proceeds with partial analysis
```

**Server Response**:

```json
{
  "request_id": "req_abc123",
  "run_id": "abc123def456",
  "status": "completed",
  "warnings": [
    "Deepchecks artifacts not found - skipping data quality analysis",
    "Dataset metadata not found - skipping dataset analysis"
  ],
  "summary": "Training analysis completed. Data quality analysis unavailable due to missing artifacts.",
  "agent_results": [
    {
      "agent_name": "TrainingArtifactsAnalyzer",
      "analysis": [...]
    }
  ]
}
```

**Client Behavior**:
- Display warnings to user
- Show available analysis results
- Suggest computing missing artifacts for future runs

---

### Error 3: Analysis Timeout

**Scenario**: Analysis exceeds max_analysis_time.

**Flow**:

```
Server starts analysis at T=0
├─ TrainingAnalyzer: completes at T=15s ✅
├─ DatasetAnalyzer: completes at T=25s ✅
├─ CrossArtifact: completes at T=40s ✅
└─ OptimizationAdvisor: timeout at T=60s ⏱️
```

**Server Response**:

```json
{
  "request_id": "req_abc123",
  "status": "partial",
  "warnings": [
    "Analysis timed out after 60s",
    "OptimizationAdvisor did not complete"
  ],
  "summary": "Partial analysis available. Training issues identified but optimization advice incomplete.",
  "agent_results": [
    {"agent_name": "TrainingArtifactsAnalyzer", ...},
    {"agent_name": "DatasetArtifactsAnalyzer", ...},
    {"agent_name": "CrossArtifactIntegrationAgent", ...}
  ],
  "execution_time": 60.0
}
```

**Client Behavior**:
- Accept partial results
- Display warnings
- Suggest increasing timeout for complex analyses

---

### Error 4: Server Unavailable

**Scenario**: DeepFix server is down or unreachable.

**Flow**:

```
┌──────────┐      ┌─────────┐
│ Client   │      │ Server  │
└────┬─────┘      └────┬────┘
     │                 │
     │ POST /analyze   │
     ├────────────────>│
     │                 │
     │ ❌ Connection   │
     │    Failed       │
     │                 │
     ├─┐               │
     │ │ Retry 1       │
     │<┘               │
     │                 │
     │ ❌ Connection   │
     │    Failed       │
     │                 │
     ├─┐               │
     │ │ Fallback to   │
     │ │ offline mode  │
     │<┘               │
     │                 │
```

**Client Behavior**:

```python
try:
    result = client.analyze(run_id=run_id, mlflow_uri=uri)
except ServerUnavailableError as e:
    logger.warning(
        f"DeepFix server unavailable: {e}. "
        "Skipping analysis. Training metrics logged to MLflow."
    )
    # Continue without analysis
```

**Fallback Strategy**:
- **Mode 1: Strict**: Fail if server unavailable
- **Mode 2: Fallback** (default): Warn and continue without analysis
- **Mode 3: Local**: Run local analysis (future feature)

---

### Error 5: Invalid Request

**Scenario**: Client sends malformed request.

**Server Response**:

```json
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid request: missing required field 'mlflow_tracking_uri'",
  "details": {
    "field": "mlflow_tracking_uri",
    "constraint": "required",
    "provided": null
  },
  "timestamp": "2025-10-15T10:30:00Z"
}
```

**Client Behavior**:
- Validate requests before sending (fail fast)
- Log error for debugging
- Don't retry (invalid request won't succeed)

---

## Retry Policies

### Client-Side Retry Policy

```python
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 10.0
    backoff_factor: float = 2.0
    retryable_statuses: List[int] = [408, 429, 500, 502, 503, 504]
    
    def get_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter"""
        delay = min(
            self.base_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter
```

**Retry Matrix**:

| Error Type | Retry? | Max Attempts | Strategy |
|------------|--------|--------------|----------|
| 400 Bad Request | ❌ No | 0 | Client error - fix and retry |
| 404 Not Found | ❌ No | 0 | Run doesn't exist |
| 408 Timeout | ✅ Yes | 3 | Exponential backoff |
| 429 Too Many Requests | ✅ Yes | 3 | Respect Retry-After |
| 500 Server Error | ✅ Yes | 3 | Exponential backoff |
| 503 Service Unavailable | ✅ Yes | 3 | Respect Retry-After |
| Network Error | ✅ Yes | 3 | Exponential backoff |

---

## Polling Strategy

### Status Polling (Async Mode)

```python
class PollingConfig:
    initial_interval: float = 1.0  # Start with 1s
    max_interval: float = 10.0     # Cap at 10s
    max_wait_time: float = 300.0   # Give up after 5min
    backoff_factor: float = 1.5
    
async def poll_until_complete(request_id: str) -> AnalysisResult:
    start_time = time.time()
    interval = initial_interval
    
    while time.time() - start_time < max_wait_time:
        status = await client.get_status(request_id)
        
        if status.status == "completed":
            return await client.get_result(request_id)
        elif status.status == "failed":
            raise AnalysisError(status.error_message)
        
        # Backoff
        await asyncio.sleep(interval)
        interval = min(interval * backoff_factor, max_interval)
    
    raise TimeoutError(f"Analysis did not complete within {max_wait_time}s")
```

---

## Concurrency Patterns

### Server-Side: Parallel Agent Execution

```python
async def execute_analysis(context: AgentContext) -> AnalysisResult:
    # Phase 1: Parallel artifact analysis
    artifact_agents = [
        TrainingArtifactsAnalyzer(),
        DatasetArtifactsAnalyzer(),
        DeepchecksArtifactsAnalyzer()
    ]
    
    results = await asyncio.gather(
        *[agent.run(context) for agent in artifact_agents],
        return_exceptions=True
    )
    
    # Phase 2: Sequential integration
    context.agent_results = {r.agent_name: r for r in results if not isinstance(r, Exception)}
    
    integration_result = await CrossArtifactIntegrationAgent().run(context)
    context.agent_results["CrossArtifact"] = integration_result
    
    optimization_result = await OptimizationAdvisorAgent().run(context)
    context.agent_results["OptimizationAdvisor"] = optimization_result
    
    return create_analysis_result(context)
```

### Client-Side: Multiple Concurrent Requests

```python
# Client can send multiple concurrent requests
client = DeepFixClient(server_url="...")

# Analyze multiple runs in parallel
run_ids = ["run1", "run2", "run3"]
results = await asyncio.gather(*[
    client.analyze_async(run_id=rid, mlflow_uri=uri)
    for rid in run_ids
])
```

**Server Constraint**: Maximum 10 concurrent analyses (configurable)

---

## State Management

### Server: Stateless Design

```python
# ❌ DON'T: Store per-user state
class AnalysisService:
    def __init__(self):
        self.user_sessions = {}  # NO!
        
# ✅ DO: Pure functions
async def analyze(request: AnalysisRequest) -> AnalysisResult:
    """Stateless analysis - no side effects"""
    context = create_context(request)
    result = await execute_analysis(context)
    cleanup_temp_files(context)
    return result
```

**Allowed State**:
- ✅ Knowledge base cache (shared across all requests)
- ✅ Temporary files (cleaned up after request)
- ❌ User sessions
- ❌ Analysis history (future: external database)

---

## Performance Optimization

### Caching Strategy

**Knowledge Base Cache**:
```python
cache_key = hash(query + domain + query_type)
if cache_key in knowledge_cache:
    return knowledge_cache[cache_key]

result = knowledge_bridge.query(query, domain, query_type)
knowledge_cache[cache_key] = result
return result
```

**Cache Metrics**:
- Target hit rate: >70%
- TTL: 24 hours
- Max size: 1000 entries or 1GB

---

## Success Criteria

- ✅ Training analysis completes in <70s (including artifact logging)
- ✅ Knowledge queries respond in <2s
- ✅ Server handles 10 concurrent requests
- ✅ Client retry success rate >80% on transient failures
- ✅ Offline mode works gracefully when server down
- ✅ Partial analysis provided when some agents fail
- ✅ Clear error messages with recovery suggestions

---

**Document Version**: 1.0  
**Last Updated**: October 15, 2025  
**Status**: Specification Complete

