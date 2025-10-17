# Data Contract Specification

## Overview

This document defines all data schemas, validation rules, and compatibility requirements for the DeepFix client-server architecture. It serves as the authoritative source for data structure definitions and ensures consistency across components.

---

## Artifact Schemas

### Training Artifacts

Training artifacts contain metrics and parameters from the training process.

#### Structure

```
training_artifacts/
├── metrics.csv          (required)
└── params.yaml          (required)
```

#### metrics.csv Schema

**Format**: CSV with header row

**Required Columns**:

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `step` | integer | Training step number | >= 0, monotonically increasing |
| `epoch` | integer or float | Training epoch | >= 0 |
| `train_loss` | float | Training loss value | >= 0 |

**Optional Columns** (recommended):

| Column | Type | Description |
|--------|------|-------------|
| `val_loss` | float | Validation loss value |
| `train_accuracy` | float | Training accuracy (0-1 or 0-100) |
| `val_accuracy` | float | Validation accuracy |
| `learning_rate` | float | Current learning rate |
| `grad_norm` | float | Gradient norm |
| `{metric_name}` | float | Any custom metric |

**Validation Rules**:
- ✅ Must contain at least 3 rows (minimum training history)
- ✅ No missing values in required columns
- ✅ `step` and `epoch` must be monotonically increasing
- ✅ Loss values must be non-negative (or detect anomalies)
- ⚠️ Warn if <10 epochs (insufficient for overfitting detection)

**Example**:

```csv
step,epoch,train_loss,val_loss,train_accuracy,val_accuracy,learning_rate
0,0,2.3045,2.2987,0.1234,0.1256,0.001
100,1,1.8234,1.9012,0.3456,0.3234,0.001
200,2,1.2345,1.4567,0.5678,0.5234,0.001
300,3,0.8901,1.1234,0.7123,0.6789,0.0005
```

#### params.yaml Schema

**Format**: YAML

**Required Fields**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `model_name` | string | Model architecture name | "ResNet50" |
| `optimizer` | string | Optimizer type | "Adam" |
| `learning_rate` | float | Initial learning rate | 0.001 |

**Optional Fields** (recommended):

| Field | Type | Description |
|-------|------|-------------|
| `batch_size` | integer | Training batch size |
| `num_epochs` | integer | Total epochs trained |
| `weight_decay` | float | L2 regularization |
| `dropout_rate` | float | Dropout probability |
| `scheduler` | string | LR scheduler type |
| `loss_function` | string | Loss function name |
| `augmentation` | object | Data augmentation config |

**Validation Rules**:
- ✅ Must be valid YAML
- ✅ Required fields present
- ✅ Numeric values in valid ranges (e.g., lr > 0)

**Example**:

```yaml
model_name: "ResNet50"
optimizer: "Adam"
learning_rate: 0.001
batch_size: 32
num_epochs: 50
weight_decay: 0.0001
dropout_rate: 0.5
scheduler: "CosineAnnealingLR"
loss_function: "CrossEntropyLoss"
augmentation:
  random_crop: true
  horizontal_flip: true
  color_jitter: true
```

---

### Dataset Artifacts

Dataset artifacts contain metadata and statistics about the training data.

#### Structure

```
dataset/
├── metadata.yaml        (required)
└── statistics.json      (optional)
```

#### metadata.yaml Schema

**Required Fields**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `dataset_name` | string | Unique dataset identifier | "food_waste_v2" |
| `num_train_samples` | integer | Training set size | 10000 |
| `num_classes` | integer | Number of classes | 5 |

**Optional Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `num_val_samples` | integer | Validation set size |
| `num_test_samples` | integer | Test set size |
| `class_names` | list[string] | Class labels |
| `imbalance_ratio` | float | Max/min class ratio |
| `image_size` | string | Image dimensions (e.g., "224x224") |
| `color_mode` | string | "RGB" or "grayscale" |

**Validation Rules**:
- ✅ `num_train_samples` > 0
- ✅ `num_classes` >= 2
- ⚠️ Warn if `imbalance_ratio` > 10 (severe imbalance)
- ⚠️ Warn if `num_val_samples` < 0.2 * `num_train_samples`

**Example**:

```yaml
dataset_name: "food_waste_v2"
num_train_samples: 8000
num_val_samples: 2000
num_test_samples: 1000
num_classes: 5
class_names: ["bread", "dairy", "fruit", "meat", "vegetable"]
imbalance_ratio: 3.5
image_size: "224x224"
color_mode: "RGB"
```

---

### Deepchecks Artifacts

Deepchecks artifacts contain data quality analysis results.

#### Structure

```
deepchecks/
├── results.yaml         (required)
└── config.yaml          (optional)
```

#### results.yaml Schema

**Format**: YAML with nested structure

**Structure**:

```yaml
dataset_name: string
results:
  train_test_validation:
    - header: string          # Check name
      json_result: object     # Check-specific results
      display_txt: string     # Text summary
      display_images: list    # Base64 images (optional)
  data_integrity:
    - header: string
      json_result: object
      display_txt: string
```

**Known Check Headers**:

| Suite | Check Header | Purpose |
|-------|--------------|---------|
| Train-Test | "Label Drift" | Detect label distribution changes |
| Train-Test | "Image Dataset Drift" | Detect image distribution changes |
| Train-Test | "Image Property Drift" | Detect property changes |
| Train-Test | "New Labels" | Identify new labels in test set |
| Integrity | "Image Property Outliers" | Find outlier images |
| Integrity | "Property Label Correlation" | Check feature-label relationships |
| Integrity | "Class Performance" | Analyze per-class metrics |

**Validation Rules**:
- ✅ Must be valid YAML
- ✅ Contains at least one check result
- ✅ Each result has `header` and `json_result`
- ⚠️ Warn if critical checks missing (e.g., Label Drift)

**Example**:

```yaml
dataset_name: "food_waste_v2"
results:
  train_test_validation:
    - header: "Label Drift"
      json_result:
        drift_score: 0.023
        method: "PSI"
        passed: true
      display_txt: "No significant label drift detected (PSI=0.023)"
      
    - header: "Image Dataset Drift"
      json_result:
        drift_score: 0.15
        passed: false
      display_txt: "Significant drift detected between train and test sets"
      
  data_integrity:
    - header: "Class Performance"
      json_result:
        per_class_accuracy:
          bread: 0.92
          dairy: 0.88
          fruit: 0.95
          meat: 0.76
          vegetable: 0.89
      display_txt: "Meat class shows lower performance (0.76)"
```

---

### Model Checkpoint Artifacts

Model checkpoint artifacts are optional and used for model-specific analysis.

#### Structure

```
best_checkpoint/
└── model.ckpt           (PyTorch Lightning checkpoint)
```

**Validation Rules**:
- ✅ File exists and is readable
- ✅ Valid PyTorch checkpoint format
- ⚠️ Checkpoint analysis not required for v1

---

## API Request/Response Schemas

### AnalysisRequest

```typescript
interface AnalysisRequest {
  run_id: string;                    // Required: MLflow run ID
  mlflow_tracking_uri: string;       // Required: MLflow server URI
  analysis_options?: {               // Optional
    enable_training_analysis?: boolean;  // default: true
    enable_dataset_analysis?: boolean;   // default: true
    enable_deepchecks_analysis?: boolean; // default: true
    optimization_focus?: string[];       // ["overfitting", "data_quality", ...]
    max_analysis_time?: number;          // seconds, default: 60
  };
  context?: {                        // Optional
    dataset_name?: string;
    model_type?: string;             // "cnn", "transformer", etc.
    task_type?: string;              // "classification", "regression", etc.
  };
}
```

**Validation Rules**:
- ✅ `run_id` is non-empty string
- ✅ `mlflow_tracking_uri` is valid URI (http/https)
- ✅ `max_analysis_time` in range [30, 300]
- ✅ `optimization_focus` items in allowed set
- ✅ `model_type` in enum (if provided)

---

### AnalysisResult

```typescript
interface AnalysisResult {
  request_id: string;
  run_id: string;
  status: "completed" | "partial" | "failed";
  summary?: string;                  // Natural language summary
  agent_results: AgentResult[];
  execution_time: number;            // seconds
  timestamp: string;                 // ISO 8601
  warnings?: string[];               // Non-fatal warnings
}

interface AgentResult {
  agent_name: string;
  analysis: Analysis[];
  analyzed_artifacts?: string[];
  retrieved_knowledge?: string[];
  additional_outputs?: Record<string, any>;
  error_message?: string;
}

interface Analysis {
  findings: Finding;
  recommendations: Recommendation;
}

interface Finding {
  description: string;
  evidence: string;
  severity: "low" | "medium" | "high";
  confidence: number;                // 0.0 to 1.0
}

interface Recommendation {
  action: string;
  rationale: string;
  priority: "low" | "medium" | "high";
  confidence: number;                // 0.0 to 1.0
}
```

**Validation Rules**:
- ✅ `status` in enum
- ✅ `confidence` in range [0.0, 1.0]
- ✅ `severity` and `priority` in enum
- ✅ `timestamp` is valid ISO 8601
- ✅ At least one agent_result if status is "completed" or "partial"

---

### KnowledgeQuery

```typescript
interface KnowledgeQuery {
  query: string;                     // Required
  domain?: "training" | "data_quality" | "architecture" | "optimization" | "global";
  query_type: "best_practice" | "diagnostic" | "solution" | "validation";
  max_results?: number;              // default: 5, range: [1, 20]
  min_confidence?: number;           // default: 0.7, range: [0.0, 1.0]
}

interface KnowledgeResponse {
  query: string;
  results: KnowledgeItem[];
}

interface KnowledgeItem {
  content: string;
  source: string;
  confidence?: number;               // 0.0 to 1.0
  relevance_score?: number;          // 0.0 to 1.0
  metadata?: Record<string, any>;
}
```

**Validation Rules**:
- ✅ `query` is non-empty string
- ✅ `query_type` in enum
- ✅ `max_results` in range [1, 20]
- ✅ `min_confidence` in range [0.0, 1.0]

---

## Backward Compatibility

### Version Strategy

**API Versioning**: Semantic versioning in URL path (`/api/v1/...`)

**Breaking Changes** (require new major version):
- Removing required fields
- Changing field types
- Removing endpoints
- Changing authentication

**Non-Breaking Changes** (can stay in same version):
- Adding optional fields
- Adding new endpoints
- Deprecating (not removing) fields
- Adding enum values

### Legacy Artifact Support

**v0 Format (Pre-Refactoring)**:
- Artifacts stored without structured organization
- Server must handle both old and new formats

**Migration Path**:

```python
def load_training_artifacts(path: str) -> TrainingArtifacts:
    """Load artifacts with backward compatibility"""
    # Try new format first
    if (Path(path) / "training_artifacts" / "metrics.csv").exists():
        return load_v1_format(path)
    
    # Fall back to legacy format
    if (Path(path) / "metrics.csv").exists():
        return load_v0_format(path)
    
    raise ArtifactNotFoundError(f"No training artifacts found at {path}")
```

### Schema Evolution

**Adding New Fields**:

```yaml
# ✅ OK: Add optional fields with defaults
params.yaml (v1):
  model_name: "ResNet50"
  optimizer: "Adam"
  learning_rate: 0.001

params.yaml (v2):
  model_name: "ResNet50"
  optimizer: "Adam"
  learning_rate: 0.001
  mixed_precision: false  # New optional field
```

**Deprecating Fields**:

```yaml
# ⚠️ Deprecate with warning, support for 2 versions
params.yaml (v2):
  model_name: "ResNet50"  # Deprecated: use 'architecture' instead
  architecture: "ResNet50"
```

---

## Validation Rules Summary

### Client-Side Validation

**Before Sending Request**:
- ✅ Required fields present
- ✅ Field types correct
- ✅ Values in valid ranges
- ✅ URIs properly formatted

**Purpose**: Fail fast, reduce server load

### Server-Side Validation

**On Request Receipt**:
- ✅ Full schema validation (Pydantic models)
- ✅ Business logic validation (e.g., run_id exists in MLflow)
- ✅ Rate limiting checks
- ✅ Authorization checks (future)

**Purpose**: Security, data integrity

### Artifact Validation

**On Artifact Load**:
- ✅ File exists and readable
- ✅ Format valid (CSV, YAML, JSON)
- ✅ Required fields present
- ⚠️ Optional fields missing (warn, continue)
- ⚠️ Values out of expected range (warn, analyze anyway)

**Purpose**: Graceful degradation, useful error messages

---

## Error Response Format

All errors follow this structure:

```typescript
interface ErrorResponse {
  error: string;                     // Error code (UPPERCASE_SNAKE_CASE)
  message: string;                   // Human-readable message
  details?: Record<string, any>;     // Additional context
  timestamp: string;                 // ISO 8601
}
```

**Standard Error Codes**:

| Code | HTTP Status | Description | Retry? |
|------|-------------|-------------|--------|
| `VALIDATION_ERROR` | 400 | Invalid request data | ❌ |
| `RUN_NOT_FOUND` | 404 | MLflow run doesn't exist | ❌ |
| `ARTIFACT_NOT_FOUND` | 404 | Required artifact missing | ❌ |
| `MLFLOW_CONNECTION_ERROR` | 503 | Can't reach MLflow server | ✅ |
| `ANALYSIS_TIMEOUT` | 504 | Analysis exceeded time limit | ⚠️ |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error | ✅ |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | ✅ |

**Example**:

```json
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid request: max_analysis_time must be between 30 and 300 seconds",
  "details": {
    "field": "analysis_options.max_analysis_time",
    "provided": 600,
    "constraint": "30 <= value <= 300"
  },
  "timestamp": "2025-10-15T10:30:00Z"
}
```

---

## Data Quality Metrics

### Completeness

**Metric**: Percentage of required fields present

**Targets**:
- Training artifacts: 100% required fields
- Dataset artifacts: >80% recommended fields
- Deepchecks artifacts: >5 checks executed

### Validity

**Metric**: Percentage of values passing validation

**Targets**:
- Numeric ranges: 100% valid
- Enum values: 100% valid
- Format constraints: 100% valid

### Timeliness

**Metric**: Artifact freshness

**Targets**:
- Artifacts logged <1 hour after training
- Analysis requested <24 hours after logging

---

## Testing Data

### Example Valid Request

```json
{
  "run_id": "abc123def456",
  "mlflow_tracking_uri": "http://localhost:5000",
  "analysis_options": {
    "enable_training_analysis": true,
    "enable_dataset_analysis": true,
    "enable_deepchecks_analysis": false,
    "optimization_focus": ["overfitting"],
    "max_analysis_time": 120
  },
  "context": {
    "dataset_name": "food_waste_v2",
    "model_type": "cnn",
    "task_type": "classification"
  }
}
```

### Example Valid Response

```json
{
  "request_id": "req_abc123",
  "run_id": "abc123def456",
  "status": "completed",
  "summary": "Training shows signs of overfitting after epoch 15. Validation loss increased 12% while training loss decreased 25%. Consider early stopping or regularization.",
  "agent_results": [
    {
      "agent_name": "TrainingArtifactsAnalyzer",
      "analysis": [
        {
          "findings": {
            "description": "Overfitting detected",
            "evidence": "Val loss increased from 0.45 to 0.52 over epochs 15-20",
            "severity": "high",
            "confidence": 0.92
          },
          "recommendations": {
            "action": "Add dropout (rate 0.3-0.5) or L2 regularization (weight_decay 0.001)",
            "rationale": "Regularization constrains model complexity and reduces overfitting",
            "priority": "high",
            "confidence": 0.88
          }
        }
      ],
      "analyzed_artifacts": ["TrainingArtifacts"],
      "retrieved_knowledge": ["Regularization best practices"]
    }
  ],
  "execution_time": 45.3,
  "timestamp": "2025-10-15T10:30:45Z",
  "warnings": []
}
```

---

## Migration Checklist

When adding new data contracts:

- [ ] Define schema in this document
- [ ] Add Pydantic models in code
- [ ] Update OpenAPI spec
- [ ] Add validation rules
- [ ] Create test fixtures
- [ ] Document backward compatibility
- [ ] Update client SDK
- [ ] Add migration guide

---

**Document Version**: 1.0  
**Last Updated**: October 15, 2025  
**Status**: Specification Complete

