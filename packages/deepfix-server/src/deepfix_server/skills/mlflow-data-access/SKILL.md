---
name: mlflow-data-access
description: Utilities and instructions for downloading dataset artifacts and models from MLflow and S3.
---
# MLflow Data Access

When you need to access the dataset for a given MLflow Run ID or S3 URI, use the following guidelines and tools.

## Downloading Datasets from MLflow Run

You can download and load the dataset artifact using Python or the provided script `download_dataset.py`.

### Python Snippet:
```python
import mlflow
from datasets import load_from_disk

def load_dataset_from_mlflow(run_id: str, artifact_path: str = "dataset"):
    # Download the dataset artifact from MLflow
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    # Load it via HuggingFace datasets
    dataset = load_from_disk(local_path)
    return dataset
```

### CLI Command:
```bash
python mlflow-data-access/download_dataset.py --run-id "<MLFLOW_RUN_ID>" --artifact-path "dataset" --output-dir "./data"
```

## Downloading Datasets from S3

If the dataset was synchronized to an S3 URI (e.g. `s3://<bucket>/<dataset_name>/...`):
```python
import os
import boto3

# Use standard AWS credentials injected into your environment:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL
```
