---
name: mlflow-data-access
description: Utilities and instructions for downloading dataset artifacts from MLflow.
---
# MLflow Data Access

When you need to access the dataset for a given MLflow Run ID, use the following snippet to download and load it from the artifact store:

```python
import mlflow
from datasets import load_from_disk

def load_dataset_from_mlflow(run_id: str, artifact_path: str = "dataset"):
    # Download the dataset artifact
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    # Load it via HuggingFace datasets
    dataset = load_from_disk(local_path)
    return dataset
```

Make sure you have `mlflow` and `datasets` installed in your environment before running this code.
