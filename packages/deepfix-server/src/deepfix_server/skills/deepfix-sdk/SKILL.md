---
name: deepfix-sdk
description: Instructions and CLI commands on how to use deepfix-sdk to run diagnostics on datasets, evaluate models, and iterate on fixes.
---
# DeepFix SDK Skill

This skill provides comprehensive instructions on using the `deepfix-sdk` CLI and Python SDK inside the sandbox environment to diagnose datasets, evaluate model quality, and train candidate models.

## 1. CLI Commands

### Diagnose a Dataset
Run automated diagnostic checks (integrity, multicollinearity, class imbalance, leakage) on a registered dataset:
```bash
deepfix-sdk diagnose --dataset "<DATASET_NAME>"
```

### Diagnose Dataset with a Baseline Model
Evaluate an existing model against dataset checks:
```bash
deepfix-sdk diagnose --dataset "<DATASET_NAME>" --model "<MODEL_NAME_OR_URI>"
```

---

## 2. Python SDK API Usage

You can also import and use `deepfix_sdk` directly in your Python training and evaluation scripts (`train.py`):

### Loading and Preparing Tabular Datasets:
```python
import pandas as pd
from deepfix_sdk.data import TabularDataset

# Load tabular dataset from CSV or DataFrame
df = pd.read_csv("data.csv")
dataset = TabularDataset(
    dataset=df,
    dataset_name="my_dataset",
    label="target",
    cat_features=["category_col"]
)
```

### Running In-Memory Diagnosis:
```python
from deepfix_sdk.client import DeepFixClient

client = DeepFixClient(api_url="http://host.docker.internal:4141/v1/analyse")
diagnosis = client.get_diagnosis(
    train_data=train_dataset,
    test_data=test_dataset,
    model=trained_model,
    model_name="candidate_model"
)
print(diagnosis.to_text())
```

---

## 3. Recommended Iterative Fix Workflow in `train.py`

1. **Step 1 - Inspect Diagnostics:** Run `deepfix-sdk diagnose --dataset <name>` or review provided diagnostic findings to identify multicollinear features, class imbalances, and distribution shifts.
2. **Step 2 - Apply Preprocessing Remediations:**
   - **Multicollinearity:** Drop high-PPS redundant features, use PCA, or apply Ridge/L2/tree-based architectures.
   - **Class Imbalance:** Compute class weights (e.g. `compute_class_weight`) or use stratified splitting (`StratifiedKFold`).
   - **Small Sample Size:** Use cross-validation to prevent overfitting.
3. **Step 3 - Train & Evaluate:** Train candidate model and evaluate validation metrics (accuracy, F1, ROC-AUC).
4. **Step 4 - Persist & Report:**
   - Push weights: `python s3-weights-storage/push_weights_to_s3.py --weights-path ./model.pt --job-id "$DEEPFIX_JOB_ID"`
   - Notify server: `python deepfix-communication/report_completion.py --job-id "$DEEPFIX_JOB_ID" --status COMPLETED ...`
