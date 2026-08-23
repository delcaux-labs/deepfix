---
name: deepfix-communication
description: Instructions and script to report completion status and metrics back to DeepFix Server webhook.
---
# DeepFix Communication

When you have finished iterating and have either successfully fixed the model or exhausted all options, you MUST report your completion back to the DeepFix Server using the provided `report_completion.py` script.

## Script Location
`deepfix-communication/report_completion.py`

## Usage Examples

### Successful Fix with S3 Weights Uploaded:
```bash
python deepfix-communication/report_completion.py \
    --job-id "$DEEPFIX_JOB_ID" \
    --success \
    --status COMPLETED \
    --s3-weights-uri "s3://my-bucket/job_123/weights/model.pt" \
    --final-metrics '{"accuracy": 0.96, "val_loss": 0.12}' \
    --applied-fixes "Resolved multicollinearity with PCA" "Added class weights" \
    --summary "Model validation accuracy improved from 0.82 to 0.96 exceeding target 0.90."
```

### Failed / Plateaued Fix:
```bash
python deepfix-communication/report_completion.py \
    --job-id "$DEEPFIX_JOB_ID" \
    --status FAILED \
    --applied-fixes "Tried hyperparameter tuning" \
    --summary "Target metric could not be reached after maximum iterations."
```

The script will submit an HTTP POST request containing a `FinalFixReport` payload to the DeepFix Server webhook endpoint.
