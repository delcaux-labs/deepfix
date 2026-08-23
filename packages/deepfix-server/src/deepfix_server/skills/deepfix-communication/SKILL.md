---
name: deepfix-communication
description: Instructions on how to communicate the final status back to DeepFix Server once you are finished iterating.
---
# DeepFix Communication

When you have finished iterating and have either successfully fixed the model or exhausted all options, you MUST report your completion back to the DeepFix Server using the provided `report_completion.py` script.

The script is located at: `deepfix-communication/report_completion.py`

Usage:
```bash
python deepfix-communication/report_completion.py \
    --webhook-url "http://host.docker.internal:8844/webhook/completion" \
    --job-id "YOUR_JOB_ID" \
    --success \
    --final-run-id "MLFLOW_RUN_ID" \
    --fixes "Applied class balancing" "Tuned learning rate" \
    --final-metrics '{"accuracy": 0.95}'
```

If you failed, omit `--success` and pass an empty list of fixes (or the fixes you tried).
