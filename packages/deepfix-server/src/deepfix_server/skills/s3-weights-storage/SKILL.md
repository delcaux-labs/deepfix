---
name: s3-weights-storage
description: Instructions and CLI utility for uploading trained model checkpoints and weights directly to S3.
---
# S3 Weights Storage

When you have trained or fine-tuned a model checkpoint and need to persist it into the designated S3 bucket, use the provided `push_weights_to_s3.py` script.

## CLI Usage

```bash
python s3-weights-storage/push_weights_to_s3.py \
    --weights-path "./model.pt" \
    --job-id "$DEEPFIX_JOB_ID" \
    --s3-bucket "$AWS_S3_BUCKET"
```

## Python API Usage

```python
import os
import boto3

def upload_weights(local_path: str, s3_bucket: str, job_id: str) -> str:
    filename = os.path.basename(local_path)
    s3_key = f"{job_id}/weights/{filename}"
    s3_client = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))
    s3_client.upload_file(local_path, s3_bucket, s3_key)
    return f"s3://{s3_bucket}/{s3_key}"
```

The script will print the canonical S3 URI (`s3://<bucket>/<job_id>/weights/<filename>`) to stdout. You should then pass this URI to `report_completion.py` when notifying the DeepFix Server.
