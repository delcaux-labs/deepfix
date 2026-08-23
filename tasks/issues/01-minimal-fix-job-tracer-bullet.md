# 01 — Minimal End-to-End Fix Job Tracer Bullet

**What to build:** An end-to-end tracer bullet that allows an ML developer to trigger a model fix job via the CLI or Python SDK and track its lifecycle to completion. When `deepfix-sdk fix --dataset <dataset>` (or `deepfix-sdk diagnose --fix`) is executed, the SDK synchronizes datasets and model weights to S3, submits a job request to the DeepFix Server, the server persists the job in SQLite, simulates/executes initial job handling, and the CLI polls the server until receiving a completed status report, displaying the results in stdout.

**Blocked by:** None — can start immediately.

**Status:** completed

- [x] Domain models `FixJobStatus`, `FixJob`, `FinalFixReport`, and `FixJobRequest` exist in `deepfix-core` and are exported cleanly.
- [x] S3 storage & retrieval for dataset artifacts via `push_to_s3()` and `from_s3()` across `BaseDataset`, `TabularDataset`, and `NLPDataset`.
- [x] Hugging Face `datasets` integration (`to_hf_dataset()`, `from_hf_dataset()`) with metadata preservation in Parquet schema.
- [x] Automatic model weights S3 upload helper `push_model_to_s3()` supporting PyTorch, Scikit-learn, and local weights files.
- [x] `POST /v2/fix` endpoint on DeepFix Server accepts fix job parameters (`dataset_name`, `model_name`, `dataset_uri`, `model_uri`, `target_metric`, `target_value`, `max_iterations`, `s3_bucket`), creates a record in SQLite, and returns a unique `job_id`.
- [x] `GET /v2/fix/{job_id}` endpoint returns current job status, iteration count, dataset/model URIs, and result payload.
- [x] `POST /webhook/completion` endpoint updates `FixJobRecord` with final metrics, applied fixes, and S3 weights URI.
- [x] `deepfix-sdk fix` CLI command sends the job request to the server and polls status until completion or failure.
- [x] Artifact staging in `./deepfix_output/<job_id>/` generates `metrics.json`, `summary_report.md`, and `train_fixed.py`.
- [x] Typecheck and lint pass across modified packages.

