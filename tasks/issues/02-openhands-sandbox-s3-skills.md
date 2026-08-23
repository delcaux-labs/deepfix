# 02 — OpenHands Sandbox Skills with S3 Weights Push & Webhook Callback

**What to build:** An autonomous execution sandbox equipping the OpenHands agent with required runtime capabilities and domain skills. The Docker sandbox environment is configured to propagate AWS/S3 credentials and OpenTelemetry traces into MLflow. Inside the container, the agent has access to skills for pulling datasets from MLflow (`mlflow-data-access`), uploading trained model weights directly to an S3 bucket path (`s3-weights-storage` / `push_weights_to_s3.py`), and sending the final completion report payload (`FinalFixReport`) back to the DeepFix Server via `POST /webhook/completion` (`deepfix-communication`).

**Blocked by:** 01 — Minimal End-to-End Fix Job Tracer Bullet

**Status:** completed

- [x] `mlflow-data-access` skill created under `deepfix-kb` with instructions and utilities for dataset retrieval from MLflow.
- [x] `s3-weights-storage` skill created under `deepfix-kb` with `push_weights_to_s3.py` script allowing the agent to upload model checkpoints to `s3://<bucket>/<job_id>/...`.
- [x] `deepfix-communication` skill created under `deepfix-kb` with `report_completion.py` script sending `FinalFixReport` (including `s3_weights_uri`, `final_metrics`, `applied_fixes`, and `status`) to DeepFix Server webhook.
- [x] DeepFix Server implements `POST /webhook/completion` endpoint to receive and persist final reports into SQLite.
- [x] `AutonomousFixConfig` manages Docker sandbox settings, S3 credentials injection, and OTEL MLflow tracing environment variables.
- [x] OpenHands sandbox skills loading and environment variables verified.
- [x] Typecheck and lint pass across modified packages.

