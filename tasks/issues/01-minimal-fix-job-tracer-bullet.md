# 01 — Minimal End-to-End Fix Job Tracer Bullet

**What to build:** An end-to-end tracer bullet that allows an ML developer to trigger a model fix job via the CLI and track its lifecycle to completion. When `deepfix-sdk fix --dataset <dataset>` (or `deepfix-sdk diagnose --fix`) is executed, the CLI submits a job request to the DeepFix Server, the server persists the job in SQLite, simulates/executes initial job handling, and the CLI polls the server until receiving a completed status report, displaying the results in stdout.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Domain models `FixJobStatus`, `FixJob`, and `FinalFixReport` exist in `deepfix-core` and are exported cleanly.
- [ ] `POST /v2/fix` endpoint on DeepFix Server accepts fix job parameters (`dataset_name`, `model_name`, `target_metric`, `target_value`, `max_iterations`, `s3_bucket`), creates a record in SQLite, and returns a unique `job_id`.
- [ ] `GET /v2/fix/{job_id}` endpoint returns current job status, iteration count, and result payload.
- [ ] `deepfix-sdk fix` CLI command sends the job request to the server and polls status until completion or failure.
- [ ] Unit and integration tests verify the end-to-end submission and polling cycle.
- [ ] Typecheck and lint pass across modified packages.
