# PRD: Autonomous Fix System CLI

## 1. Introduction / Overview

DeepFix provides automated multi-agent diagnostic capabilities that analyze ML datasets and models to detect critical data integrity defects, feature-feature multicollinearity, sample leakage, class imbalance, and performance degradation.

The **Autonomous Fix System CLI** extends DeepFix from passive diagnostics to proactive, end-to-end autonomous model repair. Through the CLI (`deepfix-sdk fix` or `deepfix-sdk diagnose --fix`), users can submit an ML task, dataset, baseline model, and target optimization metrics. The CLI delegates the fix workflow to the DeepFix Server, which runs an autonomous OpenHands engineer inside a sandboxed Docker execution environment. 

The agent iteratively diagnoses, synthesizes candidate training script fixes (e.g., handling severe multicollinearity, class weighting, stratified k-fold cross-validation), executes experiments with MLflow tracking, **pushes trained model weights to a designated S3 bucket**, and delivers the improved model code, S3 artifact pointers, and evaluation reports into a dedicated `./deepfix_output/` staged directory.

---

## 2. Goals

- **Automated End-to-End Remediation:** Transform diagnostic findings (from `DiagnosticSystem` / `engine.py`) into concrete code and modeling fixes without manual developer intervention.
- **Server-Delegated Sandboxed Iteration:** Offload the heavy compute and iterative agent execution to DeepFix Server and sandboxed OpenHands Docker containers.
- **S3 Model Weight Persistence:** Empower the OpenHands agent to push large model checkpoints and weights directly to a configured S3 bucket, recording canonical S3 URIs in MLflow and fix reports.
- **CI/CD & Scriptable CLI Experience:** Provide a clean, robust CLI stream with formatted Rich tables, progress bars, and exit codes suitable for automated pipelines and interactive developer terminals.
- **Safe Staged Artifact Delivery:** Export fixed training scripts, S3 weight pointers, performance metrics diffs, and remediation summaries to `./deepfix_output/` without altering source workspace files directly.
- **Observable Progress:** Provide live streaming status updates via polling or server-sent event / websocket channels, logging intermediate metrics and iterations.

---

## 3. User Stories

### US-001: Trigger Autonomous Fix from CLI
**Description:** As an ML engineer, I want to run a single CLI command with my dataset and target metrics so that DeepFix autonomously repairs my model pipeline.

**Acceptance Criteria:**
- [ ] CLI command `deepfix-sdk fix --dataset <name> --model <name> --target-metric <metric> --target-value <float> --max-iterations <int> [--s3-bucket <bucket>]` is available.
- [ ] CLI verifies connectivity to the DeepFix Server (`--api-url`, defaulting to `http://localhost:4141`) before initiating the job.
- [ ] CLI displays initial job submission details (Job ID, Dataset, Target Metric, Baseline, S3 Target) in a clean terminal table.
- [ ] Returns non-zero exit code on submission failure or invalid arguments.
- [ ] Typecheck and lint pass across `deepfix-sdk`.

### US-002: Stream Real-Time Fix Progress and Iteration Status
**Description:** As an ML engineer, I want to see streaming log output and progress indicators in my terminal so that I know what iteration the autonomous agent is executing and what metrics it has achieved.

**Acceptance Criteria:**
- [ ] CLI displays a live progress bar tracking iteration count against `--max-iterations`.
- [ ] CLI streams structured status updates: agent phase (e.g., `Diagnosing`, `Synthesizing Fix`, `Training`, `Evaluating`, `Uploading to S3`, `Completed`).
- [ ] Intermediate MLflow metric runs (e.g., iteration step, loss, validation F1/ROC-AUC) are formatted into a summary table on stdout.
- [ ] Graceful handling of Ctrl+C with prompt to abort server job or detach monitoring.
- [ ] Typecheck and lint pass.

### US-003: Deliver Staged Output Artifacts & S3 Weight References
**Description:** As an ML engineer, I want the final fixed training script, S3 weight URIs, and summary report to be staged into a dedicated output directory so that my original source files remain safe.

**Acceptance Criteria:**
- [ ] Creates `./deepfix_output/<job_id>/` (or custom path via `--output-dir`) upon job completion.
- [ ] Staged directory contains:
  - `train_fixed.py`: The executable standalone training script incorporating all remediations.
  - `summary_report.md`: Markdown summary detailing initial failure modes, fixes applied, baseline vs. final metrics, MLflow run links, and S3 weight URIs (`s3://<bucket>/<job_id>/model.pt`).
  - `metrics.json`: Machine-readable dictionary of baseline, final evaluation metrics, and `s3_model_uri`.
  - `model_artifacts/`: Downloaded local copy of model checkpoint/weights (or symlinked reference if configured).
- [ ] CLI prints a concise summary banner with relative paths and the S3 weights location.
- [ ] Typecheck and lint pass.

### US-004: Server API Autonomous Fix Job Lifecycle & Webhook Handling
**Description:** As a platform developer, I want the DeepFix Server to expose endpoints to create, track, and complete fix jobs via OpenHands so that the execution loop runs asynchronously and reliably.

**Acceptance Criteria:**
- [ ] `POST /v2/fix` endpoint accepts dataset name, model pointer, target metrics, S3 bucket configurations, and initializes a `FixJob` in SQLite.
- [ ] Server launches `OpenHandsExecutor` asynchronously in the background.
- [ ] `GET /v2/fix/{job_id}` endpoint returns current job status, current iteration, logs, metrics, and S3 status.
- [ ] `POST /webhook/completion` receives `FinalFixReport` (including `s3_weights_uri`) from the agent runtime and transitions status to `COMPLETED` or `FAILED`.
- [ ] Server logs OTEL traces for the autonomous session to MLflow.
- [ ] Unit and integration tests pass for the new endpoints.

### US-005: OpenHands Autonomous Engineer Sandbox Skills & S3 Access
**Description:** As the autonomous fix agent, I need specialized skills in my execution environment so that I can download data from MLflow, train models, push weights to S3, and report back completion.

**Acceptance Criteria:**
- [ ] `mlflow-data-access` skill provides instructions and helpers for downloading dataset dictionaries and baseline artifacts from MLflow.
- [ ] `s3-weights-storage` skill (or CLI utility `push_weights_to_s3.py`) provides clear instructions and scripts for uploading trained model checkpoints directly to the configured S3 bucket path (`s3://<bucket>/<job_id>/...`).
- [ ] `deepfix-communication` skill provides `report_completion.py` inside the container to post final results and S3 URI to `/webhook/completion`.
- [ ] OpenHands agent runs in sandboxed `DockerWorkspace` with injected AWS/S3 environment variables, terminal, file editor, and task tracker tools.
- [ ] Agent prompt includes structured diagnostic findings (e.g. multicollinearity, class weighting, CV strategy) and explicit instructions for S3 weight persistence.
- [ ] End-to-end sandbox execution passes verification test with mock dataset and S3 upload.

### US-006: Cancel / Stop Autonomous Fix Job from CLI
**Description:** As an ML engineer, I want a CLI command to stop an ongoing fix job by its ID so that I can terminate runaway or unnecessary background agent executions on the server immediately.

**Acceptance Criteria:**
- [ ] CLI command `deepfix-sdk cancel <job_id>` (and alias `deepfix-sdk stop <job_id>`) is available.
- [ ] Server endpoint `POST /v2/fix/{job_id}/cancel` halts the background OpenHands sandbox container/executor and transitions the `FixJob` status in SQLite to `CANCELLED`.
- [ ] CLI prints a status confirmation indicating whether the job was successfully terminated.
- [ ] Typecheck and lint pass.

---

## 4. Functional Requirements

- **FR-1:** The CLI must provide the command `deepfix-sdk fix` with arguments:
  - `--dataset` / `-d` (required): Name of the dataset registered in MLflow / DeepFix.
  - `--model` / `-m` (optional): Name or URI of the baseline model artifact.
  - `--target-metric` (optional, default: `accuracy`): Target metric key to optimize.
  - `--target-value` (optional, default: `0.90`): Threshold value to consider the fix successful.
  - `--max-iterations` (optional, default: `5`): Maximum autonomous refinement loops.
  - `--s3-bucket` (optional): Target S3 bucket name for saving fixed model weights.
  - `--api-url` (optional, default: `http://localhost:4141`): DeepFix Server URL.
  - `--output-dir` / `-o` (optional, default: `./deepfix_output`): Path to stage output artifacts.
  - `--poll-interval` (optional, default: `2.0`): Polling frequency in seconds.
- **FR-2:** The CLI must also support the alias flag `--fix` on `deepfix-sdk diagnose` for unified diagnostic + fix invocation.
- **FR-3:** When triggered, the CLI must send a `POST /v2/fix` request to the server, obtain a `job_id`, and transition into streaming/polling mode.
- **FR-4:** The CLI must format streaming events to stdout using Rich console output, printing timestamps, iteration numbers, current loss/metric values, and agent thoughts/status.
- **FR-5:** The DeepFix Server must execute `OpenHandsExecutor` in the background, provisioning an isolated Docker workspace for code execution with S3 credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET`, `AWS_ENDPOINT_URL`).
- **FR-6:** The Server must inject the comprehensive diagnostic findings (from `DiagnosticSystem.arun`) into the OpenHands system prompt, specifically noting issues such as:
  - Structural multicollinearity (e.g. geometric feature redundancy, PPS scores).
  - Skewness and missing EDA preprocessing (e.g. log transformations, scaling).
  - Class imbalance and metric optimization (e.g. class weighting, recall prioritization).
  - Small sample size mitigation (e.g. stratified k-fold cross-validation).
- **FR-7:** The OpenHands agent must upload the winning model weights to `s3://<bucket>/<job_id>/weights/` upon hitting target metrics or exhausting iterations.
- **FR-8:** Upon completion or error, the server must persist the `FinalFixReport` (including `s3_weights_uri`) and make the output artifacts available for download.
- **FR-9:** The CLI must pull the final artifacts from the server / MLflow / S3 store, write them into `./deepfix_output/<job_id>/`, and exit with code `0` on success or non-zero on failure.
- **FR-10:** The CLI must provide `deepfix-sdk cancel <job_id>` (alias `deepfix-sdk stop <job_id>`) calling `POST /v2/fix/{job_id}/cancel` on the DeepFix Server, terminating the underlying OpenHands session and updating SQLite job status to `CANCELLED`.
- **FR-11:** In dataset-only fix mode (when `--model` is omitted), the autonomous fix loop must focus strictly on **dataset partitioning and preprocessing repair** (eliminating train-test sample leakage, constructing stratified splits preserving minority classes, addressing multicollinear redundancy via feature selection/transforms), re-running dataset integrity validation checks, and saving the improved partitioned dataset to S3.

---

## 5. Non-Goals (Out of Scope)

- **Direct In-Place Workspace Editing:** The CLI will NOT automatically modify or overwrite existing local user scripts or git branches; all outputs must remain safely staged in `./deepfix_output/`.
- **Interactive TUI / Textual App:** No full-screen interactive curses/Textual dashboard in this version; output uses standard stdout/stderr streams compatible with CI/CD runners.
- **Local Bare-Metal Agent Subprocess:** OpenHands execution will NOT run directly uncontained on the user's host OS; execution is delegated to the server and contained Docker workspaces.
- **Human-in-the-Loop Intermediary Prompts:** The fix process runs fully autonomously until completion or hitting `max_iterations` without prompting the user for intermediate code approvals.
- **Direct S3 Bucket Administration:** The system assumes the S3 bucket already exists; automatic bucket provisioning or IAM role management is out of scope.

---

## 6. Technical Considerations

- **Dependencies:**
  - Client: `typer`, `rich`, `pydantic`, `httpx` or `requests`, `mlflow`, `boto3` / `s3fs` (optional).
  - Server: `fastapi`, `sqlmodel`, `openhands-sdk`, `openhands-tools`, `openhands-workspace`, `docker`, `boto3`.
- **S3 Storage & Credentials:**
  - S3 environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL`, `DEEPFIX_S3_BUCKET`) passed securely to the OpenHands Docker container.
  - S3-compatible endpoints supported (AWS S3, MinIO, Cloudflare R2, Ceph).
- **Concurrency & Sandboxing:**
  - Each fix job receives a dedicated Docker container workspace (`ghcr.io/openhands/agent-server:latest-python`).
  - Container communication with the server uses a bridge network or host gateway (`host.docker.internal`).
- **Telemetry & Observability:**
  - OTEL environment variables (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`) are configured per job to export OpenHands agent traces into MLflow.
- **Direct S3 Retrieval:** The CLI downloads model weights, checkpoints, and partitioned datasets directly from S3 using configured local/environment AWS S3 credentials (`boto3`), consistent with `TabularDataset.from_s3()` and `push_model_to_s3()` SDK conventions.

---

## 7. Success Metrics

- **Fix Automation Success Rate:** > 80% of fix jobs successfully produce an improved model exceeding baseline metrics or resolving identified diagnostic warnings on benchmark datasets (e.g., Wisconsin Breast Cancer).
- **S3 Persistence Reliability:** 100% of successful fix runs upload valid model weight artifacts to S3 with verifiable checksums and canonical URIs.
- **Zero Host Workspace Pollution:** 100% of generated candidate code and files isolated in sandbox and delivered strictly to `./deepfix_output/`.
- **CLI Latency & Responsiveness:** Time from CLI invocation to initial server job dispatch < 1.5 seconds.
- **CI/CD Compatibility:** CLI reliably exits with status `0` on met target metrics and status `1` on failed runs with machine-readable outputs.

---

## 8. Open Questions

*None — all open design, architectural, and operational questions have been resolved.*


