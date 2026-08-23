# Autonomous Fix System — Implementation Plan (Lean Architecture)

> **Goal**: Extend DeepFix from diagnosis-only to autonomous fix generation and iterative improvement using a **Lean Agent-Centric Architecture**. OpenHands handles all iterations, metrics checking, and fix application internally, while DeepFix Server simply provisions the sandbox, provides the diagnostic context, and waits for a webhook callback.

---

## Phase 1 — Foundation (3 Streams in Parallel)

### Stream 1: Core Domain Models
**Package**: `packages/deepfix-core/src/deepfix_core/models/`
**Assignable to**: 1 subagent
**Depends on**: Nothing

Create new file `packages/deepfix-core/src/deepfix_core/models/fixes.py` with simplified tracking models:

| Model | Fields | Purpose |
|---|---|---|
| `FixJobStatus` (StrEnum) | `PENDING`, `IN_PROGRESS`, `COMPLETED`, `FAILED` | Job lifecycle status |
| `FixJob` (BaseModel) | `job_id: str`, `status: FixJobStatus`, `started_at: datetime`, `baseline_metrics: dict` | Track the overall autonomous run |
| `FinalFixReport` (BaseModel) | `success: bool`, `final_metrics: dict`, `applied_fixes: list[str]`, `run_id: str` | Payload submitted by OpenHands to webhook |

Extend `APIResponse` in `api.py`:
- Add optional field `fix_report: Optional[FinalFixReport] = None`
- Add optional field `job_id: Optional[str] = None`

**Deliverables**:
- [ ] `fixes.py` with models above
- [ ] Updated `__init__.py` with exports
- [ ] Updated `APIResponse`

---

### Stream 2: OpenHands Environment Skills
**Package**: `packages/deepfix-kb/`
**Assignable to**: 1 subagent
**Depends on**: Nothing

Create specific OpenHands AgentSkills under `packages/deepfix-kb/src/deepfix_kb/skills/` to empower the agent to run autonomously:

#### 1. `mlflow-data-access` Skill
Provides documentation and snippets to the agent on how to properly download the dataset from the MLflow artifact store (`mlflow.artifacts.download_artifacts`) and load it via `datasets.load_from_disk`.

#### 2. `deepfix-communication` Skill
Provides a Python script (`report_completion.py`) that the agent is instructed to run when it decides it has finished iterating. 
- The script should accept arguments like `--success`, `--final-run-id`, and `--fixes-summary`.
- It will make a `POST` request to the DeepFix Server's webhook endpoint (`/webhook/completion`) to signal that the agent has finished.

**Deliverables**:
- [ ] `skills/mlflow-data-access/SKILL.md`
- [ ] `skills/deepfix-communication/SKILL.md` + `report_completion.py`

---

### Stream 3: Infrastructure, Config, Test Env & OTEL Observability
**Location**: Root project files + deepfix-server config
**Assignable to**: 1 subagent
**Depends on**: Nothing

1. **Update `.env.example`**:
   ```env
   # === Autonomous Fix System ===
   OPENHANDS_LLM_API_KEY=              # Provided at runtime
   OPENHANDS_LLM_MODEL=anthropic/claude-sonnet-4-5-20250929
   OPENHANDS_DOCKER_IMAGE=ghcr.io/openhands/agent-server:latest-python
   OPENHANDS_SANDBOX_PORT=8010         

   # === OpenHands OTEL Observability (MLflow Tracing) ===
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5000
   OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id=0
   OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
   ```

2. **Update `test.env` and `test.env.example`**: 
   Add DeepFix webhook and fix endpoint URLs (`DEEPFIX_TEST_FIX_API_URL`, `DEEPFIX_TEST_WEBHOOK_URL`).

3. **Create config model** in `packages/deepfix-server/src/deepfix_server/config.py`: 
   `AutonomousFixConfig` with fields for OpenHands settings, OTEL parameters, and `setup_otel_environment()`.

4. **Add OpenHands dependencies** to `packages/deepfix-server/pyproject.toml`:
   - `openhands-sdk`
   - `openhands-tools`
   - `openhands-workspace`

**Deliverables**:
- [ ] Updated `.env.example`, `test.env`, `test.env.example`
- [ ] `AutonomousFixConfig` model
- [ ] Updated `pyproject.toml` dependencies

---

## Phase 2 — Core Components (2 Streams in Parallel)

### Stream 4: Sandbox Executor & System Prompt Builder
**Package**: `packages/deepfix-server/src/deepfix_server/`
**Assignable to**: 1 subagent
**Depends on**: Streams 1, 2, 3

Create `openhands_executor.py`:
- Fetches diagnostic findings from SQLite.
- Provisions `DockerWorkspace` and exports OTEL env vars.
- Constructs a comprehensive **System Prompt** for the OpenHands agent, injecting the findings, baseline metrics, and instructions to use the `deepfix-communication` Python script when done.
- Launches the agent as a background task.

**Deliverables**:
- [ ] `openhands_executor.py` implementation

---

### Stream 5: API & Webhook Extensions
**Packages**: `deepfix-server` + `deepfix-sdk`
**Assignable to**: 1 subagent
**Depends on**: Streams 1, 4

1. **Server API (`deepfix-server/api.py`)**:
   - `POST /v2/fix`: Accepts dataset pointers, saves a new `FixJob` to SQLite, calls `openhands_executor` in the background, and returns the `job_id`.
   - `POST /webhook/completion`: Receives `FinalFixReport` payload from the agent's Python script, updates SQLite, and marks the job complete.
2. **SDK (`deepfix-sdk/client.py`)**:
   - Implement `diagnose_and_fix()` which triggers `/v2/fix` and polls until completion.
3. **CLI (`deepfix-sdk/cli.py`)**: Add `--fix` flag.

**Deliverables**:
- [ ] `api.py` with `/v2/fix` and `/webhook/completion`
- [ ] `client.py` with `diagnose_and_fix()`
- [ ] `cli.py` with `--fix`

---

## Execution Schedule

| Phase | Streams | Parallelism | Estimated Effort | Blocked By |
|---|---|---|---|---|
| **Phase 1** | Streams 1, 2, 3 | **3 parallel** | Small | Nothing |
| **Phase 2** | Streams 4, 5 | **2 parallel** | Medium | Streams 1, 2, 3 |
| **Phase 3** | Stream 6 | 1 agent | Medium | All above |
