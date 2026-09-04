# PRD: Autonomous Fix CLI Evaluation Harness & LLM-as-a-Judge

## 1. Introduction / Overview

Following the implementation of the Autonomous Fix System CLI ([prd-autonomous-fix-cli.md](file:///home/fadelco/workspace/deepfix/tasks/prd-autonomous-fix-cli.md)), DeepFix can autonomously diagnose and repair ML pipelines using an OpenHands agent running inside sandboxed Docker containers. However, evaluating the reliability, correctness, and safety of these autonomous fix rollouts requires systematic, real-world benchmarking.

The **Autonomous Fix CLI Evaluation Harness** provides an automated benchmarking and evaluation framework for DeepFix. It ingests configurable benchmark manifests (covering tabular, multicollinearity, data leakage, and class imbalance scenarios), executes the autonomous fix CLI (`deepfix-sdk fix`), captures comprehensive rollout traces (CLI streaming events, server-side OpenHands trajectory logs, and staged deliverables in `./deepfix_output/`), and applies an **LLM-as-a-Judge** scoring engine across a standardized multi-dimensional rubric. All rollout traces, judge evaluations, and metrics are automatically logged to MLflow and rendered into Rich terminal dashboards and persistent `./deepfix_eval_output/` reports.

---

## 2. Goals

- **Manifest-Driven Benchmark Orchestration:** Define and load benchmark suites via declarative YAML/JSON manifests specifying datasets, defect profiles, baseline models, target metrics, and expected remediation outcomes.
- **End-to-End Rollout Trace Capture:** Capture complete agent rollouts across both client and server boundaries (CLI Rich stdout streams, OpenHands reasoning thoughts and bash/editor tool actions, MLflow metric runs, and final staged artifacts).
- **Multi-Dimensional LLM-as-a-Judge:** Assess agent rollouts using a structured evaluation rubric covering Root-Cause Remediation Soundness, Trajectory Efficiency, Tool/Skill Usage Compliance, Code Quality, and Metric Target Attainment.
- **Unified MLflow Observability:** Export all rollout trajectories, prompt interactions, judge rubrics, and comparative diffs into MLflow Evaluation Experiments for experiment tracking and version comparison.
- **Rich Terminal UX & Report Generation:** Deliver real-time visual progress and summary tables via Rich console formatting, alongside machine-readable JSON and human-readable Markdown evaluation reports staged in `./deepfix_eval_output/`.
- **Benchmark Regression Detection:** Enable side-by-side performance comparisons across agent prompt revisions, underlying LLM models, and server releases to detect behavioral regressions.

---

## 3. User Stories

### US-001: Benchmark Suite Manifest Schema & Loader
**Description:** As an ML platform engineer, I want to define benchmark test suites in declarative YAML/JSON manifests so that I can systematically configure evaluation datasets, defects, and expected fix criteria.

**Acceptance Criteria:**
- [ ] Pydantic schema `BenchmarkManifest` and `BenchmarkTestCase` models validate test configurations (id, name, dataset_uri, model_uri, target_metric, target_value, max_iterations, expected_defects, timeout_seconds).
- [ ] `expected_defects` is a list of diagnostic defect tags (e.g., `MULTICOLLINEARITY`, `LEAKAGE`, `CLASS_IMBALANCE`). The harness verifies addressed defects by checking `FinalFixReport.applied_fixes` against the expected set, and the judge rubric dimension "Remediation Soundness" penalizes any unaddressed expected defects.
- [ ] Loader supports single test case files, directory-based test suites, and filtering by tags (e.g., `tag: tabular`, `tag: leakage`, `tag: smoke`).
- [ ] Ships with canonical benchmark manifests (located in `packages/deepfix-sdk/benchmarks/manifests/`):
  - `breast_cancer_multicollinearity.yaml`: Wisconsin Breast Cancer dataset with known geometric multicollinearity and small-sample validation needs.
  - `breast_cancer_dataset_only.yaml`: Wisconsin Breast Cancer dataset-only fix mode (no baseline model), focusing on dataset partitioning and preprocessing repair.
  - `synthetic_leakage.yaml`: Tabular dataset with target-leaking synthetic features.
  - `imbalanced_credit.yaml`: Severe class imbalance requiring cost-sensitive weighting or stratified sampling.
- [ ] Typecheck and lint pass across `deepfix-sdk`.

### US-002: Autonomous Fix CLI Execution Runner & Trace Collector
**Description:** As an evaluation engineer, I want the harness to invoke the autonomous fix CLI for each benchmark test case and harvest comprehensive traces across client and server boundaries.

**Acceptance Criteria:**
- [ ] Harness executes `deepfix-sdk fix` per benchmark case with configurable timeout and isolation.
- [ ] Captures client-side CLI stdout/stderr logs and structured streaming events (phases, iterations, intermediate metrics).
- [ ] Retrieves OpenHands agent trajectory (thoughts, tool calls, bash commands, file edits) via the MLflow Python SDK, querying the MLflow tracking server for OTEL traces and spans logged during the fix session.
- [ ] Harvests staged output artifacts from `./deepfix_output/<job_id>/` (`train_fixed.py`, `summary_report.md`, `metrics.json`, `s3_model_uri`).
- [ ] Assembles all collected artifacts into an in-memory `RolloutTrace` data structure.
- [ ] Typecheck and lint pass.

### US-003: LLM-as-a-Judge Multi-Dimensional Evaluation Engine
**Description:** As an ML researcher, I want an LLM-as-a-judge to inspect the rollout traces against a rigorous evaluation rubric so that I can grade the agent's reasoning, code fixes, and safety.

**Acceptance Criteria:**
- [ ] Judge engine invokes a configured LLM (via LiteLLM / OpenHands LLM client) with a specialized evaluation system prompt and the collected `RolloutTrace`.
- [ ] Evaluates five distinct rubric dimensions (scored 1 to 5 with detailed qualitative feedback):
  1. **Remediation Soundness:** Did the code directly resolve the flagged diagnostic defect without shortcuts or spurious transforms?
  2. **Trajectory Efficiency:** Did the agent plan effectively without thrashing, tool hallucinations, or redundant loops?
  3. **Tool & Skill Compliance:** Did the agent correctly use S3 upload scripts, MLflow data access, and `report_completion.py`?
  4. **Code Quality & Reproducibility:** Is `train_fixed.py` clean, robust, executable, and properly commented?
  5. **Metric Attainment & Leakage Safety:** Did the model reach the target metric legitimately without introducing train-test leakage?
- [ ] Returns a structured Pydantic `JudgeVerdict` containing dimension scores, overall score (0–100), boolean `pass_status`, identified anti-patterns, and executive rationale.
- [ ] Unit tests pass with mock LLM judge responses.

### US-004: MLflow Trace & Benchmark Experiment Logging
**Description:** As an MLOps engineer, I want the evaluation harness to log all benchmark test runs, agent traces, and judge verdicts to MLflow so that our team can track agent performance trends over time.

**Acceptance Criteria:**
- [ ] Harness logs each benchmark suite execution under a designated MLflow experiment (default: `deepfix-fix-benchmarks`).
- [ ] Test cases are logged as individual MLflow runs tagged with test ID, agent model, prompt version, and dataset name.
- [ ] Logs judge dimension scores and final metrics as standard MLflow metrics.
- [ ] Logs rollout artifacts to MLflow: raw agent trajectory JSON, client CLI logs, `train_fixed.py`, `summary_report.md`, and the full judge evaluation report.
- [ ] Typecheck and lint pass.

### US-005: Rich CLI Evaluation Interface & Multi-Format Reporting
**Description:** As a developer running benchmarks locally or in CI, I want a dedicated CLI command with live terminal progress, summary tables, and persistent reports so that I can immediately understand evaluation results.

**Acceptance Criteria:**
- [ ] CLI command `deepfix-sdk eval run [OPTIONS]` is available with parameters:
  - `--manifest` / `-m` (path to benchmark YAML/directory).
  - `--suite` / `-s` (suite name or glob pattern).
  - `--judge-model` (LLM model for evaluation; defaults to the `OPENHANDS_LLM_MODEL` environment variable for consistency with the autonomous fix agent).
  - `--api-url` (DeepFix Server URL, default: `http://localhost:4141`).
  - `--output-dir` / `-o` (directory for reports, default: `./deepfix_eval_output`).
  - `--mlflow-experiment` (MLflow experiment name).
  - `--fail-under` (minimum overall score threshold to exit with code 0, default: `70`).
  - `--concurrency` (number of benchmark cases to run in parallel, default: `1`).
- [ ] Rich console renders live progress across test cases and prints a formatted summary table (Test ID, Dataset, Target Metric, Achieved Metric, Judge Score, Pass/Fail, Duration).
- [ ] Generates persistent output in `./deepfix_eval_output/<eval_run_id>/` (where `eval_run_id` follows the format `eval_{timestamp}_{uuid[:8]}`, matching the existing `fix_` ID convention):
  - `eval_summary.md`: Human-readable markdown executive summary.
  - `eval_results.json`: Complete machine-readable results dictionary.
  - `rollouts/`: Subdirectory containing raw trajectory traces per benchmark.
- [ ] Exits with code `0` if all tests pass and meet `--fail-under`, or non-zero on failure.

### US-006: Benchmark Run Comparison & Regression Detection
**Description:** As a developer tuning the autonomous fix prompt or server skills, I want to compare a new evaluation run against a baseline run so that I can instantly detect regressions or metric improvements.

**Acceptance Criteria:**
- [ ] CLI command `deepfix-sdk eval compare --baseline <run_id_or_path> --candidate <run_id_or_path>` is available.
- [ ] Compares judge dimension scores, success rates, and iteration counts across runs.
- [ ] Displays a side-by-side Rich diff table highlighting score deltas (green for improvement, red for regression).
- [ ] Flags any test case where status flipped from PASS to FAIL as a regression.
- [ ] Typecheck and lint pass.

---

## 4. Functional Requirements

- **FR-1:** The evaluation harness must support benchmark definitions in YAML/JSON format adhering to the `BenchmarkManifest` schema, specifying test case metadata, dataset pointer (local path, MLflow URI, or S3 URI), baseline model pointer (optional), target metric name and threshold, expected diagnostic defects, and timeout limits.
- **FR-2:** The CLI must expose `deepfix-sdk eval run` and `deepfix-sdk eval compare` commands registered as a Typer sub-application (`eval_app = typer.Typer(name="eval")`) added to the main `deepfix_sdk.cli` app via `app.add_typer(eval_app)`. The `eval` sub-app and all benchmark dependencies are installed via the optional `deepfix-sdk[benchmark]` extra.
- **FR-3:** For each benchmark case, the harness must invoke the autonomous fix workflow either via the CLI client runner or directly via DeepFix Server API `POST /v2/fix`, monitoring progress until completion, timeout, or failure.
- **FR-4:** The trace collector must capture:
  1. CLI terminal stream events and stdout/stderr.
  2. OpenHands agent trajectory steps (thoughts, tool calls, bash execution logs, file diffs) retrieved from MLflow via the MLflow Python SDK (`mlflow.client.MlflowClient`), querying OTEL traces and spans exported by the DeepFix Server during the fix session.
  3. Intermediate and final MLflow run metrics recorded by the agent.
  4. Staged outputs in `./deepfix_output/<job_id>/` (`train_fixed.py`, `summary_report.md`, `metrics.json`, `s3_model_uri`).
- **FR-5:** The LLM-as-a-Judge engine must construct a structured evaluation prompt that includes:
  1. The benchmark task specification and expected defect remediations.
  2. The initial diagnostic findings injected into the fix agent.
  3. The full chronological agent trajectory (actions, tool calls, errors encountered, recovery attempts).
  4. The candidate `train_fixed.py` source code and final achieved metrics.
- **FR-6:** The LLM-as-a-Judge engine must score the rollout on five 5-point dimensions:
  1. **Remediation Soundness** (Weight: 25%)
  2. **Trajectory Efficiency** (Weight: 20%)
  3. **Tool & Skill Usage Compliance** (Weight: 15%)
  4. **Code Quality & Reproducibility** (Weight: 20%)
  5. **Metric Attainment & Safety** (Weight: 20%)
- **FR-7:** The judge engine must output a strictly validated JSON payload matching the `JudgeVerdict` schema, containing dimension scores, normalized 0–100 score, pass/fail flag, qualitative critique, detected failure modes (e.g. `DATA_LEAKAGE`, `SYNTAX_ERROR`, `THRASHING`, `METRIC_GAMING`), and recommendations.
- **FR-8:** All evaluation runs, rollout trajectories, judge metrics, and artifact diffs must be logged to an MLflow experiment configured via `--mlflow-experiment` (default: `deepfix-fix-benchmarks`).
- **FR-9:** The harness must render real-time Rich progress indicators during execution and output a terminal results table with colored status indicators (green/yellow/red).
- **FR-10:** The harness must write a complete evaluation artifact bundle into `./deepfix_eval_output/<eval_run_id>/` containing `eval_summary.md`, `eval_results.json`, and individual rollout traces.
- **FR-11:** The harness must handle benchmark job failures, container crashes, and timeouts gracefully without halting the entire benchmark suite, recording a failed verdict and continuing to subsequent test cases.
- **FR-12:** The regression comparator (`deepfix-sdk eval compare`) must calculate delta metrics and generate a Rich differential comparison table between two evaluation run JSON files or MLflow run IDs.
- **FR-13:** The `deepfix-sdk` package must include at least four pre-packaged seed benchmark manifests under `packages/deepfix-sdk/benchmarks/manifests/`:
  - `tabular_breast_cancer.yaml`
  - `breast_cancer_dataset_only.yaml`
  - `synthetic_leakage.yaml`
  - `class_imbalance.yaml`

---

## 5. Non-Goals (Out of Scope)

- **Online Human Grading Intermediaries:** The evaluation harness is fully automated; human-in-the-loop annotation interfaces or manual review queues are out of scope.
- **Model Training During Evaluation:** The LLM-as-a-judge inspects the generated artifacts, metrics diffs, and execution traces; the harness does not independently re-fit the model from scratch during the judging step.
- **Replacing Standard Unit/Integration Tests:** The evaluation harness is designed for agent behavioral benchmarking and real-world defect remediation; standard unit/integration tests remain under `tests/`.
- **Arbitrary Code Sandbox Execution by the Judge:** The judge operates strictly as an LLM evaluation agent; it does not execute untrusted candidate code outside of the existing OpenHands Docker sandbox.
- **Support for Non-DeepFix Fix Systems:** The harness is tailored specifically to DeepFix CLI, OpenHands trajectories, and DeepFix Server API contracts.

---

## 6. Technical Considerations

- **Package & Module Layout:**
  - All evaluation harness code lives in the `deepfix-sdk` package, installed via the optional extra `pip install deepfix-sdk[benchmark]`.
  - Pydantic models (`BenchmarkManifest`, `BenchmarkTestCase`, `RolloutTrace`, `JudgeVerdict`) → `packages/deepfix-sdk/src/deepfix_sdk/eval/models.py`.
  - Harness engine, runner, and judge modules → `packages/deepfix-sdk/src/deepfix_sdk/eval/`.
  - CLI commands → `packages/deepfix-sdk/src/deepfix_sdk/eval/cli.py` (registered as a Typer sub-app).
  - Judge prompt templates → `packages/deepfix-sdk/src/deepfix_sdk/eval/prompts/` (versioned as string constants in a dedicated module; prompt version tracked as an MLflow run tag `prompt_version`).
  - Seed benchmark manifests → `packages/deepfix-sdk/benchmarks/manifests/`.
- **Dependencies (in `[benchmark]` extra):**
  - Client / CLI: `typer`, `rich`, `pydantic`, `pyyaml`, `httpx`, `mlflow`.
  - LLM Evaluation: `litellm` (or existing DeepFix LLM client / LiteLLM proxy), allowing plug-and-play evaluation across OpenAI, Anthropic, Gemini, and local Ollama models.
- **Trace Ingestion Architecture:**
  - Client side: Captures Rich output via `rich.console.Console(record=True)` or subprocess stdout pipes.
  - Server side: Retrieves OpenHands agent trajectory data (OTEL traces and spans) from the MLflow tracking server via the MLflow Python SDK (`mlflow.client.MlflowClient`). The DeepFix Server already exports OTEL traces to MLflow during fix sessions via configured `OTEL_EXPORTER_OTLP_ENDPOINT`.
- **Telemetry & MLflow Synchronization:**
  - Evaluation runs log directly to MLflow using `mlflow.start_run(nested=True)` for sub-benchmark runs.
  - Trajectory JSON files and code diffs uploaded as MLflow artifacts.
- **Resilience & Rate Limiting:**
  - LLM judge calls configured with exponential backoff and retry logic.
  - Sequential or controlled concurrent benchmark execution (configurable via `--concurrency`, default: `1` to avoid Docker container port conflicts and LLM rate limit exhaustion).
- **Execution Environment:**
  - Seamlessly runs in local developer environments (WSL/Linux/macOS) and CI/CD runners with access to Docker and the DeepFix server.

---

## 7. Success Metrics

- **Harness Execution Reliability:** > 99% completed benchmark runs without harness-level runtime crashes or unhandled exceptions.
- **Harness Overhead:** Evaluation harness overhead (manifest loading, trace ingestion, judge invocation, report generation) < 30 seconds per benchmark case, excluding agent execution time.
- **Judge Scoring Consistency:** Inter-evaluator score variance < 5% across identical rollout traces on deterministic temperature settings (`temperature=0.0`).
- **Trace Capture Completeness:** 100% of benchmark runs capture both CLI execution output, MLflow-sourced OpenHands trajectory, and staged deliverables.
- **Actionable Defect Detection:** Successfully detects and penalizes simulated failure modes (e.g. data leakage, failing unit tests, hallucinated libraries) with a score < 50% and specific failure mode tags.

---

## 8. Open Questions

- *None — all foundational questions regarding manifest loading, trace collection, judge rubric dimensions, and MLflow integration have been resolved.*
