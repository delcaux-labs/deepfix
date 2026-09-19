# PRD: Migrate Knowledge Base Tools to Deepfix-Server

## Introduction

Currently, external search and knowledge retrieval capabilities in DeepFix reside in a separate package (`packages/deepfix-kb`), which relies on custom wrappers (`KnowledgeBridge`, custom async tool executors) and passes pre-fetched knowledge as raw strings to agents. 

This project deprecates and deletes `packages/deepfix-kb`, consolidating agent tools and knowledge management directly inside `packages/deepfix-server` using Agno's native ecosystem:
1. **Agno Built-in Search Tools**: Provide `DuckDuckGoTools` (default, zero API key) and `TavilyTools` (for high-fidelity web search when `TAVILY_API_KEY` is provided) directly attached to reasoning agents.

3. **Workspace Simplification**: Remove `packages/deepfix-kb` from workspace dependencies and clean up `DiagnosticSystem` / `AnalysisWorkflow` to drop legacy `KnowledgeBridge` references.

---

## Goals

- Eliminate maintenance overhead and inter-package coupling caused by `packages/deepfix-kb`.
- Enable Agno agents (specifically `CrossArtifactReasoningAgent`) to autonomously use Agno built-in tools (`DuckDuckGoTools`, `TavilyTools`) during reasoning.
- Provide embedded domain knowledge retrieval inside `deepfix-server` using Agno's `AgentKnowledge`.
- Modernize `deepfix-server` configuration so tools and knowledge are loaded cleanly from settings/environment variables without manual `KnowledgeBridge` plumbing.
- Maintain clean package boundaries and verify all type checks, linters, and tests pass after removal of `deepfix-kb`.

---

## User Stories

### US-001: Configure Agno Search Tools in Deepfix-Server
**Description:** As a server developer, I want built-in Agno search tools (`DuckDuckGoTools`, `TavilyTools`) configured in `deepfix-server` so that agents can access web search without custom wrapper code.

**Acceptance Criteria:**
- [x] Add `duckduckgo-search` and `tavily-python` to `packages/deepfix-server/pyproject.toml` dependencies.
- [x] Implement a tool factory/manager (e.g. `deepfix_server.tools.search` or within config) that instantiates `DuckDuckGoTools` by default and `TavilyTools` if `TAVILY_API_KEY` is set or configured.
- [x] Settings in `deepfix_server/config.py` expose search configuration (e.g. `search_provider: Literal["duckduckgo", "tavily", "none"] = "duckduckgo"`).
- [x] Lint and typecheck pass (`uv run ruff check` and `uv run mypy` / `uv run pyright`).


### US-003: Equip CrossArtifactReasoningAgent with Agno Tools & Knowledge
**Description:** As an ML diagnostic user, I want the `CrossArtifactReasoningAgent` to autonomously query web search and  when diagnosing cross-artifact anomalies.

**Acceptance Criteria:**
- [x] Attach resolved search tools (`DuckDuckGoTools` / `TavilyTools`) to `CrossArtifactReasoningAgent` via Agno's `tools` parameter.
- [x] Update reasoning prompts/instructions to guide the agent on when and how to call search and knowledge tools.

### US-004: Refactor DiagnosticSystem and AnalysisWorkflow Interfaces
**Description:** As an SDK or API consumer, I want `DiagnosticSystem` and `AnalysisWorkflow` to automatically configure tools and knowledge from settings without requiring a legacy `KnowledgeBridge` parameter.

**Acceptance Criteria:**
- [x] Remove `knowledge_bridge` parameter from `DiagnosticSystem.__init__` and `AnalysisWorkflow.__init__`.
- [x] Remove `prefetch_knowledge` function from `reasoning.py` (or replace with direct agent tool delegation).
- [x] Initialize tools and knowledge internally based on `Settings` / `LLMConfig`.
- [x] Existing server entrypoints and CLI workflows run without deprecated parameters.

### US-005: Remove Deepfix-KB Package and Clean Workspace Dependencies
**Description:** As a repository maintainer, I want `packages/deepfix-kb` completely deleted and removed from the workspace configuration so that the codebase remains lean and unified.

**Acceptance Criteria:**
- [x] Delete `packages/deepfix-kb` directory from workspace.
- [x] Remove `"packages/deepfix-kb"` from root `pyproject.toml` `members` list.
- [x] Remove `deepfix-kb` from `packages/deepfix-server/pyproject.toml` dependencies and `tool.uv.sources`.
- [x] Remove any dangling imports of `deepfix_kb` across `packages/deepfix-server`.
- [x] `uv sync` succeeds without errors.

### US-006: End-to-End Validation of Migrated Reasoning & Diagnosis Workflow
**Description:** As a developer/QA, I want to run end-to-end integration tests against a live `deepfix-server` instance to verify that client diagnosis requests complete successfully with native Agno tools.

**Acceptance Criteria:**
- [x] Start local server using `uv run --env-file .env deepfix-server launch`.
- [x] Run tabular workflow E2E test via `uv run --env-file test.env pytest tests/test_tabular_workflow_e2e.py::TestTabularWorkflowE2E::test_tabular_diagnosis_workflow -s -v`.
- [x] Verify server logs indicate clean startup without `deepfix-kb`, proper tool/knowledge initialization, and successful reasoning agent execution.
- [x] Ensure `APIResponse` returned to the client contains summary and agent results without errors.

---

## Functional Requirements

- **FR-1**: The system must provide configurable web search tools (`DuckDuckGoTools` or `TavilyTools`) without requiring custom HTTP query wrappers.
- **FR-2**: When `TAVILY_API_KEY` is present and search provider is set to Tavily, the system must utilize Agno's `TavilyTools`.
- **FR-3**: When `TAVILY_API_KEY` is absent or provider is set to DuckDuckGo, the system must default to `DuckDuckGoTools` requiring zero credentials.
- **FR-4**: The `CrossArtifactReasoningAgent` must have direct access to configured search tools and knowledge base to execute dynamic queries.
- **FR-5**: `DiagnosticSystem` and `AnalysisWorkflow` must instantiate and run without requiring external `KnowledgeBridge` instances.
- **FR-6**: The workspace must build, lock, and typecheck cleanly after the deletion of `packages/deepfix-kb`.
- **FR-7**: The full tabular diagnosis E2E test suite must pass against a running server.

---

## Non-Goals

- Migrating legacy Perplexity custom API wrappers (`perplexity_client.py`); research and web search are handled natively by Agno's toolkits.
- Updating container deployment files (`Dockerfile`, `docker-compose.dev.yml`) and documentation site (`docs-mint/`) in this iteration (scoped for a dedicated DevOps/docs pass per user choice 4B).

---

## Technical Considerations
- **Tool Calling Support**: Tool calling requires LLM models that support function calling / tools. When tools are enabled, model temperature and JSON mode schema settings should remain consistent with Agno agent requirements.


---

## Testing & Verification Strategy

The primary validation gate for this migration is End-to-End (E2E) testing against a live local server instance, ensuring that server tool integration, agent reasoning, and SDK client communication work end-to-end.

### 1. Server Launch (Terminal 1)
Run the server loaded with active environment settings:
```bash
uv run --env-file .env deepfix-server launch
```
*Verification Check:* Confirm server logs show:
- FastAPI startup on the configured host/port (e.g. `http://0.0.0.0:8000`).
- No import warnings or references to `deepfix_kb`.
- Successful initialization of search tools.

### 2. E2E Test Execution (Terminal 2)
Execute the end-to-end tabular workflow test using `test.env`:
```bash
uv run --env-file test.env pytest tests/test_tabular_workflow_e2e.py::TestTabularWorkflowE2E::test_tabular_diagnosis_workflow -s -v
```
*Expected Outcomes:*
1. **Client Initialization & Ingestion**: `DeepFixClient` connects to `DEEPFIX_TEST_API_URL` and posts train/validation tabular datasets.
2. **Model Training & Diagnostics**: Baseline model is fitted, deepchecks suites run, and the payload is sent to the server diagnosis endpoint.
3. **Agent Reasoning Execution**: `CrossArtifactReasoningAgent` on the server receives diagnostic signals and executes reasoning with native Agno tools (`DuckDuckGoTools`/`TavilyTools`) .
4. **Response Validation**: `check_response` verifies the response is a valid `APIResponse` containing a non-empty summary and agent results.
5. **Output Artifact**: Test saves diagnosis output to `breast_cancer_classification_e2e.txt`.

---

## Success Metrics

- `packages/deepfix-kb` directory completely deleted with zero workspace lock conflicts.
- 100% replacement of `KnowledgeBridge` usages in `deepfix-server` with native Agno tools.
- `uv sync` succeeds without errors.
- **E2E Validation**: `uv run --env-file test.env pytest tests/test_tabular_workflow_e2e.py::TestTabularWorkflowE2E::test_tabular_diagnosis_workflow` completes with exit code 0 against a live local server.
- Agents retrieve external web information autonomously without custom retrieval plumbing.

---

## Open Questions / Resolved Decisions

- **Embedding Model Configuration (Resolved)**: An OpenAI-compatible embedder configuration (`EmbedderConfig`) is defined in `deepfix_server.config` and loaded via environment variables (`DEEPFIX_EMBEDDER_API_KEY`, `DEEPFIX_EMBEDDER_BASE_URL`, `DEEPFIX_EMBEDDER_MODEL_NAME`, `DEEPFIX_EMBEDDER_DIMENSIONS`), falling back to LLM settings if not explicitly specified.
- **Initial Knowledge Ingestion (Resolved)**: The vector store will be left empty on initialization without static document seeding. It is designed to be populated dynamically by a scheduled web scraping agent.



