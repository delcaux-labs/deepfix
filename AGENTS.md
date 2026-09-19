# Agent Guidelines & Engineering Standards

> [!NOTE]
> **Read the README for System Architecture & Context**:
> Whenever you need information on system architecture, workspace packages, deployment, or usage examples, always consult and read [`README.md`](README.md).

## 1. Tooling & Command Execution with `uv`

Whenever `uv` is installed, always use it. Never invoke bare `python` or bare `pytest`.

- **Running Python Scripts**:
  ```bash
  uv run script.py
  ```
- **Running Tests**:
  Always pass `--env-file test.env` to load test environment variables:
  ```bash
  uv run --env-file test.env pytest tests/test_tabular_workflow_e2e.py -v -s
  ```
  To run a specific test by name:
  ```bash
  uv run --env-file test.env pytest tests/test_tabular_workflow_e2e.py::TestTabularWorkflowE2E::test_tabular_diagnosis_workflow -v -s
  ```
- **Code Quality & Formatting**:
  ```bash
  uv run ruff check . --fix
  uv run ruff format .
  ```
- **Launching Local Server**:
  ```bash
  uv run --env-file .env deepfix-server launch
  ```
- **Dependency Management**:
  ```bash
  uv sync
  uv add <package-name>
  ```

---

## 2. Software Design Patterns & Architecture

DeepFix follows the **Deep Module** philosophy and **Ports & Adapters** architecture. Design modules that maximize leverage for callers, locality for maintainers, and natural testability. For an overview of system components and package relationships, refer to [`README.md`](README.md).

### Core Principles
- **Deep Modules (Small Interface, Deep Implementation)**:
  - Expose a minimal, high-leverage interface (few methods, simple parameter types, clear return types).
  - Hide internal complexity, coordination, and algorithms inside the module.
  - Avoid shallow pass-through modules where the interface is as complex as the implementation.
- **Clean Seams & Ports/Adapters**:
  - Place seams (interfaces) where behavior truly varies across boundaries (e.g. transport, storage, or external APIs).
  - Core diagnostic and reasoning logic must remain independent of external frameworks (FastAPI, CLI, external SDKs).
  - "One adapter means a hypothetical seam; two adapters means a real one." Do not create abstractions unless variation is real.
- **Designing for Testability**:
  - **Accept dependencies, don't instantiate them**: Use dependency injection. Pass clients, configs, or repositories to callers or constructors rather than instantiating concrete external dependencies internally.
  - **Return results, avoid hidden side-effects**: Methods should return observable data structures or models rather than mutating hidden global state.
  - **High cohesion, low coupling**: Ensure each module has a singular, well-defined responsibility.

---

## 3. Testing Standards: Integration & E2E Only (Zero Mocks)

All testing in DeepFix must prove that real functionalities work end-to-end with real data.

### Mandatory Rules
1. **Integration and E2E Tests Only**:
   - Write integration tests across multi-component boundaries and end-to-end tests exercising complete user workflows.
   - Do not write isolated unit tests that test trivial implementation details.
2. **Strict Zero-Mock Mandate**:
   - **NEVER** use `unittest.mock`, `MagicMock`, `AsyncMock`, `patch`, or `mocker`.
   - Never create fake stand-ins or dummy objects that simulate business logic, database queries, or server responses.
3. **Always Real Data & Real Functionalities**:
   - Always use real datasets: load them via `deepfix_sdk.zoo` (e.g., `load_tweet_emotion_classification`), use actual tabular CSVs, or provide real image/annotation files.
   - Execute real code paths: train baseline models, run real data integrity/drift checks, execute live client requests, and verify real responses.
4. **Service Dependencies & Skips**:
   - When a test requires a live service (e.g., `DEEPFIX_TEST_API_URL`) or credentials (e.g., `TAVILY_API_KEY`) that are not present, explicitly skip the test using `@pytest.mark.skipif(...)` or let it fail.
   - **Never mock an external service just to make a test pass.**
5. **Test Organization**:
   - Place all integration and E2E workflow tests in the root `tests/` directory (e.g., `test_*_workflow_e2e.py`).
   - Use shared fixtures defined in `tests/conftest.py` (e.g., `api_url`, `deepfix_timeout`, `check_response`).

### DOs and DON'Ts

#### ❌ DON'T: Shallow Mock Testing (Strictly Forbidden)
```python
# BAD: Mocking the client and simulating responses
from unittest.mock import MagicMock, patch

def test_diagnosis_workflow_bad():
    mock_client = MagicMock()
    mock_client.get_diagnosis.return_value = {"status": "ok", "summary": "fake"}
    
    result = mock_client.get_diagnosis(train_data="dummy", test_data="dummy")
    assert result["status"] == "ok"
    mock_client.get_diagnosis.assert_called_once()
```

#### ✅ DO: Real Integration / E2E Testing with Real Data
```python
# GOOD: Real data, real SDK client, live server, real response assertion
import pytest
from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import TabularDataset
from deepfix_sdk.zoo.datasets import load_breast_cancer_classification

class TestTabularWorkflowE2E:
    def test_tabular_diagnosis_workflow(self, api_url: str, check_response: callable, deepfix_timeout: int):
        # 1. Initialize real client
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)
        
        # 2. Load real dataset from zoo
        train_df, val_df = load_breast_cancer_classification(as_train_test=True)
        train_data = TabularDataset(dataset_name="breast_cancer", dataset=train_df, target_column="target")
        val_data = TabularDataset(dataset_name="breast_cancer", dataset=val_df, target_column="target")
        
        # 3. Execute real diagnosis against live server
        response = client.get_diagnosis(
            train_data=train_data,
            test_data=val_data,
        )
        
        # 4. Assert on real observable output
        assert check_response(response)
        assert response.summary is not None
        assert len(response.agent_results) > 0
```

---

## 4. Commit Messages

Use conventional-style prefixes:

- `feat: ...` – new feature.
- `fix: ...` – bug fix.
- `docs: ...` – documentation only.
- `test: ...` – tests only.
- `refactor: ...` – internal refactor.
- `chore: ...` – maintenance / tooling.

Example:
```bash
git commit -m "feat: add new dataset analyzer"
```

When creating a commit message, be short and to the point.
