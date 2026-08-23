## DeepFix Server

DeepFix Server provides an HTTP API and CLI to analyze ML experiment artifacts (datasets, deepchecks reports, model checkpoints, training logs) using agentic reasoning and LLM-powered analysis. It orchestrates specialized analyzer agents, synthesizes cross-artifact insights, and returns structured findings and recommendations.

### Key Features
- Agentic analysis of multiple artifact types: dataset, deepchecks, model checkpoints, training
- Cross-artifact reasoning to synthesize holistic insights and a concise summary
- Configurable LLM backend via environment variables
- Simple HTTP API served with FastAPI
- Typer-based CLI to launch the server locally

### Requirements
- Python 3.11

### Install
You can install the server in editable mode from the monorepo root or from this package directory.

```bash
# From repo root using pip
pip install -e deepfix-core
pip install -e deepfix-server
```


### Configure LLM Backend
Set the following environment variables (or pass an env file to the CLI) to configure the LLM used by the agents:

- `DEEPFIX_LLM_API_KEY`: API key for your LLM provider
- `DEEPFIX_LLM_BASE_URL`: Base URL of the LLM API
- `DEEPFIX_LLM_MODEL_NAME`: Model name to use (e.g., gpt-4o, llama-3.1, etc.)
- `DEEPFIX_LLM_TEMPERATURE`: Float (e.g., 0.7)
- `DEEPFIX_LLM_MAX_TOKENS`: Integer (e.g., 8000)
- `DEEPFIX_LLM_CACHE`: Boolean-like string ("1"/"0", "true"/"false")
- `DEEPFIX_LLM_TRACK_USAGE`: Boolean-like string

Example `.env` file:

```bash
DEEPFIX_LLM_API_KEY=sk-...
DEEPFIX_LLM_BASE_URL=https://api.your-llm.com/v1
DEEPFIX_LLM_MODEL_NAME=gpt-4o
DEEPFIX_LLM_TEMPERATURE=0.4
DEEPFIX_LLM_MAX_TOKENS=6000
DEEPFIX_LLM_CACHE=true
DEEPFIX_LLM_TRACK_USAGE=true
```

### Run the Server (CLI)
The package exposes a CLI entrypoint `deepfix-server`.

```bash
# Show version
deepfix-server version

# Launch server (defaults: host 127.0.0.1, port 8844)
deepfix-server launch -host 0.0.0.0 -port 8844 -e .env
```

Startup message: "Starting DeepFix server on <host>:<port>".

### HTTP API
The API is hosted via FastAPI. The Analyse Artifacts endpoint is available at:

- POST `/v1/analyse`

Request and response models reuse `deepfix-core` Pydantic types.

Request body schema (selected fields):
- `dataset_artifacts` (optional): `deepfix_core.models.DatasetArtifacts`
- `training_artifacts` (optional): `deepfix_core.models.TrainingArtifacts`
- `deepchecks_artifacts` (optional): `deepfix_core.models.DeepchecksArtifacts`
- `model_checkpoint_artifacts` (optional): `deepfix_core.models.ModelCheckpointArtifacts`
- `dataset_name` (optional): string

Response body schema (selected fields):
- `agent_results`: map of agent name to `deepfix_core.models.AgentResult`
- `summary`: cross-artifact summary string
- `additional_outputs`: extra outputs from agents
- `error_messages`: optional map of agent to error string

#### cURL Example
```bash
curl -X POST \
  http://localhost:8844/v1/analyse \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "my-dataset",
    "dataset_artifacts": { /* see deepfix-core schemas */ },
    "deepchecks_artifacts": { /* deepchecks results */ },
    "model_checkpoint_artifacts": { /* checkpoint metadata */ },
    "training_artifacts": { /* metrics/params */ }
  }'
```

#### Python Example (with deepfix-sdk)
```python
from deepfix_sdk.client import DeepFixClient
from deepfix_core.models import APIRequest

client = DeepFixClient(base_url="http://localhost:8844")

resp = client.diagnose_dataset(dataset_name=...)
print(result.to_text())
```

### How It Works (High-Level)
- The API endpoint receives an `APIRequest`, decodes it into an `AgentContext` and runs the `DiagnosticSystem`.
- The coordinator dispatches artifacts to specialized analyzer agents in parallel via LangGraph:
  - `DeepchecksArtifactsAnalyzer`: data quality, drift, integrity
  - `DatasetArtifactsAnalyzer`: dataset stats, class balance, anomalies
  - `ModelCheckpointArtifactsAnalyzer`: checkpoint integrity, config consistency, deployment readiness
  - `TrainingArtifactsAnalyzer`: training dynamics
- `CrossArtifactReasoningAgent` fans in all agent findings and performs multi-chain reasoning comparison and synthesis.
- Response is returned as `deepfix_core.models.APIResponse` with structured findings and summary.

```mermaid
graph TD
    START([START]) -->|Dynamic Fan-Out| deepchecks_analyzer[DeepchecksArtifactsAnalyzer]
    START -->|Dynamic Fan-Out| dataset_analyzer[DatasetArtifactsAnalyzer]
    START -->|Dynamic Fan-Out| checkpoint_analyzer[ModelCheckpointArtifactsAnalyzer]
    START -->|Dynamic Fan-Out| training_analyzer[TrainingArtifactsAnalyzer]

    deepchecks_analyzer -->|Fan-In| cross_artifact_reasoner[CrossArtifactReasoningAgent<br/>Multi-Chain Reasoning & Synthesis]
    dataset_analyzer -->|Fan-In| cross_artifact_reasoner
    checkpoint_analyzer -->|Fan-In| cross_artifact_reasoner
    training_analyzer -->|Fan-In| cross_artifact_reasoner

    cross_artifact_reasoner --> END([END])
```

### Logging
Logging is configured via `deepfix_server.logging`. By default logs are printed to stdout. You can integrate it into your application and customize formats or levels if embedding the server.

### Development
Run tests (from the monorepo root, if applicable):

```bash
pytest
```

Local development install:

```bash
uv pip install -e .
```

### Notes and Limitations
- `TrainingArtifactsAnalyzer` exists as a scaffold and includes detailed analysis primitives, but its main `_run` method is not finalized. The server will still run and process other artifact types.
- The LLM configuration is required for best results; without it, DSPy will need to be configured globally, otherwise some agents will warn or operate in limited mode.

### License
This project is licensed under the terms specified in the repository LICENSE.
