# Refactor deepfix-server + deepfix-kb from DSPy to Pydantic AI

> **For Hermes:** Use plan skill guidance for execution — TDD, bite-sized tasks, frequent commits.

**Goal:** Replace DSPy with Pydantic AI (https://ai.pydantic.dev/) across `deepfix-server` and `deepfix-kb` packages while preserving all existing behavior, agent architecture, and API contracts.

**Architecture:** DSPy currently provides three abstractions: (1) `dspy.LM` for LLM invocation, (2) `dspy.Signature` / `dspy.Module` / `dspy.ChainOfThought` / `dspy.ReAct` for structured prompting, and (3) `dspy.Tool` / `dspy.clients.Cache` for tool integration and caching. Pydantic AI replaces all three: `Agent` replaces `dspy.Module`/`dspy.ChainOfThought`, `@agent.tool` replaces `dspy.Tool`, and Pydantic models + `result_type` replace `dspy.Signature` I/O fields. The `dspy.context(lm=...)` pattern becomes Pydantic AI's `Model` passed to Agent constructors.

**Tech Stack:** Pydantic AI (`pydantic-ai`), existing: Python 3.11+, FastAPI, uv, Pydantic v2, SQLAlchemy, MLflow, LlamaIndex.

**Scope:** ~14 files across two packages. DSPy is used in:
- `deepfix-server`: agents (`base.py`, `artifact_analyzers.py`, `cross_artifact_reasoning.py`, `optimizationadvisor.py`), signatures (`signatures.py`), cache (`dspy_cache.py`), config (`config.py`), logging (`logging.py`), tests
- `deepfix-kb`: tools (`dspy_tools.py`), perplexity client (`perplexity_client.py`)

---

## Phase 1: Pydantic AI foundation — server config + LLM abstraction (no agent logic yet)

### Task 1.1: Add pydantic-ai dependency and create LLM factory

**Objective:** Install `pydantic-ai` and build a simple LLM factory that replaces `dspy.LM` + `dspy.context(lm=...)`.

**Files:**
- Modify: `packages/deepfix-server/pyproject.toml`
- Create: `packages/deepfix-server/src/deepfix_server/llm.py`
- Modify: `packages/deepfix-server/src/deepfix_server/config.py`

**Step 1: Add dependency**

Add `"pydantic-ai>=0.0.50"` to `pyproject.toml` dependencies. Remove `"dspy>=3.0.3"` (do NOT remove yet — we'll do that at the very end after all tests pass without it).

**Step 2: Create `deepfix_server/llm.py`**

```python
"""Pydantic AI model factory for DeepFix server agents."""
from __future__ import annotations

from typing import Optional

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models import Model, OpenAIModel

from .config import LLMConfig


def create_model(config: LLMConfig) -> Model:
    """Create a Pydantic AI Model from an LLMConfig.

    Args:
        config: LLM configuration with model_name, api_key, base_url, etc.

    Returns:
        A Pydantic AI Model configured for the given provider.
    """
    # Pydantic AI's OpenAIModel is compatible with any OpenAI-compatible API
    # (OpenRouter, LiteLLM, etc.) by setting base_url
    return OpenAIModel(
        model_name=config.model_name,
        api_key=config.api_key,
        base_url=config.base_url,
    )


def create_agent(
    config: LLMConfig,
    result_type: type | None = None,
    system_prompt: str = "",
) -> PydanticAgent:
    """Create a Pydantic AI Agent from an LLMConfig.

    Args:
        config: LLM configuration.
        result_type: Optional Pydantic model for structured output.
        system_prompt: System prompt string.

    Returns:
        A configured PydanticAgent.
    """
    model = create_model(config)
    return PydanticAgent(
        model=model,
        result_type=result_type,
        system_prompt=system_prompt,
    )
```

**Step 3: Verify it imports**

```bash
cd /root/workspace/deepfix && uv run python -c "from deepfix_server.llm import create_model, create_agent; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add packages/deepfix-server/pyproject.toml packages/deepfix-server/src/deepfix_server/llm.py
git commit -m "feat: add pydantic-ai dependency and LLM factory module"
```

### Task 1.2: Add pydantic-ai to deepfix-kb dependencies

**Objective:** Add `pydantic-ai` to the KB package too, since perplexity_client.py also uses DSPy.

**Files:**
- Modify: `packages/deepfix-kb/pyproject.toml`

**Step 1: Add dependency**

Add `"pydantic-ai>=0.0.50"` to `deepfix-kb/pyproject.toml`.

**Step 2: Verify install**

```bash
cd /root/workspace/deepfix && uv pip install -e packages/deepfix-kb
```

**Step 3: Commit**

```bash
git add packages/deepfix-kb/pyproject.toml
git commit -m "feat: add pydantic-ai dependency to deepfix-kb"
```

---

## Phase 2: Replace dspy.Signature with Pydantic models + Agent result_type

### Task 2.1: Create Pydantic response models for artifact analysis

**Objective:** Replace `dspy.Signature` OutputFields with Pydantic BaseModel classes that `pydantic_ai.Agent` uses as `result_type`.

**Files:**
- Create: `packages/deepfix-server/src/deepfix_server/agent_models.py`
- Modify (later): `packages/deepfix-server/src/deepfix_server/agents/signatures.py` — keep for reference, delete at end

**Step 1: Create the models file**

```python
"""Pydantic AI result models replacing DSPy signatures."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from deepfix_core.models import Analysis


class ArtifactAnalysisResult(BaseModel):
    """Result of a single artifact analysis."""
    analysis: List[Analysis] = Field(
        description="Findings and recommendations based on the artifacts"
    )


class CrossArtifactReasoningResult(BaseModel):
    """Result of cross-artifact reasoning."""
    analysis: List[Analysis] = Field(
        description="Consolidated analysis with cross-artifact insights"
    )
    summary: str = Field(
        description="Summary of the cross-artifact reasoning and analysis"
    )
```

**Step 2: Verify it imports**

```bash
cd /root/workspace/deepfix && uv run python -c "from deepfix_server.agent_models import ArtifactAnalysisResult, CrossArtifactReasoningResult; print('OK')"
```

**Step 3: Commit**

```bash
git add packages/deepfix-server/src/deepfix_server/agent_models.py
git commit -m "feat: add pydantic result models for agent analysis"
```

---

## Phase 3: Refactor ArtifactAnalyzer base class

### Task 3.1: Rewrite agents/base.py — ArtifactAnalyzer on Pydantic AI

**Objective:** Replace the `dspy.Module` base class, `dspy.ChainOfThought(signature)`, and `self.llm.acall()` with Pydantic AI Agent.

**Files:**
- Modify: `packages/deepfix-server/src/deepfix_server/agents/base.py`
- Modify: `packages/deepfix-server/src/deepfix_server/agent_models.py`

**Key changes:**
1. Remove `import dspy`. Import `from pydantic_ai import Agent as PydanticAgent`.
2. `Agent(dspy.Module)` → `Agent` (plain class, no DSPy base).
3. Remove `dspy.context(lm=...)` — no longer needed; Pydantic AI Agent carries its own model.
4. `self.llm = dspy.ChainOfThought(signature)` → `self.agent = PydanticAgent(model=..., result_type=ArtifactAnalysisResult, system_prompt=self.system_prompt)`.
5. `self.llm.acall(artifacts=prompt, output_language=...)` → `self.agent.run(user_prompt=prompt)`.
6. `response.analysis` → `response.data.analysis` (Pydantic AI wraps structured output in `.data`).
7. Keep `PromptBuilder` unchanged — it still produces string prompts.
8. The `_llm_context` method is deleted entirely.

**Step 1: Write tests first**

Create/modify `packages/deepfix-server/tests/test_base_agent.py`:

```python
import pytest
from deepfix_server.agents.base import ArtifactAnalyzer
from deepfix_server.config import LLMConfig
from deepfix_core.models import DatasetArtifacts

class DummyAnalyzer(ArtifactAnalyzer):
    @property
    def system_prompt(self) -> str:
        return "You are a test analyzer."
    
    @property
    def supported_artifact_types(self):
        return DatasetArtifacts

def test_artifact_analyzer_initialization():
    config = LLMConfig(model_name="test-model", api_key="fake")
    analyzer = DummyAnalyzer(config=config)
    assert analyzer.agent_name == "DummyAnalyzer"
    assert analyzer.system_prompt == "You are a test analyzer."
    # Pydantic AI agent should be initialized
    assert analyzer.agent is not None
```

**Step 2: Run test to confirm failure**

```bash
cd /root/workspace/deepfix && uv run pytest packages/deepfix-server/tests/test_base_agent.py -v
```

**Step 3: Rewrite base.py**

The critical refactored `ArtifactAnalyzer`:

```python
class ArtifactAnalyzer(Agent):
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        config_prompt_builder: Optional[PromptConfig] = None,
    ):
        super().__init__(config=config)
        self.prompt_builder = PromptBuilder(config=config_prompt_builder)
        self.agent = create_pydantic_agent(config, self.system_prompt)

    async def aforward(self, context: AgentContext) -> AgentResult:
        LOGGER.info(f"Running {self.agent_name} agent...")
        self._check_artifacts(context.artifacts)
        prompt = self.prompt_builder.build_prompt(
            artifacts=context.artifacts, context=None
        )
        # Pydantic AI: combine prompt with language instruction
        user_message = f"Output language: {context.language}\n\n{prompt}"
        result = await self.agent.run(user_message)
        return AgentResult(
            agent_name=self.agent_name,
            analysis=result.data.analysis,
            analyzed_artifacts=[type(a).__name__ for a in context.artifacts],
        )
```

**Also update the `Agent` base class** — remove `dspy.Module` parent, remove `_llm_context`.

**Step 4: Create the `create_pydantic_agent` helper in llm.py.**

```python
def create_agent_for_analysis(config: LLMConfig | None, system_prompt: str) -> PydanticAgent:
    """Create a Pydantic AI Agent configured for artifact analysis.
    
    If config is None, uses environment defaults.
    """
    from .agent_models import ArtifactAnalysisResult
    model_config = config or LLMConfig(
        model_name=os.getenv("DEEPFIX_LLM_MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("DEEPFIX_LLM_API_KEY"),
        base_url=os.getenv("DEEPFIX_LLM_BASE_URL"),
    )
    model = OpenAIModel(
        model_name=model_config.model_name,
        api_key=model_config.api_key,
        base_url=model_config.base_url,
    )
    return PydanticAgent(
        model=model,
        result_type=ArtifactAnalysisResult,
        system_prompt=system_prompt,
    )
```

**Step 5: Run test to verify pass**

```bash
cd /root/workspace/deepfix && uv run pytest packages/deepfix-server/tests/test_base_agent.py -v
```

**Step 6: Commit**

```bash
git add packages/deepfix-server/src/deepfix_server/agents/base.py packages/deepfix-server/src/deepfix_server/llm.py packages/deepfix-server/src/deepfix_server/agent_models.py packages/deepfix-server/tests/test_base_agent.py
git commit -m "refactor: replace DSPy Module/ChainOfThought with Pydantic AI Agent in ArtifactAnalyzer base"
```

---

## Phase 4: Refactor individual analyzers

### Task 4.1: Update artifact_analyzers.py subclasses

**Objective:** The `DeepchecksArtifactsAnalyzer`, `DatasetArtifactsAnalyzer`, `ModelCheckpointArtifactsAnalyzer`, `TrainingArtifactsAnalyzer` subclasses already inherit from `ArtifactAnalyzer`. After Phase 3, they should "just work" with minimal changes.

**Files:**
- Modify: `packages/deepfix-server/src/deepfix_server/agents/artifact_analyzers.py`

**Key changes:**
1. Remove `import dspy` (if present at top of base — not in this file directly).
2. Constructor signatures: remove `llm: Optional[dspy.Module] = None` parameter. Replace with just `config: Optional[LLMConfig] = None`.
3. Ensure `super().__init__(config=config)` is called correctly (no `llm=` kwarg).
4. All four analyzer `__init__` methods currently pass `llm=llm, config=config` to super — change to just `config=config`.

**Step 1: Make the changes**

In each analyzer's `__init__`:
```python
# Before:
def __init__(self, config=None, llm=None):
    super().__init__(llm=llm, config=config)

# After:
def __init__(self, config=None):
    super().__init__(config=config)
```

**Step 2: Verify tests pass**

```bash
cd /root/workspace/deepfix && uv run pytest packages/deepfix-server/tests/ -v -k "not cross_artifact and not e2e"
```

**Step 3: Commit**

```bash
git add packages/deepfix-server/src/deepfix_server/agents/artifact_analyzers.py
git commit -m "refactor: remove DSPy llm param from artifact analyzer constructors"
```

---

## Phase 5: Refactor CrossArtifactReasoningAgent

### Task 5.1: Rewrite cross_artifact_reasoning.py

**Objective:** Replace `dspy.ReAct` / `dspy.ChainOfThought` / `dspy.MultiChainComparison` with Pydantic AI Agent + manual self-consistency loop.

**Files:**
- Modify: `packages/deepfix-server/src/deepfix_server/agents/cross_artifact_reasoning.py`

**Key changes:**
1. Remove `import dspy`. Import `from pydantic_ai import Agent as PydanticAgent`.
2. Replace `self.predict = dspy.ReAct(signature, tools=tools)` / `dspy.ChainOfThought(signature)` with a single Pydantic AI `Agent`.
3. Replace `self.compare = dspy.MultiChainComparison(signature, M=num_attempts)` with a manual loop: run the agent N times, then run a final "consolidator" pass.
4. Tools (from `create_knowledge_tools`) need to be refactored separately (Phase 7). For now, if knowledge_bridge is provided, pass tool functions to Pydantic AI via `@agent.tool` decorator pattern or `tools=` parameter.

**Step 1: Rewrite the constructor**

```python
class CrossArtifactReasoningAgent(Agent):
    def __init__(self, llm_config=None, knowledge_bridge=None, num_attempts=3):
        super().__init__(config=llm_config)
        self.knowledge_bridge = knowledge_bridge
        self.num_attempts = num_attempts
        
        # Build the Pydantic AI agent
        from ..agent_models import CrossArtifactReasoningResult
        model = create_model(llm_config) if llm_config else None
        tools_list = []
        if self.knowledge_bridge:
            tools_list = create_knowledge_tools(self.knowledge_bridge, include_hybrid=False)
        self.agent = PydanticAgent(
            model=model,
            result_type=CrossArtifactReasoningResult,
            system_prompt=self.system_prompt,
            tools=tools_list,  # Will work once Phase 7 is done
        )
```

**Step 2: Rewrite aforward for self-consistency**

```python
async def aforward(self, previous_analyses, output_language="english"):
    LOGGER.info("Running cross-artifact reasoning agent...")
    assert len(previous_analyses) > 0
    
    # Run N times for self-consistency
    completions = []
    for _ in range(self.num_attempts):
        prompt = self._build_reasoning_prompt(previous_analyses, output_language)
        result = await self.agent.run(prompt)
        completions.append(result.data)
    
    # Consolidation: use the last one or a simple voting/merge strategy
    # For simplicity, use the agent one more time with all completions
    consolidate_prompt = self._build_consolidation_prompt(completions, previous_analyses, output_language)
    final = await self.agent.run(consolidate_prompt)
    
    # Compute analyzed_artifacts, retrieved_knowledge from previous analyses
    ...
    return AgentResult(
        agent_name=self.agent_name,
        analysis=final.data.analysis,
        ...
    )
```

**Step 3: Add helper methods for prompt building**

```python
def _build_reasoning_prompt(self, previous_analyses, output_language):
    analyses_text = json.dumps(
        {name: {"analysis": r.analysis, "error": r.error_message}
         for name, r in previous_analyses.items()},
        default=str, indent=2
    )
    return f"Previous analyses:\n{analyses_text}\n\nOutput language: {output_language}"

def _build_consolidation_prompt(self, completions, previous_analyses, output_language):
    comp_text = "\n\n---\n\n".join(
        f"Attempt {i+1}:\n{c.model_dump_json(indent=2)}"
        for i, c in enumerate(completions)
    )
    return f"You ran cross-artifact reasoning {self.num_attempts} times. Here are the results:\n\n{comp_text}\n\nConsolidate these into a single coherent analysis. Output language: {output_language}"
```

**Step 4: Update tests**

```bash
cd /root/workspace/deepfix && uv run pytest packages/deepfix-server/tests/test_cross_artifact_reasoning.py -v
```

Tests will fail initially — update them to work with the new API.

**Step 5: Commit**

```bash
git add packages/deepfix-server/src/deepfix_server/agents/cross_artifact_reasoning.py packages/deepfix-server/tests/test_cross_artifact_reasoning.py
git commit -m "refactor: replace DSPy ReAct/MultiChainComparison with Pydantic AI self-consistency"
```

---

## Phase 6: Refactor OptimizationAdvisorAgent

### Task 6.1: Rewrite optimizationadvisor.py

**Objective:** Replace `dspy.ChainOfThought` with Pydantic AI Agent.

**Files:**
- Modify: `packages/deepfix-server/src/deepfix_server/agents/optimizationadvisor.py`

**Key changes:**
1. Remove `import dspy`.
2. Replace `self.llm = dspy.ChainOfThought(signature)` with `PydanticAgent`.
3. Replace `self.llm.acall(...)` with `self.agent.run(...)`.

**Step 1: Apply changes, commit**

```bash
git add packages/deepfix-server/src/deepfix_server/agents/optimizationadvisor.py
git commit -m "refactor: replace DSPy ChainOfThought with Pydantic AI in OptimizationAdvisorAgent"
```

---

## Phase 7: Refactor deepfix-kb tools

### Task 7.1: Rewrite dspy_tools.py — remove DSPy Tool dependency

**Objective:** The KB tools are currently wrapped as `dspy.Tool(func=..., name=..., desc=..., args=...)`. Pydantic AI uses plain functions with `@agent.tool` decorator. The tool classes are already callable — we just need to remove the `dspy.Tool` wrapping.

**Files:**
- Modify: `packages/deepfix-kb/src/deepfix_kb/tools/dspy_tools.py`
- Rename to: `packages/deepfix-kb/src/deepfix_kb/tools/kb_tools.py` (or just update in place)
- Modify: `packages/deepfix-kb/src/deepfix_kb/tools/__init__.py`

**Key changes:**
1. Remove `import dspy`.
2. Remove all `to_dspy_tool()` methods.
3. Add Pydantic AI `@tool` decorator pattern — or simpler: keep the callable classes and register them with Pydantic AI Agent's `tools=` parameter directly (Pydantic AI accepts plain callables and infers the tool signature from the function).

**Step 1: Update __init__.py**

Change `create_knowledge_tools` to return plain callables instead of DSPy Tool objects:

```python
def create_knowledge_tools(bridge, include_hybrid=False):
    tools = [
        WebSearchTool(bridge),
        ResearchTool(bridge),
        KnowledgeLookupTool(bridge),
    ]
    if include_hybrid:
        tools.append(HybridSearchTool(bridge))
    return tools
```

**Step 2: Remove DSPy import from dspy_tools.py**

Remove `import dspy` and all `to_dspy_tool` methods. Keep the `__call__` and `_search`/`_research`/`_lookup` methods.

**Step 3: Commit**

```bash
git add packages/deepfix-kb/src/deepfix_kb/tools/
git commit -m "refactor: remove DSPy Tool wrapping from KB tools"
```

### Task 7.2: Rewrite perplexity_client.py

**Objective:** Replace `dspy.Signature`, `dspy.Module`, `dspy.ChainOfThought`, `dspy.LM`, `dspy.context(lm=...)` with Pydantic AI.

**Files:**
- Modify: `packages/deepfix-kb/src/deepfix_kb/retrieval/perplexity_client.py`

**Key changes:**
1. `PerplexityResearchModule(dspy.Module)` → `PerplexityResearchModule` (plain class using Pydantic AI).
2. `dspy.Signature` classes (`ResearchQuery`, `BriefResearchQuery`, etc.) → Pydantic `BaseModel` result types or just `str` result_type since they all output a single `answer: str`.
3. `dspy.ChainOfThought(Signature)` → `PydanticAgent(result_type=str, system_prompt=...)`.
4. `self.researcher(topic=query)` → `await self.researcher_agent.run(query)`.
5. `dspy.LM(model=..., api_key=..., api_base=...)` → `OpenAIModel(model_name=..., api_key=..., base_url=...)`.
6. `dspy.context(lm=self.lm)` → no longer needed; agent carries model.

**Step 1: Rewrite PerplexityResearchModule**

```python
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIModel

class PerplexityResearchModule:
    def __init__(self, model: OpenAIModel):
        self.researcher = PydanticAgent(
            model=model,
            result_type=str,
            system_prompt="You are a helpful research assistant...",
        )
        # ... similar for brief, detailed, comprehensive variants
    
    async def aforward(self, query, context="", depth="detailed"):
        user_msg = f"Research query: {query}"
        if context:
            user_msg += f"\n\nContext: {context}"
        if depth == "brief":
            result = await self.brief_researcher.run(user_msg)
        elif depth == "comprehensive":
            result = await self.comprehensive_researcher.run(user_msg)
        else:
            result = await self.detailed_researcher.run(user_msg)
        # Return a simple namespace to preserve compatibility
        return type("Prediction", (), {"answer": result.data})()
```

**Step 2: Update PerplexitySonarRetriever**

Replace `dspy.LM` with `OpenAIModel`, replace `dspy.context(lm=...)` with direct agent calls.

**Step 3: Commit**

```bash
git add packages/deepfix-kb/src/deepfix_kb/retrieval/perplexity_client.py
git commit -m "refactor: replace DSPy with Pydantic AI in PerplexitySonarRetriever"
```

---

## Phase 8: Refactor server config and cache

### Task 8.1: Remove DSPy from config.py

**Objective:** Remove `import dspy` and `dspy.configure_cache(...)` from config.py.

**Files:**
- Modify: `packages/deepfix-server/src/deepfix_server/config.py`

**Change:** Delete line `import dspy` and line `dspy.configure_cache(enable_disk_cache=settings.llm_cache)`. Caching will be handled separately or via Pydantic AI's built-in mechanisms.

### Task 8.2: Rewrite or remove dspy_cache.py

**Objective:** `DSPyDatabaseCache` subclasses `dspy.clients.Cache`. Pydantic AI doesn't have a drop-in cache replacement with the same API. Options:
- (a) Port to a standalone SQLAlchemy-based cache that wraps `OpenAIModel` calls via middleware.
- (b) Remove for now, rely on disk cache via `DSPY_CACHEDIR` env var (will need replacement).
- (c) Keep but remove the `dspy.clients.Cache` parent class — make it a standalone cache that the model factory can use.

**Recommended: Option (c)** — convert to a standalone class. The cache stores request/response pairs keyed by SHA256 hash, which is framework-agnostic.

**Files:**
- Modify: `packages/deepfix-server/src/deepfix_server/agents/dspy_cache.py` → rename to `llm_cache.py`

### Task 8.3: Remove DSPy from logging.py

**Objective:** Remove `setup_dspy_logging` references. Replace `mlflow.dspy.autolog()` with — nothing. MLflow tracing for Pydantic AI would be a future enhancement.

**Files:**
- Modify: `packages/deepfix-server/src/deepfix_server/logging.py`
- Rename: `setup_dspy_logging` → `setup_llm_logging` (keep the MLflow parts, just drop `.dspy.autolog()`)

---

## Phase 9: Remove DSPy dependency, final cleanup

### Task 9.1: Remove DSPy from pyproject.toml files

**Files:**
- Modify: `packages/deepfix-server/pyproject.toml` — remove `"dspy>=3.0.3"`
- Modify: `packages/deepfix-kb/pyproject.toml` — remove `"dspy>=3.0.3"`

### Task 9.2: Delete signatures.py (no longer needed)

**Files:**
- Delete: `packages/deepfix-server/src/deepfix_server/agents/signatures.py`

### Task 9.3: Run full test suite

```bash
cd /root/workspace/deepfix && uv run pytest packages/deepfix-server/tests/ -v
cd /root/workspace/deepfix && uv run ruff check packages/deepfix-server/src packages/deepfix-kb/src
```

### Task 9.4: Final commit

```bash
git add -A
git commit -m "refactor: complete DSPy → Pydantic AI migration, remove DSPy dependency"
```

---

## Files Likely to Change (complete list)

| File | Action |
|------|--------|
| `packages/deepfix-server/pyproject.toml` | Add `pydantic-ai`, remove `dspy` |
| `packages/deepfix-kb/pyproject.toml` | Add `pydantic-ai`, remove `dspy` |
| `packages/deepfix-server/src/deepfix_server/llm.py` | **Create** — LLM factory |
| `packages/deepfix-server/src/deepfix_server/agent_models.py` | **Create** — Pydantic result types |
| `packages/deepfix-server/src/deepfix_server/agents/base.py` | Rewrite — `dspy.Module` → Pydantic AI Agent |
| `packages/deepfix-server/src/deepfix_server/agents/artifact_analyzers.py` | Minor — remove `llm` kwarg |
| `packages/deepfix-server/src/deepfix_server/agents/cross_artifact_reasoning.py` | Rewrite — ReAct/ChainOfThought → Pydantic AI + self-consistency |
| `packages/deepfix-server/src/deepfix_server/agents/optimizationadvisor.py` | Rewrite — ChainOfThought → Pydantic AI Agent |
| `packages/deepfix-server/src/deepfix_server/config.py` | Remove `import dspy`, `dspy.configure_cache` |
| `packages/deepfix-server/src/deepfix_server/logging.py` | Remove `mlflow.dspy.autolog()` |
| `packages/deepfix-server/src/deepfix_server/agents/signatures.py` | **Delete** |
| `packages/deepfix-server/src/deepfix_server/agents/dspy_cache.py` | Rewrite to `llm_cache.py` |
| `packages/deepfix-kb/src/deepfix_kb/tools/dspy_tools.py` | Remove DSPy Tool wrapping |
| `packages/deepfix-kb/src/deepfix_kb/retrieval/perplexity_client.py` | Rewrite — DSPy → Pydantic AI |
| `packages/deepfix-server/tests/test_base_agent.py` | **Create** — tests for refactored base |
| `packages/deepfix-server/tests/test_cross_artifact_reasoning.py` | Update for new API |

---

## Risks, Tradeoffs, and Open Questions

### Risks
1. **Structured output reliability**: DSPy's `ChainOfThought` with Signature had strong structured-output guarantees. Pydantic AI's `result_type` is also strong (it uses constrained generation / retries) but needs testing with the actual LLM provider (OpenRouter).
2. **Self-consistency replacement**: `dspy.MultiChainComparison` is a sophisticated module that compares multiple completions intelligently. Our manual consolidation loop is simpler. We may lose some quality here — this is the biggest behavioral change.
3. **Cache**: `dspy.clients.Cache` is deeply integrated. Pydantic AI doesn't have an equivalent pluggable cache layer. We'll need to use model-level middleware or a wrapper.
4. **Tool calling**: DSPy's `ReAct` with tools is battle-tested for agentic tool use. Pydantic AI's `@agent.tool` is newer but functional. Needs live testing.

### Tradeoffs
- **Losing MultiChainComparison** — the consolidation quality may degrade somewhat.
- **Gaining simpler code** — Pydantic AI is significantly less abstract than DSPy; the codebase will be more readable and easier to onboard contributors.
- **Losing built-in MLflow autolog** — `mlflow.dspy.autolog()` goes away; tracing will need a separate effort.

### Open Questions
1. Should we keep the `TrainingArtifactsAnalyzer` stub or fix it during this refactor? (Out of scope — keep as-is.)
2. Should the KB tools use Pydantic AI's `Tool` type or plain callables? (Plain callables work with `tools=` parameter.)
3. Do we want to add `temperature` / `max_tokens` to the Pydantic AI model config? (Yes — pass them to `OpenAIModel` constructor if supported.)

---

## Verification Checklist

- [ ] All server unit tests pass
- [ ] `ruff check` clean on both packages
- [ ] `uv run deepfix-server launch -port 8844` starts without DSPy import errors
- [ ] A real `POST /v1/analyse` call returns valid structured results
- [ ] Cross-artifact reasoning produces coherent consolidated analysis
- [ ] KB research tool returns citations as before