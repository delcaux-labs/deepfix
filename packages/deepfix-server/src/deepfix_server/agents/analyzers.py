"""Specialized Agno artifact analyzer agents and execution helpers."""

from __future__ import annotations

import traceback
from typing import Optional
import json
from agno.agent import Agent
from agno.models.base import Model
from deepfix_core.models import AgentResult, Artifacts

from ..config import LLMConfig
from ..logging import get_logger
from ..prompt_builders import PromptBuilder
from .models import create_agno_model
from .prompts import (
    CHECKPOINT_SYSTEM_PROMPT,
    DATASET_SYSTEM_PROMPT,
    DEEPCHECKS_SYSTEM_PROMPT,
    TRAINING_SYSTEM_PROMPT,
)
from .schemas import ArtifactAnalysisResult

LOGGER = get_logger(__name__)


def _resolve_model(
    model: Optional[Model] = None, llm_config: Optional[LLMConfig] = None
) -> Optional[Model]:
    if model is not None:
        return model
    try:
        if llm_config is not None:
            return create_agno_model(llm_config)
        from ..config import settings
        return create_agno_model(settings.get_llm_config())
    except Exception as exc:
        LOGGER.warning("Could not resolve Agno model: %s", exc)
        return None


def create_dataset_analyzer(
    model: Optional[Model] = None, llm_config: Optional[LLMConfig] = None
) -> Agent:
    """Create a DatasetArtifactsAnalyzer Agno agent."""
    return Agent(
        id="dataset_artifacts_analyzer",
        name="DatasetArtifactsAnalyzer",
        model=_resolve_model(model, llm_config),
        description="Analyzes dataset artifacts, data distributions, and feature properties.",
        instructions=DATASET_SYSTEM_PROMPT,
        output_schema=ArtifactAnalysisResult,
        use_json_mode=True
    )


def create_training_analyzer(
    model: Optional[Model] = None, llm_config: Optional[LLMConfig] = None
) -> Agent:
    """Create a TrainingArtifactsAnalyzer Agno agent."""
    return Agent(
        id="training_artifacts_analyzer",
        name="TrainingArtifactsAnalyzer",
        model=_resolve_model(model, llm_config),
        description="Analyzes model training dynamics, loss curves, and optimization parameters.",
        instructions=TRAINING_SYSTEM_PROMPT,
        output_schema=ArtifactAnalysisResult,
        use_json_mode=True
    )


def create_checkpoint_analyzer(
    model: Optional[Model] = None, llm_config: Optional[LLMConfig] = None
) -> Agent:
    """Create a ModelCheckpointArtifactsAnalyzer Agno agent."""
    return Agent(
        id="model_checkpoint_artifacts_analyzer",
        name="ModelCheckpointArtifactsAnalyzer",
        model=_resolve_model(model, llm_config),
        description="Analyzes model checkpoints, weights, and architecture properties.",
        instructions=CHECKPOINT_SYSTEM_PROMPT,
        output_schema=ArtifactAnalysisResult,
        use_json_mode=True
    )


def create_deepchecks_analyzer(
    model: Optional[Model] = None, llm_config: Optional[LLMConfig] = None
) -> Agent:
    """Create a DeepchecksArtifactsAnalyzer Agno agent."""
    return Agent(
        id="deepchecks_artifacts_analyzer",
        name="DeepchecksArtifactsAnalyzer",
        model=_resolve_model(model, llm_config),
        description="Analyzes Deepchecks data validation test suites and integrity checks.",
        instructions=DEEPCHECKS_SYSTEM_PROMPT,
        output_schema=ArtifactAnalysisResult,
        use_json_mode=True
    )


async def run_artifact_analyzer(
    agent: Agent,
    artifact: Optional[Artifacts] = None,
    artifacts: Optional[Artifacts] = None,
    language: str = "english",
    output_language: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    dataset_name: Optional[str] = None,
) -> AgentResult:
    """Execute analysis for a single artifact using an Agno analyzer agent.

    Args:
        agent: The Agno Agent to execute.
        artifact: The artifact to analyze.
        artifacts: Alias for artifact for backward compatibility.
        language: Desired output language.
        output_language: Alias for language.
        prompt_builder: Optional custom prompt builder.
        dataset_name: Optional dataset name.

    Returns:
        AgentResult containing structured findings or an error message.
    """
    target_artifact = artifact or artifacts
    if target_artifact is None:
        raise ValueError("An artifact must be provided to run_artifact_analyzer.")

    target_language = output_language or language
    builder = prompt_builder or PromptBuilder()
    agent_name = agent.name or type(agent).__name__
    try:
        LOGGER.debug("Running Agno analyzer agent %s...", agent_name)
        prompt = builder.build_prompt(artifacts=[target_artifact], context=None)
        user_message = f"Output language: {target_language}\n\n{prompt}"

        run_output = await agent.arun(user_message)

        content = run_output.content
        if isinstance(content, str):
            content = json.loads(content)
        
        if isinstance(content, dict):
            content = ArtifactAnalysisResult.model_validate(content)
        
        if isinstance(content, ArtifactAnalysisResult):
            analysis = content.analysis
        else:
            msg = f"Unexpected content type from Agno agent {agent_name}: {type(content)}"
            LOGGER.error(msg)
            raise ValueError(msg)

        return AgentResult(
            agent_name=agent_name,
            analysis=analysis,
            analyzed_artifacts=[type(target_artifact).__name__],
        )
    except Exception as e:
        LOGGER.error("Error in %s: %s", agent_name, traceback.format_exc())
        return AgentResult(
            agent_name=agent_name,
            error_message=str(e),
            analyzed_artifacts=[type(target_artifact).__name__],
        )
