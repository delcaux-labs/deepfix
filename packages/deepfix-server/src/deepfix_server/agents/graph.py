"""LangGraph state graph for orchestrating multi-agent artifact analysis.

Implements dynamic fan-out to specialized analyzer nodes in parallel and
fan-in to a centralized cross-artifact reasoning node.
"""

from __future__ import annotations

import traceback
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from deepfix_core.models import AgentResult, Artifacts
from deepfix_kb import KnowledgeBridge
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..config import LLMConfig
from ..logging import get_logger
from ..prompt_builders import PromptBuilder
from .prompts import (
    CHECKPOINT_SYSTEM_PROMPT,
    DATASET_SYSTEM_PROMPT,
    DEEPCHECKS_SYSTEM_PROMPT,
    TRAINING_SYSTEM_PROMPT,
)
from .reasoning import run_cross_artifact_reasoning
from .schemas import AgentContext, ArtifactAnalysisResult

LOGGER = get_logger(__name__)


def create_chat_model(config: Optional[LLMConfig] = None) -> BaseChatModel:
    """Create a LangChain BaseChatModel (ChatOpenAI) from an LLMConfig.

    Supports direct OpenAI models or LiteLLM / custom OpenAI-compatible endpoints
    via `base_url`.

    Args:
        config: LLM configuration.

    Returns:
        Configured BaseChatModel instance.
    """
    if config is None:
        raise ValueError("No LLM configuration provided.")

    if not config.api_key:
        raise ValueError("No LLM API key configured. Please provide LLM configuration.")

    kwargs: dict = {
        "model": config.model_name,
        "api_key": config.api_key,
        "temperature": config.temperature if config.temperature is not None else 0.7,
        "max_tokens": config.max_tokens or 8000,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url

    return ChatOpenAI(**kwargs)


async def run_artifact_analysis_node(
    artifact: Artifacts,
    agent_name: str,
    system_prompt: str,
    language: str,
    llm: BaseChatModel,
    prompt_builder: Optional[PromptBuilder] = None,
) -> AgentResult:
    """Execute analysis for a single artifact type using structured LLM outputs."""
    builder = prompt_builder or PromptBuilder()
    try:
        LOGGER.info("Running %s agent...", agent_name)
        prompt = builder.build_prompt(artifacts=[artifact], context=None)
        user_message = f"Output language: {language}\n\n{prompt}"

        structured_llm = llm.with_structured_output(ArtifactAnalysisResult)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        result: ArtifactAnalysisResult = await structured_llm.ainvoke(messages)

        return AgentResult(
            agent_name=agent_name,
            analysis=result.analysis,
            analyzed_artifacts=[type(artifact).__name__],
        )
    except Exception as e:
        LOGGER.error("Error in %s: %s", agent_name, traceback.format_exc())
        return AgentResult(
            agent_name=agent_name,
            error_message=str(e),
            analyzed_artifacts=[type(artifact).__name__],
        )


def merge_agent_results(
    left: Dict[str, AgentResult], right: Dict[str, AgentResult]
) -> Dict[str, AgentResult]:
    """Reducer to merge agent results dictionary across parallel branches."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def merge_errors(left: List[str], right: List[str]) -> List[str]:
    """Reducer to concatenate error messages across parallel branches."""
    return list(left or []) + list(right or [])


class AnalysisGraphState(TypedDict):
    """State definition for the artifact analysis LangGraph."""

    context: AgentContext
    agent_results: Annotated[Dict[str, AgentResult], merge_agent_results]
    cross_artifact_result: Optional[AgentResult]
    errors: Annotated[List[str], merge_errors]
    summary: Optional[str]


def route_initial_artifacts(state: AnalysisGraphState) -> Sequence[str]:
    """Dynamically route execution to active artifact analyzers based on input context."""
    context = state["context"]
    nodes = []

    if context.deepchecks_artifacts is not None:
        nodes.append("deepchecks_analyzer")
    if context.dataset_artifacts is not None:
        nodes.append("dataset_analyzer")
    if context.model_checkpoint_artifacts is not None:
        nodes.append("checkpoint_analyzer")
    if context.training_artifacts is not None:
        nodes.append("training_analyzer")

    if not nodes:
        raise ValueError("No artifacts provided in AgentContext for analysis.")

    return nodes


def create_analyzer_node(
    artifact_attr: str,
    agent_name: str,
    system_prompt: str,
    llm: BaseChatModel,
    prompt_builder: PromptBuilder,
):
    """Factory creating a LangGraph node function for a specific artifact type."""

    async def node_func(state: AnalysisGraphState) -> Dict[str, Any]:
        context = state["context"]
        artifact = getattr(context, artifact_attr, None)
        if artifact is None:
            return {}

        result = await run_artifact_analysis_node(
            artifact=artifact,
            agent_name=agent_name,
            system_prompt=system_prompt,
            language=context.language,
            llm=llm,
            prompt_builder=prompt_builder,
        )

        errors = []
        if result.error_message:
            errors.append(f"{agent_name}: {result.error_message}")

        return {
            "agent_results": {agent_name: result},
            "errors": errors,
        }

    return node_func


def create_cross_artifact_node(
    llm: BaseChatModel,
    knowledge_bridge: Optional[KnowledgeBridge] = None,
    num_chains: int = 3,
):
    """Factory creating the cross-artifact reasoning synthesis node."""

    async def node_func(state: AnalysisGraphState) -> Dict[str, Any]:
        context = state["context"]
        agent_results = state.get("agent_results", {})

        result = await run_cross_artifact_reasoning(
            previous_analyses=agent_results,
            llm=llm,
            knowledge_bridge=knowledge_bridge,
            output_language=context.language,
            num_chains=num_chains,
        )

        summary = (
            result.additional_outputs.get("summary")
            if result.additional_outputs
            else None
        )

        errors = []
        if result.error_message:
            errors.append(f"CrossArtifactReasoningAgent: {result.error_message}")

        return {
            "cross_artifact_result": result,
            "agent_results": {"CrossArtifactReasoningAgent": result},
            "summary": summary,
            "errors": errors,
        }

    return node_func


def build_analysis_graph(
    llm_config: Optional[LLMConfig] = None,
    knowledge_bridge: Optional[KnowledgeBridge] = None,
    chat_model: Optional[BaseChatModel] = None,
    num_chains: int = 3,
) -> CompiledStateGraph:
    """Build and compile the multi-agent analysis LangGraph StateGraph.

    Args:
        llm_config: Optional LLM configuration.
        knowledge_bridge: Optional KnowledgeBridge for domain knowledge.
        chat_model: Optional pre-configured LangChain chat model (for testing/mocking).
        num_chains: Number of parallel reasoning chains for cross-artifact reasoning.

    Returns:
        CompiledStateGraph ready for async invocation (`ainvoke`).
    """
    llm = chat_model or create_chat_model(llm_config)
    prompt_builder = PromptBuilder()

    workflow = StateGraph(AnalysisGraphState)

    # 1. Add parallel analyzer nodes
    workflow.add_node(
        "deepchecks_analyzer",
        create_analyzer_node(
            artifact_attr="deepchecks_artifacts",
            agent_name="DeepchecksArtifactsAnalyzer",
            system_prompt=DEEPCHECKS_SYSTEM_PROMPT,
            llm=llm,
            prompt_builder=prompt_builder,
        ),
    )
    workflow.add_node(
        "dataset_analyzer",
        create_analyzer_node(
            artifact_attr="dataset_artifacts",
            agent_name="DatasetArtifactsAnalyzer",
            system_prompt=DATASET_SYSTEM_PROMPT,
            llm=llm,
            prompt_builder=prompt_builder,
        ),
    )
    workflow.add_node(
        "checkpoint_analyzer",
        create_analyzer_node(
            artifact_attr="model_checkpoint_artifacts",
            agent_name="ModelCheckpointArtifactsAnalyzer",
            system_prompt=CHECKPOINT_SYSTEM_PROMPT,
            llm=llm,
            prompt_builder=prompt_builder,
        ),
    )
    workflow.add_node(
        "training_analyzer",
        create_analyzer_node(
            artifact_attr="training_artifacts",
            agent_name="TrainingArtifactsAnalyzer",
            system_prompt=TRAINING_SYSTEM_PROMPT,
            llm=llm,
            prompt_builder=prompt_builder,
        ),
    )

    # 2. Add synthesis node
    workflow.add_node(
        "cross_artifact_reasoner",
        create_cross_artifact_node(
            llm=llm,
            knowledge_bridge=knowledge_bridge,
            num_chains=num_chains,
        ),
    )

    # 3. Dynamic Fan-out edges
    workflow.add_conditional_edges(
        START,
        route_initial_artifacts,
        [
            "deepchecks_analyzer",
            "dataset_analyzer",
            "checkpoint_analyzer",
            "training_analyzer",
        ],
    )

    # 4. Fan-in edges to cross-artifact reasoner
    workflow.add_edge("deepchecks_analyzer", "cross_artifact_reasoner")
    workflow.add_edge("dataset_analyzer", "cross_artifact_reasoner")
    workflow.add_edge("checkpoint_analyzer", "cross_artifact_reasoner")
    workflow.add_edge("training_analyzer", "cross_artifact_reasoner")

    # 5. Output to END
    workflow.add_edge("cross_artifact_reasoner", END)

    return workflow.compile()
