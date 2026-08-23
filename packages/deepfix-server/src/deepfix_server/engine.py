from __future__ import annotations

import traceback
from typing import Optional

import mlflow
from deepfix_kb import KnowledgeBridge

from .agents.graph import build_analysis_graph
from .agents.schemas import AgentContext, AgentResult
from .config import LLMConfig
from .logging import get_logger
from .models import Result

LOGGER = get_logger(__name__)


class DiagnosticSystem:
    """Main orchestrator that coordinates specialized analyzer agents via LangGraph."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        knowledge_bridge: Optional[KnowledgeBridge] = None,
    ):
        """Initialize the diagnostic system with LangGraph state graph.

        Args:
            config: Optional LLM configuration.
            knowledge_bridge: Optional KnowledgeBridge instance.
        """
        self.llm_config = config
        self.agent_name = self.__class__.__name__
        self.knowledge_bridge = knowledge_bridge or KnowledgeBridge()
        self.graph = build_analysis_graph(
            llm_config=self.llm_config,
            knowledge_bridge=self.knowledge_bridge,
        )

    @mlflow.trace(name="DiagnosticSystem.arun")
    async def arun(self, context: AgentContext) -> Result:
        """Run artifact analysis graph asynchronously.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            Result containing analysis results and summary.
        """
        try:
            LOGGER.info(
                f"Starting LangGraph analysis for dataset {context.dataset_name} "
                f"with {len(context.artifacts)} artifacts..."
            )
            initial_state = {
                "context": context,
                "agent_results": dict(context.agent_results or {}),
                "cross_artifact_result": None,
                "errors": [],
                "summary": None,
            }

            final_state = await self.graph.ainvoke(initial_state)

            # Update context with results from graph execution
            agent_results = final_state.get("agent_results", {})
            for name, res in agent_results.items():
                context.agent_results[name] = res

            summary = final_state.get("summary")
            return Result(
                context=context,
                summary=summary,
            )

        except Exception as e:
            LOGGER.error(
                f"Error in {self.agent_name}:\n {traceback.format_exc()}"
            )
            error_result = AgentResult(agent_name=self.agent_name, error_message=str(e))
            context.agent_results[self.agent_name] = error_result
            return Result(
                context=context,
                summary=None,
            )
