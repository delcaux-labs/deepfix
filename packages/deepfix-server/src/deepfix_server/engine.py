from __future__ import annotations

import traceback
from typing import Optional

import mlflow
from deepfix_kb import KnowledgeBridge

from .agents.schemas import AgentContext, AgentResult
from .agents.workflow import AnalysisWorkflow
from .config import LLMConfig
from .logging import get_logger
from .models import Result
from .prompt_builders import PromptBuilder

LOGGER = get_logger(__name__)


class DiagnosticSystem:
    """Main orchestrator that coordinates specialized analyzer agents via Agno AnalysisWorkflow."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        knowledge_bridge: Optional[KnowledgeBridge] = None,
        num_chains: int = 3,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        """Initialize the diagnostic system with Agno AnalysisWorkflow.

        Args:
            config: Optional LLM configuration.
            knowledge_bridge: Optional KnowledgeBridge instance.
            num_chains: Number of reasoning chains for cross-artifact synthesis.
            prompt_builder: Optional custom prompt builder instance.
        """
        self.llm_config = config
        self.agent_name = self.__class__.__name__
        self.knowledge_bridge = knowledge_bridge
        self.num_chains = num_chains
        self.workflow = AnalysisWorkflow(
            llm_config=self.llm_config,
            knowledge_bridge=self.knowledge_bridge,
            num_chains=self.num_chains,
            prompt_builder=prompt_builder,
        )

    @mlflow.trace(name="DiagnosticSystem.arun")
    async def arun(self, context: AgentContext) -> Result:
        """Run artifact analysis workflow asynchronously.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            Result containing analysis results and summary.
        """
        try:
            LOGGER.info(
                f"Starting Agno AnalysisWorkflow for dataset '{context.dataset_name}' "
                f"with {len(context.artifacts)} artifacts..."
            )
            return await self.workflow.run_analysis(context)

        except Exception as e:
            LOGGER.error(f"Error in {self.agent_name}:\n {traceback.format_exc()}")
            error_result = AgentResult(agent_name=self.agent_name, error_message=str(e))
            if context.agent_results is None:
                context.agent_results = {}
            context.agent_results[self.agent_name] = error_result
            return Result(
                context=context,
                summary=None,
            )
