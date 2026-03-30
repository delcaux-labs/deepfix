import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from deepfix_core.models import Artifacts

from .agents.artifact_analyzers import (
    DatasetArtifactsAnalyzer,
    DeepchecksArtifactsAnalyzer,
    ModelCheckpointArtifactsAnalyzer,
)
from .agents.base import Agent, ArtifactAnalyzer
from .agents.cross_artifact_reasoning import CrossArtifactReasoningAgent
from .config import LLMConfig
from .logging import get_logger
from .models import AgentContext, AgentResult, ArtifactAnalysisResult

LOGGER = get_logger(__name__)


class ArtifactAnalysisCoordinator(Agent):
    """Main orchestrator that coordinates specialized analyzer agents."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
    ):
        super().__init__(config=config)

        # TODO: fix KnowledgeBridge
        # Initialize KnowledgeBridge for OptimizationAdvisor
        # self.knowledge_bridge = KnowledgeBridge()

        # initialize agents and loaders
        self.analyzer_agents = self._initialize_analyzer_agents()
        self.cross_artifact_reasoning_agent = CrossArtifactReasoningAgent(
            llm_config=self._llm_config
        )

        # TODO: fix KnowledgeBridge
        # self.optimization_advisor_agent = OptimizationAdvisorAgent(
        #     knowledge_bridge=self.knowledge_bridge,
        #     llm_config=self._llm_config,
        # )

    async def _analyze_one_artifact(self, artifact: Artifacts) -> AgentResult:
        agent_name = None
        try:
            analyzer_agent = self._get_analyzer_agent(artifact)
            agent_name = analyzer_agent.agent_name
            if analyzer_agent:
                focused_context = self._create_focused_context(artifact)
                result = await analyzer_agent.arun(focused_context)
                return result
        except Exception as e:
            LOGGER.error(f"Error with agent {agent_name}:\n {traceback.format_exc()}")
            raise e

    async def aforward(self, context: AgentContext) -> ArtifactAnalysisResult:
        """Analyze artifacts asynchronously and return results.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            ArtifactAnalysisResult containing analysis results from all agents.
        """
        # 1. Analyze artifacts
        LOGGER.info(
            f"Analyzing {len(context.artifacts)} artifacts linked to dataset {context.dataset_name}..."
        )
        results = await asyncio.gather(
            *[self._analyze_one_artifact(artifact) for artifact in context.artifacts]
        )
        for result in results:
            context.agent_results[result.agent_name] = result

        # 2. Cross-artifact reasoning
        LOGGER.info("Cross-artifact reasoning...")
        cross_artifact_result = await self.cross_artifact_reasoning_agent.arun(
            previous_analyses=context.agent_results, output_language=context.language
        )
        context.agent_results[cross_artifact_result.agent_name] = cross_artifact_result

        # TODO: fix KnowledgeBridge
        # 3. Optimization recommendations (grounded with KnowledgeBridge)
        # LOGGER.info("Generating optimization recommendations...")
        # optimization_result = await self.optimization_advisor_agent.arun(
        #     artifacts_analysis=cross_artifact_result.analysis,
        #     constraints=None,
        # )
        # context.agent_results[optimization_result.agent_name] = optimization_result

        # 4. Output results
        output = ArtifactAnalysisResult(
            context=context,
            summary=cross_artifact_result.additional_outputs.get("summary", None),
        )
        return output

    def forward(self, context: AgentContext) -> ArtifactAnalysisResult:
        """Run aforward synchronously in a separate thread to avoid event loop conflicts."""
        return asyncio.run(self.aforward(context))

    async def arun(self, context: AgentContext) -> ArtifactAnalysisResult:
        """Run the coordinator asynchronously with error handling.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            ArtifactAnalysisResult with analysis results or error message if execution fails.
        """
        try:
            return await self.acall(context)
        except Exception as e:
            LOGGER.error(
                f"Error with coordinator {self.agent_name}:\n {traceback.format_exc()}"
            )
            error_result = AgentResult(agent_name=self.agent_name, error_message=str(e))
            context.agent_results[self.agent_name] = error_result
            return ArtifactAnalysisResult(
                context=context,
                summary=None,
            )

    def run(self, context: AgentContext) -> ArtifactAnalysisResult:
        """Run the coordinator (alias for arun for backward compatibility).

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            ArtifactAnalysisResult containing analysis results from all agents.
        """
        return asyncio.run(self.arun(context))

    def _get_analyzer_agent(self, artifact: Artifacts) -> ArtifactAnalyzer:
        for analyzer_agent in self.analyzer_agents:
            if analyzer_agent.supports_artifact(artifact):
                return analyzer_agent
        raise ValueError(
            f"No analyzer agent found for artifact of type: {type(artifact)}"
        )

    def _create_focused_context(self, artifact: Artifacts) -> AgentContext:
        ctx = AgentContext()
        ctx.insert_artifact(artifact)
        return ctx

    def _initialize_analyzer_agents(self) -> List[ArtifactAnalyzer]:
        """Initialize specialized analyzer agents"""
        agents = [
            DeepchecksArtifactsAnalyzer(config=self._llm_config),
            DatasetArtifactsAnalyzer(config=self._llm_config),
            ModelCheckpointArtifactsAnalyzer(config=self._llm_config),
        ]
        return agents
