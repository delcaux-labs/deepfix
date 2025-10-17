from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from .models import AgentResult, AgentContext, ArtifactAnalysisResult
from .agents.artifact_analyzers import (DeepchecksArtifactsAnalyzer, 
DatasetArtifactsAnalyzer, 
ModelCheckpointArtifactsAnalyzer,
)
from .agents.base import ArtifactAnalyzer
from .config import  LLMConfig
from .logging import get_logger
from .agents.cross_artifact_reasoning import CrossArtifactReasoningAgent

LOGGER = get_logger(__name__)

class ArtifactAnalysisCoordinator:
    """Main orchestrator that coordinates specialized analyzer agents."""

    def __init__(
        self,
        llm_config: Optional[LLMConfig]=None,
    ):
        self.llm_config = llm_config        

        #initialize agents and loaders
        self.analyzer_agents = self._initialize_analyzer_agents()
        self.cross_artifact_reasoning_agent = CrossArtifactReasoningAgent(llm_config=self.llm_config)

           
    def _analyze_one_artifact(self, artifact) -> AgentResult:
        try:
            analyzer_agent = self._get_analyzer_agent(artifact)
            if analyzer_agent:
                focused_context = self._create_focused_context(artifact)
                result = analyzer_agent(focused_context)
                return result
        except Exception as e:
            LOGGER.error(f"Error analyzing artifact {artifact}: {e}")
            return AgentResult(agent_name="None", error_message=str(e))

    def run(self,context:AgentContext,max_workers:int=3) -> ArtifactAnalysisResult:   

        #1. Analyze artifacts
        LOGGER.info(f"Analyzing {len(context.artifacts)} artifacts linked to dataset {context.dataset_name}...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(self._analyze_one_artifact, context.artifacts):
                context.agent_results[result.agent_name] = result

        #2. Cross-artifact reasoning
        LOGGER.info(f"Cross-artifact reasoning...")
        out = self.cross_artifact_reasoning_agent.run(previous_analyses=context.agent_results)
        context.agent_results[out.agent_name] = out

        #3. Output results
        output = ArtifactAnalysisResult(context=context, 
                                        summary=out.additional_outputs['summary'],
                                    )
        return output
        
    def _get_analyzer_agent(self, artifact) -> ArtifactAnalyzer:
        for analyzer_agent in self.analyzer_agents:
            if analyzer_agent.supports_artifact(artifact):
                return analyzer_agent
        raise ValueError(f"No analyzer agent found for artifact of type: {type(artifact)}")
    
    def _create_focused_context(self, artifact) -> AgentContext:
        return AgentContext(
            artifacts=[artifact],
        )

    def _initialize_analyzer_agents(self) -> List[ArtifactAnalyzer]:
        """Initialize specialized analyzer agents."""
        agents = [DeepchecksArtifactsAnalyzer(config=self.llm_config),
                  DatasetArtifactsAnalyzer(config=self.llm_config),
                  ModelCheckpointArtifactsAnalyzer(config=self.llm_config)]
        return agents


