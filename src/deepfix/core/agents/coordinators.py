from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List


from .models import AgentResult, AnalyzerTypes,AgentContext, Artifacts
from .artifact_analyzers import (DeepchecksArtifactsAnalyzer, 
DatasetArtifactsAnalyzer, 
ModelCheckpointArtifactsAnalyzer)
from .base import ArtifactAnalyzer, Agent
from ..pipelines import Pipeline
from ..pipelines.loaders import LoadDeepchecksArtifacts, LoadModelCheckpoint, LoadTrainingArtifact, LoadDatasetArtifact
from ..config import MLflowConfig
from ...integrations import MLflowManager
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)

class ArtifactAnalysisCoordinator:
    """Main orchestrator that coordinates specialized analyzer agents."""

    def __init__(
        self,
        artifact_sqlite_path: str,
        dataset_name: str,
        mlflow_tracking_uri: str,
        mlflow_experiment_name: Optional[str]=None,
        mlflow_run_id: Optional[str]=None,
        mlflow_run_name: Optional[str]=None,
    ):
        self.analyzer_agents = self._initialize_analyzer_agents()
        self.mlflow_config = MLflowConfig(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name,
            run_id=mlflow_run_id,
            run_name=mlflow_run_name,
        )
        self.artifact_sqlite_path = artifact_sqlite_path
        self.dataset_name = dataset_name
        self.artifacts_loader = self._initialize_artifacts_loader()
    
    def _analyze_one_artifact(self, artifact: Artifacts) -> AgentResult:
        analyzer_agent = self._get_analyzer_agent(artifact)
        if analyzer_agent:
            focused_context = self._create_focused_context(artifact)
            result = analyzer_agent(focused_context)
            return result

    def run(self,) -> AgentContext:   
        # Create context  
        context = self.create_context()
        # Analyze artifacts
        LOGGER.info(f"Analyzing {len(context.artifacts)} artifacts linked to dataset {context.dataset_name}...")
        with ThreadPoolExecutor() as executor:
            for result in executor.map(self._analyze_one_artifact, context.artifacts):
                context.agent_results[result.agent_name] = result
        LOGGER.info(f"Analyzed {len(context.agent_results)} artifacts successfully.")
        return context
    
    def create_context(self,) -> AgentContext:
        output = self.artifacts_loader.run()
        if "artifacts" not in output:
            raise KeyError("No artifacts loaded. Please check the artifacts loader.")
        artifacts = output["artifacts"]
        return AgentContext(artifacts=artifacts, dataset_name=self.dataset_name)
    
    def _get_analyzer_agent(self, artifact: Artifacts) -> ArtifactAnalyzer:
        for analyzer_agent in self.analyzer_agents:
            if analyzer_agent.supports_artifact(artifact):
                return analyzer_agent
        raise ValueError(f"No analyzer agent found for artifact of type: {type(artifact)}")
    
    def _create_focused_context(self, artifact: Artifacts) -> AgentContext:
        return AgentContext(
            artifacts=[artifact],
            dataset_name=self.dataset_name
        )

    def _initialize_analyzer_agents(self) -> List[ArtifactAnalyzer]:
        """Initialize specialized analyzer agents."""
        agents = [DeepchecksArtifactsAnalyzer(),
                  DatasetArtifactsAnalyzer(),
                  ModelCheckpointArtifactsAnalyzer()]
        return agents
    
    def _initialize_artifacts_loader(self)->Pipeline:
        mlflow_manager = MLflowManager.from_config(self.mlflow_config)
        cfg = dict(mlflow_manager=mlflow_manager, 
                artifact_sqlite_path=self.artifact_sqlite_path
                )
        steps = [LoadDeepchecksArtifacts(**cfg),
                LoadModelCheckpoint(**cfg),
                LoadTrainingArtifact(**cfg),
                LoadDatasetArtifact(dataset_name=self.dataset_name,**cfg)
        ]
        return Pipeline(steps=steps)


class AgentCoordinator:
    def __init__(self,):
        self.agents = self._load_agents()
        self.knowledge_bridge = KnowledgeBridge(intelligence_config)

    def run(self, context: dict) -> dict:
        agent_context = AgentContext.from_pipeline_context(context)

        # Run applicable agents sequentially
        for agent in self._get_applicable_agents(agent_context):
            try:
                result = agent.run(agent_context)
                agent_context.agent_results[agent.name] = result
                agent_context.completed_agents.append(agent.name)
            except Exception as e:
                self._handle_agent_failure(agent, e, agent_context)

        # Synthesize results
        synthesizer = NarrativeSynthesizer(self.knowledge_bridge)
        advisor_result = synthesizer.synthesize(agent_context)

        context["advisor_result"] = advisor_result
        return context

    def _load_agents(self):
        pass

    def _get_applicable_agents(self, agent_context: AgentContext):
        pass

    def _handle_agent_failure(
        self, agent: Agent, e: Exception, agent_context: AgentContext
    ):
        pass
