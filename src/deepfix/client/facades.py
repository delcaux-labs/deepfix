from typing import Optional
from .config import LLMConfig, MLflowConfig, ArtifactConfig
import mlflow

class ArtifactAnalysisCoordinator:
    """Main orchestrator that coordinates specialized analyzer agents."""

    def __init__(
        self,
        artifact_sqlite_path: Optional[str]=None,
        dataset_name: Optional[str]=None,
        mlflow_tracking_uri: Optional[str]=None,
        mlflow_experiment_name: Optional[str]=None,
        mlflow_run_id: Optional[str]=None,
        mlflow_run_name: Optional[str]=None,
        mlflow_config: Optional[MLflowConfig]=None,
        trace_llm_requests: bool=True,
        artifact_config: Optional[ArtifactConfig]=None,
        llm_config: Optional[LLMConfig]=None,
        env_file: Optional[str]=None,
    ):
        self.mlflow_config = mlflow_config
        self.artifact_config = artifact_config
        self.llm_config = llm_config        
        self.training_dynamics_config = training_dynamics_config

        if llm_config is None:
            assert env_file is not None, "env_file must be provided if llm_config is not provided"
            self.llm_config = LLMConfig.load_from_env(env_file=env_file)

        if mlflow_config is None:
            self.mlflow_config = MLflowConfig(tracking_uri=mlflow_tracking_uri,
                                            experiment_name=mlflow_experiment_name,
                                            run_id=mlflow_run_id,
                                            run_name=mlflow_run_name,
                                            trace_dspy=trace_llm_requests
                                        )

        if artifact_config is None:
            self.artifact_config = ArtifactConfig(dataset_name=dataset_name,
                                                sqlite_path=artifact_sqlite_path,
                                                load_dataset_metadata=True,
                                                load_checks=True,
                                                load_model_checkpoint=True,
                                                load_training=True,
                                                download_if_missing=True,
                                                cache_enabled=True
                                                )
        
                
        #initialize agents and loaders
        self.dataset_name = self.artifact_config.dataset_name
        self.artifacts_loader = self._initialize_artifacts_loader()
        self.analyzer_agents = self._initialize_analyzer_agents()
        self.trace_llm_requests = self.mlflow_config.trace_dspy

        # Initialize tracing
        self._initialize_tracing()    

    def _initialize_tracing(self):
        if self.trace_llm_requests:
            mlflow.dspy.autolog()
            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            mlflow.set_experiment("DSPy-tracing")           
    
    @classmethod
    def from_config(cls, mlflow_config: MLflowConfig, 
                    artifact_config: ArtifactConfig, 
                    llm_config: Optional[LLMConfig] = None, 
                    env_file: Optional[str] = None) -> "ArtifactAnalysisCoordinator":
        return cls(mlflow_config=mlflow_config, artifact_config=artifact_config, llm_config=llm_config, env_file=env_file)
    
    
    def run(self,max_workers:int=3) -> ArtifactAnalysisResult:   
        #1. Create context  
        LOGGER.info(f"Creating context for dataset {self.dataset_name}...")
        context = self.create_context()

        #2. Analyze artifacts
        LOGGER.info(f"Analyzing {len(context.artifacts)} artifacts linked to dataset {context.dataset_name}...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(self._analyze_one_artifact, context.artifacts):
                context.agent_results[result.agent_name] = result

        #3. Cross-artifact reasoning
        LOGGER.info(f"Cross-artifact reasoning...")
        out = self.cross_artifact_reasoning_agent(previous_analyses=context.agent_results)
        context.agent_results[out.agent_name] = out

        #4. Output results
        output = ArtifactAnalysisResult(context=context, 
                                        summary=out.additional_outputs['summary'],
                                    )
        return output
    
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
        agents = [DeepchecksArtifactsAnalyzer(config=self.llm_config),
                  DatasetArtifactsAnalyzer(config=self.llm_config),
                  ModelCheckpointArtifactsAnalyzer(config=self.llm_config),
                  TrainingArtifactsAnalyzer(llm_config=self.llm_config,config=self.training_dynamics_config)]
        return agents
    
    def _initialize_artifacts_loader(self)->ArtifactLoadingPipeline:
        return ArtifactLoadingPipeline.from_config(mlflow_config=self.mlflow_config, 
                                                    artifact_config=self.artifact_config)


class DatasetAnalyzer:

    def __init__(self,api_key: Optional[str] = None,
                    base_url: Optional[str] = None,
                    model_name: Optional[str] = None,
                    temperature: float = 0.7,
                    max_tokens: int = 8000,
                    env_file: Optional[str] = None,
                    sqlite_path: Optional[str] = None,
                    mlflow_config: Optional[MLflowConfig] = None,
                ):

        if env_file is not None:
            llm_config = LLMConfig.load_from_env(env_file=env_file)
        else:
            llm_config = LLMConfig(api_key=api_key,
                                    base_url=base_url,
                                    model_name=model_name,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    track_usage=True,
                                    cache=True
                            )
        
        if mlflow_config is not None:
            self.mlflow_config = mlflow_config
        else:
            self.mlflow_config = MLflowConfig(trace_dspy=llm_config.track_usage)        

        self.llm_config = llm_config
        self.env_file = env_file
        self.sqlite_path = sqlite_path
    
    def run(self, dataset_name: str,num_workers: int = 3):
        artifact_config = ArtifactConfig(dataset_name=dataset_name,
                                        load_dataset_metadata=True,
                                        load_checks=False,
                                        load_model_checkpoint=False,
                                        load_training=False,
                                        download_if_missing=True,
                                        cache_enabled=True,
                                    )
        if self.sqlite_path is not None:
            artifact_config.sqlite_path = self.sqlite_path

        cfg = dict(mlflow_config=self.mlflow_config, 
            artifact_config=artifact_config,
            llm_config=self.llm_config,
            env_file=self.env_file)
        
        # 1. artifact analysis
        runner = ArtifactAnalysisCoordinator.from_config(**cfg)
        result = runner.run(max_workers=num_workers)
        return result

    