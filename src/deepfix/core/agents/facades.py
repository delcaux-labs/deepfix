from typing import Optional
from ..config import LLMConfig, MLflowConfig, ArtifactConfig

from .coordinators import ArtifactAnalysisCoordinator

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

    