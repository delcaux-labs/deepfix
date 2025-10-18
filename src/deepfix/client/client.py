from typing import Optional
import requests
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live

from .pipelines import ArtifactLoadingPipeline, DatasetIngestionPipeline
from .config import MLflowConfig, ArtifactConfig
from ..shared.models import APIRequest, APIResponse,ArtifactPath

console = Console()

class DeepFixClient:
    def __init__(self,api_url: str = "http://localhost:8844",mlflow_config: Optional[MLflowConfig]=None,timeout: int = 30):

        self.mlflow_config = mlflow_config or MLflowConfig()
        self.api_url = api_url
        self.artifacts_loader: Optional[ArtifactLoadingPipeline] = None
        self.timeout = timeout

    def diagnose_dataset(self, dataset_name: str) -> APIResponse:
        artifact_config = ArtifactConfig(load_dataset_metadata=True, load_checks=False, load_model_checkpoint=False, load_training=False)
        self.artifacts_loader = ArtifactLoadingPipeline(mlflow_config=self.mlflow_config, 
                                                        artifact_config=artifact_config,
                                                        dataset_name=dataset_name)
        request = self._create_dataset_request(dataset_name=dataset_name)
        response = self._send_request(request)
        return response
    
    def ingest_dataset(self, dataset_name: str,train_data,test_data,
                    train_test_validation=True,
                    data_integrity=True,
                    batch_size=8,
                    overwrite=False):
        dataset_logging_pipeline = DatasetIngestionPipeline(dataset_name=dataset_name,
                                                train_test_validation=train_test_validation,
                                                data_integrity=data_integrity,
                                                batch_size=batch_size,
                                                overwrite=overwrite
                                                )
        dataset_logging_pipeline.run(train_data=train_data,
                                    test_data=test_data,
                                )
    
    def _create_dataset_request(self, dataset_name: str):
        output = self.artifacts_loader.run()
        cfg = {'dataset_name': dataset_name}
        value = output.get(ArtifactPath.DATASET.value)
        if value is None:
            raise ValueError(f"Dataset artifacts not found for dataset {dataset_name}")
        if not isinstance(value, dict):
            raise ValueError(f"Expected a dictionary, got {type(value)}")
        cfg["dataset_artifacts"] = value.get(ArtifactPath.DATASET.value)
        cfg["deepchecks_artifacts"] = value.get(ArtifactPath.DEEPCHECKS.value)
        return APIRequest(**cfg)
    
    def _send_request(self, request: APIRequest):
                
        with Live(Spinner("dots", text="[cyan]Running analysis...[/cyan]", style="cyan"), console=console, refresh_per_second=10):
            response = requests.post(f"{self.api_url}/v1/analyze", json=request.model_dump(), timeout=self.timeout)
                    
            if response.status_code != 200:
                console.print("[red]✗[/red] Analysis failed", style="bold red")
                raise Exception(f"Error during analysis: status code: {response.status_code} \nand message: {response.text}")
        
        console.print("[green]✓[/green] Analysis complete!", style="bold green")
        return APIResponse(**response.json())

    
    

