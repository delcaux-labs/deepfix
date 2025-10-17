from typing import Optional
import requests

from .pipelines import ArtifactLoadingPipeline, DatasetIngestionPipeline
from .config import MLflowConfig, ArtifactConfig
from ..shared.models import APIRequest, APIResponse


class DeepFixClient:
    def __init__(self,api_url: str = "http://localhost:8844",mlflow_config: Optional[MLflowConfig]=None,artifact_config: Optional[ArtifactConfig]=None):

        self.mlflow_config = mlflow_config or MLflowConfig()
        self.artifact_config = artifact_config or ArtifactConfig()
        self.api_url = api_url
        self.artifacts_loader = ArtifactLoadingPipeline.from_config(mlflow_config=self.mlflow_config, 
                                                                            artifact_config=self.artifact_config)

    def diagnose_dataset(self, dataset_name: str) -> APIResponse:
        request = self._create_request(dataset_name=dataset_name)
        response = self._send_request(request)
        return response
    
    def ingest_dataset(self, dataset_name: str,train_data,test_data):
        dataset_logging_pipeline = DatasetIngestionPipeline(dataset_name=dataset_name,
                                                train_test_validation=True,
                                                data_integrity=True,
                                                batch_size=8,
                                                overwrite=False # True -> i.e. delete and re-create
                                                )
        dataset_logging_pipeline.run(train_data=train_data,
                                    test_data=test_data,
                                )

    def _create_request(self,dataset_name: str):
        output = self.artifacts_loader.run()
        if "artifacts" not in output:
            raise KeyError("No artifacts loaded. Please check the artifacts loading pipeline.")
        artifacts = output["artifacts"]
        return APIRequest(artifacts=artifacts, dataset_name=dataset_name)
    
    def _send_request(self, request: APIRequest):
        response = requests.post(f"{self.api_url}/v1/analyze", json=request.model_dump(), timeout=30)
        if response.status_code != 200:
            raise Exception(f"Error during analysis: status code: {response.status_code} \nand message: {response.text}")
        return APIResponse(**response.json())

    
    

