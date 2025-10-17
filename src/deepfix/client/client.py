from typing import Optional
import requests

from .pipelines import ArtifactLoadingPipeline, DatasetIngestionPipeline
from .config import MLflowConfig, ArtifactConfig
from ..shared.models import APIRequest, APIResponse,ArtifactPath


class DeepFixClient:
    def __init__(self,api_url: str = "http://localhost:8844",mlflow_config: Optional[MLflowConfig]=None,artifact_config: Optional[ArtifactConfig]=None):

        self.mlflow_config = mlflow_config or MLflowConfig()
        self.artifact_config = artifact_config or ArtifactConfig()
        self.api_url = api_url
        self.artifacts_loader: Optional[ArtifactLoadingPipeline] = None

    def diagnose_dataset(self, dataset_name: str) -> APIResponse:
        self.artifacts_loader = ArtifactLoadingPipeline(mlflow_config=self.mlflow_config, 
                                                        artifact_config=self.artifact_config,
                                                        dataset_name=dataset_name)
        request = self._create_request(dataset_name=dataset_name)
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

    def _create_request(self, dataset_name: str):
        output = self.artifacts_loader.run()
        dataset_artifacts = None
        training_artifacts = None
        deepchecks_artifacts = None
        model_checkpoint_artifacts = None
        
        for key, value in output.items():
            name = ArtifactPath(key)                
            if name == ArtifactPath.DATASET:
                assert isinstance(value, dict), "Dataset artifacts must be a dict"
                dataset_artifacts = value.get(ArtifactPath.DATASET.value)
                deepchecks_artifacts = value.get(ArtifactPath.DEEPCHECKS.value)
            elif name == ArtifactPath.TRAINING:
                training_artifacts = value
            elif name == ArtifactPath.DEEPCHECKS:
                deepchecks_artifacts = value
            elif name == ArtifactPath.MODEL_CHECKPOINT:
                model_checkpoint_artifacts = value

        return APIRequest(
            dataset_artifacts=dataset_artifacts,
            training_artifacts=training_artifacts,
            deepchecks_artifacts=deepchecks_artifacts,
            model_checkpoint_artifacts=model_checkpoint_artifacts,
            dataset_name=dataset_name
        )
    
    def _send_request(self, request: APIRequest):
        response = requests.post(f"{self.api_url}/v1/analyze", json=request.model_dump(), timeout=30)
        if response.status_code != 200:
            raise Exception(f"Error during analysis: status code: {response.status_code} \nand message: {response.text}")
        return APIResponse(**response.json())

    
    

