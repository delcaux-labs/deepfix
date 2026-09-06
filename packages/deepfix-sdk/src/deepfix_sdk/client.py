import asyncio
import concurrent.futures
import json
import os
import pathlib
import time
from typing import Any, Callable, Optional, Union

import requests
from agno.client import AgentOSClient
from deepfix_core.models import (
    AgentResult,
    Analysis,
    AnalysisJobStatus,
    APIJobResponse,
    APIRequest,
    APIResponse,
    ArtifactPath,
    AutonomousFixRequest,
    DataType,
    FixJob,
    FixJobRequest,
    FixJobStatus,
)
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_delay,
    wait_fixed,
)

from .artifacts import ArtifactRepository, ArtifactStatus
from .config import ArtifactConfig, MLflowConfig
from .data.base import BaseDataset
from .logging import get_logger

console = Console()
LOGGER = get_logger(__name__)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously, safely handling existing active event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class DeepFixClient:
    """Main client for interacting with the DeepFix server.

    This client provides a high-level interface for diagnosing ML datasets,
    ingesting data with quality checks, and leveraging AI-powered recommendations
    to improve your ML workflows.

    Attributes:
        mlflow_config (MLflowConfig): Configuration for MLflow integration.
        api_url (str): Base URL of the DeepFix server.
        timeout (int): Request timeout in seconds.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8844",
        mlflow_config: Optional[MLflowConfig] = None,
        artifact_config: Optional[ArtifactConfig] = None,
        timeout: int = 360,
    ):
        """Initialize the DeepFixClient.

        Args:
            api_url (str, optional): URL of the DeepFix server. Defaults to "http://localhost:8844".
            mlflow_config (MLflowConfig, optional): MLflow configuration for experiment tracking.
                If not provided, a default MLflowConfig is created. Defaults to None.
            artifact_config (ArtifactConfig, optional): Artifact cache configuration used to discover
                stored datasets/models. Defaults to None.
            timeout (int, optional): Request timeout in seconds. Defaults to 360.

        Example:
            >>> client = DeepFixClient(
            ...     api_url="http://localhost:8844",
            ...     timeout=360
            ... )
        """
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.artifact_config = artifact_config or ArtifactConfig()
        self.api_url = api_url
        self.timeout = timeout

        self._analyze_endpoint = self.api_url
        self._artifact_repo: Optional[ArtifactRepository] = None
        self._agent_os_client: Optional[AgentOSClient] = None

    @property
    def agent_os_client(self) -> AgentOSClient:
        """Get or initialize the Agno AgentOSClient instance."""
        if self._agent_os_client is None:
            self._agent_os_client = AgentOSClient(
                base_url=self.api_url,
                timeout=float(self.timeout),
            )
        return self._agent_os_client

    def _get_auth_headers(self) -> dict[str, str]:
        """Construct authorization headers if DEEPFIX_API_KEY is configured."""
        api_key = os.getenv("DEEPFIX_API_KEY")
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _get_artifact_repository(self) -> ArtifactRepository:
        if self._artifact_repo is None:
            self._artifact_repo = ArtifactRepository(
                sqlite_path=self.artifact_config.sqlite_path
            )
        return self._artifact_repo

    def list_datasets(
        self, status: Optional[Union[str, ArtifactStatus]] = None
    ) -> list[dict[str, Any]]:
        """List datasets that have been ingested and are available for diagnosis.

        Args:
            status (ArtifactStatus | str | None): Optional filter by artifact status.

        Returns:
            List of dictionaries describing available datasets. Each record contains:
                - dataset_name: Registered run/dataset name.
                - status: Artifact registration status.
                - mlflow_run_id: Associated MLflow run, if any.
                - local_path: Path to cached artifact on disk, if downloaded.
                - updated_at / created_at: ISO8601 timestamps for auditing.
        """
        repo = self._get_artifact_repository()
        status_enum: Optional[ArtifactStatus] = None
        if status is not None:
            status_enum = (
                status if isinstance(status, ArtifactStatus) else ArtifactStatus(status)
            )
        records = repo.list_records(
            artifact_key=ArtifactPath.DATASET.value, status=status_enum
        )
        datasets = []
        for record in records:
            datasets.append(
                {
                    "dataset_name": record.run_id,
                    "status": record.status.value if record.status else None,
                    "mlflow_run_id": record.mlflow_run_id,
                    "local_path": record.local_path,
                    "created_at": record.created_at.isoformat()
                    if record.created_at
                    else None,
                    "updated_at": record.updated_at.isoformat()
                    if record.updated_at
                    else None,
                }
            )
        datasets.sort(key=lambda item: item["updated_at"] or "", reverse=True)
        return datasets

    def get_dataset_names(
        self, status: Optional[Union[str, ArtifactStatus]] = None
    ) -> list[str]:
        """Convenience method returning only dataset names for UI dropdowns."""
        return [entry["dataset_name"] for entry in self.list_datasets(status=status)]

    def get_diagnosis(
        self,
        train_data: BaseDataset,
        test_data: Optional[BaseDataset] = None,
        model: Any = None,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        language: str = "english",
    ) -> APIResponse:
        """Ingest and diagnose a model in a single operation.

        This convenience method combines ingestion and diagnosis into a single call.
        It first ingests the dataset and model (if provided), then immediately runs
        diagnosis on them to get analysis results and recommendations.

        Args:
            train_data (BaseDataset): Training dataset to ingest. Must be an instance
                of an appropriate dataset class (e.g., ImageClassificationDataset,
                TabularDataset, NLPDataset).
            test_data (BaseDataset, optional): Test/validation dataset. If provided,
                enables cross-dataset validation checks. Defaults to None.
            model (Any, optional): Model to ingest. Must be an instance of a model class.
                Defaults to None.
            model_name (str, optional): Name of the model. Defaults to None.
            batch_size (int, optional): Batch size for processing the dataset.
                Defaults to 8.
            language (str, optional): Language for analysis output. Defaults to "english".

        Returns:
            APIResponse: Response object containing:
                - Analysis results and findings
                - Actionable recommendations

        Raises:
            ValueError: If dataset with the same name exists and overwrite=False, or
                if dataset artifacts cannot be found after ingestion.
            Exception: If ingestion fails, or if the analysis request fails (non-200 status code).

        Example:
            >>> from deepfix_sdk.data import TabularDataset
            >>> import pandas as pd
            >>> df = pd.read_csv("train.csv")
            >>> label = "target"
            >>> cat_features = ["cat_feature1", "cat_feature2"]
            >>> dataset_name = "my-dataset"
            >>> train_dataset = TabularDataset(dataset=df, dataset_name=dataset_name, label=label, cat_features=cat_features)
            >>> response = client.get_diagnosis(
            ...     model_name="my-model",
            ...     train_data=train_dataset,
            ...     batch_size=16
            ... )
            >>> print(response.to_text())
        """
        assert isinstance(train_data, BaseDataset), (
            "train_data must be an instance of BaseDataset"
        )
        assert test_data is None or isinstance(test_data, BaseDataset), (
            "test_data must be an instance of BaseDataset"
        )

        dataset_name = self.get_dataset_name(train_data, test_data)

        # First, ingest the dataset and model
        self.ingest(
            train_data=train_data,
            test_data=test_data,
            model=model,
            model_name=model_name,
            batch_size=batch_size,
            overwrite=True,
        )
        # Then, diagnose the ingested dataset/model
        return self.diagnose(
            dataset_name=dataset_name,
            model_name=model_name,
            language=language,
        )

    def diagnose(
        self,
        dataset_name: str,
        language: str = "english",
        model_name: Optional[str] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> APIResponse:
        """Analyze a run and return diagnostic results with recommendations.

        Args:
            dataset_name (str): Name of the dataset to analyze.
            language (str): Language for analysis output.
            model_name (str, optional): Name of the model.
            stream (bool): Whether to stream the analysis execution in real-time.
            on_chunk (Callable[[str], None], optional): Callback receiving streamed chunks.

        Returns:
            APIResponse: Response object containing findings and recommendations.
        """
        request = self._create_request(dataset_name, model_name or "", language)
        return self._run_diagnosis(request, stream=stream, on_chunk=on_chunk)

    def _load_artifacts(self, dataset_name: str, model_name: str) -> dict:
        from .pipelines import ArtifactLoadingPipeline

        artifact_config = self.artifact_config.model_copy()
        artifact_config.load_dataset_metadata = True
        artifact_config.load_checks = True
        artifact_config.load_model_checkpoint = True
        artifact_config.load_training = False
        return ArtifactLoadingPipeline(
            mlflow_config=self.mlflow_config,
            artifact_config=artifact_config,
            dataset_name=dataset_name,
            model_name=model_name,
        ).run()

    def ingest(
        self,
        train_data: BaseDataset,
        test_data: Optional[BaseDataset] = None,
        model: Any = None,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        overwrite: bool = False,
    ) -> None:
        """Ingest a dataset with optional quality validation.

        This method uploads a dataset to the DeepFix server and optionally performs
        validation checks on the data. Supports multiple data types including images,
        tabular data, NLP text, and general vision datasets.

        Args:
            train_data (BaseDataset): Training dataset to ingest. Must be an instance
                of an appropriate dataset class (e.g., ImageClassificationDataset,
                TabularDataset, NLPDataset). The dataset name is extracted from the
                dataset_name attribute of this object.
            test_data (BaseDataset, optional): Test/validation dataset. If provided,
                enables cross-dataset validation checks. Defaults to None.
            model (Any, optional): Model to ingest. Must be an instance of a model class.
                Defaults to None.
            model_name (str, optional): Name of the model. Defaults to None.
            batch_size (int, optional): Batch size for processing the dataset.
                Defaults to 8.
            overwrite (bool, optional): If True, overwrite existing dataset with the
                same name. If False, raise an error if dataset exists. Defaults to False.

        Raises:
            ValueError: If dataset with the same name exists and overwrite=False.
            Exception: If data validation fails or ingestion fails.

        Example:
            >>> from deepfix_sdk.tabular import TabularDataset
            >>> import pandas as pd
            >>> df = pd.read_csv("train.csv")
            >>> train_dataset = TabularDataset(
            ...     dataset_name="my-dataset",
            ...     data=df
            ... )
            >>> client.ingest(
            ...     train_data=train_dataset,
            ...     batch_size=16
            ... )
        """
        from .pipelines import IngestionPipeline

        data_type = self._get_data_type(train_data, test_data)
        dataset_name = self.get_dataset_name(train_data, test_data)

        dataset_logging_pipeline = IngestionPipeline(
            dataset_name=dataset_name,
            data_type=data_type,
            mlflow_tracking_uri=self.mlflow_config.tracking_uri,
            train_test_validation=test_data is not None,
            data_integrity=True,
            model_evaluation=model is not None,
            batch_size=batch_size,
            overwrite=overwrite,
            model_name=model_name,
        )
        dataset_logging_pipeline.run(
            train_data=train_data, test_data=test_data, model=model
        )

    def _create_request(
        self,
        dataset_name: str,
        model_name: str,
        language: str = "english",
    ):
        """Create an API request for analysis.

        Internal method that loads dataset artifacts and constructs an APIRequest
        object for sending to the DeepFix server.

        Args:
            dataset_name (str): Name of the dataset.
            model_name (str): Name of the model.
            language (str, optional): Language for analysis. Defaults to "english".
            loaded_artifacts (dict): Loaded artifacts from the server.
        Returns:
            APIRequest: Request object configured with dataset artifacts and language.

        Raises:
            ValueError: If dataset artifacts are not found or have unexpected format.
        """
        loaded_artifacts = self._load_artifacts(
            dataset_name=dataset_name, model_name=model_name
        )

        cfg = {
            "dataset_name": dataset_name,
            "language": language,
            "model_name": model_name,
        }
        request = APIRequest(**cfg)
        dataset_artifacts = loaded_artifacts.get(ArtifactPath.DATASET.value, None)
        if dataset_artifacts is not None:
            request.dataset_artifacts = dataset_artifacts.to_dict()

        request.deepchecks_artifacts = loaded_artifacts.get(
            ArtifactPath.DEEPCHECKS.value, None
        )
        request.model_checkpoint_artifacts = loaded_artifacts.get(
            ArtifactPath.MODEL_CHECKPOINT.value, None
        )
        return request

    def _run_diagnosis(
        self,
        request: APIRequest,
        stream: bool = False,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> APIResponse:
        """Execute diagnosis via Agno AgentOSClient workflow execution with REST fallback."""
        
        #headers = self._get_auth_headers()

        try:
            async def run():
                return await self.agent_os_client.run_workflow(
                    workflow_id="analysisworkflow",
                    message=request.to_agent_context().model_dump_json(),
                    #headers=headers,
                )

            wf_output = _run_async(run())
            content = wf_output.content if hasattr(wf_output, "content") else wf_output
            if isinstance(content, APIResponse):
                return content
            if isinstance(content, str):
                return APIResponse.model_validate_json(content)
            return APIResponse.model_validate(content)

        except Exception as agno_err:
            raise agno_err

    def _get_data_type(
        self, train_data: BaseDataset, test_data: Optional[BaseDataset] = None
    ) -> DataType:
        data_type = train_data.data_type
        if test_data is not None:
            test_data_type = test_data.data_type
            if test_data_type != data_type:
                raise ValueError(
                    f"Test data type {test_data_type} does not match train data type {data_type}"
                )
        return data_type

    def get_dataset_name(
        self, train_data: BaseDataset, test_data: Optional[BaseDataset] = None
    ) -> str:
        dataset_name = train_data.name
        if test_data is not None:
            if test_data.name != dataset_name:
                dataset_name = f"{dataset_name}_vs_{test_data.name}"
        return dataset_name
