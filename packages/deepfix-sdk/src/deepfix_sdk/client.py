from __future__ import annotations

import os
from typing import Any, Optional, Union

import httpx
import requests
from deepfix_core.models import APIRequest, APIResponse, ArtifactPath, DataType
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from .artifacts import ArtifactRepository, ArtifactStatus
from .config import ArtifactConfig, MLflowConfig
from .data.datasets import BaseDataset

console = Console()


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
        api_url: str = "https://deepfix.delcaux.com/api/v1/analyse",
        mlflow_config: Optional[MLflowConfig] = None,
        artifact_config: Optional[ArtifactConfig] = None,
        timeout: int = 30,
    ):
        """Initialize the DeepFixClient.

        Args:
            api_url (str, optional): URL of the DeepFix server. Defaults to "http://localhost:8844".
            mlflow_config (MLflowConfig, optional): MLflow configuration for experiment tracking.
                If not provided, a default MLflowConfig is created. Defaults to None.
            artifact_config (ArtifactConfig, optional): Artifact cache configuration used to discover
                stored datasets/models. Defaults to None.
            timeout (int, optional): Request timeout in seconds. Defaults to 30.

        Example:
            >>> client = DeepFixClient(
            ...     api_url="http://localhost:8844",
            ...     timeout=120
            ... )
        """
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.artifact_config = artifact_config or ArtifactConfig()
        self.api_url = api_url
        self.timeout = timeout

        self._analyze_endpoint = self.api_url
        self._artifact_repo: Optional[ArtifactRepository] = None

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

    async def aget_diagnosis(
        self,
        train_data: BaseDataset,
        test_data: Optional[BaseDataset] = None,
        model: Any = None,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        language: str = "english",
    ) -> APIResponse:
        """Asynchronously ingest and diagnose a model in a single operation."""
        assert isinstance(train_data, BaseDataset), (
            "train_data must be an instance of BaseDataset"
        )
        assert test_data is None or isinstance(test_data, BaseDataset), (
            "test_data must be an instance of BaseDataset"
        )

        dataset_name = self.get_dataset_name(train_data, test_data)

        # Ingestion is primarily local disk operations and MLflow tracking,
        # so keeping it synchronous is acceptable for now.
        self.ingest(
            train_data=train_data,
            test_data=test_data,
            model=model,
            model_name=model_name,
            batch_size=batch_size,
            overwrite=True,
        )
        return await self.adiagnose(
            dataset_name=dataset_name,
            model_name=model_name,
            language=language,
        )

    def diagnose(
        self,
        dataset_name: str,
        language: str = "english",
        model_name: Optional[str] = None,
    ) -> APIResponse:
        """Analyze a run and return diagnostic results with recommendations.

        This method performs a comprehensive analysis of the specified run to identify
        potential issues, quality problems, and provides AI-powered recommendations for
        improvement.

        Args:
            dataset_name (str): Name of the dataset to analyze. Must match a dataset
                that has been previously ingested.
            language (str, optional): Language for analysis output. Defaults to "english".
            model_name (str, optional): Name of the model. Defaults to None.
        Returns:
            APIResponse: Response object containing:
                - Analysis results and findings
                - Quality metrics
                - Actionable recommendations
                - Dataset statistics

        Raises:
            ValueError: If dataset artifacts cannot be found for the specified dataset.
            Exception: If the analysis request fails (non-200 status code).

        Example:
            >>> response = client.diagnose(dataset_name="my-dataset")
            >>> print(response.to_text())
        """
        request = self._create_request(
            dataset_name=dataset_name,
            model_name=model_name,
            language=language,
        )
        response = self._send_request(request)
        return response

    async def adiagnose(
        self,
        dataset_name: str,
        language: str = "english",
        model_name: Optional[str] = None,
    ) -> APIResponse:
        """Asynchronously analyze a run and return diagnostic results."""
        request = self._create_request(
            dataset_name=dataset_name,
            model_name=model_name,
            language=language,
        )
        response = await self._asend_request(request)
        return response

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
            >>> from deepfix_sdk.data.datasets import TabularDataset
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

    def _send_request(self, request: APIRequest) -> APIResponse:
        """Send an analysis request to the DeepFix server.

        Internal method that sends the API request to the server and handles the response.
        Displays a progress spinner during the request and returns the parsed response.

        Args:
            request (APIRequest): The API request object to send to the server.

        Returns:
            APIResponse: Parsed response object from the server containing analysis results.

        Raises:
            Exception: If the server returns a non-200 status code or if the request times out.

        Note:
            Requires the DEEPFIX_API_KEY environment variable to be set for authentication.
        """
        console.print(
            f"[dim]Connecting to: {self._analyze_endpoint}[/dim]",
            style="dim",
        )
        with Live(
            Spinner("dots", text="[cyan]Running analysis...[/cyan]", style="cyan"),
            console=console,
            refresh_per_second=10,
        ):
            payload = request.model_dump()
            # headers = {"X-API-Key": os.getenv("DEEPFIX_API_KEY")}
            headers = {"Authorization": f"Bearer {os.getenv('DEEPFIX_API_KEY')}"}
            response = requests.post(
                self._analyze_endpoint,
                json=payload,
                timeout=self.timeout,
                headers=headers,
            )

            if response.status_code != 200:
                console.print("[red]✗[/red] Analysis failed", style="bold red")
                raise RuntimeError(
                    f"Error during analysis: status code: {response.status_code} \nand message: {response.text}"
                )
            out = APIResponse(**response.json())

        if isinstance(out.error_messages, dict) and any(out.error_messages.values()):
            console.print("[red]✗[/red] Errors during analysis", style="bold red")
            console.print(
                f"Error during analysis: {out.error_messages}"
            )

        console.print("[green]✓[/green] Analysis complete!", style="bold green")
        return out

    async def _asend_request(self, request: APIRequest) -> APIResponse:
        """Asynchronously send an analysis request to the DeepFix server."""
        console.print(
            f"[dim]Connecting to: {self._analyze_endpoint} (Async)[/dim]",
            style="dim",
        )
        with Live(
            Spinner("dots", text="[cyan]Running analysis...[/cyan]", style="cyan"),
            console=console,
            refresh_per_second=10,
        ):
            payload = request.model_dump()
            headers = {"Authorization": f"Bearer {os.getenv('DEEPFIX_API_KEY')}"}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self._analyze_endpoint,
                    json=payload,
                    headers=headers,
                )

            if response.status_code != 200:
                console.print("[red]✗[/red] Analysis failed", style="bold red")
                raise RuntimeError(
                    f"Error during analysis: status code: {response.status_code} \nand message: {response.text}"
                )
            out = APIResponse(**response.json())

        if isinstance(out.error_messages, dict) and any(out.error_messages.values()):
            console.print("[red]✗[/red] Errors during analysis", style="bold red")
            console.print(
                f"Error during analysis: {out.error_messages}"
            )

        console.print("[green]✓[/green] Analysis complete!", style="bold green")
        return out

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
