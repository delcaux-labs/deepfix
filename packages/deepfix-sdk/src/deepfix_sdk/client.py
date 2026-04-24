from __future__ import annotations

import os
from typing import Any, Optional, Union

import requests
from deepfix_core.models import (
    APIRequest,
    APIResponse,
    APIJobResponse,
    ArtifactPath,
    DataType,
    AnalysisJobStatus,
)
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
import time
from tenacity import (
    Retrying,
    stop_after_delay,
    wait_fixed,
    retry_if_result,
    retry_if_exception_type,
    RetryError,
)

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
        api_url: str = "https://deepfix.delcaux.com/api/v2/analyse",
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
            timeout (int, optional): Request timeout in seconds. Defaults to 30.

        Example:
            >>> client = DeepFixClient(
            ...     api_url="http://localhost:8844/api/v2/analyse",
            ...     timeout=360
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

    def diagnose(
        self,
        dataset_name: str,
        language: str = "english",
        model_name: Optional[str] = None,
    ) -> APIResponse:
        """Analyze a run and return diagnostic results with recommendations.

        Args:
            dataset_name (str): Name of the dataset to analyze.
            language (str): Language for analysis output.
            model_name (str, optional): Name of the model.
        Returns:
            APIResponse: Response object containing findings and recommendations.
        """
        request = self._create_request(dataset_name, model_name or "", language)
        return self._send_request(request)

    def get_result(self, job_id: str, polling_interval: float = 5.0) -> APIResponse:
        """Fetch the results of an existing analysis job by its ID.

        This method will block and poll the server if the job is still in progress.

        Args:
            job_id (str): The ID of the analysis job.
            polling_interval (float): Seconds between polling attempts.

        Returns:
            APIResponse: The analysis results once completed.
        """
        return self._poll_for_results(job_id, polling_interval=polling_interval)

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

        Internal method that handles both synchronous (v1) and asynchronous (v2)
        interactions. Skips polling if results are returned immediately.

        Args:
            request (APIRequest): The API request object to send to the server.

        Returns:
            APIResponse: Parsed response object from the server containing analysis results.

        Raises:
            RuntimeError: If the job submission or polling fails, or if the analysis fails.
        """
        # 1. Submit the job
        job_data = self._submit_job(request)

        # 2. Check if results are immediate (synchronous v1)
        if job_data.status == AnalysisJobStatus.COMPLETED.value and job_data.result:
            out = job_data.result
        else:
            # 3. Poll for results (asynchronous v2)
            out = self._poll_for_results(job_data.job_id, polling_interval=5.0)

        if isinstance(out.error_messages, dict) and any(out.error_messages.values()):
            console.print("[red]x[/red] Errors during analysis", style="bold red")
            console.print(f"Error details: {out.error_messages}")

        console.print("[green]v[/green] Analysis complete!", style="bold green")
        return out

    def _submit_job(self, request: APIRequest) -> APIJobResponse:
        """Submit an analysis job to the server.

        Args:
            request (APIRequest): The analysis request.

        Returns:
            APIJobResponse: The response from the server containing job metadata.
        """
        headers = {"Authorization": f"Bearer {os.getenv('DEEPFIX_API_KEY')}"}

        console.print(
            f"[dim]Submitting analysis job to: {self.api_url}[/dim]",
            style="dim",
        )

        request_timeout = self.timeout if "/v1/" in self.api_url else 5.0

        try:
            response = requests.post(
                self.api_url,
                json=request.model_dump(),
                timeout=request_timeout,
                headers=headers,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to DeepFix server: {str(e)}")

        if response.status_code not in [200, 202]:
            console.print("[red]x[/red] Request failed", style="bold red")
            raise RuntimeError(
                f"Error from DeepFix server: {response.status_code} - {response.text}"
            )

        job_data = APIJobResponse.model_validate(response.json())
        if not job_data.job_id:
            raise RuntimeError("Server did not return a job_id")

        console.print(
            f"[dim]Request accepted (ID: {job_data.job_id})[/dim]", style="dim"
        )

        return job_data

    def _poll_for_results(
        self, job_id: str, polling_interval: float = 10.0
    ) -> APIResponse:
        """Poll the server for the results of a background job.

        Args:
            job_id (str): The ID of the job to poll.

        Returns:
            APIResponse: The final analysis results.
        """
        headers = {"Authorization": f"Bearer {os.getenv('DEEPFIX_API_KEY')}"}

        # Determine base URL for polling (e.g., from .../api/v2/analyse to .../api)
        base_url = self.api_url.rstrip("/")
        if "/v2/analyse" in base_url:
            base_url = base_url.split("/v2/analyse")[0]
        elif "/v1/analyse" in base_url:
            base_url = base_url.split("/v1/analyse")[0]

        def is_not_finished(job_data: Optional[APIJobResponse]) -> bool:
            if job_data is None:
                return True
            return not job_data.is_finished

        start_time = time.time()
        with Live(
            Spinner("dots", text="[cyan]Analysis pending...[/cyan]", style="cyan"),
            console=console,
            refresh_per_second=10,
        ) as live:
            try:
                for attempt in Retrying(
                    stop=stop_after_delay(self.timeout),
                    wait=wait_fixed(polling_interval),
                    retry=retry_if_result(is_not_finished)
                    | retry_if_exception_type((requests.RequestException, IOError)),
                    reraise=False,
                ):
                    with attempt:
                        polling_url = f"{base_url}/v2/jobs/{job_id}"
                        job_response = requests.get(
                            polling_url,
                            headers=headers,
                            timeout=2.0,
                        )
                        if job_response.status_code != 200:
                            raise requests.RequestException(
                                f"Polling failed: {job_response.status_code}"
                            )

                        job_data = APIJobResponse.model_validate(job_response.json())
                        status = job_data.status

                        # Update spinner with current status
                        live.update(
                            Spinner(
                                "dots",
                                text=f"[cyan]Analysis in progress ({status.lower()})...[/cyan]",
                                style="cyan",
                            )
                        )

                        if status == AnalysisJobStatus.COMPLETED.value:
                            live.update(
                                Spinner(
                                    "dots", text="[green]Processing results...[/green]"
                                )
                            )
                            if job_data.result is None:
                                raise RuntimeError(
                                    "Job completed but no result data found"
                                )
                            return job_data.result

                        elif status == AnalysisJobStatus.FAILED.value:
                            error_msg = job_data.error or "Unknown error"
                            live.update(
                                Spinner(
                                    "dots",
                                    text=f"[red]Analysis failed: {error_msg}[/red]",
                                )
                            )
                            raise RuntimeError(f"DeepFix analysis failed: {error_msg}")

                # If the loop finishes without returning, it means it timed out
                raise RuntimeError("Analysis timed out")

            except Exception as e:
                if isinstance(e, (RuntimeError, RetryError)):
                    if isinstance(e, RetryError):
                        raise RuntimeError("Analysis timed out") from e
                    raise e
                raise RuntimeError(f"Analysis polling failed: {str(e)}")

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
