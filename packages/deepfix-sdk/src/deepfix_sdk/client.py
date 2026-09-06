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
        api_url: str = "http://localhost:4141",
        mlflow_config: Optional[MLflowConfig] = None,
        artifact_config: Optional[ArtifactConfig] = None,
        timeout: int = 360,
    ):
        """Initialize the DeepFixClient.

        Args:
            api_url (str, optional): URL of the DeepFix server. Defaults to "http://localhost:4141".
            mlflow_config (MLflowConfig, optional): MLflow configuration for experiment tracking.
                If not provided, a default MLflowConfig is created. Defaults to None.
            artifact_config (ArtifactConfig, optional): Artifact cache configuration used to discover
                stored datasets/models. Defaults to None.
            timeout (int, optional): Request timeout in seconds. Defaults to 360.

        Example:
            >>> client = DeepFixClient(
            ...     api_url="http://localhost:4141",
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
    def server_base_url(self) -> str:
        """Derive the root base URL for AgentOS from api_url."""
        base_url = self.api_url.rstrip("/")
        for suffix in (
            "/api/v2/analyse",
            "/api/v1/analyse",
            "/api/v2/fix",
            "/v2/analyse",
            "/v1/analyse",
            "/v2/fix",
        ):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        return base_url or "http://localhost:4141"

    @property
    def agent_os_client(self) -> AgentOSClient:
        """Get or initialize the Agno AgentOSClient instance."""
        if self._agent_os_client is None:
            self._agent_os_client = AgentOSClient(
                base_url=self.server_base_url,
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

    def _parse_agent_result_entry(self, key: str, value: Any) -> AgentResult:
        """Parse individual agent result entry into a validated AgentResult."""
        if isinstance(value, AgentResult):
            return value
        if isinstance(value, dict):
            if "analysis" not in value and ("findings" in value or "recommendations" in value):
                f = value.get("findings")
                r = value.get("recommendations")
                analysis_list = [Analysis(findings=f, recommendations=r)] if f and r else []
                return AgentResult(
                    agent_name=value.get("agent_name", key),
                    analysis=analysis_list,
                    analyzed_artifacts=value.get("analyzed_artifacts", []),
                    retrieved_knowledge=value.get("retrieved_knowledge"),
                    additional_outputs=value.get("additional_outputs", {"summary": value.get("summary")}),
                    error_message=value.get("error_message"),
                )
            try:
                return AgentResult.model_validate(value)
            except Exception as e:
                LOGGER.warning("Could not validate AgentResult for %s: %s", key, e)
                return AgentResult(agent_name=key, additional_outputs={"raw": value})
        return AgentResult(agent_name=key, additional_outputs={"raw": value})

    def _transform_workflow_output_to_api_response(
        self,
        raw_output: Any,
        dataset_name: Optional[str] = None,
    ) -> APIResponse:
        """Transform Agno WorkflowRunOutput or dictionary payload into SDK APIResponse."""
        content = getattr(raw_output, "content", raw_output)

        if isinstance(content, str):
            try:
                content = json.loads(content)
            except Exception:
                try:
                    import ast
                    content = ast.literal_eval(content)
                except Exception:
                    try:
                        import json_repair
                        content = json_repair.loads(content)
                    except Exception:
                        pass

        if hasattr(content, "model_dump"):
            content = content.model_dump(mode="json")
        elif not isinstance(content, dict):
            if hasattr(content, "context"):
                content = {
                    "context": getattr(content, "context"),
                    "summary": getattr(content, "summary", None),
                    "additional_outputs": getattr(content, "additional_outputs", {}),
                }
            else:
                return APIResponse(summary=str(content), dataset_name=dataset_name)

        ctx = content.get("context") if isinstance(content.get("context"), dict) else {}
        raw_agent_results = content.get("agent_results") or ctx.get("agent_results") or {}

        agent_results = {k: self._parse_agent_result_entry(k, v) for k, v in raw_agent_results.items()}
        summary = content.get("summary") or ctx.get("summary")
        if not summary:
            findings_count = sum(
                len(ar.analysis) for ar in agent_results.values() if ar.analysis
            )
            summary = (
                f"DeepFix diagnostic analysis completed for '{ds_name}'. "
                f"Identified {findings_count} finding(s) across analyzed artifacts."
            )
        error_messages = content.get("error_messages") or ctx.get("error_messages")
        ds_name = dataset_name or content.get("dataset_name") or ctx.get("dataset_name")

        return APIResponse(
            agent_results=agent_results,
            summary=summary,
            error_messages=error_messages,
            dataset_name=ds_name,
            additional_outputs=content.get("additional_outputs", {}),
        )

    async def _execute_streaming_workflow(
        self,
        message_str: str,
        headers: dict[str, str],
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, Optional[dict]]:
        """Stream workflow execution yielding text chunks and returning final payload."""
        full_chunks: list[str] = []
        final_payload: Optional[dict] = None
        async for event in self.agent_os_client.run_workflow_stream(
            workflow_id="analysisworkflow",
            message=message_str,
            headers=headers,
        ):
            chunk_text = None
            if hasattr(event, "content") and event.content:
                if isinstance(event.content, str):
                    chunk_text = event.content
                elif isinstance(event.content, dict):
                    final_payload = event.content
            elif hasattr(event, "delta") and event.delta:
                chunk_text = str(event.delta)

            if chunk_text:
                full_chunks.append(chunk_text)
                if on_chunk:
                    on_chunk(chunk_text)
        return "".join(full_chunks), final_payload

    def _run_diagnosis(
        self,
        request: APIRequest,
        stream: bool = False,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> APIResponse:
        """Execute diagnosis via Agno AgentOSClient workflow execution with REST fallback."""
        req_dump = request.model_dump(mode="json")
        message_dict = {
            "context": {
                "dataset_name": req_dump.get("dataset_name"),
                "dataset_artifacts": req_dump.get("dataset_artifacts"),
                "training_artifacts": req_dump.get("training_artifacts"),
                "deepchecks_artifacts": req_dump.get("deepchecks_artifacts"),
                "model_checkpoint_artifacts": req_dump.get("model_checkpoint_artifacts"),
                "language": req_dump.get("language"),
                "model_name": req_dump.get("model_name"),
            }
        }
        message_str = json.dumps(message_dict)
        headers = self._get_auth_headers()

        try:
            if stream:
                text_out, final_data = _run_async(
                    self._execute_streaming_workflow(message_str, headers, on_chunk)
                )
                output = final_data if final_data is not None else text_out
                return self._transform_workflow_output_to_api_response(
                    output, dataset_name=request.dataset_name
                )

            async def _run_sync():
                return await self.agent_os_client.run_workflow(
                    workflow_id="analysisworkflow",
                    message=message_str,
                    headers=headers,
                )

            wf_output = _run_async(_run_sync())
            return self._transform_workflow_output_to_api_response(
                wf_output, dataset_name=request.dataset_name
            )

        except Exception as agno_err:
            LOGGER.warning(
                "AgentOS workflow run failed (%s). Attempting fallback to /v2/analyse endpoint.",
                agno_err,
            )
            try:
                return self._send_request(request)
            except Exception as rest_err:
                raise RuntimeError(
                    f"Failed to communicate with DeepFix server at {self.server_base_url}: {agno_err}"
                ) from rest_err

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
        out = job_data.result        

        if isinstance(out.error_messages, dict) and any(out.error_messages.values()):
            console.print("[red]x[/red] Errors during analysis", style="bold red")
            console.print(f"Error details: {out.error_messages}")

        console.print("[green]v[/green] Analysis complete!", style="bold green")
        return out

    def _submit_job(self, request: APIRequest, url: Optional[str] = None) -> APIJobResponse:
        """Submit an analysis job to the server.

        Args:
            request (APIRequest): The analysis request.
            url (Optional[str]): Optional custom endpoint URL.

        Returns:
            APIJobResponse: The response from the server containing job metadata.
        """
        endpoint = url or self.api_url
        headers = {"Authorization": f"Bearer {os.getenv('DEEPFIX_API_KEY')}"}

        console.print(
            f"[dim]Submitting analysis job to: {endpoint}[/dim]",
            style="dim",
        )

        request_timeout = self.timeout

        try:
            response = requests.post(
                endpoint,
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
