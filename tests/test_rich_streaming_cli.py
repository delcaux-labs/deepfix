import json
from unittest.mock import MagicMock, patch

import pytest
from deepfix_core.models import FinalFixReport, FixJob, FixJobRequest, FixJobStatus
from deepfix_sdk.cli import _build_metrics_table, _get_phase_icon, app
from deepfix_sdk.client import DeepFixClient
from deepfix_server.api import app as fastapi_app
from fastapi.testclient import TestClient
from typer.testing import CliRunner


@pytest.fixture
def api_client(tmp_path):
    """TestClient for FastAPI DeepFix Server with initialized test SQLite DB."""
    from deepfix_server.config import settings
    from deepfix_server.database import Base, get_engine, init_database

    test_db = f"sqlite:///{tmp_path}/test.db"
    settings.database_url = test_db
    init_database(test_db, database_echo=False)
    engine = get_engine()
    if engine:
        Base.metadata.create_all(bind=engine)
    with TestClient(fastapi_app) as client:
        yield client


@pytest.fixture
def cli_runner():
    """Typer CliRunner for testing CLI commands."""
    return CliRunner()


class TestRichStreamingModels:
    """Test FixJob model schema enhancements."""

    def test_fix_job_defaults_and_fields(self):
        job = FixJob(
            job_id="fix_test_123",
            dataset_name="breast_cancer",
            target_metric="f1",
            target_value=0.92,
            max_iterations=5,
        )
        assert job.job_id == "fix_test_123"
        assert job.status == FixJobStatus.PENDING
        assert job.phase is None
        assert job.events == []
        assert job.intermediate_metrics == []

    def test_fix_job_with_events_and_intermediate_metrics(self):
        events = [
            {"timestamp": "2026-08-23T23:00:00Z", "phase": "Diagnosing", "message": "Analyzing EDA"},
            {"timestamp": "2026-08-23T23:00:10Z", "phase": "Synthesizing Fix", "message": "Applying LightGBM"},
        ]
        metrics = [
            {"iteration": 1, "loss": 0.45, "accuracy": 0.88, "f1": 0.86, "roc_auc": 0.90},
            {"iteration": 2, "loss": 0.28, "accuracy": 0.94, "f1": 0.93, "roc_auc": 0.96},
        ]
        job = FixJob(
            job_id="fix_test_456",
            status=FixJobStatus.IN_PROGRESS,
            phase="Training Run #2",
            events=events,
            intermediate_metrics=metrics,
            iteration=2,
            max_iterations=5,
        )
        assert job.phase == "Training Run #2"
        assert len(job.events) == 2
        assert len(job.intermediate_metrics) == 2
        assert job.intermediate_metrics[1]["accuracy"] == 0.94


class TestServerFixJobEndpoints:
    """Test server-side fix endpoints with phase, events, and cancellation."""

    def test_submit_and_get_fix_job_with_phases_and_events(self, api_client):
        # 1. Submit job
        req = FixJobRequest(
            dataset_name="synthetic_tabular",
            target_metric="accuracy",
            target_value=0.90,
            max_iterations=3,
        )
        resp = api_client.post("/v2/fix", json=req.model_dump())
        assert resp.status_code == 202
        data = resp.json()
        job_id = data["job_id"]
        assert data["status"] == "PENDING"
        assert data["phase"] == "Pending"
        assert len(data["events"]) >= 1

        # 2. Get job status
        get_resp = api_client.get(f"/v2/fix/{job_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["job_id"] == job_id
        assert get_data["phase"] is not None

    def test_step_reporting_and_metrics_accumulation(self, api_client):
        # Submit a job first
        req = FixJobRequest(
            dataset_name="synthetic_dataset",
            target_metric="accuracy",
            target_value=0.95,
            max_iterations=5,
        )
        resp = api_client.post("/v2/fix", json=req.model_dump())
        job_id = resp.json()["job_id"]

        # Report step 1
        step1_resp = api_client.post(
            f"/v2/fix/{job_id}/step",
            json={
                "iteration": 1,
                "phase": "Training Run #1",
                "message": "Completed fold 1 with LightGBM",
                "metrics": {"loss": 0.35, "accuracy": 0.89, "f1": 0.88, "roc_auc": 0.92},
            },
        )
        assert step1_resp.status_code == 200

        # Verify job status has step 1 metrics
        job_resp = api_client.get(f"/v2/fix/{job_id}")
        job_data = job_resp.json()
        assert job_data["iteration"] == 1
        assert job_data["phase"] == "Training Run #1"
        assert len(job_data["intermediate_metrics"]) == 1
        assert job_data["intermediate_metrics"][0]["accuracy"] == 0.89

    def test_cancel_fix_job_endpoint(self, api_client):
        async def slow_fix(*args, **kwargs):
            import asyncio
            await asyncio.sleep(5)

        with patch("deepfix_server.openhands_executor.OpenHandsExecutor.launch_autonomous_fix", side_effect=slow_fix):
            req = FixJobRequest(
                dataset_name="test_cancel_dataset",
                target_metric="accuracy",
                target_value=0.90,
                max_iterations=5,
            )
            resp = api_client.post("/v2/fix", json=req.model_dump())
            job_id = resp.json()["job_id"]

            # Cancel the job
            cancel_resp = api_client.post(f"/v2/fix/{job_id}/cancel")
            assert cancel_resp.status_code == 200
            cancel_data = cancel_resp.json()
            assert cancel_data["status"] == "CANCELLED"
            assert cancel_data["phase"] == "Cancelled"

            # Verify persistence
            get_resp = api_client.get(f"/v2/fix/{job_id}")
            assert get_resp.json()["status"] == "CANCELLED"

    def test_webhook_completion_with_intermediate_metrics(self, api_client):
        req = FixJobRequest(
            dataset_name="test_webhook_dataset",
            target_metric="accuracy",
            target_value=0.90,
            max_iterations=3,
        )
        resp = api_client.post("/v2/fix", json=req.model_dump())
        job_id = resp.json()["job_id"]

        # OpenHands completes via webhook
        webhook_payload = {
            "job_id": job_id,
            "success": True,
            "status": "COMPLETED",
            "iteration": 2,
            "phase": "Completed",
            "final_metrics": {"loss": 0.15, "accuracy": 0.96, "f1": 0.95, "roc_auc": 0.98},
            "intermediate_metrics": [
                {"iteration": 1, "loss": 0.32, "accuracy": 0.88, "f1": 0.86, "roc_auc": 0.91},
                {"iteration": 2, "loss": 0.15, "accuracy": 0.96, "f1": 0.95, "roc_auc": 0.98},
            ],
            "applied_fixes": ["StratifiedKFold", "LightGBM Classifier", "Class Weighting"],
            "s3_weights_uri": "s3://test-bucket/model.pt",
            "summary": "Achieved target metric 0.96 >= 0.90",
        }
        wh_resp = api_client.post("/webhook/completion", json=webhook_payload)
        assert wh_resp.status_code == 200

        # Verify job details
        job_resp = api_client.get(f"/v2/fix/{job_id}")
        job_data = job_resp.json()
        assert job_data["status"] == "COMPLETED"
        assert job_data["phase"] == "Completed"
        assert job_data["iteration"] == 2
        assert len(job_data["intermediate_metrics"]) == 2
        assert job_data["result"]["s3_weights_uri"] == "s3://test-bucket/model.pt"


class TestRichStreamingCLIComponents:
    """Test CLI formatting components (icons, table building, and commands)."""

    def test_get_phase_icon(self):
        assert _get_phase_icon("Diagnosing") == "🔍"
        assert _get_phase_icon("Synthesizing Fix") == "🛠️"
        assert _get_phase_icon("Training Run #1") == "🏃"
        assert _get_phase_icon("Evaluating Model") == "📈"
        assert _get_phase_icon("Uploading Weights to S3") == "☁️"
        assert _get_phase_icon("Completed") == "✅"
        assert _get_phase_icon("Failed") == "❌"
        assert _get_phase_icon("Cancelled") == "🛑"
        assert _get_phase_icon("Initializing") == "📋"
        assert _get_phase_icon(None) == "⚙️"

    def test_build_metrics_table_renders_rows(self):
        job = FixJob(
            job_id="fix_test_table",
            status=FixJobStatus.COMPLETED,
            target_metric="accuracy",
            target_value=0.90,
            baseline_metrics={"loss": 0.65, "accuracy": 0.72, "f1": 0.70, "roc_auc": 0.75},
            intermediate_metrics=[
                {"iteration": 1, "loss": 0.40, "accuracy": 0.85, "f1": 0.83, "roc_auc": 0.88},
                {"iteration": 2, "loss": 0.20, "accuracy": 0.95, "f1": 0.94, "roc_auc": 0.97},
            ],
            iteration=2,
            max_iterations=5,
        )
        table = _build_metrics_table(job)
        assert table is not None
        assert table.row_count == 3  # Baseline + 2 iterations

    def test_cli_cancel_command(self, cli_runner):
        mock_job = FixJob(
            job_id="fix_cli_cancel_123",
            dataset_name="breast_cancer",
            status=FixJobStatus.CANCELLED,
            phase="Cancelled",
            iteration=1,
            max_iterations=5,
        )
        with patch.object(DeepFixClient, "cancel_fix_job", return_value=mock_job):
            result = cli_runner.invoke(app, ["cancel", "fix_cli_cancel_123", "--api-url", "http://localhost:4141"])
            assert result.exit_code == 0
            assert "fix_cli_cancel_123" in result.stdout
            assert "CANCELLED" in result.stdout

    def test_cli_stop_alias_command(self, cli_runner):
        mock_job = FixJob(
            job_id="fix_cli_stop_456",
            dataset_name="breast_cancer",
            status=FixJobStatus.CANCELLED,
            phase="Cancelled",
            iteration=2,
            max_iterations=5,
        )
        with patch.object(DeepFixClient, "cancel_fix_job", return_value=mock_job):
            result = cli_runner.invoke(app, ["stop", "fix_cli_stop_456", "--api-url", "http://localhost:4141"])
            assert result.exit_code == 0
            assert "fix_cli_stop_456" in result.stdout
            assert "CANCELLED" in result.stdout

    def test_cli_fix_streaming_workflow_e2e(self, cli_runner, tmp_path):
        initial_job = FixJob(
            job_id="fix_cli_stream_789",
            dataset_name="tabular_data",
            status=FixJobStatus.PENDING,
            phase="Pending",
            target_metric="accuracy",
            target_value=0.90,
            max_iterations=2,
        )
        intermediate_job = FixJob(
            job_id="fix_cli_stream_789",
            dataset_name="tabular_data",
            status=FixJobStatus.IN_PROGRESS,
            phase="Training Run #1",
            events=[
                {"timestamp": "2026-08-23T23:10:00Z", "phase": "Diagnosing", "message": "Pre-computing diagnostics"},
                {"timestamp": "2026-08-23T23:10:05Z", "phase": "Training", "message": "Iteration 1 training"},
            ],
            intermediate_metrics=[
                {"iteration": 1, "loss": 0.35, "accuracy": 0.88, "f1": 0.87, "roc_auc": 0.91}
            ],
            iteration=1,
            max_iterations=2,
        )
        completed_job = FixJob(
            job_id="fix_cli_stream_789",
            dataset_name="tabular_data",
            status=FixJobStatus.COMPLETED,
            phase="Completed",
            events=[
                {"timestamp": "2026-08-23T23:10:00Z", "phase": "Diagnosing", "message": "Pre-computing diagnostics"},
                {"timestamp": "2026-08-23T23:10:05Z", "phase": "Training", "message": "Iteration 1 training"},
                {"timestamp": "2026-08-23T23:10:15Z", "phase": "Completed", "message": "Target reached: 0.94 >= 0.90"},
            ],
            intermediate_metrics=[
                {"iteration": 1, "loss": 0.35, "accuracy": 0.88, "f1": 0.87, "roc_auc": 0.91},
                {"iteration": 2, "loss": 0.18, "accuracy": 0.94, "f1": 0.93, "roc_auc": 0.96},
            ],
            iteration=2,
            max_iterations=2,
            result=FinalFixReport(
                success=True,
                final_metrics={"loss": 0.18, "accuracy": 0.94, "f1": 0.93, "roc_auc": 0.96},
                applied_fixes=["StratifiedKFold", "LightGBM"],
                s3_weights_uri="s3://test-bucket/model.pt",
            ),
        )

        with patch.object(DeepFixClient, "submit_fix_job", return_value=initial_job), \
             patch.object(DeepFixClient, "poll_fix_job_stream", return_value=[intermediate_job, completed_job]), \
             patch.object(DeepFixClient, "stage_output_artifacts", return_value=tmp_path / "fix_cli_stream_789"):

            result = cli_runner.invoke(
                app,
                [
                    "fix",
                    "--dataset", "tabular_data",
                    "--target-metric", "accuracy",
                    "--target-value", "0.90",
                    "--max-iterations", "2",
                    "--poll-interval", "0.01",
                    "--output-dir", str(tmp_path),
                ],
            )
            assert result.exit_code == 0
            assert "fix_cli_stream_789" in result.stdout
            assert "COMPLETED" in result.stdout
            assert "0.94" in result.stdout

    def test_cli_fix_keyboard_interrupt_cancel_choice(self, cli_runner, tmp_path):
        initial_job = FixJob(
            job_id="fix_cli_interrupt_1",
            dataset_name="tabular_data",
            status=FixJobStatus.PENDING,
            phase="Pending",
        )
        cancelled_job = FixJob(
            job_id="fix_cli_interrupt_1",
            dataset_name="tabular_data",
            status=FixJobStatus.CANCELLED,
            phase="Cancelled",
        )

        def raise_keyboard_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        with patch.object(DeepFixClient, "submit_fix_job", return_value=initial_job), \
             patch.object(DeepFixClient, "poll_fix_job_stream", side_effect=raise_keyboard_interrupt), \
             patch.object(DeepFixClient, "cancel_fix_job", return_value=cancelled_job):

            # Input 'c' to cancel
            result = cli_runner.invoke(
                app,
                ["fix", "--dataset", "tabular_data"],
                input="c\n",
            )
            assert result.exit_code == 130
            assert "Cancelling fix job" in result.stdout

    def test_cli_fix_keyboard_interrupt_detach_choice(self, cli_runner, tmp_path):
        initial_job = FixJob(
            job_id="fix_cli_interrupt_2",
            dataset_name="tabular_data",
            status=FixJobStatus.PENDING,
            phase="Pending",
        )

        def raise_keyboard_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        with patch.object(DeepFixClient, "submit_fix_job", return_value=initial_job), \
             patch.object(DeepFixClient, "poll_fix_job_stream", side_effect=raise_keyboard_interrupt):

            # Input 'd' (or default) to detach
            result = cli_runner.invoke(
                app,
                ["fix", "--dataset", "tabular_data"],
                input="d\n",
            )
            assert result.exit_code == 0
            assert "Detached from fix job" in result.stdout
            assert "deepfix-sdk cancel fix_cli_interrupt_2" in result.stdout
