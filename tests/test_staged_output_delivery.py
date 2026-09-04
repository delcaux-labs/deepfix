import json
import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from deepfix_core.models import FinalFixReport, FixJob, FixJobStatus
from deepfix_sdk.cli import _display_fix_results, app
from deepfix_sdk.client import DeepFixClient
from typer.testing import CliRunner


@pytest.fixture
def sample_completed_job():
    """Create a sample completed FixJob with full diagnostic and remediation reports."""
    report = FinalFixReport(
        success=True,
        final_metrics={"loss": 0.1425, "accuracy": 0.9650, "f1": 0.9580, "roc_auc": 0.9850},
        applied_fixes=[
            "Stratified K-Fold Cross-Validation (k=5)",
            "HistGradientBoostingClassifier with balanced class weighting",
            "VarianceThreshold & Correlation-based Multicollinearity Filtering",
            "RobustScaler Standardization",
        ],
        run_id="mlflow_run_xyz123",
        s3_weights_uri="s3://deepfix-models/fix_test_job_001/weights/model.pt",
        summary="Model repaired successfully. Validation accuracy improved from 0.7800 to 0.9650 exceeding target 0.90.",
    )
    return FixJob(
        job_id="fix_test_job_001",
        dataset_name="breast_cancer_diagnostic",
        model_name="HistGradientBoostingClassifier",
        target_metric="accuracy",
        target_value=0.90,
        max_iterations=5,
        iteration=3,
        status=FixJobStatus.COMPLETED,
        phase="Completed",
        baseline_metrics={"loss": 0.5820, "accuracy": 0.7800, "f1": 0.7600, "roc_auc": 0.8100},
        intermediate_metrics=[
            {"iteration": 1, "loss": 0.3500, "accuracy": 0.8600, "f1": 0.8400, "roc_auc": 0.8900},
            {"iteration": 2, "loss": 0.2200, "accuracy": 0.9200, "f1": 0.9100, "roc_auc": 0.9400},
            {"iteration": 3, "loss": 0.1425, "accuracy": 0.9650, "f1": 0.9580, "roc_auc": 0.9850},
        ],
        diagnosis="Severe multicollinearity among perimeter and area features. Moderate class imbalance (357 vs 212).",
        result=report,
    )


class TestStagedOutputDelivery:
    """Tests for artifact packaging and staging delivery."""

    def test_stage_output_artifacts_creates_directory_structure(self, tmp_path, sample_completed_job):
        client = DeepFixClient()
        staged_dir = client.stage_output_artifacts(sample_completed_job, output_dir=str(tmp_path))

        assert staged_dir == tmp_path / sample_completed_job.job_id
        assert staged_dir.is_dir()

        # Check expected files and subdirectories
        train_fixed_path = staged_dir / "train_fixed.py"
        summary_md_path = staged_dir / "summary_report.md"
        metrics_json_path = staged_dir / "metrics.json"
        model_artifacts_dir = staged_dir / "model_artifacts"

        assert train_fixed_path.exists()
        assert summary_md_path.exists()
        assert metrics_json_path.exists()
        assert model_artifacts_dir.exists()
        assert model_artifacts_dir.is_dir()

    def test_metrics_json_content_and_deltas(self, tmp_path, sample_completed_job):
        client = DeepFixClient()
        staged_dir = client.stage_output_artifacts(sample_completed_job, output_dir=str(tmp_path))
        metrics_file = staged_dir / "metrics.json"

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["job_id"] == "fix_test_job_001"
        assert data["status"] == "COMPLETED"
        assert data["dataset_name"] == "breast_cancer_diagnostic"
        assert data["target_metric"] == "accuracy"
        assert data["target_value"] == 0.90
        assert data["iterations_run"] == 3

        # Check baseline and final metrics
        assert data["baseline_metrics"]["accuracy"] == 0.7800
        assert data["final_metrics"]["accuracy"] == 0.9650

        # Check calculated metric deltas
        assert "accuracy" in data["metric_deltas"]
        assert pytest.approx(data["metric_deltas"]["accuracy"], 0.0001) == 0.1850
        assert pytest.approx(data["metric_deltas"]["loss"], 0.0001) == -0.4395

        # Check trajectory and S3 URI
        assert len(data["intermediate_metrics"]) == 3
        assert data["s3_weights_uri"] == "s3://deepfix-models/fix_test_job_001/weights/model.pt"
        assert data["run_id"] == "mlflow_run_xyz123"

    def test_summary_report_markdown_content(self, tmp_path, sample_completed_job):
        client = DeepFixClient()
        staged_dir = client.stage_output_artifacts(sample_completed_job, output_dir=str(tmp_path))
        summary_file = staged_dir / "summary_report.md"

        with open(summary_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check Job info
        assert "fix_test_job_001" in content
        assert "breast_cancer_diagnostic" in content
        assert "s3://deepfix-models/fix_test_job_001/weights/model.pt" in content
        assert "mlflow_run_xyz123" in content

        # Check Diagnostics section
        assert "Diagnostic Issues & Initial Defects" in content
        assert "Severe multicollinearity" in content

        # Check Remediations section
        assert "Remediations & Applied Fixes" in content
        assert "Stratified K-Fold Cross-Validation" in content
        assert "HistGradientBoostingClassifier with balanced class weighting" in content

        # Check Metrics table with deltas
        assert "Performance & Metric Deltas" in content
        assert "| Metric | Baseline | Final | Delta | Target Status |" in content
        assert "0.7800" in content
        assert "0.9650" in content
        assert "+0.1850" in content
        assert "✅ Met" in content

        # Check Deliverables list
        assert "train_fixed.py" in content
        assert "model_artifacts/" in content

    def test_train_fixed_py_is_standalone_and_syntax_valid(self, tmp_path, sample_completed_job):
        client = DeepFixClient()
        staged_dir = client.stage_output_artifacts(sample_completed_job, output_dir=str(tmp_path))
        train_script = staged_dir / "train_fixed.py"

        with open(train_script, "r", encoding="utf-8") as f:
            code = f.read()

        # Syntax check via compile()
        compiled = compile(code, str(train_script), "exec")
        assert compiled is not None

        # Verify key modules and logic exist
        assert "StratifiedKFold" in code
        assert "HistGradientBoostingClassifier" in code
        assert "class_weight" in code
        assert "train_fixed_model" in code

    def test_train_fixed_py_uses_custom_fixed_code_when_present(self, tmp_path):
        custom_script = "print('Hello from custom agent fix script!')\n"
        job = FixJob(
            job_id="fix_custom_code_123",
            dataset_name="custom_dataset",
            status=FixJobStatus.COMPLETED,
            result=FinalFixReport(
                success=True,
                fixed_code=custom_script,
            ),
        )
        client = DeepFixClient()
        staged_dir = client.stage_output_artifacts(job, output_dir=str(tmp_path))
        train_script = staged_dir / "train_fixed.py"

        with open(train_script, "r", encoding="utf-8") as f:
            content = f.read()

        assert content.strip() == custom_script.strip()

    def test_s3_weights_download_invocation(self, tmp_path, sample_completed_job):
        client = DeepFixClient()
        mock_boto3_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_boto3_session.client.return_value = mock_s3_client

        with patch("boto3.Session", return_value=mock_boto3_session):
            staged_dir = client.stage_output_artifacts(sample_completed_job, output_dir=str(tmp_path))
            mock_s3_client.download_file.assert_called_once_with(
                "deepfix-models",
                "fix_test_job_001/weights/model.pt",
                str(staged_dir / "model_artifacts" / "model.pt"),
            )


class TestCLISummaryBannerAndExitCodes:
    """Tests for CLI display banner and exit codes."""

    def test_cli_display_fix_results_completed_banner(self, sample_completed_job, tmp_path, capsys):
        staged_dir = tmp_path / sample_completed_job.job_id
        staged_dir.mkdir(parents=True, exist_ok=True)

        _display_fix_results(sample_completed_job, staged_dir)
        # Note: Console output is rendered directly to Rich console

    def test_cli_fix_command_renders_banner_and_exits_0(self, tmp_path, sample_completed_job):
        runner = CliRunner()
        with patch.object(DeepFixClient, "submit_fix_job", return_value=sample_completed_job), \
             patch.object(DeepFixClient, "poll_fix_job_stream", return_value=[sample_completed_job]), \
             patch.object(DeepFixClient, "stage_output_artifacts", return_value=tmp_path / sample_completed_job.job_id):

            result = runner.invoke(
                app,
                [
                    "fix",
                    "--dataset", "breast_cancer_diagnostic",
                    "--target-metric", "accuracy",
                    "--target-value", "0.90",
                    "--output-dir", str(tmp_path),
                ],
            )
            assert result.exit_code == 0
            assert "Autonomous Model Repair Summary" in result.stdout
            assert "Staged Deliverable Artifacts" in result.stdout
            assert "train_fixed.py" in result.stdout
            assert "summary_report.md" in result.stdout
            assert "metrics.json" in result.stdout
            assert "model_artifacts/" in result.stdout
            assert "Model successfully repaired and packaged!" in result.stdout

    def test_cli_fix_command_exits_nonzero_on_failure(self, tmp_path):
        runner = CliRunner()
        failed_job = FixJob(
            job_id="fix_failed_999",
            dataset_name="difficult_dataset",
            status=FixJobStatus.FAILED,
            error="Could not reach target accuracy 0.99 after 5 iterations",
        )
        with patch.object(DeepFixClient, "submit_fix_job", return_value=failed_job), \
             patch.object(DeepFixClient, "poll_fix_job_stream", return_value=[failed_job]), \
             patch.object(DeepFixClient, "stage_output_artifacts", return_value=tmp_path / failed_job.job_id):

            result = runner.invoke(
                app,
                [
                    "fix",
                    "--dataset", "difficult_dataset",
                    "--target-metric", "accuracy",
                    "--target-value", "0.99",
                    "--output-dir", str(tmp_path),
                ],
            )
            assert result.exit_code == 1
            assert "Fix Failed" in result.stdout
            assert "Could not reach target accuracy 0.99" in result.stdout

    def test_cli_fix_command_exits_130_on_cancelled(self, tmp_path):
        runner = CliRunner()
        cancelled_job = FixJob(
            job_id="fix_cancelled_888",
            dataset_name="cancelled_dataset",
            status=FixJobStatus.CANCELLED,
            phase="Cancelled",
        )
        with patch.object(DeepFixClient, "submit_fix_job", return_value=cancelled_job), \
             patch.object(DeepFixClient, "poll_fix_job_stream", return_value=[cancelled_job]), \
             patch.object(DeepFixClient, "stage_output_artifacts", return_value=tmp_path / cancelled_job.job_id):

            result = runner.invoke(
                app,
                [
                    "fix",
                    "--dataset", "cancelled_dataset",
                    "--output-dir", str(tmp_path),
                ],
            )
            assert result.exit_code == 130
            assert "Job Cancelled" in result.stdout
