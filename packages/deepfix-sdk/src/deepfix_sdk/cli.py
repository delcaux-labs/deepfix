import subprocess
import sys
from typing import Any, Optional

import typer
from deepfix_core.models import FixJob, FixJobStatus
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import DefaultPaths

console = Console()

app = typer.Typer(
    name="deepfix-sdk",
    help="DeepFix SDK",
    add_completion=False,
)


@app.command(name="version")
def version() -> None:
    """Print the version of the DeepFix SDK."""
    typer.echo("DeepFix SDK version: 0.1.0")


@app.command(name="launch-mlflow")
def launch_mlflow_server(
    port: int = typer.Option(5000, "-port", help="Port to run MLflow server on"),
    host: str = typer.Option("0.0.0.0", "-host", help="Host to bind MLflow server to"),
) -> None:
    """Launch MLflow tracking server."""
    try:
        # Build MLflow server command
        cmd = ["mlflow", "server"]

        # Add port
        cmd.extend(["--port", str(port)])

        # Add host
        cmd.extend(["--host", host])

        # Add backend store URI (always use the provided/default value)
        cmd.extend(["--backend-store-uri", DefaultPaths.MLFLOW_TRACKING_URI.value])

        # Add default artifact root if provided
        cmd.extend(
            ["--default-artifact-root", DefaultPaths.MLFLOW_DEFAULT_ARTIFACT_ROOT.value]
        )

        typer.echo(f"🚀 Starting MLflow server on {host}:{port}")
        typer.echo(f"📊 Backend store: {DefaultPaths.MLFLOW_TRACKING_URI.value}")
        typer.echo(
            f"📁 Artifact root: {DefaultPaths.MLFLOW_DEFAULT_ARTIFACT_ROOT.value}"
        )

        # Start the MLflow server
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Failed to start MLflow server: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        typer.echo("\n👋 MLflow server stopped.")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


def _get_phase_icon(phase: Optional[str]) -> str:
    """Return appropriate visual icon for the execution phase."""
    if not phase:
        return "⚙️"
    phase_lower = phase.lower()
    if "diagnos" in phase_lower:
        return "🔍"
    if "synth" in phase_lower or "generat" in phase_lower:
        return "🛠️"
    if "train" in phase_lower:
        return "🏃"
    if "eval" in phase_lower or "metric" in phase_lower:
        return "📈"
    if "upload" in phase_lower or "s3" in phase_lower:
        return "☁️"
    if "complet" in phase_lower or "success" in phase_lower:
        return "✅"
    if "fail" in phase_lower or "error" in phase_lower:
        return "❌"
    if "cancel" in phase_lower or "stop" in phase_lower:
        return "🛑"
    if "init" in phase_lower:
        return "📋"
    return "⚙️"


def _build_metrics_table(job: FixJob) -> Optional[Table]:
    """Construct Rich Table displaying intermediate MLflow metric progression."""
    has_baseline = bool(job.baseline_metrics)
    has_intermediate = bool(job.intermediate_metrics)
    has_final = bool(job.result and job.result.final_metrics)

    if not (has_baseline or has_intermediate or has_final):
        return None

    table = Table(
        title="📊 MLflow Metric Progression",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Run / Iteration", style="bold")
    table.add_column("Loss", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("Status", justify="center")

    def _fmt(val: Any) -> str:
        if val is None:
            return "-"
        try:
            return f"{float(val):.4f}"
        except (ValueError, TypeError):
            return str(val)

    def _extract_metric(metrics_dict: dict, keys: list[str]) -> str:
        for k in keys:
            if k in metrics_dict and metrics_dict[k] is not None:
                return _fmt(metrics_dict[k])
            for actual_k, val in metrics_dict.items():
                if actual_k.lower() == k.lower() and val is not None:
                    return _fmt(val)
        return "-"

    # 1. Baseline row
    if has_baseline:
        b = job.baseline_metrics
        table.add_row(
            "[dim]Baseline[/dim]",
            _extract_metric(b, ["loss", "val_loss", "test_loss"]),
            _extract_metric(b, ["accuracy", "val_accuracy", "acc"]),
            _extract_metric(b, ["f1", "f1_score", "val_f1", "macro_f1"]),
            _extract_metric(b, ["roc_auc", "val_roc_auc", "auc"]),
            "[dim]Initial[/dim]",
        )

    # 2. Intermediate runs
    if has_intermediate:
        for idx, run_m in enumerate(job.intermediate_metrics):
            it_num = run_m.get("iteration", idx + 1)
            target_key = (job.target_metric or "accuracy").lower()
            target_val = job.target_value or 0.90

            target_met = False
            for k, v in run_m.items():
                if target_key in k.lower():
                    try:
                        if float(v) >= target_val:
                            target_met = True
                    except (ValueError, TypeError):
                        pass

            status_col = "[bold green]🎯 Target Met[/bold green]" if target_met else "[yellow]Iterating[/yellow]"

            table.add_row(
                f"Iteration #{it_num}",
                _extract_metric(run_m, ["loss", "val_loss", "test_loss"]),
                _extract_metric(run_m, ["accuracy", "val_accuracy", "acc"]),
                _extract_metric(run_m, ["f1", "f1_score", "val_f1", "macro_f1"]),
                _extract_metric(run_m, ["roc_auc", "val_roc_auc", "auc"]),
                status_col,
            )

    # 3. Final metrics row if completed and not duplicate
    if has_final and not has_intermediate:
        f = job.result.final_metrics  # type: ignore
        table.add_row(
            f"[bold]Final (Run #{job.iteration})[/bold]",
            _extract_metric(f, ["loss", "val_loss", "test_loss"]),
            _extract_metric(f, ["accuracy", "val_accuracy", "acc"]),
            _extract_metric(f, ["f1", "f1_score", "val_f1", "macro_f1"]),
            _extract_metric(f, ["roc_auc", "val_roc_auc", "auc"]),
            "[bold green]✅ Met[/bold green]" if job.result.success else "[red]❌ Finished[/red]",  # type: ignore
        )

    return table


def _track_live_fix_session(
    client: Any,
    job: FixJob,
    poll_interval: float = 2.0,
) -> FixJob:
    """Streams real-time fix execution progress, activity events, and metrics using Rich."""
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    progress = Progress(
        SpinnerColumn(spinner_name="dots", style="bold cyan"),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=32, complete_style="green", finished_style="bold green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task(
        "Autonomous Fix Refinement",
        total=job.max_iterations or 5,
        completed=job.iteration,
    )

    seen_event_keys: set[str] = set()
    completed_job: Optional[FixJob] = None

    console.print("\n[bold cyan]📡 Streaming Autonomous Fix Agent Activity...[/bold cyan]\n")

    def _render_event(evt: dict[str, Any]) -> None:
        raw_ts = evt.get("timestamp", "")
        if "T" in raw_ts:
            try:
                time_str = raw_ts.split("T")[1][:8]
            except Exception:
                time_str = raw_ts[:8]
        else:
            time_str = raw_ts[:8] if raw_ts else "--:--:--"

        phase = evt.get("phase", "In Progress")
        icon = _get_phase_icon(phase)
        msg = evt.get("message", "")
        console.print(f"[dim]{time_str}[/dim] {icon} [bold cyan][{phase}][/bold cyan] {msg}")

    try:
        progress.start()
        for updated_job in client.poll_fix_job_stream(
            job_id=job.job_id,
            polling_interval=poll_interval,
        ):
            # 1. Update progress bar
            curr_it = updated_job.iteration or 0
            max_it = updated_job.max_iterations or 5
            curr_phase = updated_job.phase or "In Progress"
            phase_icon = _get_phase_icon(curr_phase)

            progress.update(
                task_id,
                completed=min(curr_it, max_it),
                total=max_it,
                description=f"{phase_icon} {curr_phase} (Iter {curr_it}/{max_it})",
            )
            progress.refresh()

            # 2. Print new events
            for evt in updated_job.events:
                evt_key = f"{evt.get('timestamp')}_{evt.get('message')}"
                if evt_key not in seen_event_keys:
                    seen_event_keys.add(evt_key)
                    _render_event(evt)

            completed_job = updated_job
            if updated_job.status not in (FixJobStatus.PENDING, FixJobStatus.IN_PROGRESS):
                break

    except KeyboardInterrupt:
        progress.stop()
        console.print("\n[bold yellow]⚠️  Polling interrupted by user (Ctrl+C).[/bold yellow]")
        choice = typer.prompt(
            "Choose action: [c]ancel server job / [d]etach and keep running in background",
            default="d",
        ).strip().lower()

        if choice in ("c", "cancel", "yes", "y"):
            console.print(f"[bold yellow]🛑 Cancelling fix job {job.job_id} on server...[/bold yellow]")
            try:
                cancelled_job = client.cancel_fix_job(job.job_id)
            except Exception as c_err:
                console.print(f"[bold red]❌ Failed to cancel job on server:[/bold red] {c_err}")
                raise typer.Exit(code=1) from c_err

            console.print(
                Panel(
                    f"🛑 [bold yellow]Fix Job {cancelled_job.job_id} successfully cancelled.[/bold yellow]",
                    border_style="yellow",
                )
            )
            raise typer.Exit(code=130)
        else:
            console.print(
                Panel(
                    f"[bold cyan]Detached from fix job [bold]{job.job_id}[/bold].[/bold cyan]\n"
                    "The autonomous agent will continue running in the background on the DeepFix server.\n\n"
                    f"To stop or check the job later, run:\n  [bold green]deepfix-sdk cancel {job.job_id}[/bold green]",
                    title="Detached Monitoring",
                    border_style="cyan",
                )
            )
            raise typer.Exit(code=0)
    finally:
        progress.stop()

    if completed_job is None:
        raise RuntimeError("No response received from fix job polling")

    # Display intermediate metrics table if available
    metrics_table = _build_metrics_table(completed_job)
    if metrics_table:
        console.print("\n")
        console.print(metrics_table)

    return completed_job


def _run_fix_workflow(
    dataset_name: str,
    model_name: Optional[str] = None,
    target_metric: str = "accuracy",
    target_value: float = 0.90,
    max_iterations: int = 5,
    s3_bucket: Optional[str] = None,
    api_url: str = "http://localhost:4141",
    output_dir: str = "./deepfix_output",
    poll_interval: float = 2.0,
) -> None:
    """Execute autonomous fix job workflow from CLI."""
    from .client import DeepFixClient

    client = DeepFixClient(api_url=api_url)

    # 1. Submit fix job
    console.print("\n[bold cyan]🤖 Submitting Autonomous Fix Job...[/bold cyan]")
    try:
        job = client.submit_fix_job(
            dataset_name=dataset_name,
            model_name=model_name,
            target_metric=target_metric,
            target_value=target_value,
            max_iterations=max_iterations,
            s3_bucket=s3_bucket,
        )
    except Exception as exc:
        console.print(f"[bold red]❌ Failed to submit fix job:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    # 2. Display submission table
    table = Table(title="Autonomous Fix Job Details", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Job ID", job.job_id)
    table.add_row("Dataset", job.dataset_name or dataset_name)
    if job.dataset_uri:
        table.add_row("Dataset URI", job.dataset_uri)
    table.add_row("Model", job.model_name or "N/A")
    if job.model_uri:
        table.add_row("Model URI", job.model_uri)
    table.add_row("Target Metric", f"{job.target_metric} >= {job.target_value}")
    table.add_row("Max Iterations", str(job.max_iterations))
    table.add_row("S3 Target Bucket", job.s3_bucket or "None (local staging)")
    table.add_row("Initial Status", job.status.value)
    console.print(table)

    # 3. Stream fix lifecycle with progress bar, events, and metrics
    completed_job = _track_live_fix_session(
        client=client,
        job=job,
        poll_interval=poll_interval,
    )

    # 4. Stage output artifacts
    staged_path = client.stage_output_artifacts(completed_job, output_dir=output_dir)

    # 5. Display completion results
    _display_fix_results(completed_job, staged_path)


def _display_fix_results(completed_job: FixJob, staged_path: Any) -> None:
    """Display prominent summary banner and tabular output of staged fix execution results."""
    import pathlib

    staged_dir = pathlib.Path(staged_path)
    try:
        rel_staged = staged_dir.relative_to(pathlib.Path.cwd())
    except ValueError:
        rel_staged = staged_dir

    if completed_job.status == FixJobStatus.COMPLETED:
        report = completed_job.result

        # 1. Results overview table
        results_table = Table(
            title="🎯 Autonomous Model Repair Summary",
            show_header=True,
            header_style="bold green",
            border_style="green",
        )
        results_table.add_column("Property", style="cyan", no_wrap=True)
        results_table.add_column("Details", style="bold")

        results_table.add_row("Status", "[bold green]COMPLETED (Target Met)[/bold green]")
        results_table.add_row("Job ID", completed_job.job_id)
        results_table.add_row("Dataset", completed_job.dataset_name or "N/A")
        results_table.add_row("Iterations Run", f"{completed_job.iteration} / {completed_job.max_iterations}")

        if report:
            if report.final_metrics:
                metrics_str = ", ".join(f"{k}: {v}" for k, v in report.final_metrics.items())
                results_table.add_row("Final Metrics", metrics_str)
            if report.applied_fixes:
                results_table.add_row("Applied Fixes", "\n".join(f"• {fix_desc}" for fix_desc in report.applied_fixes))
            if report.s3_weights_uri:
                results_table.add_row("S3 Model Weights", f"[link={report.s3_weights_uri}]{report.s3_weights_uri}[/link]")
            if report.run_id:
                results_table.add_row("MLflow Run ID", report.run_id)

        console.print("\n")
        console.print(results_table)

        # 2. Prominent Staged Artifacts Delivery Table
        artifacts_table = Table(
            title="📦 Staged Deliverable Artifacts",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
        )
        artifacts_table.add_column("Artifact", style="bold yellow")
        artifacts_table.add_column("Relative Path", style="green")
        artifacts_table.add_column("Description", style="dim")

        train_fixed_rel = str(rel_staged / "train_fixed.py")
        summary_md_rel = str(rel_staged / "summary_report.md")
        metrics_json_rel = str(rel_staged / "metrics.json")
        model_artifacts_rel = str(rel_staged / "model_artifacts")

        artifacts_table.add_row(
            "train_fixed.py",
            train_fixed_rel,
            "Clean, standalone, runnable Python training script incorporating fixes",
        )
        artifacts_table.add_row(
            "summary_report.md",
            summary_md_rel,
            "Comprehensive Markdown report with defects, remediations & metric deltas",
        )
        artifacts_table.add_row(
            "metrics.json",
            metrics_json_rel,
            "Structured machine-readable metrics before and after the fix",
        )
        artifacts_table.add_row(
            "model_artifacts/",
            model_artifacts_rel,
            "Downloaded model checkpoint and weights (from S3 / MLflow)",
        )

        console.print("\n")
        console.print(artifacts_table)

        # 3. Prominent Banner
        banner_text = (
            f"🎉 [bold green]Model successfully repaired and packaged![/bold green]\n\n"
            f"📁 Staged Output Directory: [bold]{staged_dir.resolve()}[/bold]\n"
            f"🐍 Standalone Script:      [cyan]{train_fixed_rel}[/cyan]\n"
            f"📊 Diagnostic Report:       [cyan]{summary_md_rel}[/cyan]\n"
            f"📈 Metrics JSON:           [cyan]{metrics_json_rel}[/cyan]\n"
        )
        if report and report.s3_weights_uri:
            banner_text += f"☁️  S3 Model Weights:       [link={report.s3_weights_uri}]{report.s3_weights_uri}[/link]\n"

        console.print(Panel(banner_text.strip(), title="DeepFix Fix Delivery", border_style="bold green"))
    elif completed_job.status == FixJobStatus.CANCELLED:
        console.print(
            Panel(
                f"🛑 [bold yellow]Fix Job '{completed_job.job_id}' was cancelled.[/bold yellow]\nStaged diagnostic outputs at {rel_staged}",
                title="Job Cancelled",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=130)
    else:
        err_msg = completed_job.error or "Fix job failed or could not meet threshold."
        console.print(
            Panel(
                f"❌ [bold red]Fix Failed:[/bold red] {err_msg}\nStaged diagnostic outputs at {rel_staged}",
                title="Fix Failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@app.command(name="fix")
def fix(
    dataset_name: str = typer.Option(..., "--dataset", "-d", help="Name of the dataset"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="Name or URI of baseline model"),
    target_metric: str = typer.Option("accuracy", "--target-metric", help="Target metric key to optimize"),
    target_value: float = typer.Option(0.90, "--target-value", help="Target metric threshold value"),
    max_iterations: int = typer.Option(5, "--max-iterations", help="Maximum autonomous refinement loops"),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="Target S3 bucket for model weights"),
    api_url: str = typer.Option("http://localhost:4141", "--api-url", help="DeepFix Server API URL"),
    output_dir: str = typer.Option("./deepfix_output", "--output-dir", "-o", help="Path to stage output artifacts"),
    poll_interval: float = typer.Option(2.0, "--poll-interval", help="Polling frequency in seconds"),
) -> None:
    """Trigger autonomous model repair workflow and poll for results."""
    _run_fix_workflow(
        dataset_name=dataset_name,
        model_name=model_name,
        target_metric=target_metric,
        target_value=target_value,
        max_iterations=max_iterations,
        s3_bucket=s3_bucket,
        api_url=api_url,
        output_dir=output_dir,
        poll_interval=poll_interval,
    )


@app.command(name="diagnose")
def diagnose(
    dataset_name: str = typer.Option(..., "--dataset", "-d", help="Name of the dataset"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="Name of the model"),
    fix: bool = typer.Option(False, "--fix", help="Trigger autonomous fix loop"),
    api_url: str = typer.Option("http://localhost:4141", "--api-url", help="DeepFix API URL"),
    target_metric: str = typer.Option("accuracy", "--target-metric", help="Target metric name"),
    target_value: float = typer.Option(0.90, "--target-value", help="Target metric value"),
    max_iterations: int = typer.Option(5, "--max-iterations", help="Maximum fix iterations"),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="Target S3 bucket for model weights"),
    output_dir: str = typer.Option("./deepfix_output", "--output-dir", "-o", help="Path to stage output artifacts"),
    poll_interval: float = typer.Option(2.0, "--poll-interval", help="Polling frequency in seconds"),
) -> None:
    """Diagnose ML artifacts and optionally run autonomous fix loop."""
    if fix:
        _run_fix_workflow(
            dataset_name=dataset_name,
            model_name=model_name,
            target_metric=target_metric,
            target_value=target_value,
            max_iterations=max_iterations,
            s3_bucket=s3_bucket,
            api_url=api_url,
            output_dir=output_dir,
            poll_interval=poll_interval,
        )
    else:
        from .client import DeepFixClient

        client = DeepFixClient(api_url=api_url)
        typer.echo(f"🔍 Diagnosing dataset: {dataset_name}...")
        response = client.diagnose(dataset_name=dataset_name, model_name=model_name)
        typer.echo(response.to_text())


@app.command(name="cancel")
def cancel_job(
    job_id: str = typer.Argument(..., help="ID of the fix job to cancel"),
    api_url: str = typer.Option("http://localhost:4141", "--api-url", help="DeepFix Server API URL"),
) -> None:
    """Cancel an ongoing autonomous fix job on the server."""
    from .client import DeepFixClient

    client = DeepFixClient(api_url=api_url)
    try:
        job = client.cancel_fix_job(job_id)
        table = Table(title="Fix Job Cancellation Confirmation", show_header=True, header_style="bold yellow")
        table.add_column("Property", style="dim")
        table.add_column("Value", style="bold")
        table.add_row("Job ID", job.job_id)
        table.add_row("Dataset", job.dataset_name or "N/A")
        table.add_row("Status", f"[yellow]{job.status.value}[/yellow]")
        table.add_row("Phase", job.phase or "Cancelled")
        table.add_row("Iterations Run", str(job.iteration))
        console.print(table)
        console.print(Panel(f"🛑 [bold yellow]Autonomous fix job '{job_id}' has been cancelled.[/bold yellow]", border_style="yellow"))
    except Exception as exc:
        console.print(f"[bold red]❌ Failed to cancel job {job_id}:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="stop")
def stop_job(
    job_id: str = typer.Argument(..., help="ID of the fix job to stop"),
    api_url: str = typer.Option("http://localhost:4141", "--api-url", help="DeepFix Server API URL"),
) -> None:
    """Stop an ongoing autonomous fix job (alias for cancel)."""
    cancel_job(job_id=job_id, api_url=api_url)


def main():
    app()
