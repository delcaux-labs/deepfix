import subprocess
import sys
from typing import Optional

import typer
from .config import DefaultPaths

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


@app.command(name="diagnose")
def diagnose(
    dataset_name: str = typer.Option(..., "--dataset", "-d", help="Name of the dataset"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="Name of the model"),
    fix: bool = typer.Option(False, "--fix", help="Trigger autonomous fix loop"),
    api_url: str = typer.Option("http://localhost:4141/v2/analyse", "--api-url", help="DeepFix API URL"),
    target_metric: str = typer.Option("accuracy", "--target-metric", help="Target metric name"),
    target_value: float = typer.Option(0.90, "--target-value", help="Target metric value"),
    max_iterations: int = typer.Option(5, "--max-iterations", help="Maximum fix iterations"),
) -> None:
    """Diagnose ML artifacts and optionally run autonomous fix loop."""
    from .client import DeepFixClient

    client = DeepFixClient(api_url=api_url)
    if fix:
        typer.echo(f"🤖 Starting autonomous fix loop for dataset: {dataset_name}...")
        response = client.diagnose_and_fix(
            train_data=None,
            dataset_name=dataset_name,
            model_name=model_name,
            target_metric=target_metric,
            target_value=target_value,
            max_iterations=max_iterations,
        )
    else:
        typer.echo(f"🔍 Diagnosing dataset: {dataset_name}...")
        response = client.diagnose(dataset_name=dataset_name, model_name=model_name)

    typer.echo(response.to_text())


def main():
    app()

