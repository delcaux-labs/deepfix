import subprocess
import sys

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


def main():
    app()
