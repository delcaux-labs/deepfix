"""CLI commands for DeepFix."""

import subprocess
import sys

import typer
from ..core.config import DefaultPaths

mlflow_app = typer.Typer()

@mlflow_app.command()
def launch_server(
    port: int = typer.Option(5000, help="Port to run MLflow server on"),
    host: str = typer.Option("127.0.0.1", help="Host to bind MLflow server to"),
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
        cmd.extend(["--default-artifact-root", DefaultPaths.MLFLOW_DEFAULT_ARTIFACT_ROOT.value])
        
        typer.echo(f"🚀 Starting MLflow server on {host}:{port}")
        typer.echo(f"📊 Backend store: {DefaultPaths.MLFLOW_TRACKING_URI.value}")
        typer.echo(f"📁 Artifact root: {DefaultPaths.MLFLOW_DEFAULT_ARTIFACT_ROOT.value}")
        
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
