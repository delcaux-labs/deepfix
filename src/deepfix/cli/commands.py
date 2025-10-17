"""CLI commands for DeepFix."""

import subprocess
import sys

import typer
from ..shared.models import DefaultPaths

commands_app = typer.Typer()

@commands_app.command(name="launch-mlflow-server")
def launch_mlflow_server(
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

@commands_app.command(name="launch-deepfix-server")
def launch_deepfix_server(
    port: int = typer.Option(8844, help="Port to run DeepFix server on"),
    host: str = typer.Option("127.0.0.1", help="Host to bind DeepFix server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload on code changes (for development)"),
) -> None:
    """Launch DeepFix server."""
    try:
        # Build uvicorn server command
        cmd = ["uvicorn", "deepfix.server.api:app"]
        
        # Add host and port
        cmd.extend(["--host", host])
        cmd.extend(["--port", str(port)])
        
        # Add reload flag if requested
        if reload:
            cmd.append("--reload")
            typer.echo("🔄 Auto-reload enabled - server will restart on code changes")
        
        typer.echo(f"🚀 Starting DeepFix server on http://{host}:{port}")
        typer.echo(f"📚 API docs available at http://{host}:{port}/docs")
        typer.echo(f"ℹ️  Server info endpoint: http://{host}:{port}/info")
        
        # Start the FastAPI server with uvicorn
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Failed to start DeepFix server: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        typer.echo("\n👋 DeepFix server stopped.")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)
        