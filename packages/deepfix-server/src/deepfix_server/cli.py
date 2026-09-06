import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .api import run_analyse_artifacts_api
from .config import settings

app = typer.Typer(
    name="deepfix-server",
    help="DeepFix server for artifact analysis and diagnosis",
    add_completion=False,
)

def display_settings():
    """Display current settings in a beautiful way using rich."""
    console = Console()

    # LLM Settings Table
    llm_table = Table(show_header=False, box=box.SIMPLE_HEAD)
    llm_table.add_column("Property", style="bold cyan", width=20)
    llm_table.add_column("Value")
    llm_table.add_row("Model", settings.llm_model_name)
    llm_table.add_row("Base URL", settings.llm_base_url or "[dim]Default[/dim]")
    llm_table.add_row("Temperature", str(settings.llm_temperature))
    llm_table.add_row("Max Tokens", str(settings.llm_max_tokens))
    llm_table.add_row("Cache", "[green]Enabled[/green]" if settings.llm_cache else "[yellow]Disabled[/yellow]")
    llm_table.add_row("Track Usage", "[green]Enabled[/green]" if settings.llm_track_usage else "[yellow]Disabled[/yellow]")

    # Mask API key if present
    api_key_display = "[red]Missing[/red]"
    if settings.llm_api_key:
        api_key_display = f"{settings.llm_api_key[:4]}..."
    llm_table.add_row("API Key", api_key_display)

    # Database Settings Table
    db_table = Table(show_header=False, box=box.SIMPLE_HEAD)
    db_table.add_column("Property", style="bold green", width=20)
    db_table.add_column("Value")
    db_table.add_row("URL", settings.database_url)
    db_table.add_row("Echo", "[green]On[/green]" if settings.database_echo else "[dim]Off[/dim]")
    db_table.add_row("Job TTL", f"{settings.job_ttl_hours} hours")

    console.print("\n")
    console.print(Panel(llm_table, title="[bold cyan]LLM Configuration[/]", border_style="cyan", padding=(1, 2)))
    console.print(Panel(db_table, title="[bold green]Database Configuration[/]", border_style="green", padding=(1, 2)))
    console.print("\n")


@app.command(name="version")
def version() -> None:
    """Print the version of the DeepFix Server."""
    typer.echo("DeepFix Server version: 0.1.0")


@app.command(name="launch")
def launch(
    port: int = typer.Option(8844, "-port", help="Port to run DeepFix server on"),
    host: str = typer.Option("0.0.0.0", "-host", help="Host to bind DeepFix server to"),
    workers: int = typer.Option(1, "-workers", help="Number of worker processes"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Launch DeepFix server."""

    typer.echo(f"🚀 Starting DeepFix server on {host}:{port}")
    display_settings()

    run_analyse_artifacts_api(
        port=port,
        host=host,
        workers=workers,
        reload=reload,
        reload_dirs=["packages"] if reload else None,
    )


def main():
    app()
