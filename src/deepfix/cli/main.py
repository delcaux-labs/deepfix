"""Main CLI application for DeepFix."""

import typer
from .commands import commands_app

# Create the main CLI app
app = typer.Typer(
    name="deepfix",
    help="DeepFix - A copilot for machine learning model training and debugging",
    add_completion=False,
)

# Register command groups
app.add_typer(
    commands_app, help="Commands for DeepFix"
)

def main() -> None:
    """Entry point for the CLI application."""
    app()
