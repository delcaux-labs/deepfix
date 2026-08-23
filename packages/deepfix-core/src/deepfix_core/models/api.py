from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from io import StringIO
from typing import Any, Dict, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analysis import AgentResult, Severity
from .artifacts import (
    # DatasetArtifacts,
    DeepchecksArtifacts,
    ModelCheckpointArtifacts,
    TrainingArtifacts,
)
from .fixes import FinalFixReport


class AnalysisJobStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# API Models
class APIResponse(BaseModel):
    """Response from the DeepFix analysis API.

    Attributes:
        agent_results: Dictionary mapping agent names to their analysis results.
        summary: Optional overall summary of the analysis.
        additional_outputs: Dictionary of additional outputs from agents.
        error_messages: Optional dictionary mapping agent names to error messages
            if they failed.
        job_id: Optional unique identifier for the fix job if triggered.
        fix_report: Optional result of the autonomous fix session.
    """

    agent_results: Dict[str, AgentResult] = Field(
        default={}, description="Results of the agents"
    )
    summary: Optional[str] = Field(default=None, description="Summary of the analysis")
    additional_outputs: Dict[str, Any] = Field(
        default={}, description="Additional outputs from the agents"
    )
    error_messages: Optional[Dict[str, Optional[str]]] = Field(
        default=None, description="Error messages if the agents failed"
    )
    dataset_name: Optional[str] = Field(
        default=None, description="Name of the dataset analyzed"
    )
    job_id: Optional[str] = Field(
        default=None, description="Unique identifier for the fix job"
    )
    fix_report: Optional[FinalFixReport] = Field(
        default=None, description="Result of the autonomous fix session if executed"
    )

    def get_results_as_dataframe(self) -> pd.DataFrame:
        """Convert all agent results to a single pandas DataFrame.

        Returns:
            DataFrame containing all findings and recommendations from all agents.
        """
        dfs = [result.to_dataframe() for result in self.agent_results.values() if hasattr(result, "to_dataframe")]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs).reset_index(drop=True)

    def get_results_as_text(self) -> str:
        """Convert all agent results to a formatted text string.

        Returns:
            Formatted text report with summary statistics, findings by severity,
            and agent-specific analysis.
        """
        df = self.get_results_as_dataframe()
        if df.empty or "agent_name" not in df.columns:
            if self.fix_report is not None:
                status = "Success" if self.fix_report.success else "Failed"
                return f"Autonomous Fix Session Result:\nStatus: {status}\nApplied Fixes: {len(self.fix_report.applied_fixes)}"
            return "No analysis results found."
        summary = "=" * 80
        summary += "\nSUMMARY STATISTICS"
        summary += "\n" + "=" * 80

        summary += f"\nTotal findings: {len(df)}"
        summary += f"\nAgents involved: {df['agent_name'].unique().tolist()}"
        summary += "\nSeverity distribution:"
        summary += f"\n{df['finding_severity'].value_counts().to_dict()}"

        summary += "\nPriority distribution:"
        # summary += f"\n{df['recommendation_priority'].value_counts().to_dict()}"

        for severity in df["finding_severity"].unique():
            summary += "\n" + "=" * 80
            summary += f"\n{severity.upper()} SEVERITY ISSUES"
            summary += "\n" + "=" * 80

            df_severity = df[df["finding_severity"] == severity]
            for i, (_, row) in enumerate(df_severity.iterrows()):
                summary += (
                    f"\n{i + 1}. [{row['agent_name']}] {row['finding_description']}"
                )
                summary += f"\n   Evidence: {row['finding_evidence']}"
                summary += f"\n   Action: {row['recommendation_action']}"
                summary += f"\n   Rationale: {row['recommendation_rationale']}"

        summary += "\n" + "=" * 80
        summary += "\nAGENT-SPECIFIC ANALYSIS"
        summary += "\n" + "=" * 80

        for agent in df["agent_name"].unique():
            agent_df = df[df["agent_name"] == agent]
            summary += f"\n{agent}:"
            summary += f"\n  - Findings: {len(agent_df)}"
            summary += f"\n  - Artifacts analyzed: {agent_df['analyzed_artifacts'].iloc[0] if not agent_df.empty else 'None'}"
            if agent_df["summary"].iloc[0]:
                summary += f"\n  - Summary: {agent_df['summary'].iloc[0]}"

        return summary

    def to_text(self, verbose: bool = False) -> str:
        """Generate a beautifully formatted analysis report using Rich.

        Args:
            verbose: If False, only show results from CrossArtifactReasoningAgent.
                If True, show results from all agents.

        Returns:
            Formatted text report with Rich formatting for terminal display.

        Raises:
            ValueError: If verbose=False and no CrossArtifactReasoningAgent results
                are found.
        """
        # Create a string buffer to capture Rich output
        buffer = StringIO()
        console = Console(file=buffer, width=120, force_terminal=True)

        df = self.get_results_as_dataframe()

        if df.empty or "agent_name" not in df.columns:
            if self.fix_session_result is not None:
                return f"Autonomous Fix Session Result:\nStatus: {self.fix_session_result.stop_reason}\nTotal Iterations: {len(self.fix_session_result.iterations)}"
            raise ValueError(
                f"No analysis results found. Error messages: {self.error_messages}"
            )

        if not verbose:
            mask = df["agent_name"].isin(["CrossArtifactReasoningAgent"])
            if not any(mask):
                raise ValueError(
                    f"No analysis results found for ``CrossArtifactReasoningAgent``. Available agents: {', '.join(self.agent_results.keys())}"
                )
            df = df[mask]

        # Header Panel
        header_text = Text(
            "DEEPFIX ANALYSIS RESULT", style="bold blue", justify="center"
        )
        console.print(Panel(header_text, style="bold blue"))
        console.print()

        # Context Information (if available)
        if self.additional_outputs.get(
            "optimization_areas"
        ) or self.additional_outputs.get("constraints"):
            context_table = Table(show_header=False, box=None, padding=(0, 2))
            context_table.add_column("Label", style="blue bold")
            context_table.add_column("Value", style="black")

            if self.additional_outputs.get("optimization_areas"):
                context_table.add_row(
                    "Optimization Areas",
                    str(self.additional_outputs["optimization_areas"]),
                )
            if self.additional_outputs.get("constraints"):
                context_table.add_row(
                    "Constraints", str(self.additional_outputs["constraints"])
                )

            console.print(
                Panel(
                    context_table,
                    title="[bold blue]Context[/bold blue]",
                    border_style="blue",
                )
            )
            console.print()

        # Summary Panel
        if self.summary:
            summary_text = Text(self.summary, style="black")
            console.print(
                Panel(
                    summary_text,
                    title="[bold green]Summary[/bold green]",
                    border_style="green",
                )
            )
            console.print()

        # Summary Statistics Table
        stats_table = self._summary_table(df, verbose=verbose)
        console.print(stats_table)
        console.print()

        # Issues by Severity
        for severity in sorted(
            df["finding_severity"].unique(),
            key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x, 3),
        ):
            severity_color = {"high": "red", "medium": "yellow", "low": "green"}.get(
                severity, "black"
            )
            df_severity = df[df["finding_severity"] == severity]

            # Create a table for issues of this severity
            issues_table = self._issues_table(
                df_severity, severity, severity_color, verbose=verbose
            )
            console.print(issues_table)
            console.print()

        # Agent-Specific Analysis
        # if verbose:
        #    agent_table = self._agent_table(df)
        #    console.print(agent_table)

        # Get the string output
        return buffer.getvalue()

    def _summary_table(self, df: pd.DataFrame, verbose: bool = False) -> Table:
        """Create a Rich table with summary statistics.

        Args:
            df: DataFrame containing agent results.
            verbose: If True, include agents involved in the summary.

        Returns:
            Rich Table object with summary statistics.
        """
        stats_table = Table(
            title="Summary Statistics",
            show_header=True,
            header_style="bold blue",
            box=None,
        )
        stats_table.add_column("Metric", style="blue bold", width=30)
        stats_table.add_column("Value", style="black", width=60)

        stats_table.add_row("Total Findings", str(len(df)))
        if verbose:
            stats_table.add_row(
                "Agents Involved", ", ".join(df["agent_name"].unique().tolist())
            )

        # Severity distribution with color coding
        severity_counts = df["finding_severity"].value_counts().to_dict()
        severity_text = Text()
        for severity, count in severity_counts.items():
            color = {"high": "red", "medium": "yellow", "low": "green"}.get(
                severity, "black"
            )
            severity_text.append(
                f"{severity.upper()}: {count}  ", style=f"bold {color}"
            )
        stats_table.add_row("Severity Distribution", severity_text)

        # Priority distribution with color coding
        # priority_counts = df['recommendation_priority'].value_counts().to_dict()
        # priority_text = Text()
        # for priority, count in priority_counts.items():
        #    color = {"high": "red", "medium": "yellow", "low": "green"}.get(priority, "black")
        #    priority_text.append(f"{priority.upper()}: {count}  ", style=f"bold {color}")
        # stats_table.add_row("Priority Distribution", priority_text)

        return stats_table

    def _issues_table(
        self,
        df_severity: pd.DataFrame,
        severity: Severity,
        severity_color: str,
        verbose: bool = False,
    ) -> Table:
        """Create a Rich table for issues of a specific severity.

        Args:
            df_severity: DataFrame filtered to issues of the given severity.
            severity: Severity level (high, medium, low).
            severity_color: Color to use for the table border and headers.
            verbose: If True, include agent name column.

        Returns:
            Rich Table object with issues of the specified severity.
        """
        issues_table = Table(
            title=f"{severity.upper()} Severity Issues ({len(df_severity)})",
            show_header=True,
            header_style=f"bold {severity_color}",
            border_style=severity_color,
            expand=False,
        )
        issues_table.add_column("#", style="dim", width=3)
        if verbose:
            issues_table.add_column("Agent", style="blue bold", width=30)
        issues_table.add_column("Finding", style="black", width=40)
        issues_table.add_column("Action", style="black", width=40)

        for i, (_, row) in enumerate(df_severity.iterrows(), 1):
            if verbose:
                issues_table.add_row(
                    str(i),
                    row["agent_name"],
                    f"{row['finding_description']}\n[dim]Evidence: {row['finding_evidence']}[/dim]",
                    f"{row['recommendation_action']}\n[dim italic]{row['recommendation_rationale']}[/dim italic]",
                )
            else:
                issues_table.add_row(
                    str(i),
                    f"{row['finding_description']}\n[dim]Evidence: {row['finding_evidence']}[/dim]",
                    f"{row['recommendation_action']}\n[dim italic]{row['recommendation_rationale']}[/dim italic]",
                )
        return issues_table

    def _agent_table(self, df: pd.DataFrame) -> Table:
        """Create a Rich table with agent-specific analysis summary.

        Args:
            df: DataFrame containing agent results.

        Returns:
            Rich Table object with agent-specific statistics.
        """
        # Agent-Specific Analysis
        agent_table = Table(
            title="Agent-Specific Analysis",
            show_header=True,
            header_style="bold blue",
            border_style="blue",
        )
        # agent_table.add_column("Agent", style="blue bold", width=30)
        agent_table.add_column("Findings", justify="center", style="yellow", width=10)
        agent_table.add_column("Artifacts", style="black", width=30)
        agent_table.add_column("Summary", style="black", width=40)

        for agent in df["agent_name"].unique():
            agent_df = df[df["agent_name"] == agent]
            artifacts = (
                agent_df["analyzed_artifacts"].iloc[0] if not agent_df.empty else "None"
            )
            summary = (
                agent_df["summary"].iloc[0] if agent_df["summary"].iloc[0] else "N/A"
            )

            agent_table.add_row(
                # agent,
                str(len(agent_df)),
                artifacts,
                summary,
            )
        return agent_table


class APIJobResponse(BaseModel):
    """Response model for the DeepFix analysis API.

    Attributes:
        job_id: The ID of the analysis job.
        status: The status of the analysis job.
        result: Optional overall summary of the analysis.
        error: Optional error messages if the job failed.
        created_at: Timestamp of when the job was created.
        updated_at: Timestamp of when the job was last updated.
    """

    job_id: str
    status: AnalysisJobStatus
    result: Optional[APIResponse] = None
    error: Optional[str] = None
    created_at: Optional[Union[datetime, str]] = None
    updated_at: Optional[Union[datetime, str]] = None

    @property
    def is_finished(self) -> bool:
        return self.status in [
            AnalysisJobStatus.COMPLETED,
            AnalysisJobStatus.FAILED,
        ]


class APIRequest(BaseModel):
    """Request model for the DeepFix analysis API.

    Attributes:
        dataset_artifacts: Optional dataset statistics and metadata.
        training_artifacts: Optional training metrics and parameters.
        deepchecks_artifacts: Optional Deepchecks validation results.
        model_checkpoint_artifacts: Optional model checkpoint information.
        dataset_name: Optional name of the dataset being analyzed.
        model_name: Optional name of the model being analyzed.
        language: Language for the analysis output. Defaults to "english".
    """

    dataset_artifacts: Optional[Dict[str, Any]] = Field(
        default=None, description="Dataset artifacts"
    )
    training_artifacts: Optional[TrainingArtifacts] = Field(
        default=None, description="Training artifacts"
    )
    deepchecks_artifacts: Optional[DeepchecksArtifacts] = Field(
        default=None, description="Deepchecks artifacts"
    )
    model_checkpoint_artifacts: Optional[ModelCheckpointArtifacts] = Field(
        default=None, description="Model checkpoint artifacts"
    )
    dataset_name: Optional[str] = Field(default=None, description="Name of the dataset")
    model_name: Optional[str] = Field(default=None, description="Name of the model")
    language: str = Field(default="english", description="Language of the analysis")


class AutonomousFixRequest(APIRequest):
    """Request model for the autonomous fix loop endpoint.

    Attributes:
        baseline_run_id: MLflow run ID of the baseline model run.
        model_class: Full module path or name of the model class.
        dataset_load_code: Python code snippet used to load dataset.
        experiment_name: MLflow experiment name for fix iterations.
        mlflow_experiment_id: MLflow experiment ID.
        hf_dataset_dir: Directory path for Hugging Face dataset.
        hf_dataset_name: Name of Hugging Face dataset.
        dataset_name: Name of dataset logged in MLflow.
        dataset_digest: Digest of dataset.
        dataset_uri: URI of dataset.
        label_column: Name of label/target column.
    """

    baseline_run_id: str = Field(
        description="MLflow run ID of the baseline model run"
    )
    model_class: str = Field(description="Model class name or path")
    dataset_load_code: Optional[str] = Field(
        default=None, description="Python code to load the dataset"
    )
    experiment_name: str = Field(
        default="deepfix-autonomous", description="MLflow experiment name"
    )
    mlflow_experiment_id: str = Field(
        default="0", description="MLflow experiment ID"
    )
    hf_dataset_dir: Optional[str] = Field(
        default=None, description="Directory path for Hugging Face dataset"
    )
    hf_dataset_name: Optional[str] = Field(
        default=None, description="Name of Hugging Face dataset"
    )
    dataset_name: str = Field(
        default="deepfix_dataset", description="Name of the dataset"
    )
    dataset_digest: Optional[str] = Field(
        default=None, description="Digest of dataset"
    )
    dataset_uri: Optional[str] = Field(
        default=None, description="URI of dataset"
    )
    label_column: Optional[str] = Field(
        default=None, description="Name of label column"
    )

