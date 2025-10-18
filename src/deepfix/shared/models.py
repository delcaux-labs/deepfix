from __future__ import annotations
from abc import abstractmethod, ABC
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import StrEnum
import os
from omegaconf import DictConfig
import pandas as pd
import yaml
from omegaconf import OmegaConf
from platformdirs import (
                user_data_dir,
                user_cache_dir,
                user_log_dir,
            )
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from io import StringIO

# Defaults
def _get_base_dirs() -> Dict[str, Path]:
    """Resolve base directories with precedence:
    1) DEEPFIX_HOME env var
    2) platform-appropriate user dirs (via platformdirs)
    3) fallback to ~/.deepfix

    Ensures directories exist.
    """
    env_home = os.environ.get("DEEPFIX_HOME")
    if env_home:
        base = Path(env_home).expanduser()
        data_dir = base / "data"
        cache_dir = base / "cache"
        log_dir = base / "logs"
    else:
        try:
            data_dir = Path(user_data_dir("deepfix", "deepfix"))
            cache_dir = Path(user_cache_dir("deepfix", "deepfix"))
            log_dir = Path(user_log_dir("deepfix", "deepfix"))
        except:
            base = Path("~/.deepfix").expanduser()
            data_dir = base / "data"
            cache_dir = base / "cache"
            log_dir = base / "logs"

    for d in (data_dir, cache_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data": data_dir,
        "cache": cache_dir,
        "log": log_dir,
    }

def _default_mlflow_tracking_uri(data_dir: Path) -> str:
    mlruns_dir = data_dir / "deepfix_mlflow"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    # Use an OS-correct file:// URI (e.g., file:///C:/... on Windows)
    return mlruns_dir.resolve().as_uri()

def _default_mlflow_downloads_dir(data_dir: Path) -> str:
    downloads = data_dir / "mlflow_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    return str(downloads)

def _default_mlflow_artifact_root(data_dir: Path) -> str:
    artifact_root = data_dir / "mlflow_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return str(artifact_root)

def _default_sqlite_path(data_dir: Path) -> str:
    sqlite_path = data_dir / "tmp" / "artifacts.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return str(sqlite_path)

def _default_output_dir(data_dir: Path) -> str:
    out = data_dir / "advisor_output"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)

def _default_knowledge_base_dir(data_dir: Path) -> str:
    knowledge_base_dir = data_dir / "knowledge_base"
    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    return str(knowledge_base_dir)

def _default_knowledge_base_indices_dir(data_dir: Path) -> str:
    p = _default_knowledge_base_dir(data_dir)
    knowledge_base_indices_dir = Path(p) / "indices"
    knowledge_base_indices_dir.mkdir(parents=True, exist_ok=True)
    return str(knowledge_base_indices_dir)

def _default_knowledge_base_documents_dir(data_dir: Path) -> str:
    p = _default_knowledge_base_dir(data_dir)
    knowledge_base_documents_dir = Path(p) / "documents"
    knowledge_base_documents_dir.mkdir(parents=True, exist_ok=True)
    return str(knowledge_base_documents_dir)

_BASE_DIRS = _get_base_dirs()


class DefaultPaths(StrEnum):
    MLFLOW_TRACKING_URI = _default_mlflow_tracking_uri(_BASE_DIRS["data"])
    MLFLOW_DOWNLOADS = _default_mlflow_downloads_dir(_BASE_DIRS["data"])
    MLFLOW_RUN_NAME = "default"
    MLFLOW_DEFAULT_ARTIFACT_ROOT = _default_mlflow_artifact_root(_BASE_DIRS["data"])

    DATASETS_EXPERIMENT_NAME = "deepfix_datasets"

    ARTIFACTS_SQLITE_PATH = _default_sqlite_path(_BASE_DIRS["data"])

    ADVISOR_OUTPUT_DIR = _default_output_dir(_BASE_DIRS["data"])

    KNOWLEDGE_BASE_DIR = _default_knowledge_base_dir(_BASE_DIRS["data"])
    KNOWLEDGE_BASE_INDICES_DIR = _default_knowledge_base_indices_dir(_BASE_DIRS["data"])
    KNOWLEDGE_BASE_DOCUMENTS_DIR = _default_knowledge_base_documents_dir(_BASE_DIRS["data"])


# MLTest configs
class DeepchecksConfig(BaseModel):
    train_test_validation: bool = Field(
        default=True, description="Whether to run the train_test_validation suite"
    )
    data_integrity: bool = Field(
        default=True, description="Whether to run the data_integrity suite"
    )
    model_evaluation: bool = Field(
        default=False, description="Whether to run the model_evaluation suite"
    )
    max_samples: Optional[int] = Field(
        default=None, description="Maximum number of samples to run the suites on"
    )
    random_state: int = Field(
        default=42, description="Random seed to use for the suites"
    )
    save_results: bool = Field(default=False, description="Whether to save the results")
    output_dir: Optional[str] = Field(
        default=None, description="Output directory to save the results"
    )
    batch_size: int = Field(default=16, description="Batch size to use for the suites")

    @classmethod
    def from_dict(cls, config: Union[Dict[str, Any], DictConfig]) -> "DeepchecksConfig":
        return cls(**config)

    @classmethod
    def from_file(cls, file_path: str) -> "DeepchecksConfig":
        return cls.from_dict(OmegaConf.load(file_path))


# Artifacts models
class ArtifactPath(StrEnum):
    # training artifacts
    TRAINING = "training_artifacts"
    TRAINING_METRICS = "metrics.csv"
    MODEL_CHECKPOINT = "best_checkpoint"
    TRAINING_PARAMS = "params.yaml"
    # deepchecks artifacts
    DEEPCHECKS = "deepchecks"
    # dataset artifacts
    DATASET = "dataset"



## Deepchecks
class DeepchecksResultHeaders(StrEnum):
    # Train-Test Validation
    LabelDrift = "Label Drift"
    ImageDatasetDrift = "Image Dataset Drift"
    ImagePropertyDrift = "Image Property Drift"
    PropertyLabelCorrelationChange = "Property Label Correlation Change"
    HeatmapComparison = "Heatmap Comparison"
    NewLabels = "New Labels"
    # Data Integrity
    ImagePropertyOutliers = "Image Property Outliers"
    PropertyLabelCorrelation = "Property Label Correlation"
    LabelPropertyOutliers = "Label Property Outliers"
    ClassPerformance = "Class Performance"


class DeepchecksParsedResult(BaseModel):
    header: str = Field(description="Header of the result")
    json_result: Dict[str, Any] = Field(description="JSON result of the result")
    display_images: Optional[List[str]] = Field(
        default=None,
        description="Display images of the result as base64 encoded strings",
    )
    display_txt: Optional[str] = Field(
        default=None, description="Display text of the result"
    )

    def to_dict(self, exclude_images: bool = False) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        dumped_dict["header"] = dumped_dict["header"]
        dumped_dict.pop("display_txt")
        if exclude_images:
            dumped_dict.pop("display_images")
        return dumped_dict

    @classmethod
    def from_dict(
        self, d: Union[Dict[str, Any], DictConfig]
    ) -> "DeepchecksParsedResult":
        return DeepchecksParsedResult(
            header=d["header"],
            json_result=d["json_result"],
            display_images=d.get("display_images", None),
            display_txt=d.get("display_txt", None),
        )


class Artifacts(BaseModel):
    
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("to_dict method to be implemented in the subclass")

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


class DeepchecksArtifacts(Artifacts):
    dataset_name: str = Field(description="Name of the dataset")
    results: Dict[str, List[DeepchecksParsedResult]] = Field(
        description="Results of the artifact"
    )
    config: Optional[DeepchecksConfig] = Field(
        default=None, description="Config of the artifact"
    )

    def to_dict(self) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        dumped_dict["results"] = {
            k: [r.to_dict() for r in v] for k, v in self.results.items()
        }
        dumped_dict["config"] = self.config.model_dump() if self.config else None
        return dumped_dict

    @classmethod
    def from_dict(self, d: Union[Dict[str, Any], DictConfig]) -> "DeepchecksArtifacts":
        results = {
            k: [DeepchecksParsedResult.from_dict(r) for r in v]
            for k, v in d["results"].items()
        }
        config = None
        if d.get("config"):
            config = DeepchecksConfig.from_dict(d["config"])
        return DeepchecksArtifacts(
            dataset_name=d["dataset_name"], results=results, config=config
        )

    @classmethod
    def from_file(
        cls, file_path: Optional[str] = None, dir_path: Optional[str] = None
    ) -> "DeepchecksArtifacts":
        assert (file_path is None) ^ (dir_path is None), (
            "Either file_path or dir_path must be provided"
        )

        file_path = (
            file_path
            if (file_path is not None)
            else os.path.join(dir_path, ArtifactPath.DEEPCHECKS_ARTIFACTS.value)
        )
        with open(file_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)

        artifacts = cls.from_dict(d)
        if dir_path is not None:
            artifacts.config = DeepchecksConfig.from_file(
                os.path.join(dir_path, ArtifactPath.DEEPCHECKS_CONFIG.value)
            )

        return artifacts


class ModelCheckpointArtifacts(Artifacts):
    path: str = Field(description="Path to the model checkpoint")
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Config of the model"
    )

    def to_dict(self) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict):
        return cls(model_path=d["model_path"], model_config=d.get("config"))

    @classmethod
    def from_file(cls, path: str) -> "ModelCheckpointArtifacts":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


## Training Artifacts
class TrainingArtifacts(Artifacts):
    model_config = {"arbitrary_types_allowed": True}

    metrics_path: Optional[str] = Field(
        default=None, description="Path to the metrics file"
    )
    metrics_values: Optional[pd.DataFrame] = Field(
        default=None, description="Metrics of the artifact"
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters of the training routine"
    )

    def to_dict(self) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        if self.metrics_values is not None:
            dumped_dict["metrics_values"] = self.metrics_values.to_dict(orient="list")
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict):
        if d.get("metrics_values"):
            metrics_values = pd.DataFrame.from_dict(d.get("metrics_values"))
        elif d.get("metrics_path"):
            metrics_values = pd.read_csv(d.get("metrics_path"))
        else:
            metrics_values = None
        return cls(
            metrics_path=d.get("metrics_path"),
            metrics_values=metrics_values,
            params=d.get("params"),
        )

    @classmethod
    def from_file(cls, metrics_path: str) -> "TrainingArtifacts":
        return cls(metrics_path=metrics_path, metrics_values=pd.read_csv(metrics_path))


## Dataset
class DatasetArtifacts(Artifacts):
    dataset_name: str = Field(..., description="Name of the dataset")
    statistics: Optional[Dict[str, Any]] = Field(
        default=None, description="Statistics of the dataset"
    )

    def to_dict(self) -> Dict[str, Any]:
        dumped_dict = self.model_dump()
        return dumped_dict

    @classmethod
    def from_file(cls, path: str) -> "DatasetArtifacts":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls(dataset_name=d["dataset_name"], statistics=d["statistics"])



# ============================================================================
# Analysis data Models
# ============================================================================
class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class AnalyzerTypes(StrEnum):
    TRAINING = "training"
    DEEPCHECKS = "deepchecks"
    DATASET = "dataset"
    MODEL_CHECKPOINT = "model_checkpoint"
    
class Finding(BaseModel):
    description: str = Field(default=...,description="Short Description of the finding")
    evidence: str = Field(default=...,description="Evidence of the finding")
    severity: Severity = Field(default=...,description="Severity of the finding")
    confidence: float = Field(ge=0.0, le=1.0,description="Confidence in the finding and severity")

class Recommendation(BaseModel):
    action: str =  Field(default=...,description="Action to take to address the finding")
    rationale: str =  Field(default=...,description="Rationale for the recommendation")
    #priority: Severity = Field(default=...,description="Priority of the recommendation")
    confidence: float = Field(ge=0.0, le=1.0,description="Confidence in the recommendation")

class Analysis(BaseModel):
    findings: Finding = Field(default=...,description="Finding of the analysis")
    recommendations: Recommendation = Field(default=...,description="Recommendation based on the findings")

class AgentResult(BaseModel):
    agent_name: str
    analysis: List[Analysis] = Field(default=[],description="List of Analysis elements")
    analyzed_artifacts: Optional[List[str]] = Field(default=None,description="List of artifacts analyzed by the agent")
    retrieved_knowledge: Optional[List[str]] = Field(default=None,description="External knowledge relevant to the analysis")
    additional_outputs: Dict[str, Any] = Field(default={},description="Additional outputs from the agent")
    error_message: Optional[str] = Field(default=None,description="Error message if the agent failed")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for analysis in self.analysis:
            # Extract findings and recommendations from the Analysis object
            findings = analysis.findings
            recommendations = analysis.recommendations
            
            # Create a row combining findings and recommendations
            row = {
                'agent_name': self.agent_name,
                'analyzed_artifacts': ', '.join(self.analyzed_artifacts) if self.analyzed_artifacts else '',
                'retrieved_knowledge': ', '.join(self.retrieved_knowledge) if self.retrieved_knowledge else '',
                'summary': self.additional_outputs.get('summary', ''),
                'finding_description': findings.description,
                'finding_evidence': findings.evidence,
                'error_message': self.error_message,
                'finding_severity': findings.severity.value,
                'finding_confidence': findings.confidence,
                'recommendation_action': recommendations.action,
                'recommendation_rationale': recommendations.rationale,
                #'recommendation_priority': recommendations.priority.value,
                'recommendation_confidence': recommendations.confidence
            }
            rows.append(row)
        
        return pd.DataFrame(rows)

# API Models
class APIResponse(BaseModel):
    agent_results: Dict[str, AgentResult] = Field(default={},description="Results of the agents")
    summary: Optional[str] = Field(default=None,description="Summary of the analysis")
    additional_outputs: Dict[str, Any] = Field(default={},description="Additional outputs from the agents")
    error_messages: Optional[Dict[str, Optional[str]]] = Field(default=None,description="Error messages if the agents failed")

    def get_results_as_dataframe(self) -> pd.DataFrame:
        dfs = [result.to_dataframe() for result in self.agent_results.values()]                
        return pd.concat(dfs).reset_index(drop=True)

    def get_results_as_text(self) -> str:
        df = self.get_results_as_dataframe()
        summary = "="*80
        summary += "\nSUMMARY STATISTICS"
        summary += ("\n" + "="*80)

        summary += (f"\nTotal findings: {len(df)}")
        summary += (f"\nAgents involved: {df['agent_name'].unique().tolist()}")
        summary += ("\nSeverity distribution:")
        summary += f"\n{df['finding_severity'].value_counts().to_dict()}"

        summary += (f"\nPriority distribution:")
        #summary += f"\n{df['recommendation_priority'].value_counts().to_dict()}"
        
        for severity in df['finding_severity'].unique():
            summary += ("\n" + "="*80)
            summary += (f"\n{severity.upper()} SEVERITY ISSUES")
            summary += ("\n" + "="*80)

            df_severity = df[df['finding_severity'] == severity]
            for i, (_, row) in enumerate(df_severity.iterrows()):
                summary += (f"\n{i+1}. [{row['agent_name']}] {row['finding_description']}")
                summary += (f"\n   Evidence: {row['finding_evidence']}")
                summary += (f"\n   Action: {row['recommendation_action']}")
                summary += (f"\n   Rationale: {row['recommendation_rationale']}")

        summary += ("\n" + "="*80)
        summary += ("\nAGENT-SPECIFIC ANALYSIS")
        summary += ("\n" + "="*80)

        for agent in df['agent_name'].unique():
            agent_df = df[df['agent_name'] == agent]
            summary += (f"\n{agent}:")
            summary += (f"\n  - Findings: {len(agent_df)}")
            summary += (f"\n  - Artifacts analyzed: {agent_df['analyzed_artifacts'].iloc[0] if not agent_df.empty else 'None'}")
            if agent_df['summary'].iloc[0]:
                summary += (f"\n  - Summary: {agent_df['summary'].iloc[0]}")

        return summary

    def to_text(self) -> str:
        """Generate a beautifully formatted analysis report using Rich."""
        # Create a string buffer to capture Rich output
        buffer = StringIO()
        console = Console(file=buffer, width=120, force_terminal=True)
        
        df = self.get_results_as_dataframe()
        
        # Header Panel
        header_text = Text("DEEPFIX ANALYSIS RESULT", style="bold blue", justify="center")
        console.print(Panel(header_text, style="bold blue"))
        console.print()
        
        # Context Information (if available)
        if self.additional_outputs.get('optimization_areas') or self.additional_outputs.get('constraints'):
            context_table = Table(show_header=False, box=None, padding=(0, 2))
            context_table.add_column("Label", style="blue bold")
            context_table.add_column("Value", style="black")
            
            if self.additional_outputs.get('optimization_areas'):
                context_table.add_row("Optimization Areas", str(self.additional_outputs['optimization_areas']))
            if self.additional_outputs.get('constraints'):
                context_table.add_row("Constraints", str(self.additional_outputs['constraints']))
            
            console.print(Panel(context_table, title="[bold blue]Context[/bold blue]", border_style="blue"))
            console.print()
        
        # Summary Panel
        if self.summary:
            summary_text = Text(self.summary, style="black")
            console.print(Panel(summary_text, title="[bold green]Summary[/bold green]", border_style="green"))
            console.print()
        
        # Summary Statistics Table
        stats_table = Table(title="Summary Statistics", show_header=True, header_style="bold blue", box=None)
        stats_table.add_column("Metric", style="blue bold", width=30)
        stats_table.add_column("Value", style="black", width=60)
        
        stats_table.add_row("Total Findings", str(len(df)))
        stats_table.add_row("Agents Involved", ", ".join(df['agent_name'].unique().tolist()))
        
        # Severity distribution with color coding
        severity_counts = df['finding_severity'].value_counts().to_dict()
        severity_text = Text()
        for severity, count in severity_counts.items():
            color = {"high": "red", "medium": "yellow", "low": "green"}.get(severity, "black")
            severity_text.append(f"{severity.upper()}: {count}  ", style=f"bold {color}")
        stats_table.add_row("Severity Distribution", severity_text)
        
        # Priority distribution with color coding
        #priority_counts = df['recommendation_priority'].value_counts().to_dict()
        #priority_text = Text()
        #for priority, count in priority_counts.items():
        #    color = {"high": "red", "medium": "yellow", "low": "green"}.get(priority, "black")
        #    priority_text.append(f"{priority.upper()}: {count}  ", style=f"bold {color}")
        #stats_table.add_row("Priority Distribution", priority_text)
        
        console.print(stats_table)
        console.print()
        
        # Issues by Severity
        for severity in sorted(df['finding_severity'].unique(), key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x, 3)):
            severity_color = {"high": "red", "medium": "yellow", "low": "green"}.get(severity, "black")
            df_severity = df[df['finding_severity'] == severity]
            
            # Create a table for issues of this severity
            issues_table = Table(
                title=f"{severity.upper()} Severity Issues ({len(df_severity)})",
                show_header=True,
                header_style=f"bold {severity_color}",
                border_style=severity_color,
                expand=False
            )
            issues_table.add_column("#", style="dim", width=3)
            issues_table.add_column("Agent", style="blue bold", width=30)
            issues_table.add_column("Finding", style="black", width=40)
            issues_table.add_column("Action", style="black", width=40)
            
            for i, (_, row) in enumerate(df_severity.iterrows(), 1):
                issues_table.add_row(
                    str(i),
                    row['agent_name'],
                    f"{row['finding_description']}\n[dim]Evidence: {row['finding_evidence']}[/dim]",
                    f"{row['recommendation_action']}\n[dim italic]{row['recommendation_rationale']}[/dim italic]"
                )
            
            console.print(issues_table)
            console.print()
        
        # Agent-Specific Analysis
        agent_table = Table(
            title="Agent-Specific Analysis",
            show_header=True,
            header_style="bold blue",
            border_style="blue"
        )
        agent_table.add_column("Agent", style="blue bold", width=30)
        agent_table.add_column("Findings", justify="center", style="yellow", width=10)
        agent_table.add_column("Artifacts", style="black", width=30)
        agent_table.add_column("Summary", style="black", width=40)
        
        for agent in df['agent_name'].unique():
            agent_df = df[df['agent_name'] == agent]
            artifacts = agent_df['analyzed_artifacts'].iloc[0] if not agent_df.empty else 'None'
            summary = agent_df['summary'].iloc[0] if agent_df['summary'].iloc[0] else 'N/A'
            
            agent_table.add_row(
                agent,
                str(len(agent_df)),
                artifacts,
                summary
            )
        
        console.print(agent_table)
        
        # Get the string output
        return buffer.getvalue()

class APIRequest(BaseModel):
    dataset_artifacts: Optional[DatasetArtifacts] = Field(default=None,description="Dataset artifacts")
    training_artifacts: Optional[TrainingArtifacts] = Field(default=None,description="Training artifacts")
    deepchecks_artifacts: Optional[DeepchecksArtifacts] = Field(default=None,description="Deepchecks artifacts")
    model_checkpoint_artifacts: Optional[ModelCheckpointArtifacts] = Field(default=None,description="Model checkpoint artifacts")
    dataset_name: Optional[str] = Field(default=None,description="Name of the dataset")