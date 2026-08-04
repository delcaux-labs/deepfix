
from typing import Any, Dict, List, Optional

from deepfix_core.models import (
    AgentResult,
    Analysis,
    Artifacts,
    DatasetArtifacts,
    DeepchecksArtifacts,
    ModelCheckpointArtifacts,
    TrainingArtifacts,
)
from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Context for agent execution.

    Contains all artifacts, configuration, and intermediate results needed
    for agent analysis.

    Attributes:
        dataset_artifacts: Optional dataset statistics and metadata.
        training_artifacts: Optional training metrics and parameters.
        deepchecks_artifacts: Optional Deepchecks validation results.
        model_checkpoint_artifacts: Optional model checkpoint information.
        dataset_name: Optional name of the dataset being analyzed.
        language: Language for analysis output. Defaults to "english".
        agent_results: Dictionary of results from previously run agents.
        knowledge_cache: Dictionary for caching knowledge retrieval results.
    """

    dataset_artifacts: Optional[DatasetArtifacts] = None
    training_artifacts: Optional[TrainingArtifacts] = None
    deepchecks_artifacts: Optional[DeepchecksArtifacts] = None
    model_checkpoint_artifacts: Optional[ModelCheckpointArtifacts] = None
    dataset_name: Optional[str] = None
    language: str = Field(default="english", description="Language of the analysis")
    agent_results: Dict[str, AgentResult] = Field(
        default={}, description="Results of the agents"
    )
    knowledge_cache: Dict[str, Any] = Field(default={})

    @property
    def artifacts(
        self,
    ) -> List[Artifacts]:
        """Get list of all non-None artifacts.

        Returns:
            List of all artifacts that are not None.
        """
        artifacts = [
            self.dataset_artifacts,
            self.training_artifacts,
            self.deepchecks_artifacts,
            self.model_checkpoint_artifacts,
        ]
        return [a for a in artifacts if a is not None]

    def insert_artifact(self, artifact: Artifacts):
        """Insert an artifact into the appropriate context field.

        Args:
            artifact: Artifact to insert. Must be one of: DatasetArtifacts,
                TrainingArtifacts, DeepchecksArtifacts, ModelCheckpointArtifacts.

        Raises:
            ValueError: If artifact type is not recognized.
        """
        if isinstance(artifact, DatasetArtifacts):
            self.dataset_artifacts = artifact
        elif isinstance(artifact, TrainingArtifacts):
            self.training_artifacts = artifact
        elif isinstance(artifact, DeepchecksArtifacts):
            self.deepchecks_artifacts = artifact
        elif isinstance(artifact, ModelCheckpointArtifacts):
            self.model_checkpoint_artifacts = artifact
        else:
            raise ValueError(f"Invalid artifact type: {type(artifact)}")

class ArtifactAnalysisResult(BaseModel):
    """Result from artifact analysis.

    Attributes:
        summary: Optional overall summary of the analysis.
        analysis: List of analysis results.
    """

    summary: Optional[str] = Field(default=..., description="Summary of the analysis")
    analysis: List[Analysis] = Field(default=[], description="List of Analysis elements")

class CrossArtifactReasoningResult(BaseModel):
    """Structured output for the cross-artifact reasoning agent."""

    analysis: List[Analysis] = Field(
        description="Consolidated analysis with cross-artifact insights and recommendations",
        default_factory=list,
    )
    summary: str = Field(
        description="Summary of the cross-artifact reasoning and analysis",
        default="",
    )
