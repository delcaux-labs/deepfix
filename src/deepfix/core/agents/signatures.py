import dspy
from typing import List, Dict

from .models import Analysis, AgentResult


class TrainingArtifactsAnalysisSignature(dspy.Signature):
    artifacts: str = dspy.InputField(description="Training artifacts to analyze")
    analysis: List[Analysis] = dspy.OutputField(
        description="List of Analysis elements"
    )


class DeepchecksArtifactsAnalysisSignature(dspy.Signature):
    artifacts: str = dspy.InputField(description="Deepchecks artifacts to analyze")
    analysis: List[Analysis] = dspy.OutputField(
        description="List of Analysis elements"
    )


class DatasetArtifactsAnalysisSignature(dspy.Signature):
    artifacts: str = dspy.InputField(description="Dataset artifacts to analyze")
    analysis: List[Analysis] = dspy.OutputField(
        description="List of Analysis elements"
    )


class ModelCheckpointArtifactsAnalysisSignature(dspy.Signature):
    artifacts: str = dspy.InputField(
        description="Model checkpoint artifacts to analyze"
    )
    analysis: List[Analysis] = dspy.OutputField(
        description="List of Analysis elements"
    )


class ArtifactsAnalysisSignature(dspy.Signature):
    previous_analyses: Dict[str, AgentResult] = dspy.InputField(
        description="Separate analyses of artifacts from specialized agents"
    )    

    refined_analysis: str = dspy.OutputField(
        description="Refined analysis of the artifacts using all the available context"
    )
