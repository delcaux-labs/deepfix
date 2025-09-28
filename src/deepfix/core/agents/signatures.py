import dspy
from typing import List, Optional, Dict

from .models import Analysis


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
    previous_analyses: Dict[str, List[Optional[Analysis]]] = dspy.InputField(
        description="Separate analyses of artifacts from specialized agents"
    )    

    refined_analysis: str = dspy.OutputField(
        description="Refined analysis of the artifacts using all the available context"
    )
