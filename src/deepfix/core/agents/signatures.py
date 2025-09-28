import dspy
from typing import List

from .models import Finding, Recommendation


class TrainingArtifactsAnalysisSignature(dspy.Signature):
    """Signature for training artifacts analysis"""
    training_artifacts: str = dspy.InputField(description="Training artifacts to analyze")
    findings: List[Finding] = dspy.OutputField(description="List of findings from analyzing training artifacts")
    recommendations: List[Recommendation] = dspy.OutputField(description="List of recommendations based on the findings")

class DeepchecksArtifactsAnalysisSignature(dspy.Signature):
    """Signature for deepchecks artifacts analysis"""
    deepchecks_artifacts: str = dspy.InputField(description="Deepchecks artifacts to analyze")
    findings: List[Finding] = dspy.OutputField(description="List of findings from analyzing deepchecks artifacts")
    recommendations: List[Recommendation] = dspy.OutputField(description="List of recommendations based on the findings")

class DatasetArtifactsAnalysisSignature(dspy.Signature):
    """Signature for dataset artifacts analysis"""
    dataset_artifacts: str = dspy.InputField(description="Dataset artifacts to analyze")
    findings: List[Finding] = dspy.OutputField(description="List of findings from analyzing dataset artifacts")
    recommendations: List[Recommendation] = dspy.OutputField(description="List of recommendations based on the findings")

class ModelCheckpointArtifactsAnalysisSignature(dspy.Signature):
    """Signature for model checkpoint artifacts analysis"""
    model_checkpoint_artifacts: str = dspy.InputField(description="Model checkpoint artifacts to analyze")
    findings: List[Finding] = dspy.OutputField(description="List of findings from analyzing model checkpoint artifacts")
    recommendations: List[Recommendation] = dspy.OutputField(description="List of recommendations based on the findings")

class ArtifactsAnalysisSignature(dspy.Signature):
    """Signature for artifacts analysis"""
    training_analysis: str = dspy.InputField(description="Training artifacts analysis")
    deepchecks_analysis: str = dspy.InputField(description="Deepchecks artifacts analysis")
    dataset_analysis: str = dspy.InputField(description="Dataset artifacts analysis")
    model_checkpoint_analysis: str = dspy.InputField(description="Model checkpoint artifacts analysis")

    findings: List[Finding] = dspy.OutputField(description="List of findings from analyzing artifacts")
    recommendations: List[Recommendation] = dspy.OutputField(description="List of recommendations based on the findings")
