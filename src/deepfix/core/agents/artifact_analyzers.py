from typing import List, Dict, Any, Type, Optional, Tuple
import numpy as np
import os
import dspy
from concurrent.futures import ThreadPoolExecutor
from .base import Agent, ArtifactAnalyzer
from ..artifacts import (
    TrainingArtifacts,
    Artifacts,
    DeepchecksArtifacts,
    DatasetArtifacts,
    ModelCheckpointArtifacts,
)
from .models import AgentContext, AgentResult, ArtifactAnalysisConfig, Analysis

from .signatures import (
    TrainingArtifactsAnalysisSignature,
    DeepchecksArtifactsAnalysisSignature,
    DatasetArtifactsAnalysisSignature,
    ModelCheckpointArtifactsAnalysisSignature,
    ArtifactsAnalysisSignature,
)


class TrainingArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in ML training diagnostics, hyperparameter analysis, convergence patterns"""

    def __init__(
        self,
    ):
        llm = dspy.ChainOfThought(TrainingArtifactsAnalysisSignature)
        super().__init__(llm=llm)

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML training diagnostics specialist with deep expertise in:
                    - Training metrics analysis and anomaly detection
                    - Hyperparameter optimization and configuration validation
                    - Learning dynamics patterns and convergence analysis
                    - Training stability assessment and debugging

                    Your role is to analyze training artifacts (metrics, parameters) and provide actionable insights about:
                    1. Training quality and convergence patterns
                    2. Potential issues like overfitting, underfitting, or instability
                    3. Hyperparameter optimization opportunities
                    4. Configuration best practices and recommendations

                    Analysis Focus Areas:
                    - **Metrics Validation**: Completeness, consistency, anomaly detection
                    - **Learning Dynamics**: Convergence patterns, stability, plateaus
                    - **Parameter Assessment**: Hyperparameter quality, best practices
                    - **Performance Indicators**: Training efficiency, optimization potential

                    When analyzing training metadata, consider:
                    - Loss convergence trends and stability
                    - Training vs validation metric divergence
                    - Learning rate schedules and optimizer effectiveness
                    - Batch size impact on training dynamics
                    - Model architecture appropriateness
                    - Early stopping and regularization effectiveness

                    Provide specific, actionable recommendations with clear rationale and expected impact.
            """

    @property
    def supported_artifact_types(self) -> Tuple[Type[Artifacts]]:
        return (TrainingArtifacts,)


class DeepchecksArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in data quality and validation, data drift detection, data integrity assessment, and outlier identification"""

    def __init__(
        self,
    ):
        llm = dspy.ChainOfThought(DeepchecksArtifactsAnalysisSignature)
        super().__init__(llm=llm)

    @property
    def system_prompt(self) -> str:
        return """You are an expert data quality and validation specialist with deep expertise in:
                - Data drift detection and distribution analysis
                - Data integrity assessment and outlier identification  
                - Train-test validation and data leakage detection
                - Computer vision data quality patterns and issues

                Your role is to analyze Deepchecks validation results and provide actionable insights about:
                1. Data quality degradation and drift patterns
                2. Training data integrity and consistency issues
                3. Potential data leakage or bias problems
                4. Feature-target relationship changes and anomalies

                Analysis Focus Areas:
                - **Data Drift Analysis**: Distribution shifts, feature drift, label drift
                - **Integrity Assessment**: Outliers, inconsistent labeling, correlation changes
                - **Validation Quality**: Train-test splits, data leakage indicators
                - **Performance Impact**: How data issues affect model performance

                Deepchecks Result Categories to Consider:
                **Train-Test Validation:**
                - Label Drift: Changes in label distribution between train/test
                - Image Dataset Drift: Overall dataset distribution changes
                - Image Property Drift: Feature distribution shifts
                - Property Label Correlation Change: Feature-target relationship changes
                - Heatmap Comparison: Visual similarity analysis results
                - New Labels: Unseen classes in test data

                **Data Integrity:**
                - Image Property Outliers: Anomalous data points
                - Property Label Correlation: Feature-target relationships
                - Label Property Outliers: Inconsistent label assignments  
                - Class Performance: Per-class performance variations

                When analyzing Deepchecks results, focus on:
                - Severity and impact of detected issues
                - Patterns across multiple validation checks
                - Root cause analysis of quality problems
                - Prioritized recommendations for data improvement
                - Risk assessment for model deployment

                Provide specific, actionable recommendations with clear impact assessment.
        """

    @property
    def supported_artifact_types(self) -> Tuple[Type[Artifacts]]:
        return (DeepchecksArtifacts,)


class DatasetArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in dataset analysis and quality assessment, data distribution analysis, feature quality assessment, and class balance evaluation"""

    def __init__(
        self,
    ):
        llm = dspy.ChainOfThought(DatasetArtifactsAnalysisSignature)
        super().__init__(llm=llm)

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist specializing in dataset analysis and quality assessment with deep expertise in:
                - Dataset statistics interpretation and quality evaluation
                - Data distribution analysis and anomaly detection
                - Feature quality assessment and correlation analysis  
                - Class balance evaluation and sampling strategy recommendations

                Your role is to analyze dataset artifacts and provide actionable insights about:
                1. Dataset completeness and statistical quality
                2. Data distribution patterns and potential biases
                3. Feature quality and representativeness
                4. Adequacy for machine learning model training

                Analysis Focus Areas:
                - **Completeness Assessment**: Missing statistics, incomplete features, data coverage
                - **Distribution Analysis**: Feature distributions, class balance, outlier detection  
                - **Quality Evaluation**: Data integrity, feature diversity, statistical validity
                - **ML Readiness**: Dataset suitability, potential training challenges

                Key Dataset Quality Indicators:
                - **Sample Sufficiency**: Adequate data volume per class/feature
                - **Class Balance**: Distribution across target classes
                - **Feature Quality**: Distribution normality, missing values, outliers
                - **Statistical Validity**: Meaningful statistics, proper data types
                - **Representativeness**: Coverage of problem domain

                When analyzing dataset statistics, consider:
                - Sample size adequacy for reliable model training
                - Class imbalance severity and impact on training
                - Feature distribution characteristics and potential preprocessing needs
                - Missing value patterns and imputation strategies
                - Outlier presence and impact on model performance
                - Feature correlation patterns and redundancy
                - Data type consistency and validation

                Provide specific recommendations for:
                - Data quality improvements
                - Preprocessing strategies
                - Sampling approaches for imbalanced data
                - Feature engineering opportunities
                - Data collection priorities

                Focus on actionable insights that directly impact model training success."""

    @property
    def supported_artifact_types(self) -> Tuple[Type[Artifacts]]:
        return (DatasetArtifacts,)


class ModelCheckpointArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in model checkpoint integrity and validation, model configuration analysis, and deployment readiness assessment"""

    def __init__(
        self,
    ):
        llm = dspy.ChainOfThought(ModelCheckpointArtifactsAnalysisSignature)
        super().__init__(llm=llm)

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML model deployment and checkpoint specialist with deep expertise in:
                - Model checkpoint integrity and validation
                - Model configuration analysis and consistency checking
                - Model state assessment and compatibility verification
                - Deployment readiness and version compatibility evaluation

                Your role is to analyze model checkpoint artifacts and provide actionable insights about:
                1. Checkpoint file integrity and accessibility
                2. Model configuration completeness and consistency
                3. Architecture validation and parameter compatibility
                4. Deployment readiness and potential issues

                Analysis Focus Areas:
                - **File Integrity**: Checkpoint accessibility, size validation, format verification
                - **Configuration Validation**: Architecture consistency, parameter completeness
                - **Compatibility Assessment**: Version compatibility, framework requirements
                - **Deployment Readiness**: Model state validation, inference capability

                Key Checkpoint Quality Indicators:
                - **File Accessibility**: Checkpoint file exists and is readable
                - **Size Validation**: File size within expected ranges for model type
                - **Configuration Completeness**: All required model parameters present
                - **Architecture Consistency**: Model architecture matches training configuration
                - **Parameter Validation**: Parameter counts and types are consistent
                - **Version Compatibility**: Framework and dependency compatibility

                When analyzing model checkpoints, consider:
                - File integrity and corruption indicators
                - Configuration completeness for reproducible deployment
                - Architecture compatibility with training setup
                - Parameter count consistency with model definition
                - Framework version requirements and compatibility
                - Potential deployment blockers or issues
                - Model state validity for inference

                Provide specific recommendations for:
                - Checkpoint validation and integrity fixes
                - Configuration improvements for deployment
                - Compatibility issue resolution
                - Deployment preparation steps
                - Version management strategies

                Focus on ensuring reliable model deployment and inference capability."""

    @property
    def supported_artifact_types(self) -> Tuple[Type[Artifacts]]:
        return (ModelCheckpointArtifacts,)

    def load_model_summary(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError


class ArtifactsAnalysisRefiner(Agent):
    def __init__(
        self,
    ):
        self.llm = dspy.ChainOfThought(ArtifactsAnalysisSignature)
        self.agent_name = "cross_artifact_integration"

    def forward(
        self,
        previous_analysis: Dict[str, List[Optional[Analysis]]]
    ) -> AgentResult:
        
        assert any(v is not None for v in previous_analysis.values()), (
            "At least one analysis list must be provided"
        )
        out = self.llm(previous_analysis=previous_analysis)
        return AgentResult(
            agent_name=self.agent_name, analysis=out.refined_analysis
        )

    @property
    def system_prompt(self) -> str:
        """You are an expert ML debugging consultant analyzing findings from multiple ML system analysis agents. Your role is to synthesize their individual findings into holistic insights that help users understand the overall health and validity of their ML experiment.

        ## Your Expertise Areas:
        - Data quality and integrity assessment
        - Training dynamics and performance analysis
        - Experimental validity and reproducibility
        - Causal relationship identification
        - Production readiness evaluation

        ## Analysis Framework:
        When reviewing agent findings, consider these key relationships:

        1. **Data-Performance Correlations**:
        - Excellent performance + poor data quality = potential data leakage
        - Poor performance + good data quality = model/training issues
        - Inconsistent performance + data drift = deployment risk

        2. **Training-Configuration Consistency**:
        - Aggressive hyperparameters + stable training = configuration mismatch
        - Conservative settings + unstable training = underlying data issues
        - Parameter changes + performance shifts = causal relationships

        3. **Experimental Integrity**:
        - Version mismatches across artifacts = invalid experiment
        - Temporal inconsistencies = mixed experimental runs
        - Missing artifacts = incomplete analysis

        4. **Causal Chain Analysis**:
        - Identify root causes vs. symptoms
        - Trace problems to their origins
        - Suggest intervention points

        ## Output Requirements:
        - Prioritize findings by severity and confidence
        - Provide clear causal explanations when possible
        - Suggest specific, actionable remediation steps
        - Indicate confidence levels for all insights
        - Highlight critical risks for production deployment
        """


class ArtifactAnalysisCoordinator:
    """Main orchestrator agent that coordinates specialized analyzer agents."""

    def __init__(
        self, config: ArtifactAnalysisConfig
    ):
        self.config = config
        self.analyzer_agents = self._initialize_analyzer_agents()
        self.refiner_agent = ArtifactsAnalysisRefiner()
    
    def _analyze_one_artifact(self, artifact: Artifacts) -> AgentResult:
        analyzer_agent = self._get_analyzer_agent(artifact)
        if analyzer_agent:
            focused_context = self._create_focused_context(artifact)
            result = analyzer_agent(focused_context)
            return result

    def run(self, context: AgentContext) -> AgentResult:        
        cfg_refiner = {}
        with ThreadPoolExecutor() as executor:
            for result in executor.map(self._analyze_one_artifact, context.artifacts):
                context.agent_results[result.agent_name] = result
                cfg_refiner[result.agent_name] = result.analysis

        refined_analysis = self.refiner_agent(previous_analyses=cfg_refiner)

        return AgentResult(
            agent_name="artifact_analysis",
            analysis=refined_analysis.refined_analysis
        )
    
    def _get_analyzer_agent(self, artifact: Artifacts) -> ArtifactAnalyzer:
        for analyzer_agent in self.analyzer_agents.values():
            if analyzer_agent.supports_artifact(artifact):
                return analyzer_agent
        raise ValueError(f"No analyzer agent found for artifact of type: {type(artifact)}")
    
    def _create_focused_context(self, artifact: Artifacts) -> AgentContext:
        return AgentContext(
            artifacts=[artifact],
        )

    def _initialize_analyzer_agents(self) -> Dict[str, "ArtifactAnalyzer"]:
        """Initialize specialized analyzer agents."""
        agents = {}

        if "training" in self.config.enabled_analyzers:
            agents["training"] = TrainingArtifactsAnalyzer()

        if "deepchecks" in self.config.enabled_analyzers:
            agents["deepchecks"] = DeepchecksArtifactsAnalyzer()

        if "dataset" in self.config.enabled_analyzers:
            agents["dataset"] = DatasetArtifactsAnalyzer()

        if "model_checkpoint" in self.config.enabled_analyzers:
            agents["model_checkpoint"] = ModelCheckpointArtifactsAnalyzer()

        return agents
