from typing import Any, Dict, List, Optional

import pandas as pd
from deepfix_core.models import (
    DatasetArtifacts,
    DeepchecksArtifacts,
    Finding,
    ModelCheckpointArtifacts,
    Severity,
    TrainingArtifacts,
)

from ..config import LLMConfig
from ..logging import get_logger
from ..models import AgentContext, AgentResult
from .base import ArtifactAnalyzer
from .training_dynamics_utils import TrainingDynamicsAnalyzer

LOGGER = get_logger(__name__)


class DeepchecksArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in data quality and validation, data drift detection, data integrity assessment, and outlier identification"""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
    ):
        super().__init__(config=config)

    @property
    def system_prompt(self) -> str:
        return """You are an expert in data quality control for machine learning with deep expertise in:
                - Data drift detection and distribution analysis
                - Data integrity assessment and outlier identification
                - Train-test validation and data leakage detection
                - Data quality patterns, anomalies, and failure modes

                You are given Deepchecks test results for a dataset and model. These may include:
                - Train–test validation checks (drift, correlations, new labels, etc.)
                - Data integrity checks (outliers, label/property issues, class performance)
                - Per-check metadata such as severity, warnings, and example rows

                Your role is to:
                1. Interpret the Deepchecks results and explain what they mean in practical terms
                2. Identify issues that could harm model usability, robustness, or fairness
                3. Call out suspicious or surprising patterns that may indicate deeper problems
                4. Provide concrete, prioritized recommendations to improve data and evaluation setup

                Focus your analysis on both **data correctness** and **downstream model usability**:

                Analysis Focus Areas:
                - **Drift & Distribution Shifts**:
                  - Are there strong shifts between train and test (features, labels, image properties)?
                  - Do the shifts align with the intended deployment population, or are they suspicious?
                - **Integrity & Label Quality**:
                  - Are there many outliers, inconsistent labels, or corrupted / low-quality samples?
                  - Any tests suggesting mislabeled data, label noise, or broken feature–label relationships?
                - **Data Leakage & Evaluation Validity**:
                  - Any evidence or hints of leakage (near-duplicate samples across splits, unrealistic performance, etc.)?
                  - Are train/test splits appropriate for the claimed use case?
                - **Bias, Representativeness & Coverage**:
                  - Do Deepchecks tests indicate strong class imbalance or underrepresented subgroups?
                  - Any patterns that could lead to unfair or brittle behavior in deployment?
                - **Model Performance & Stability**:
                  - Are there classes or regions of the input space where performance is clearly degraded?
                  - Do the checks suggest the model is overfitting to artifacts rather than signal?

                When analyzing Deepchecks results, explicitly:
                - Highlight **suspicious or high-risk** findings, not just any deviation from ideal
                - Distinguish between **hard blockers** (data/eval is clearly broken) and **soft blockers** (risks or quality issues)
                - Point out **gaps or missing checks** that limit confidence in the data and evaluation

                OUTPUT FORMAT (strictly follow this structure):
                1. Summary
                   - 2–4 bullet points summarizing overall data quality, drift, and evaluation reliability.
                2. Drift & Distribution
                   - Findings about feature/label drift and how concerning they are.
                3. Integrity & Label Quality
                   - Findings about outliers, label consistency, and corrupted samples.
                4. Leakage, Bias & Representativeness
                   - Findings related to leakage risks, bias, and coverage/imbalance.
                5. Usability & Suspicious Elements
                   - Explicitly list any suspicious, surprising, or risky elements that may hinder reliable model use.
                6. Recommendations (Prioritized)
                   - A numbered list of concrete actions (data fixes, checks to add, split changes, etc.), ordered from most to least critical.

                Be specific, base your reasoning on the provided Deepchecks results, and avoid inventing tests or metrics that are not present."""

    @property
    def supported_artifact_types(self):
        return DeepchecksArtifacts


class DatasetArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in dataset analysis and quality assessment, data distribution analysis, feature quality assessment, and class balance evaluation"""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
    ):
        super().__init__(config=config)

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist specializing in dataset analysis and quality assessment with deep expertise in:
                - Dataset statistics interpretation and quality evaluation
                - Data distribution analysis and anomaly detection
                - Feature quality assessment and correlation analysis
                - Class balance evaluation and sampling strategy recommendations

                You are given dataset artifacts that summarize a dataset intended for ML training. These may include:
                - Global statistics (row/column counts, feature types, missingness)
                - Per-feature distributions, correlations, and summary metrics
                - Class/label distributions and per-class statistics

                Your role is to:
                1. Assess whether the dataset is suitable for the intended modeling task
                2. Identify weaknesses that could harm model performance or reliability
                3. Detect suspicious or surprising patterns in the data statistics
                4. Recommend concrete actions to improve data quality and usability

                Focus your analysis on both **data quality** and **practical ML usability**:

                Analysis Focus Areas:
                - **Completeness & Integrity**:
                  - Are there many missing values, invalid entries, or obviously corrupted features?
                  - Any columns with almost no information (constant, near-constant, or extremely sparse)?
                - **Distribution & Outliers**:
                  - Are distributions heavy-tailed, extremely skewed, or multi-modal in a concerning way?
                  - Are there outliers that are likely errors vs. genuine rare but important cases?
                - **Class Balance & Coverage**:
                  - Is the label distribution heavily imbalanced or missing important classes?
                  - Are there classes or regions of feature space with too few samples for reliable learning?
                - **Feature Relationships & Leakage Risks**:
                  - Do correlations suggest potential leakage (e.g., features that are almost copies of the label)?
                  - Any suspiciously perfect or near-perfect relationships that could make evaluation misleading?
                - **Task Appropriateness & Metadata**:
                  - Does the dataset structure (features/labels/types) match the claimed task (classification, regression, etc.)?
                  - Is critical metadata (label definitions, units, time ranges) missing or ambiguous?

                When analyzing dataset statistics, explicitly:
                - Call out **suspicious, surprising, or high-risk** patterns (e.g., impossible values, implausible distributions)
                - Distinguish between **hard blockers** (dataset unusable without fixes) and **soft blockers** (risk or quality issues)
                - Highlight **gaps or missing information** that prevent a confident assessment

                OUTPUT FORMAT (strictly follow this structure):
                1. Summary
                   - 2–4 bullet points summarizing overall dataset quality and suitability.
                2. Completeness & Integrity
                   - Findings about missingness, invalid values, and low-information features.
                3. Distribution, Outliers & Balance
                   - Findings about distributions, outliers, and class/label balance.
                4. Feature Relationships & Leakage
                   - Findings about correlations, redundancy, and potential leakage risks.
                5. Usability & Suspicious Elements
                   - Explicitly list any suspicious, surprising, or risky elements that may hinder training or evaluation.
                6. Recommendations (Prioritized)
                   - A numbered list of concrete actions (cleaning, resampling, feature changes, data collection), ordered from most to least critical.

                Be specific, tie your reasoning to the provided statistics, and avoid inventing features or labels that are not present."""

    @property
    def supported_artifact_types(self):
        return DatasetArtifacts


class ModelCheckpointArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in model checkpoint integrity and validation, model configuration analysis, and deployment readiness assessment"""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
    ):
        super().__init__(config=config)

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML model checkpoint and Model Card specialist with deep expertise in:
                - Model checkpoint integrity and validation
                - Model configuration and architecture analysis
                - Training configuration and hyperparameter sanity checking

                You are given model checkpoint artifacts that may include:
                - One or more checkpoint/state files (e.g. *.bin, *.safetensors, *.ckpt, *.pt)
                - Model configuration (e.g. config.json, model card, training args)
                - Training / evaluation metadata (metrics, dataset descriptions, tags)

                Your role is to:
                1. Assess checkpoint and config integrity and internal consistency
                2. Identify issues that could hinder real‑world usability
                3. Detect suspicious, surprising, or risky patterns in the artifacts
                4. Provide concrete, prioritized recommendations to improve usability and safety

                Focus your analysis on both **correctness** and **usability**:

                Analysis Focus Areas:
                - **File & Format Integrity**:
                  - Are all referenced checkpoint files present and readable?
                  - Do file sizes and counts look reasonable for the claimed model size?
                  - Any signs of partial, mixed, or incompatible checkpoints?
                - **Configuration & Architecture Validation**:
                  - Do architecture parameters (layers, hidden size, heads, vocab_size, num_labels, etc.) form a coherent model?
                  - Are there mismatches between config and checkpoint (e.g. different vocab_size, missing heads, changed num_labels)?
                  - Are required keys or sections missing or set to obviously wrong defaults?
                - **Training Configuration & Metadata**:
                  - Do training hyperparameters, objective, and head type align with the model’s intended task?
                  - Are metrics and dataset descriptions consistent with the architecture and head (e.g. classification vs regression)?
                  - Any signs the checkpoint is partially trained, mis‑labeled, or repurposed for a different task?
                - **Compatibility & Deployment Readiness**:
                  - Are there strong version or hardware assumptions (framework versions, device type, precision, quantization)?
                  - Any known‑problem settings (extreme learning rates, absurd batch sizes, invalid dropout, etc.) that suggest misconfiguration?
                  - Are there clear instructions or metadata for loading and running the model, or is critical information missing?
                - **Usability & Safety Concerns**:
                  - Anything that would make this checkpoint hard to use “out of the box” (missing tokenizer, unclear task, ambiguous labels, etc.)?
                  - Any suspicious or misleading metadata (e.g. unrealistic metrics, inconsistent task descriptions, contradictory tags)?
                  - Any hints of data leakage, evaluation contamination, or unsafe usage claims in the metadata/model card?

                When analyzing model checkpoints, explicitly:
                - Call out **suspicious, surprising, or high‑risk** elements, even if they might still work technically
                - Highlight **gaps or ambiguities** that require user decisions (e.g. unknown label mapping, missing preprocessing steps)
                - Distinguish between **hard blockers** (will likely break loading/inference) and **soft blockers** (degrade quality or reliability)

                Be specific, avoid guessing names of files or values that are not present in the artifacts, and prefer concrete, actionable guidance over generic advice."""

    def load_model_summary(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def supported_artifact_types(self):
        return ModelCheckpointArtifacts


class TrainingArtifactsAnalyzer(ArtifactAnalyzer):
    def __init__(
        self, llm_config: Optional[LLMConfig] = None
    ):
        super().__init__(config=llm_config)

        self.logger = LOGGER
        self._analysis_cache = {}
        self.analyzer = TrainingDynamicsAnalyzer()

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
    def supported_artifact_types(self):
        return TrainingArtifacts

    def _run(self, context: AgentContext) -> AgentResult:
        """Main analysis method following the specification"""

        raise NotImplementedError("TrainingArtifactsAnalyzer is not implemented yet")

        # Find training artifacts
        training_artifacts = self._get_training_artifacts(context.artifacts)
        if not training_artifacts:
            return self._no_training_data_result()

        findings = []
        recommendations = []

        try:
            # Analyze training metrics if available
            if training_artifacts.metrics_values is not None:
                findings.extend(self._analyze_training_curves(training_artifacts))
                findings.extend(self._detect_overfitting_patterns(training_artifacts))
                findings.extend(self._analyze_training_stability(training_artifacts))
                findings.extend(self._detect_gradient_anomalies(training_artifacts))

                # Add parameter-metric correlation analysis if params available
                if training_artifacts.params:
                    findings.extend(
                        self._analyze_parameter_impact(
                            training_artifacts.metrics_values, training_artifacts.params
                        )
                    )

            # Generate recommendations based on findings

            return AgentResult(
                agent_name=self.agent_name,
                analysis=self._format_analysis_output(findings, recommendations),
                analyzed_artifacts=["TrainingArtifacts"],
            )

        except (ValueError, KeyError, AttributeError) as e:
            self.logger.error("Training dynamics analysis failed: %s", str(e))
            return self._error_result(str(e))

    def _get_training_artifacts(self, artifacts: List) -> Optional[TrainingArtifacts]:
        """Extract training artifacts from context"""
        for artifact in artifacts:
            if isinstance(artifact, TrainingArtifacts):
                return artifact
        return None

    def _no_training_data_result(self) -> AgentResult:
        """Return result when no training data is available"""
        return AgentResult(
            agent_name=self.agent_name,
            analysis="No training artifacts available for analysis",
            analyzed_artifacts=[],
        )

    def _error_result(self, error_msg: str) -> AgentResult:
        """Return result when analysis fails"""
        return AgentResult(
            agent_name=self.agent_name,
            analysis=f"Training dynamics analysis failed: {error_msg}",
            analyzed_artifacts=["TrainingArtifacts"],
        )

    # ===== CORE ANALYSIS COMPONENTS =====

    def _analyze_training_curves(self, artifacts: TrainingArtifacts) -> List[Finding]:
        """Analyze overall training curve characteristics"""
        findings = []
        metrics_df = artifacts.metrics_values

        if metrics_df is None or metrics_df.empty:
            return findings

        try:
            # Focus on primary metrics (loss, accuracy)
            primary_metrics = self.analyzer.identify_primary_metrics(metrics_df)

            for metric_name, metric_series in primary_metrics.items():
                # Calculate improvement rate
                improvement_rate = self.analyzer.calculate_improvement_rate(
                    metric_series
                )

                # Detect plateau phases
                plateau_info = self.analyzer.detect_performance_plateaus(metric_series)

                # Assess overall trend quality
                trend_quality = self.analyzer.assess_trend_quality(
                    metric_series, improvement_rate
                )

                if trend_quality["concerns"]:
                    findings.append(
                        Finding(
                            description=f"Training trend concerns detected in {metric_name}",
                            evidence=f"Improvement rate: {improvement_rate:.4f}, Plateau epochs: {plateau_info['total_plateau_epochs']}, Concerns: {', '.join(trend_quality['concerns'])}",
                            severity=self.analyzer.assess_trend_severity(trend_quality),
                            confidence=trend_quality["score"],
                        )
                    )

        except (ValueError, KeyError, AttributeError) as e:
            self.logger.warning("Training curve analysis failed: %s", str(e))
            findings.append(self._analysis_failure_finding("training_curves", e))

        return findings

    def _detect_overfitting_patterns(
        self, artifacts: TrainingArtifacts
    ) -> List[Finding]:
        """Detect overfitting through multiple analytical approaches"""
        findings = []
        metrics_df = artifacts.metrics_values

        if metrics_df is None or metrics_df.empty:
            return findings

        try:
            # 1. Performance Gap Analysis
            findings.extend(self._analyze_performance_gap(metrics_df))

            # 2. Trend Divergence Detection
            findings.extend(self._analyze_trend_divergence(metrics_df))

            # 3. Plateau Detection
            findings.extend(self._detect_validation_plateaus(metrics_df))

            # 4. Early Stopping Analysis
            findings.extend(self._analyze_early_stopping_signals(metrics_df))

        except (ValueError, KeyError, AttributeError) as e:
            self.logger.warning("Overfitting detection failed: %s", str(e))
            findings.append(self._analysis_failure_finding("overfitting_detection", e))

        return findings

    def _analyze_performance_gap(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Analyze train-validation performance gaps"""
        findings = []

        # Extract train/validation metric pairs
        metric_pairs = self.analyzer.identify_metric_pairs(metrics_df)

        for train_col, val_col in metric_pairs:
            if train_col in metrics_df.columns and val_col in metrics_df.columns:
                # Calculate performance gap over time
                gap_analysis = self.analyzer.calculate_performance_gap(
                    metrics_df[train_col], metrics_df[val_col]
                )

                # Detect concerning patterns
                if (
                    gap_analysis["max_relative_gap"]
                    > self.config.overfitting_thresholds["train_val_divergence"]
                ):
                    findings.append(
                        Finding(
                            description=f"Significant train-validation gap detected in {train_col}/{val_col}",
                            evidence=f"Max relative gap: {gap_analysis['max_relative_gap']:.3f}, Divergence epoch: {gap_analysis.get('divergence_start_epoch', 'N/A')}, Trend correlation: {gap_analysis.get('trend_correlation', 'N/A'):.3f}",
                            severity=self.analyzer.assess_overfitting_severity(
                                gap_analysis
                            ),
                            confidence=min(0.9, gap_analysis["max_relative_gap"] * 5),
                        )
                    )

        return findings

    def _analyze_trend_divergence(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Detect trend divergence between train and validation metrics"""
        findings = []
        metric_pairs = self.analyzer.identify_metric_pairs(metrics_df)

        for train_col, val_col in metric_pairs:
            if train_col in metrics_df.columns and val_col in metrics_df.columns:
                # Use moving averages to smooth curves
                window = min(5, len(metrics_df) // 4)
                train_smooth = (
                    metrics_df[train_col].rolling(window=window, center=True).mean()
                )
                val_smooth = (
                    metrics_df[val_col].rolling(window=window, center=True).mean()
                )

                # Calculate correlation between trends
                correlation = train_smooth.corr(val_smooth)

                # Detect divergence points
                if correlation < 0.5:  # Low correlation indicates divergence
                    findings.append(
                        Finding(
                            description=f"Training-validation trend divergence in {train_col}/{val_col}",
                            evidence=f"Trend correlation: {correlation:.3f} (threshold: 0.5)",
                            severity=Severity.MEDIUM
                            if correlation > 0.2
                            else Severity.HIGH,
                            confidence=1.0 - correlation if correlation >= 0 else 0.9,
                        )
                    )

        return findings

    def _detect_validation_plateaus(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Detect validation metric plateaus"""
        findings = []
        val_columns = [col for col in metrics_df.columns if "val" in col.lower()]

        for val_col in val_columns:
            if val_col in metrics_df.columns:
                plateau_info = self.analyzer.detect_performance_plateaus(
                    metrics_df[val_col]
                )
                plateau_epochs = plateau_info.get("total_plateau_epochs", 0)
                threshold = self.config.overfitting_thresholds[
                    "val_loss_plateau_epochs"
                ]

                if plateau_epochs >= threshold:
                    findings.append(
                        Finding(
                            description=f"Validation plateau detected in {val_col}",
                            evidence=f"Plateau duration: {plateau_epochs} epochs (threshold: {threshold})",
                            severity=Severity.MEDIUM
                            if plateau_epochs < threshold * 2
                            else Severity.HIGH,
                            confidence=min(0.9, plateau_epochs / (threshold * 2)),
                        )
                    )

        return findings

    def _analyze_early_stopping_signals(
        self, metrics_df: pd.DataFrame
    ) -> List[Finding]:
        """Analyze early stopping signals"""
        findings = []
        val_loss_cols = [
            col
            for col in metrics_df.columns
            if "val" in col.lower() and "loss" in col.lower()
        ]

        for val_loss_col in val_loss_cols:
            if val_loss_col in metrics_df.columns:
                # Find best epoch and check if training continued significantly after
                best_epoch = metrics_df[val_loss_col].idxmin()
                total_epochs = len(metrics_df) - 1
                epochs_after_best = total_epochs - best_epoch

                patience = self.config.overfitting_thresholds["early_stopping_patience"]

                if epochs_after_best > patience:
                    findings.append(
                        Finding(
                            description=f"Training continued {epochs_after_best} epochs after best validation loss",
                            evidence=f"Best epoch: {best_epoch}, Total epochs: {total_epochs}, Recommended patience: {patience}",
                            severity=Severity.MEDIUM,
                            confidence=min(0.9, epochs_after_best / (patience * 2)),
                        )
                    )

        return findings

    def _analyze_training_stability(
        self, artifacts: TrainingArtifacts
    ) -> List[Finding]:
        """Analyze training stability through multiple metrics"""
        findings = []
        metrics_df = artifacts.metrics_values

        if metrics_df is None or metrics_df.empty:
            return findings

        try:
            findings.extend(self._analyze_loss_variance(metrics_df))
        except (ValueError, KeyError, AttributeError) as e:
            self.logger.warning("Training stability analysis failed: %s", str(e))
            findings.append(self._analysis_failure_finding("training_stability", e))

        return findings

    def _analyze_loss_variance(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Analyze loss variance for stability assessment"""
        findings = []
        loss_columns = [col for col in metrics_df.columns if "loss" in col.lower()]

        for loss_col in loss_columns:
            if loss_col in metrics_df.columns and len(metrics_df[loss_col]) > 10:
                window_size = min(
                    self.config.stability_thresholds["metric_volatility_window"],
                    len(metrics_df) // 2,
                )
                rolling_cv = self.analyzer.calculate_rolling_cv(
                    metrics_df[loss_col], window_size
                )
                threshold = self.config.stability_thresholds["loss_variance_threshold"]
                high_volatility_periods = rolling_cv > threshold

                if high_volatility_periods.any():
                    volatility_score = rolling_cv.max()
                    volatile_epochs = high_volatility_periods.sum()

                    findings.append(
                        Finding(
                            description=f"High training volatility detected in {loss_col}",
                            evidence=f"Max coefficient of variation: {volatility_score:.4f}, Volatile epochs: {volatile_epochs}",
                            severity=self._assess_stability_severity(volatility_score),
                            confidence=min(0.9, volatility_score / threshold),
                        )
                    )

        return findings

    def _detect_gradient_anomalies(self, artifacts: TrainingArtifacts) -> List[Finding]:
        """Detect gradient anomalies from training metrics"""
        findings = []
        metrics_df = artifacts.metrics_values

        if metrics_df is None or metrics_df.empty:
            return findings

        try:
            gradient_metrics = self.analyzer.extract_gradient_metrics(metrics_df)
            if gradient_metrics:
                findings.extend(self._detect_exploding_gradients(gradient_metrics))
            else:
                findings.extend(self._infer_gradient_issues_from_loss(metrics_df))
        except (ValueError, KeyError, AttributeError) as e:
            self.logger.warning("Gradient anomaly detection failed: %s", str(e))
            findings.append(self._analysis_failure_finding("gradient_anomalies", e))

        return findings

    def _detect_exploding_gradients(
        self, gradient_metrics: Dict[str, pd.Series]
    ) -> List[Finding]:
        """Detect exploding gradient patterns"""
        findings = []

        for metric_name, gradient_norms in gradient_metrics.items():
            threshold = self.config.gradient_thresholds["exploding_gradient_threshold"]
            exploding_episodes = gradient_norms > threshold

            if exploding_episodes.any():
                max_norm = gradient_norms.max()
                explosion_count = exploding_episodes.sum()

                findings.append(
                    Finding(
                        description=f"Exploding gradients detected in {metric_name}",
                        evidence=f"Max gradient norm: {max_norm:.2e}, Explosion episodes: {explosion_count}",
                        severity=Severity.HIGH,
                        confidence=min(0.95, max_norm / threshold),
                    )
                )

        return findings

    def _infer_gradient_issues_from_loss(
        self, metrics_df: pd.DataFrame
    ) -> List[Finding]:
        """Infer gradient issues from loss behavior"""
        findings = []
        loss_columns = [col for col in metrics_df.columns if "loss" in col.lower()]

        for loss_col in loss_columns:
            if loss_col in metrics_df.columns and len(metrics_df[loss_col]) > 5:
                loss_series = metrics_df[loss_col]
                loss_changes = loss_series.diff()
                sudden_spikes = loss_changes > (loss_series.std() * 3)

                if sudden_spikes.any():
                    spike_count = sudden_spikes.sum()
                    max_spike = loss_changes.max()

                    findings.append(
                        Finding(
                            description=f"Potential exploding gradients inferred from {loss_col} spikes",
                            evidence=f"Loss spikes detected: {spike_count}, Max spike: {max_spike:.4f}",
                            severity=Severity.MEDIUM,
                            confidence=0.6,
                        )
                    )

        return findings

    def _analyze_parameter_impact(
        self, metrics_df: pd.DataFrame, params: Dict[str, Any]
    ) -> List[Finding]:
        """Analyze correlation between parameters and training dynamics"""
        findings = []

        try:
            learning_rate = params.get("learning_rate", params.get("lr"))
            if learning_rate and self.analyzer.has_convergence_issues(metrics_df):
                if learning_rate > 0.1:
                    findings.append(
                        Finding(
                            description=f"Learning rate may be too high ({learning_rate})",
                            evidence="High learning rate combined with training instability",
                            severity=Severity.MEDIUM,
                            confidence=0.7,
                        )
                    )
        except (ValueError, KeyError, AttributeError) as e:
            self.logger.warning("Parameter impact analysis failed: %s", str(e))

        return findings

    # ===== UTILITY METHODS =====

    def _assess_stability_severity(self, volatility_score: float) -> Severity:
        threshold = self.config.stability_thresholds["loss_variance_threshold"]
        if volatility_score > threshold * 4:
            return Severity.HIGH
        elif volatility_score > threshold * 2:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _analysis_failure_finding(
        self, analysis_type: str, error: Exception
    ) -> Finding:
        return Finding(
            description=f"Could not complete {analysis_type} analysis",
            evidence=f"Error: {type(error).__name__}: {str(error)}",
            severity=Severity.LOW,
            confidence=0.1,
        )
