from typing import List, Dict, Any, Optional
import pandas as pd
import dspy
from .base import ArtifactAnalyzer
from ..models import AgentContext, AgentResult
from ..config import LLMConfig
from .training_dynamics_utils import TrainingDynamicsAnalyzer
from ...shared.models import DeepchecksArtifacts, DatasetArtifacts, ModelCheckpointArtifacts, TrainingArtifacts,Finding, Severity
from ..logging import get_logger

LOGGER = get_logger(__name__)

class DeepchecksArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in data quality and validation, data drift detection, data integrity assessment, and outlier identification"""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        llm: Optional[dspy.Module] = None,
    ):
        super().__init__(llm=llm, config=config)

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
    def supported_artifact_types(self):
        return DeepchecksArtifacts

class DatasetArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in dataset analysis and quality assessment, data distribution analysis, feature quality assessment, and class balance evaluation"""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        llm: Optional[dspy.Module] = None,
    ):
        super().__init__(llm=llm, config=config)

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
    def supported_artifact_types(self):
        return DatasetArtifacts
    
class ModelCheckpointArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in model checkpoint integrity and validation, model configuration analysis, and deployment readiness assessment"""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        llm: Optional[dspy.Module] = None,
    ):
        super().__init__(llm=llm, config=config)

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

    def load_model_summary(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def supported_artifact_types(self):
        return ModelCheckpointArtifacts

class TrainingArtifactsAnalyzer(ArtifactAnalyzer):
    def __init__(self, llm_config: Optional[LLMConfig]=None,llm: Optional[dspy.Module] = None):
        super().__init__(llm=llm,config=llm_config)
        
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
                    findings.extend(self._analyze_parameter_impact(
                        training_artifacts.metrics_values, training_artifacts.params
                    ))
        
        # Generate recommendations based on findings
            
            return AgentResult(
                agent_name=self.agent_name,
                analysis=self._format_analysis_output(findings, recommendations),
                analyzed_artifacts=["TrainingArtifacts"]
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
            analyzed_artifacts=[]
        )
    
    def _error_result(self, error_msg: str) -> AgentResult:
        """Return result when analysis fails"""
        return AgentResult(
            agent_name=self.agent_name,
            analysis=f"Training dynamics analysis failed: {error_msg}",
            analyzed_artifacts=["TrainingArtifacts"]
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
                improvement_rate = self.analyzer.calculate_improvement_rate(metric_series)
                
                # Detect plateau phases
                plateau_info = self.analyzer.detect_performance_plateaus(metric_series)
                
                # Assess overall trend quality
                trend_quality = self.analyzer.assess_trend_quality(metric_series, improvement_rate)
                
                if trend_quality["concerns"]:
                    findings.append(Finding(
                        description=f"Training trend concerns detected in {metric_name}",
                        evidence=f"Improvement rate: {improvement_rate:.4f}, Plateau epochs: {plateau_info['total_plateau_epochs']}, Concerns: {', '.join(trend_quality['concerns'])}",
                        severity=self.analyzer.assess_trend_severity(trend_quality),
                        confidence=trend_quality["score"]
                    ))
        
        except (ValueError, KeyError, AttributeError) as e:
            self.logger.warning("Training curve analysis failed: %s", str(e))
            findings.append(self._analysis_failure_finding("training_curves", e))
        
        return findings
    
    def _detect_overfitting_patterns(self, artifacts: TrainingArtifacts) -> List[Finding]:
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
                    metrics_df[train_col], 
                    metrics_df[val_col]
                )
                
                # Detect concerning patterns
                if gap_analysis["max_relative_gap"] > self.config.overfitting_thresholds["train_val_divergence"]:
                    findings.append(Finding(
                        description=f"Significant train-validation gap detected in {train_col}/{val_col}",
                        evidence=f"Max relative gap: {gap_analysis['max_relative_gap']:.3f}, Divergence epoch: {gap_analysis.get('divergence_start_epoch', 'N/A')}, Trend correlation: {gap_analysis.get('trend_correlation', 'N/A'):.3f}",
                        severity=self.analyzer.assess_overfitting_severity(gap_analysis),
                        confidence=min(0.9, gap_analysis["max_relative_gap"] * 5)
                    ))
        
        return findings

    def _analyze_trend_divergence(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Detect trend divergence between train and validation metrics"""
        findings = []
        metric_pairs = self.analyzer.identify_metric_pairs(metrics_df)
        
        for train_col, val_col in metric_pairs:
            if train_col in metrics_df.columns and val_col in metrics_df.columns:
                # Use moving averages to smooth curves
                window = min(5, len(metrics_df) // 4)
                train_smooth = metrics_df[train_col].rolling(window=window, center=True).mean()
                val_smooth = metrics_df[val_col].rolling(window=window, center=True).mean()
                
                # Calculate correlation between trends
                correlation = train_smooth.corr(val_smooth)
                
                # Detect divergence points
                if correlation < 0.5:  # Low correlation indicates divergence
                    findings.append(Finding(
                        description=f"Training-validation trend divergence in {train_col}/{val_col}",
                        evidence=f"Trend correlation: {correlation:.3f} (threshold: 0.5)",
                        severity=Severity.MEDIUM if correlation > 0.2 else Severity.HIGH,
                        confidence=1.0 - correlation if correlation >= 0 else 0.9
                    ))
        
        return findings
    
    def _detect_validation_plateaus(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Detect validation metric plateaus"""
        findings = []
        val_columns = [col for col in metrics_df.columns if 'val' in col.lower()]
        
        for val_col in val_columns:
            if val_col in metrics_df.columns:
                plateau_info = self.analyzer.detect_performance_plateaus(metrics_df[val_col])
                plateau_epochs = plateau_info.get("total_plateau_epochs", 0)
                threshold = self.config.overfitting_thresholds["val_loss_plateau_epochs"]
                
                if plateau_epochs >= threshold:
                    findings.append(Finding(
                        description=f"Validation plateau detected in {val_col}",
                        evidence=f"Plateau duration: {plateau_epochs} epochs (threshold: {threshold})",
                        severity=Severity.MEDIUM if plateau_epochs < threshold * 2 else Severity.HIGH,
                        confidence=min(0.9, plateau_epochs / (threshold * 2))
                    ))
        
        return findings
    
    def _analyze_early_stopping_signals(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Analyze early stopping signals"""
        findings = []
        val_loss_cols = [col for col in metrics_df.columns if 'val' in col.lower() and 'loss' in col.lower()]
        
        for val_loss_col in val_loss_cols:
            if val_loss_col in metrics_df.columns:
                # Find best epoch and check if training continued significantly after
                best_epoch = metrics_df[val_loss_col].idxmin()
                total_epochs = len(metrics_df) - 1
                epochs_after_best = total_epochs - best_epoch
                
                patience = self.config.overfitting_thresholds["early_stopping_patience"]
                
                if epochs_after_best > patience:
                    findings.append(Finding(
                        description=f"Training continued {epochs_after_best} epochs after best validation loss",
                        evidence=f"Best epoch: {best_epoch}, Total epochs: {total_epochs}, Recommended patience: {patience}",
                        severity=Severity.MEDIUM,
                        confidence=min(0.9, epochs_after_best / (patience * 2))
                    ))
        
        return findings
    
    def _analyze_training_stability(self, artifacts: TrainingArtifacts) -> List[Finding]:
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
        loss_columns = [col for col in metrics_df.columns if 'loss' in col.lower()]
        
        for loss_col in loss_columns:
            if loss_col in metrics_df.columns and len(metrics_df[loss_col]) > 10:
                window_size = min(self.config.stability_thresholds["metric_volatility_window"], len(metrics_df) // 2)
                rolling_cv = self.analyzer.calculate_rolling_cv(metrics_df[loss_col], window_size)
                threshold = self.config.stability_thresholds["loss_variance_threshold"]
                high_volatility_periods = rolling_cv > threshold
                
                if high_volatility_periods.any():
                    volatility_score = rolling_cv.max()
                    volatile_epochs = high_volatility_periods.sum()
                    
                    findings.append(Finding(
                        description=f"High training volatility detected in {loss_col}",
                        evidence=f"Max coefficient of variation: {volatility_score:.4f}, Volatile epochs: {volatile_epochs}",
                        severity=self._assess_stability_severity(volatility_score),
                        confidence=min(0.9, volatility_score / threshold)
                    ))
        
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
    
    def _detect_exploding_gradients(self, gradient_metrics: Dict[str, pd.Series]) -> List[Finding]:
        """Detect exploding gradient patterns"""
        findings = []
        
        for metric_name, gradient_norms in gradient_metrics.items():
            threshold = self.config.gradient_thresholds["exploding_gradient_threshold"]
            exploding_episodes = gradient_norms > threshold
            
            if exploding_episodes.any():
                max_norm = gradient_norms.max()
                explosion_count = exploding_episodes.sum()
                
                findings.append(Finding(
                    description=f"Exploding gradients detected in {metric_name}",
                    evidence=f"Max gradient norm: {max_norm:.2e}, Explosion episodes: {explosion_count}",
                    severity=Severity.HIGH,
                    confidence=min(0.95, max_norm / threshold)
                ))
        
        return findings
    
    def _infer_gradient_issues_from_loss(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Infer gradient issues from loss behavior"""
        findings = []
        loss_columns = [col for col in metrics_df.columns if 'loss' in col.lower()]
        
        for loss_col in loss_columns:
            if loss_col in metrics_df.columns and len(metrics_df[loss_col]) > 5:
                loss_series = metrics_df[loss_col]
                loss_changes = loss_series.diff()
                sudden_spikes = loss_changes > (loss_series.std() * 3)
                
                if sudden_spikes.any():
                    spike_count = sudden_spikes.sum()
                    max_spike = loss_changes.max()
                    
                    findings.append(Finding(
                        description=f"Potential exploding gradients inferred from {loss_col} spikes",
                        evidence=f"Loss spikes detected: {spike_count}, Max spike: {max_spike:.4f}",
                        severity=Severity.MEDIUM,
                        confidence=0.6
                    ))
        
        return findings
    
    def _analyze_parameter_impact(self, metrics_df: pd.DataFrame, params: Dict[str, Any]) -> List[Finding]:
        """Analyze correlation between parameters and training dynamics"""
        findings = []
        
        try:
            learning_rate = params.get("learning_rate", params.get("lr"))
            if learning_rate and self.analyzer.has_convergence_issues(metrics_df):
                if learning_rate > 0.1:
                    findings.append(Finding(
                        description=f"Learning rate may be too high ({learning_rate})",
                        evidence="High learning rate combined with training instability",
                        severity=Severity.MEDIUM,
                        confidence=0.7
                    ))
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
    
    def _analysis_failure_finding(self, analysis_type: str, error: Exception) -> Finding:
        return Finding(
            description=f"Could not complete {analysis_type} analysis",
            evidence=f"Error: {type(error).__name__}: {str(error)}",
            severity=Severity.LOW,
            confidence=0.1
        )
        
