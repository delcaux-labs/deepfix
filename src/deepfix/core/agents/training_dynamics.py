import dspy
import pandas as pd
from typing import Tuple, Type, List, Dict, Any, Optional
import logging
from ..artifacts import TrainingArtifacts
from .signatures import TrainingArtifactsAnalysisSignature
from .base import ArtifactAnalyzer, AgentContext, AgentResult
from .models import TrainingDynamicsConfig, Finding, Recommendation, Severity
from . import training_dynamics_utils as utils


class TrainingDynamicsAgent(ArtifactAnalyzer):
    def __init__(self, config: TrainingDynamicsConfig,):
        llm = dspy.ChainOfThought(TrainingArtifactsAnalysisSignature)
        super().__init__(llm=llm)
        self.config = config
        self.agent_name = "training_dynamics"
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis cache for performance optimization
        self._analysis_cache = {} if config.lightweight_mode else None
    
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
    def supported_artifact_types(self) -> Tuple[Type[TrainingArtifacts]]:
        return (TrainingArtifacts,)
    
    def _run(self, context: AgentContext) -> AgentResult:
        """Main analysis method following the specification"""
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
            primary_metrics = utils.identify_primary_metrics(metrics_df)
            
            for metric_name, metric_series in primary_metrics.items():
                # Calculate improvement rate
                improvement_rate = utils.calculate_improvement_rate(metric_series)
                
                # Detect plateau phases
                plateau_info = utils.detect_performance_plateaus(metric_series)
                
                # Assess overall trend quality
                trend_quality = utils.assess_trend_quality(metric_series, improvement_rate)
                
                if trend_quality["concerns"]:
                    findings.append(Finding(
                        description=f"Training trend concerns detected in {metric_name}",
                        evidence=f"Improvement rate: {improvement_rate:.4f}, Plateau epochs: {plateau_info['total_plateau_epochs']}, Concerns: {', '.join(trend_quality['concerns'])}",
                        severity=utils.assess_trend_severity(trend_quality),
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
        metric_pairs = utils.identify_metric_pairs(metrics_df)
        
        for train_col, val_col in metric_pairs:
            if train_col in metrics_df.columns and val_col in metrics_df.columns:
                # Calculate performance gap over time
                gap_analysis = utils.calculate_performance_gap(
                    metrics_df[train_col], 
                    metrics_df[val_col]
                )
                
                # Detect concerning patterns
                if gap_analysis["max_relative_gap"] > self.config.overfitting_thresholds["train_val_divergence"]:
                    findings.append(Finding(
                        description=f"Significant train-validation gap detected in {train_col}/{val_col}",
                        evidence=f"Max relative gap: {gap_analysis['max_relative_gap']:.3f}, Divergence epoch: {gap_analysis.get('divergence_start_epoch', 'N/A')}, Trend correlation: {gap_analysis.get('trend_correlation', 'N/A'):.3f}",
                        severity=utils.assess_overfitting_severity(gap_analysis),
                        confidence=min(0.9, gap_analysis["max_relative_gap"] * 5)
                    ))
        
        return findings

    def _analyze_trend_divergence(self, metrics_df: pd.DataFrame) -> List[Finding]:
        """Detect trend divergence between train and validation metrics"""
        findings = []
        metric_pairs = utils.identify_metric_pairs(metrics_df)
        
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
                plateau_epochs = utils.count_plateau_epochs(metrics_df[val_col])
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
                rolling_cv = utils.calculate_rolling_cv(metrics_df[loss_col], window_size)
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
            gradient_metrics = utils.extract_gradient_metrics(metrics_df)
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
            if learning_rate and utils.has_convergence_issues(metrics_df):
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
        
    def _format_analysis_output(self, findings: List[Finding], recommendations: List[Recommendation]) -> str:
        output_lines = []
        if findings:
            output_lines.append("## Training Dynamics Analysis Findings:")
            for i, finding in enumerate(findings, 1):
                output_lines.append(f"{i}. **{finding.description}** (Severity: {finding.severity.value}, Confidence: {finding.confidence:.2f})")
                output_lines.append(f"   Evidence: {finding.evidence}")
        if recommendations:
            output_lines.append("## Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                output_lines.append(f"{i}. **{rec.action}** (Priority: {rec.priority.value})")
        if not findings and not recommendations:
            output_lines.append("No significant training dynamics issues detected.")
        return "\n".join(output_lines)