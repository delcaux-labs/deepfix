"""
Utility methods for TrainingDynamicsAgent
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from .models import Severity


def identify_primary_metrics(metrics_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Identify primary metrics (loss, accuracy) for analysis"""
    primary_metrics = {}
    
    # Look for loss metrics
    loss_cols = [col for col in metrics_df.columns if 'loss' in col.lower()]
    for col in loss_cols[:2]:  # Limit to first 2 loss metrics
        primary_metrics[col] = metrics_df[col]
    
    # Look for accuracy metrics
    acc_cols = [col for col in metrics_df.columns if 'acc' in col.lower()]
    for col in acc_cols[:2]:  # Limit to first 2 accuracy metrics
        primary_metrics[col] = metrics_df[col]
    
    return primary_metrics


def calculate_improvement_rate(metric_series: pd.Series) -> float:
    """Calculate improvement rate for a metric series"""
    if len(metric_series) < 2:
        return 0.0
    
    # For loss metrics, improvement is decrease; for accuracy, it's increase
    is_loss_metric = 'loss' in metric_series.name.lower() if hasattr(metric_series, 'name') else False
    
    start_val = metric_series.iloc[0]
    end_val = metric_series.iloc[-1]
    
    if start_val == 0:
        return 0.0
    
    if is_loss_metric:
        # For loss, improvement is reduction
        improvement = (start_val - end_val) / abs(start_val)
    else:
        # For accuracy, improvement is increase
        improvement = (end_val - start_val) / abs(start_val)
    
    return improvement


def detect_performance_plateaus(metric_series: pd.Series) -> Dict[str, Any]:
    """Detect plateau periods in metric series"""
    if len(metric_series) < 5:
        return {"total_plateau_epochs": 0, "plateau_periods": []}
    
    # Calculate rolling standard deviation to detect plateaus
    window = min(5, len(metric_series) // 3)
    rolling_std = metric_series.rolling(window=window).std()
    
    # Plateau threshold (very low standard deviation)
    plateau_threshold = metric_series.std() * 0.1
    
    plateau_mask = rolling_std < plateau_threshold
    plateau_epochs = plateau_mask.sum()
    
    return {
        "total_plateau_epochs": plateau_epochs,
        "plateau_periods": []  # Could be enhanced to identify specific periods
    }


def assess_trend_quality(metric_series: pd.Series, improvement_rate: float) -> Dict[str, Any]:
    """Assess overall trend quality"""
    concerns = []
    score = 0.8  # Start with good score
    
    # Check improvement rate
    if abs(improvement_rate) < 0.01:  # Very little improvement
        concerns.append("minimal_improvement")
        score -= 0.3
    
    # Check for high volatility
    cv = metric_series.std() / abs(metric_series.mean()) if metric_series.mean() != 0 else float('inf')
    if cv > 0.2:  # High coefficient of variation
        concerns.append("high_volatility")
        score -= 0.2
    
    # Check for trend reversal
    if len(metric_series) > 10:
        first_half_mean = metric_series.iloc[:len(metric_series)//2].mean()
        second_half_mean = metric_series.iloc[len(metric_series)//2:].mean()
        
        is_loss = 'loss' in str(metric_series.name).lower() if hasattr(metric_series, 'name') else False
        
        if is_loss and second_half_mean > first_half_mean * 1.1:
            concerns.append("trend_reversal")
            score -= 0.3
        elif not is_loss and second_half_mean < first_half_mean * 0.9:
            concerns.append("trend_reversal")
            score -= 0.3
    
    return {
        "score": max(0.0, score),
        "concerns": concerns
    }


def assess_trend_severity(trend_quality: Dict[str, Any]) -> Severity:
    """Assess severity based on trend quality"""
    score = trend_quality["score"]
    concerns = trend_quality["concerns"]
    
    if score < 0.3 or "trend_reversal" in concerns:
        return Severity.HIGH
    elif score < 0.6 or len(concerns) > 1:
        return Severity.MEDIUM
    else:
        return Severity.LOW


def identify_metric_pairs(metrics_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Identify train-validation metric pairs"""
    pairs = []
    
    # Common patterns for train-val pairs
    train_prefixes = ['train_', 'training_', '']
    val_prefixes = ['val_', 'validation_', 'valid_']
    
    for col in metrics_df.columns:
        col_lower = col.lower()
        
        # Skip if already identified as validation
        if any(prefix in col_lower for prefix in val_prefixes):
            continue
        
        # Look for corresponding validation metric
        base_name = col_lower
        for train_prefix in train_prefixes:
            if col_lower.startswith(train_prefix):
                base_name = col_lower[len(train_prefix):]
                break
        
        # Find validation counterpart
        for val_prefix in val_prefixes:
            val_name = val_prefix + base_name
            val_cols = [c for c in metrics_df.columns if c.lower() == val_name]
            if val_cols:
                pairs.append((col, val_cols[0]))
                break
    
    return pairs


def calculate_performance_gap(train_series: pd.Series, val_series: pd.Series) -> Dict[str, Any]:
    """Calculate performance gap between train and validation metrics"""
    if len(train_series) != len(val_series):
        min_len = min(len(train_series), len(val_series))
        train_series = train_series.iloc[:min_len]
        val_series = val_series.iloc[:min_len]
    
    # Calculate relative gap
    gaps = abs(train_series - val_series) / (abs(train_series) + 1e-8)
    max_gap = gaps.max()
    
    # Find divergence start (where gap starts increasing consistently)
    divergence_start = None
    if len(gaps) > 5:
        gap_trend = gaps.rolling(window=3).mean().diff()
        increasing_trend = gap_trend > 0.01
        if increasing_trend.any():
            divergence_start = increasing_trend.idxmax()
    
    # Calculate trend correlation
    correlation = train_series.corr(val_series) if len(train_series) > 1 else 1.0
    
    return {
        "max_relative_gap": max_gap,
        "divergence_start_epoch": divergence_start,
        "trend_correlation": correlation,
        "final_gap": gaps.iloc[-1] if len(gaps) > 0 else 0.0
    }


def assess_overfitting_severity(gap_analysis: Dict[str, Any]) -> Severity:
    """Assess overfitting severity based on gap analysis"""
    max_gap = gap_analysis["max_relative_gap"]
    correlation = gap_analysis.get("trend_correlation", 1.0)
    
    if max_gap > 0.3 or correlation < 0.3:
        return Severity.HIGH
    elif max_gap > 0.15 or correlation < 0.6:
        return Severity.MEDIUM
    else:
        return Severity.LOW


def count_plateau_epochs(metric_series: pd.Series) -> int:
    """Count epochs where metric shows no significant improvement"""
    if len(metric_series) < 3:
        return 0
    
    # Calculate rolling improvement
    window = min(3, len(metric_series) // 2)
    rolling_improvement = metric_series.rolling(window=window).apply(
        lambda x: abs(x.iloc[-1] - x.iloc[0]) / (abs(x.iloc[0]) + 1e-8)
    )
    
    # Count epochs with minimal improvement
    plateau_threshold = 0.01  # 1% improvement threshold
    plateau_epochs = (rolling_improvement < plateau_threshold).sum()
    
    return plateau_epochs


def calculate_rolling_cv(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling coefficient of variation"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Avoid division by zero
    rolling_cv = rolling_std / (abs(rolling_mean) + 1e-8)
    
    return rolling_cv


def detect_oscillations(series: pd.Series) -> float:
    """Detect oscillatory behavior in series"""
    if len(series) < 6:
        return 0.0
    
    # Count direction changes
    diff_series = series.diff()
    sign_changes = (diff_series.shift(1) * diff_series < 0).sum()
    
    # Normalize by series length
    oscillation_score = sign_changes / (len(series) - 1)
    
    return oscillation_score


def detect_premature_convergence(series: pd.Series) -> bool:
    """Detect if series converged too early"""
    if len(series) < 10:
        return False
    
    # Check if first 30% of training shows most improvement
    split_point = len(series) // 3
    early_improvement = abs(series.iloc[split_point] - series.iloc[0])
    late_improvement = abs(series.iloc[-1] - series.iloc[split_point])
    
    # If early improvement is much larger, might be premature convergence
    if early_improvement > 0 and late_improvement / early_improvement < 0.1:
        return True
    
    return False


def extract_gradient_metrics(metrics_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Extract gradient-related metrics from dataframe"""
    gradient_metrics = {}
    
    # Look for common gradient metric names
    gradient_patterns = ['grad_norm', 'gradient_norm', 'grad_clip', 'gradient_clip']
    
    for col in metrics_df.columns:
        col_lower = col.lower()
        for pattern in gradient_patterns:
            if pattern in col_lower:
                gradient_metrics[col] = metrics_df[col]
                break
    
    return gradient_metrics


def detect_loss_stagnation(loss_series: pd.Series) -> bool:
    """Detect if loss has stagnated (potential vanishing gradients)"""
    if len(loss_series) < 10:
        return False
    
    # Check if loss improvement in last 50% is minimal
    split_point = len(loss_series) // 2
    recent_improvement = abs(loss_series.iloc[-1] - loss_series.iloc[split_point])
    initial_loss = abs(loss_series.iloc[0])
    
    # If recent improvement is less than 1% of initial loss
    if initial_loss > 0 and recent_improvement / initial_loss < 0.01:
        return True
    
    return False


def has_convergence_issues(metrics_df: pd.DataFrame) -> bool:
    """Check if training has convergence issues"""
    loss_cols = [col for col in metrics_df.columns if 'loss' in col.lower()]
    
    for loss_col in loss_cols:
        if len(metrics_df[loss_col]) > 5:
            # Check if loss is not decreasing
            loss_series = metrics_df[loss_col]
            if loss_series.iloc[-1] >= loss_series.iloc[0] * 0.95:  # Less than 5% improvement
                return True
    
    return False


def has_stability_issues(metrics_df: pd.DataFrame) -> bool:
    """Check if training has stability issues"""
    for col in metrics_df.select_dtypes(include=[np.number]).columns:
        if len(metrics_df[col]) > 5:
            cv = metrics_df[col].std() / (abs(metrics_df[col].mean()) + 1e-8)
            if cv > 0.2:  # High coefficient of variation
                return True
    
    return False
