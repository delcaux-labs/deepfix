"""
Training dynamics analysis utilities centered around `TrainingDynamicsAnalyzer`.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from ..models import Severity


class TrainingDynamicsAnalyzer:
    """Analyzer encapsulating training dynamics computations with robust heuristics."""

    def __init__(
        self,
        *,
        min_points: int = 5,
        plateau_std_ratio: float = 0.1,
        min_improvement_rate: float = 0.01,
        high_cv_threshold: float = 0.2,
        overfit_gap_high: float = 0.30,
        overfit_gap_med: float = 0.15,
        corr_low_high: float = 0.30,
        corr_low_med: float = 0.60,
        oscillations_min_len: int = 6,
    ) -> None:
        self.min_points = min_points
        self.plateau_std_ratio = plateau_std_ratio
        self.min_improvement_rate = min_improvement_rate
        self.high_cv_threshold = high_cv_threshold
        self.overfit_gap_high = overfit_gap_high
        self.overfit_gap_med = overfit_gap_med
        self.corr_low_high = corr_low_high
        self.corr_low_med = corr_low_med
        self.oscillations_min_len = oscillations_min_len

    # ---------- helpers ----------
    def _clean_series(self, s: pd.Series) -> pd.Series:
        return s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    def _smooth(self, s: pd.Series, window: int = 3) -> pd.Series:
        window = max(1, min(window, max(1, len(s)//3)))
        return s.rolling(window=window, min_periods=1, center=False).mean()

    def _slope(self, s: pd.Series) -> float:
        s = self._clean_series(s)
        if len(s) < 2:
            return 0.0
        x = np.arange(len(s), dtype=float)
        try:
            m, _b = np.polyfit(x, s.values.astype(float), 1)
        except (np.linalg.LinAlgError, ValueError, TypeError):
            return 0.0
        return float(m)

    def identify_primary_metrics(self, metrics_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Identify primary metrics (loss, accuracy) for analysis."""
        primary_metrics: Dict[str, pd.Series] = {}
        loss_cols = [col for col in metrics_df.columns if 'loss' in col.lower()]
        for col in loss_cols[:2]:
            primary_metrics[col] = metrics_df[col]

        acc_cols = [col for col in metrics_df.columns if 'acc' in col.lower()]
        for col in acc_cols[:2]:
            primary_metrics[col] = metrics_df[col]

        return primary_metrics

    def calculate_improvement_rate(self, metric_series: pd.Series) -> float:
        """Calculate improvement rate for a metric series."""
        if len(metric_series) < 2:
            return 0.0

        series = self._clean_series(metric_series)
        if len(series) < 2:
            return 0.0

        is_loss_metric = 'loss' in str(metric_series.name).lower() if hasattr(metric_series, 'name') else False
        start_val = series.iloc[0]
        end_val = series.iloc[-1]
        denom = abs(start_val) if abs(start_val) > 1e-8 else 1.0
        if is_loss_metric:
            return float((start_val - end_val) / denom)
        return float((end_val - start_val) / denom)

    def detect_performance_plateaus(self, metric_series: pd.Series) -> Dict[str, Any]:
        """Detect plateau periods; returns counts and contiguous segments."""
        series = self._clean_series(metric_series)
        if len(series) < max(5, self.min_points):
            return {"total_plateau_epochs": 0, "plateau_periods": []}

        window = min(5, max(2, len(series) // 6))
        rolling_std = series.rolling(window=window, min_periods=2).std()
        plateau_threshold = max(1e-12, series.std() * self.plateau_std_ratio)
        mask = (rolling_std < plateau_threshold).fillna(False)

        # Extract contiguous segments
        segments: List[Tuple[int, int]] = []
        in_seg = False
        start = 0
        for idx, v in enumerate(mask.values.tolist()):
            if v and not in_seg:
                in_seg = True
                start = idx
            elif not v and in_seg:
                in_seg = False
                segments.append((start, idx - 1))
        if in_seg:
            segments.append((start, len(mask) - 1))

        return {
            "total_plateau_epochs": int(mask.sum()),
            "plateau_periods": segments
        }

    def assess_trend_quality(self, metric_series: pd.Series, improvement_rate: float) -> Dict[str, Any]:
        """Assess overall trend quality."""
        concerns: List[str] = []
        score = 0.8

        if abs(improvement_rate) < self.min_improvement_rate:
            concerns.append("minimal_improvement")
            score -= 0.3

        series = self._clean_series(metric_series)
        cv = series.std() / abs(series.mean()) if series.mean() != 0 else float('inf')
        if cv > self.high_cv_threshold:
            concerns.append("high_volatility")
            score -= 0.2

        if len(series) > 10:
            first_half_mean = series.iloc[:len(series)//2].mean()
            second_half_mean = series.iloc[len(series)//2:].mean()
            is_loss = 'loss' in str(metric_series.name).lower() if hasattr(metric_series, 'name') else False
            if is_loss and second_half_mean > first_half_mean * 1.1:
                concerns.append("trend_reversal")
                score -= 0.3
            elif not is_loss and second_half_mean < first_half_mean * 0.9:
                concerns.append("trend_reversal")
                score -= 0.3

        # Slope-based signal
        slope = self._slope(self._smooth(series))
        if (is_loss and slope > 0) or (not is_loss and slope < 0):
            concerns.append("unfavorable_trend")
            score -= 0.2

        return {
            "score": max(0.0, score),
            "concerns": concerns
        }

    def assess_trend_severity(self, trend_quality: Dict[str, Any]) -> Severity:
        """Assess severity based on trend quality."""
        score = trend_quality["score"]
        concerns = trend_quality["concerns"]
        if score < 0.3 or "trend_reversal" in concerns:
            return Severity.HIGH
        elif score < 0.6 or len(concerns) > 1:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def identify_metric_pairs(self, metrics_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Identify train-validation metric pairs."""
        pairs: List[Tuple[str, str]] = []
        train_prefixes = ['train_', 'training_', '']
        val_prefixes = ['val_', 'validation_', 'valid_']
        for col in metrics_df.columns:
            col_lower = col.lower()
            if any(prefix in col_lower for prefix in val_prefixes):
                continue
            base_name = col_lower
            for train_prefix in train_prefixes:
                if col_lower.startswith(train_prefix):
                    base_name = col_lower[len(train_prefix):]
                    break
            for val_prefix in val_prefixes:
                val_name = val_prefix + base_name
                val_cols = [c for c in metrics_df.columns if c.lower() == val_name]
                if val_cols:
                    pairs.append((col, val_cols[0]))
                    break
        return pairs

    def calculate_performance_gap(self, train_series: pd.Series, val_series: pd.Series) -> Dict[str, Any]:
        """Calculate performance gap between train and validation metrics."""
        if len(train_series) != len(val_series):
            min_len = min(len(train_series), len(val_series))
            train_series = train_series.iloc[:min_len]
            val_series = val_series.iloc[:min_len]

        gaps = abs(train_series - val_series) / (abs(train_series) + 1e-8)
        max_gap = gaps.max()
        divergence_start = None
        if len(gaps) > 5:
            gap_trend = gaps.rolling(window=3).mean().diff()
            increasing_trend = gap_trend > 0.01
            if increasing_trend.any():
                divergence_start = increasing_trend.idxmax()
        correlation = train_series.corr(val_series) if len(train_series) > 1 else 1.0
        return {
            "max_relative_gap": max_gap,
            "divergence_start_epoch": divergence_start,
            "trend_correlation": correlation,
            "final_gap": gaps.iloc[-1] if len(gaps) > 0 else 0.0
        }

    def assess_overfitting_severity(self, gap_analysis: Dict[str, Any]) -> Severity:
        """Assess overfitting severity based on gap analysis."""
        max_gap = float(gap_analysis["max_relative_gap"])
        correlation = float(gap_analysis.get("trend_correlation", 1.0))
        if max_gap > self.overfit_gap_high or correlation < self.corr_low_high:
            return Severity.HIGH
        elif max_gap > self.overfit_gap_med or correlation < self.corr_low_med:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def count_plateau_epochs(self, metric_series: pd.Series) -> int:
        """Count epochs where metric shows no significant improvement."""
        if len(metric_series) < 3:
            return 0
        window = min(3, len(metric_series) // 2)
        rolling_improvement = metric_series.rolling(window=window).apply(
            lambda x: abs(x.iloc[-1] - x.iloc[0]) / (abs(x.iloc[0]) + 1e-8)
        )
        plateau_threshold = 0.01
        plateau_epochs = (rolling_improvement < plateau_threshold).sum()
        return int(plateau_epochs)

    def calculate_rolling_cv(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling coefficient of variation."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return rolling_std / (abs(rolling_mean) + 1e-8)

    def detect_oscillations(self, series: pd.Series) -> float:
        """Detect oscillatory behavior in series."""
        if len(series) < self.oscillations_min_len:
            return 0.0
        diff_series = series.diff()
        sign_changes = (diff_series.shift(1) * diff_series < 0).sum()
        oscillation_score = sign_changes / (len(series) - 1)
        return float(oscillation_score)

    def detect_premature_convergence(self, series: pd.Series) -> bool:
        """Detect if series converged too early."""
        if len(series) < 10:
            return False
        split_point = len(series) // 3
        early_improvement = abs(series.iloc[split_point] - series.iloc[0])
        late_improvement = abs(series.iloc[-1] - series.iloc[split_point])
        if early_improvement > 0 and late_improvement / early_improvement < 0.1:
            return True
        return False

    def extract_gradient_metrics(self, metrics_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract gradient-related metrics from dataframe."""
        gradient_metrics: Dict[str, pd.Series] = {}
        gradient_patterns = ['grad_norm', 'gradient_norm', 'grad_clip', 'gradient_clip']
        for col in metrics_df.columns:
            col_lower = col.lower()
            for pattern in gradient_patterns:
                if pattern in col_lower:
                    gradient_metrics[col] = metrics_df[col]
                    break
        return gradient_metrics

    def detect_loss_stagnation(self, loss_series: pd.Series) -> bool:
        """Detect if loss has stagnated (potential vanishing gradients)."""
        if len(loss_series) < 10:
            return False
        split_point = len(loss_series) // 2
        recent_improvement = abs(loss_series.iloc[-1] - loss_series.iloc[split_point])
        initial_loss = abs(loss_series.iloc[0])
        if initial_loss > 0 and recent_improvement / initial_loss < 0.01:
            return True
        return False

    def has_convergence_issues(self, metrics_df: pd.DataFrame) -> bool:
        """Check if training has convergence issues."""
        loss_cols = [col for col in metrics_df.columns if 'loss' in col.lower()]
        for loss_col in loss_cols:
            if len(metrics_df[loss_col]) > 5:
                loss_series = self._clean_series(metrics_df[loss_col])
                if len(loss_series) < 2:
                    continue
                slope = self._slope(self._smooth(loss_series))
                # Non-decreasing loss or very small improvement
                improvement = self.calculate_improvement_rate(loss_series)
                if slope >= 0 or improvement < self.min_improvement_rate:
                    return True
        return False

    def has_stability_issues(self, metrics_df: pd.DataFrame) -> bool:
        """Check if training has stability issues."""
        for col in metrics_df.select_dtypes(include=[np.number]).columns:
            if len(metrics_df[col]) > 5:
                series = self._clean_series(metrics_df[col])
                if len(series) < 2:
                    continue
                cv = series.std() / (abs(series.mean()) + 1e-8)
                if cv > self.high_cv_threshold:
                    return True
        return False

    # ---------- high level ----------
    def analyze(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Run a concise diagnostic over available metrics.

        Returns a dictionary with key findings and severities.
        """
        summary: Dict[str, Any] = {
            "convergence_issue": self.has_convergence_issues(metrics_df),
            "stability_issue": self.has_stability_issues(metrics_df),
            "metrics": {},
            "overfitting": {},
        }

        # Per-metric analysis for primary metrics
        for name, series in self.identify_primary_metrics(metrics_df).items():
            imp = self.calculate_improvement_rate(series)
            trend_q = self.assess_trend_quality(series, imp)
            sev = self.assess_trend_severity(trend_q)
            plateaus = self.detect_performance_plateaus(series)
            oscill = self.detect_oscillations(series)
            summary["metrics"][name] = {
                "improvement_rate": imp,
                "trend_quality": trend_q,
                "trend_severity": sev.name,
                "plateaus": plateaus,
                "oscillation_score": oscill,
            }

        # Overfitting across identified pairs
        for train_col, val_col in self.identify_metric_pairs(metrics_df):
            gap = self.calculate_performance_gap(metrics_df[train_col], metrics_df[val_col])
            sev = self.assess_overfitting_severity(gap)
            summary["overfitting"][f"{train_col}__vs__{val_col}"] = {
                "gap": gap,
                "severity": sev.name,
            }

        return summary



