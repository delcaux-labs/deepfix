import numpy as np
import pandas as pd
import pytest
from deepfix_server.agents.training_dynamics_utils import TrainingDynamicsAnalyzer

@pytest.fixture
def analyzer():
    return TrainingDynamicsAnalyzer()

def test_calculate_improvement_rate_loss_happy_path(analyzer):
    """Test happy path for loss metrics (decreasing loss = positive improvement)."""
    series = pd.Series([1.0, 0.8, 0.6, 0.4], name="train_loss")
    # (start - end) / start = (1.0 - 0.4) / 1.0 = 0.6
    assert analyzer.calculate_improvement_rate(series) == pytest.approx(0.6)

def test_calculate_improvement_rate_acc_happy_path(analyzer):
    """Test happy path for non-loss metrics (increasing accuracy = positive improvement)."""
    series = pd.Series([0.5, 0.6, 0.7, 0.8], name="train_acc")
    # (end - start) / start = (0.8 - 0.5) / 0.5 = 0.6
    assert analyzer.calculate_improvement_rate(series) == pytest.approx(0.6)

def test_calculate_improvement_rate_empty_series(analyzer):
    """Test edge case: Empty series."""
    series = pd.Series([], dtype=float)
    assert analyzer.calculate_improvement_rate(series) == 0.0

def test_calculate_improvement_rate_single_element_series(analyzer):
    """Test edge case: Single element series."""
    series = pd.Series([0.5], name="train_acc")
    assert analyzer.calculate_improvement_rate(series) == 0.0

def test_calculate_improvement_rate_after_cleaning_empty(analyzer):
    """Test edge case: Series that becomes empty after cleaning (NaNs/Infs)."""
    series = pd.Series([np.nan, np.inf, -np.inf], name="train_loss")
    assert analyzer.calculate_improvement_rate(series) == 0.0

def test_calculate_improvement_rate_after_cleaning_single(analyzer):
    """Test edge case: Series that becomes single element after cleaning."""
    series = pd.Series([0.5, np.nan, np.inf], name="train_loss")
    assert analyzer.calculate_improvement_rate(series) == 0.0

def test_calculate_improvement_rate_zero_start(analyzer):
    """Test edge case: Start value is zero."""
    series = pd.Series([0.0, 0.1, 0.2], name="train_acc")
    # start_val = 0.0, end_val = 0.2, denom = 1.0 (since abs(start) <= 1e-8)
    # improvement = (0.2 - 0.0) / 1.0 = 0.2
    assert analyzer.calculate_improvement_rate(series) == pytest.approx(0.2)

def test_calculate_improvement_rate_near_zero_start(analyzer):
    """Test edge case: Start value is very small (near zero)."""
    series = pd.Series([1e-9, 0.1, 0.2], name="train_acc")
    # start_val = 1e-9, end_val = 0.2, denom = 1.0 (since abs(start) <= 1e-8)
    # improvement = (0.2 - 1e-9) / 1.0 = 0.199999999
    assert analyzer.calculate_improvement_rate(series) == pytest.approx(0.2)

def test_calculate_improvement_rate_no_name_as_non_loss(analyzer):
    """Test edge case: Series with no name (should be treated as non-loss)."""
    series = pd.Series([0.5, 0.8])
    # Treated as non-loss: (0.8 - 0.5) / 0.5 = 0.6
    assert analyzer.calculate_improvement_rate(series) == pytest.approx(0.6)

def test_calculate_improvement_rate_non_loss_name(analyzer):
    """Test edge case: Series with name not containing 'loss'."""
    series = pd.Series([0.5, 0.8], name="metric_x")
    # Treated as non-loss: (0.8 - 0.5) / 0.5 = 0.6
    assert analyzer.calculate_improvement_rate(series) == pytest.approx(0.6)

def test_detect_performance_plateaus_happy_path(analyzer):
    """Test detecting plateaus in a plateauing series."""
    # Constant values from index 3 to 8
    series = pd.Series([1.0, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.4])
    result = analyzer.detect_performance_plateaus(series)
    assert result["total_plateau_epochs"] > 0
    assert len(result["plateau_periods"]) > 0

def test_detect_performance_plateaus_no_plateau(analyzer):
    """Test detecting plateaus in a constantly improving series."""
    series = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    result = analyzer.detect_performance_plateaus(series)
    assert result["total_plateau_epochs"] == 0
    assert result["plateau_periods"] == []

def test_detect_performance_plateaus_too_short(analyzer):
    """Test detecting plateaus in a series too short to detect plateaus."""
    series = pd.Series([1.0, 0.6, 0.6, 0.6])
    result = analyzer.detect_performance_plateaus(series)
    assert result["total_plateau_epochs"] == 0
    assert result["plateau_periods"] == []

def test_assess_trend_quality_good_trend(analyzer):
    """Test assess_trend_quality with a good trend."""
    series = pd.Series([1.0, 0.8, 0.6, 0.4, 0.2, 0.1], name="train_loss")
    imp = analyzer.calculate_improvement_rate(series)
    result = analyzer.assess_trend_quality(series, imp)
    assert result["score"] >= 0.8
    assert result["concerns"] == []

def test_assess_trend_quality_minimal_improvement(analyzer):
    """Test assess_trend_quality with minimal improvement."""
    series = pd.Series([1.0, 1.0, 0.999, 1.0, 1.0], name="train_loss")
    imp = analyzer.calculate_improvement_rate(series)
    result = analyzer.assess_trend_quality(series, imp)
    assert "minimal_improvement" in result["concerns"]
    assert result["score"] < 0.8

def test_assess_trend_quality_high_volatility(analyzer):
    """Test assess_trend_quality with high volatility (CV)."""
    series = pd.Series([1.0, 2.0, 0.5, 3.0, 0.2, 4.0], name="train_loss")
    imp = analyzer.calculate_improvement_rate(series)
    result = analyzer.assess_trend_quality(series, imp)
    assert "high_volatility" in result["concerns"]

def test_assess_trend_quality_trend_reversal(analyzer):
    """Test assess_trend_quality with trend reversal."""
    # Loss decreasing then increasing significantly
    series = pd.Series([1.0, 0.8, 0.6, 0.5, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0], name="train_loss")
    imp = analyzer.calculate_improvement_rate(series) # (1.0 - 2.0) / 1.0 = -1.0
    result = analyzer.assess_trend_quality(series, imp)
    assert "trend_reversal" in result["concerns"]

def test_detect_oscillations_high(analyzer):
    """Test detecting high oscillations."""
    series = pd.Series([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5])
    score = analyzer.detect_oscillations(series)
    assert score > 0.5

def test_detect_oscillations_low(analyzer):
    """Test detecting low/no oscillations."""
    series = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    score = analyzer.detect_oscillations(series)
    assert score == 0.0

def test_calculate_performance_gap_divergent(analyzer):
    """Test calculate_performance_gap with divergent train/val series."""
    train = pd.Series([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])
    val = pd.Series([1.0, 0.9, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2])
    result = analyzer.calculate_performance_gap(train, val)
    assert result["max_relative_gap"] > 0.5
    assert result["trend_correlation"] < 0.5

def test_calculate_performance_gap_correlated(analyzer):
    """Test calculate_performance_gap with well-correlated train/val series."""
    train = pd.Series([1.0, 0.8, 0.6, 0.4, 0.2])
    val = pd.Series([1.1, 0.9, 0.7, 0.5, 0.3])
    result = analyzer.calculate_performance_gap(train, val)
    assert result["trend_correlation"] > 0.9
