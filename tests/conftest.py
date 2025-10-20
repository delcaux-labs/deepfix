import sys
import pytest

# Patch numpy early
try:
    import numpy as np
    if not hasattr(np, 'Inf'):
        np.Inf = np.inf
except Exception:
    pass

# Import DeepchecksConfig without loading the full deepchecks.py module
try:
    from deepfix.shared.models import DeepchecksConfig
except ImportError:
    # Try alternate import path
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from deepfix.shared.models import DeepchecksConfig


@pytest.fixture
def deepchecks_config_no_save() -> DeepchecksConfig:
    """
    Fixture providing a DeepchecksConfig with save_results disabled.
    This prevents tests from writing files to disk.
    """
    return DeepchecksConfig(
        save_results=False,
        train_test_validation=True,
        data_integrity=False,
        model_evaluation=False,
        max_samples=50,
        random_state=42,
    )


@pytest.fixture
def minimal_deepchecks_config() -> DeepchecksConfig:
    """
    Fixture providing minimal DeepchecksConfig for quick test execution.
    Only enables train_test_validation suite.
    """
    return DeepchecksConfig(
        save_results=False,
        train_test_validation=True,
        data_integrity=False,
        model_evaluation=False,
        max_samples=20,
        random_state=42,
    )
