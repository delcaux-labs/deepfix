import sys
import os

import pytest


# Patch numpy early
try:
    import numpy as np

    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if not hasattr(np, "NINF"):
        np.NINF = -np.inf
except Exception:
    pass

from deepfix_core.models import APIResponse, DeepchecksConfig

@pytest.fixture
def minimal_deepchecks_config() -> DeepchecksConfig:
    """
    Fixture providing minimal DeepchecksConfig for quick test execution.
    Only enables train_test_validation suite.
    """
    return DeepchecksConfig(
        save_results=False,
        train_test_validation=True,
        data_integrity=True,
        model_evaluation=True,
        random_state=42,
    )


@pytest.fixture
def api_url():
    """Fixture providing the DeepFix API URL for tests."""
    url = os.getenv("DEEPFIX_TEST_API_URL")
    if url is None:
        raise ValueError("DEEPFIX_TEST_API_URL is not set")
    return url


@pytest.fixture
def deepfix_timeout():
    return int(os.getenv("DEEPFIX_TIMEOUT", "600"))


@pytest.fixture
def coco_detection_paths() -> dict[str, str]:
    """Fixture providing COCO detection dataset paths, skipping if not set."""
    paths = {
        "tr_images": os.getenv("TR_IMAGES_DIR_PATH"),
        "tr_annotations": os.getenv("TR_ANNOTATIONS_PATH"),
        "val_images": os.getenv("VAL_IMAGES_DIR_PATH"),
        "val_annotations": os.getenv("VAL_ANNOTATIONS_PATH"),
    }

    if not all(paths.values()):
        pytest.skip(
            "Object detection dataset paths (TR_IMAGES_DIR_PATH, TR_ANNOTATIONS_PATH, "
            "VAL_IMAGES_DIR_PATH, VAL_ANNOTATIONS_PATH) not fully set"
        )

    return paths


@pytest.fixture
def check_response():

    def check(response):
        assert isinstance(response, APIResponse), (
            "Response should be an APIResponse instance"
        )
        assert response.summary is not None, "Response should have a summary"
        assert len(response.agent_results) > 0, (
            "Response should contain results from agents"
        )

        print("\nDeepFix Analysis Summary:")
        print(response.to_text())

        return True

    return check


