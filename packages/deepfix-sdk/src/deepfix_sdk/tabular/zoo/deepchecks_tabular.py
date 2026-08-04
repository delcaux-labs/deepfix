"""
DeepChecks tabular datasets utilities.

This module provides convenient loaders for pre-built tabular datasets from the DeepChecks library,
supporting classification and regression tasks.

Requires the ``[tabular]`` extra: ``pip install deepfix-sdk[tabular]``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from deepchecks.tabular import Dataset

LOGGER = logging.getLogger(__name__)


def _require_tabular():
    try:
        from deepchecks.tabular import Dataset  # noqa: F811
        from deepchecks.tabular.datasets import classification, regression  # noqa: F811
    except ImportError:
        raise ImportError(
            "Tabular dependencies are required for these datasets. "
            "Install with: pip install deepfix-sdk[tabular]"
        ) from None


# Classification Datasets
def load_iris_classification(as_train_test: bool = True) -> Tuple["Dataset", "Dataset"]:
    """
    Load Iris classification dataset from DeepChecks.

    Contains 3 classes of iris plants (setosa, versicolor, virginica).
    This is a multi-class tabular classification task.

    Args:
        as_train_test: If True, returns a tuple of (train_dataset, test_dataset)

    Returns:
        Tuple of (train_data, test_data) as TabularDataset objects
    """
    _require_tabular()
    from deepchecks.tabular.datasets import classification

    LOGGER.info("Loading Iris classification dataset")
    try:
        train_data, test_data = classification.iris.load_data(
            data_format="Dataset", as_train_test=as_train_test
        )
        return train_data, test_data
    except Exception as e:
        LOGGER.error("Failed to load Iris classification dataset: %s", str(e))
        raise e


def load_breast_cancer_classification(
    as_train_test: bool = True,
) -> Tuple["Dataset", "Dataset"]:
    """
    Load Breast Cancer classification dataset from DeepChecks.

    Contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
    This is a binary tabular classification task (malignant or benign).

    Args:
        as_train_test: If True, returns a tuple of (train_dataset, test_dataset)

    Returns:
        Tuple of (train_data, test_data) as TabularDataset objects
    """
    _require_tabular()
    from deepchecks.tabular.datasets import classification

    LOGGER.info("Loading Breast Cancer classification dataset")
    try:
        train_data, test_data = classification.breast_cancer.load_data(
            data_format="Dataset", as_train_test=as_train_test
        )
        return train_data, test_data
    except Exception as e:
        LOGGER.error("Failed to load Breast Cancer classification dataset: %s", str(e))
        raise e


def load_adult_classification(
    as_train_test: bool = True,
) -> Tuple["Dataset", "Dataset"]:
    """
    Load Adult (Census Income) classification dataset from DeepChecks.

    Predict whether income exceeds $50K/yr based on census data.
    This is a binary tabular classification task with categorical features.

    Args:
        as_train_test: If True, returns a tuple of (train_dataset, test_dataset)

    Returns:
        Tuple of (train_data, test_data) as TabularDataset objects
    """
    _require_tabular()
    from deepchecks.tabular.datasets import classification

    LOGGER.info("Loading Adult Income classification dataset")
    try:
        train_data, test_data = classification.adult.load_data(
            data_format="Dataset", as_train_test=as_train_test
        )
        return train_data, test_data
    except Exception as e:
        LOGGER.error("Failed to load Adult Income classification dataset: %s", str(e))
        raise e


# Regression Datasets
def load_california_housing_regression(
    as_train_test: bool = True,
) -> Tuple["Dataset", "Dataset"]:
    """
    Load California Housing regression dataset from DeepChecks.

    Predict median house value for California districts.
    This is a tabular regression task.

    Args:
        as_train_test: If True, returns a tuple of (train_dataset, test_dataset)

    Returns:
        Tuple of (train_data, test_data) as TabularDataset objects
    """
    _require_tabular()
    from deepchecks.tabular.datasets import regression

    LOGGER.info("Loading California Housing regression dataset")
    try:
        train_data, test_data = regression.california_housing.load_data(
            data_format="Dataset", as_train_test=as_train_test
        )
        return train_data, test_data
    except Exception as e:
        LOGGER.error(
            "Failed to load California Housing regression dataset: %s", str(e)
        )
        raise e
