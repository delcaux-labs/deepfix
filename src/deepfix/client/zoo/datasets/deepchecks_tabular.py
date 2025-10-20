"""
DeepChecks tabular datasets utilities.

This module provides convenient loaders for pre-built tabular datasets from the DeepChecks library,
supporting classification and regression tasks with automatic categorical feature detection.
"""

from typing import Optional, Tuple
import logging

from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets import classification, regression

LOGGER = logging.getLogger(__name__)


# Classification Datasets
def load_adult_classification(as_train_test:bool=False) -> Dataset:
    """
    Load Adult/Census income classification dataset from DeepChecks.
    
    Predicts whether income exceeds $50K/yr.
    
    Returns:
        DeepChecks Dataset object
    """
    LOGGER.info("Loading Adult classification dataset")
    return classification.adult.load_data(data_format="Dataset",as_train_test=as_train_test)


def load_iris_classification(as_train_test:bool=False) -> Dataset:
    """
    Load Iris flower classification dataset from DeepChecks.
    
    Returns:
        DeepChecks Dataset object
    """
    LOGGER.info("Loading Iris classification dataset")    
    return classification.iris.load_data(data_format="Dataset",as_train_test=as_train_test) 


# Regression Datasets
def load_airbnb_regression(data_size:Optional[int]=15000) -> Tuple[Dataset, Dataset]:
    """
    Load AirBnb regression dataset from DeepChecks.
    
    Returns:
    Tuple of (train_dataset, test_dataset) as DeepChecks Dataset objects
    """
    LOGGER.info("Loading AirBnb regression dataset")
    train_data,_ = regression.airbnb.load_data_and_predictions(data_format="Dataset",load_train=True,data_size=data_size) 
    test_data,_ = regression.airbnb.load_data_and_predictions(data_format="Dataset",load_train=False,data_size=data_size) 
    return train_data, test_data


def load_wine_quality_regression(as_train_test:bool=False) -> Dataset:
    """
    Load Wine quality regression dataset from DeepChecks.
    
    Returns:
    DeepChecks Dataset object
    """
    LOGGER.info("Loading Wine quality regression dataset")
    return regression.wine_quality.load_data(data_format="Dataset",as_train_test=as_train_test) 


def load_avocado_regression(as_train_test:bool=False) -> Dataset:
    """
    Load Avocado regression dataset from DeepChecks.
    
    Returns:
    DeepChecks Dataset object
    """
    LOGGER.info("Loading Avocado regression dataset")
    return regression.avocado.load_data(data_format="Dataset",as_train_test=as_train_test)


