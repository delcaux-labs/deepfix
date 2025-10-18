"""
Zoo datasets module.

Provides utilities for loading various datasets including custom food datasets
and pre-built datasets from the DeepChecks library.
"""

# DeepChecks Vision Dataset Loaders
from .deepchecks_vision import (
    load_mnist_classification,
    load_coco_detection,
    load_mask_detection,
    load_segmentation_dataset,
)

# DeepChecks Tabular Dataset Loaders
from .deepchecks_tabular import (
    load_adult_classification,
    load_iris_classification,
    load_airbnb_regression,
    load_wine_quality_regression,
    load_avocado_regression,
)

# DeepChecks NLP Dataset Loaders
from .deepchecks_nlp import (
    load_tweet_emotion_classification,
    load_just_dance_comment_classification,
    load_scierc_ner,
)

# Dataset Registry
from .registry import (
    DatasetRegistry,
    DatasetType,
    DatasetMetadata,
    get_registry,
)

__all__ = [
    # Vision datasets
    'load_mnist_classification',
    'load_coco_detection',
    'load_mask_detection',
    'load_segmentation_dataset',
    # Tabular datasets
    'load_adult_classification',
    'load_iris_classification',
    'load_airbnb_regression',
    'load_wine_quality_regression',
    'load_avocado_regression',
    # NLP datasets
    'load_tweet_emotion_classification',
    'load_just_dance_comment_classification',
    'load_scierc_ner',
    # Registry
    'DatasetRegistry',
    'DatasetType',
    'DatasetMetadata',
    'get_registry',
]
