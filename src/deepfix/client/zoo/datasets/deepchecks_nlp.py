"""
DeepChecks NLP datasets utilities.

This module provides convenient loaders for pre-built NLP datasets from the DeepChecks library,
supporting text classification and token classification (NER) tasks with TextData return types.
"""

from typing import Tuple
import logging

from deepchecks.nlp import TextData
from deepchecks.nlp.datasets import classification, token_classification

LOGGER = logging.getLogger(__name__)


# Classification Datasets
def load_tweet_emotion_classification(include_embeddings: bool = False) -> TextData:
    """
    Load Tweet Emotion classification dataset from DeepChecks.
    
    Contains tweets labeled with emotion categories (joy, sadness, anger, fear, neutral).
    This is a text classification task for sentiment/emotion analysis.
    
    Returns:
        TextData object
    """
    LOGGER.info("Loading Tweet Emotion classification dataset")
    try:
        data = classification.tweet_emotion.load_data(data_format="TextData",
                                            include_embeddings=include_embeddings,
                                            include_properties=True,
                                            as_train_test=False
                                        )  # type: ignore
        return data
    except Exception as e:
        LOGGER.error("Failed to load Tweet Emotion classification dataset: %s", str(e))
        raise e


def load_just_dance_comment_classification(include_embeddings: bool = False,use_full_size: bool = False) -> TextData:
    """
    Load Just Dance Comment Analysis classification dataset from DeepChecks.
    
    Contains YouTube comments about Just Dance videos with binary classification labels
    (relevant/irrelevant or positive/negative).
    
    Returns:
        TextData object
    """
    LOGGER.info("Loading Just Dance comment classification dataset")
    try:
        data = classification.just_dance_comment_analysis.load_data(data_format="TextData",
                                                            include_embeddings=include_embeddings,
                                                            include_properties=True,
                                                            as_train_test=False,
                                                            use_full_size=use_full_size
                                                            )  # type: ignore
        return data
    except Exception as e:
        LOGGER.error("Failed to load Just Dance comment classification dataset: %s", str(e))
        raise e


# Token Classification Datasets
def load_scierc_ner(include_embeddings: bool = False) -> Tuple[TextData, TextData]:
    """
    Load SciERC Named Entity Recognition (NER) dataset from DeepChecks.
    
    Contains scientific paper abstracts with token-level entity annotations for:
    - Method, Material, Other-ScientificTerm, Task, Generic, Metric
    
    This is a token classification task for extracting named entities from scientific text.
    
    Returns:
        Tuple of (train_data, test_data) as TextData objects for token classification
    """
    LOGGER.info("Loading SciERC NER dataset")
    try:
        train_data, test_data = token_classification.scierc_ner.load_data(data_format="TextData",
        include_embeddings=include_embeddings,
        include_properties=True)  # type: ignore
        return train_data, test_data
    except Exception as e:
        LOGGER.error("Failed to load SciERC NER dataset: %s", str(e))
        raise e
