"""
Model utilities for DeepFix SDK.

This module provides utilities for working with machine learning models,
including metadata extraction for logging and artifact management.
"""

from .utils import get_model_metadata
from .ir_model import IRLookupModel

__all__ = [
    "get_model_metadata",
    "IRLookupModel",
]
