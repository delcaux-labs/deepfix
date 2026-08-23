"""
Model utilities for extracting metadata from ML models.

This module provides utility functions for extracting metadata from various
machine learning models, particularly scikit-learn compatible models.
"""

from typing import Any

from deepfix_core.models import ModelCheckpointArtifacts


# TODO: extend to PyTorch models
def get_model_metadata(model: Any) -> ModelCheckpointArtifacts:
    if not hasattr(model, "get_params"):
        raise ValueError(
            f"Model of type {type(model).__name__} does not have a 'get_params' method. "
            "Only scikit-learn compatible models (BaseEstimator subclasses) are currently supported."
        )

    # Extract model type (class name)
    model_type = type(model).__name__
    class_docstring = model.__class__.__doc__

    # Extract hyperparameters using get_params with deep=False to avoid nested objects
    try:
        hyperparameters = model.get_params(deep=False)
    except Exception as e:
        raise ValueError(f"Failed to extract parameters from model: {str(e)}")

    # Convert any non-serializable parameter values to strings
    serialized_params = {}
    for key, value in hyperparameters.items():
        try:
            # Check if value is JSON serializable (basic types)
            if value is None or isinstance(value, (bool, int, float, str)):
                serialized_params[key] = value
            elif isinstance(value, (list, tuple)):
                # Try to serialize lists/tuples, convert to string if it fails
                try:
                    serialized_params[key] = (
                        list(value) if isinstance(value, tuple) else value
                    )
                except Exception:
                    serialized_params[key] = str(value)
            elif isinstance(value, dict):
                serialized_params[key] = value
            else:
                # For complex objects, store their string representation
                serialized_params[key] = str(value)
        except Exception:
            serialized_params[key] = str(value)

    return ModelCheckpointArtifacts(
        model_type=model_type,
        hyperparameters=serialized_params,
        context={"class_docstring": class_docstring},
    )
