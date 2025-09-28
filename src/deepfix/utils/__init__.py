"""
Shared utilities for DeepSight MLOps Copilot.

This module contains shared utility functions and classes including:
- Configuration management and loading
- Logging setup and formatting
- Input validation and sanitization
- Common helper functions
"""

from .logging import setup_logging, get_logger

__all__ = [
    "setup_logging",
    "get_logger",
]
