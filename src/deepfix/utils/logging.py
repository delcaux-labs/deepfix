"""
Logging setup and configuration for DeepSight MLOps Copilot.

This module provides centralized logging configuration with support for
different log levels, formats, and output destinations.
"""

import logging
import logging.handlers
from typing import Optional, Dict, Any
from pathlib import Path
import sys


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Setup logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Custom log format string
        config: Configuration dictionary for logging settings
    """
    # Get logging config if provided
    if config:
        logging_config = config.get("logging", {})
        level = logging_config.get("level", level)
        log_file = logging_config.get("file", log_file)
        format_string = logging_config.get("format", format_string)

    # Set default format if not provided
    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Setup file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels for external libraries
    _configure_external_loggers()


def _configure_external_loggers() -> None:
    """Configure logging levels for external libraries."""
    # Reduce noise from external libraries
    external_loggers = {
        "urllib3.connectionpool": logging.WARNING,
        "requests.packages.urllib3": logging.WARNING,
        "matplotlib": logging.WARNING,
        "PIL": logging.WARNING,
        "asyncio": logging.WARNING,
        "aiohttp": logging.WARNING,
    }

    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
