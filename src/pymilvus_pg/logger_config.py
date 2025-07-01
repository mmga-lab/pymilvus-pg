#!/usr/bin/env python
"""
Logger configuration module for pymilvus_pg.

This module provides centralized logging configuration using loguru with both
console and file output. It supports environment-based configuration and
dynamic log level adjustment.

Features:
- Automatic log directory creation
- Unique log files per execution with timestamps
- File rotation and compression
- Color-coded console output
- Environment variable configuration
- Dynamic log level adjustment
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Configure log directory with environment override capability
# Defaults to 'logs' subdirectory in current working directory
# Can be overridden via PYMILVUS_PG_LOG_DIR environment variable
_log_dir = Path(os.getenv("PYMILVUS_PG_LOG_DIR", Path.cwd() / "logs"))
_log_dir.mkdir(parents=True, exist_ok=True)

# Generate unique log file name based on execution timestamp
# This ensures each run has its own log file for easier debugging
_log_file = _log_dir / f"pymilvus_pg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Remove default logger to configure custom handlers
logger.remove()

# Configure console output handler
# Uses WARNING level by default to reduce console noise during normal operations
# INFO and DEBUG messages are captured in log files
logger.add(
    sys.stderr,
    level="INFO",  # Reduced from INFO to minimize console output
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# Configure file output handler
# Captures DEBUG and above messages with comprehensive details
# Includes automatic rotation, retention, and compression
logger.add(
    _log_file,
    level="DEBUG",  # Capture all debug information in files
    rotation="10 MB",  # Rotate when file reaches 10 MB
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress rotated files to save space
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
)


def set_logger_level(level: str) -> None:
    """
    Dynamically adjust the log level for both console and file outputs.

    This function allows runtime adjustment of logging levels, useful for
    debugging or when more verbose output is needed.

    Parameters
    ----------
    level : str
        Log level to set. Supported levels:
        - 'DEBUG': Detailed information for debugging
        - 'INFO': General information about program execution
        - 'WARNING': Warning messages for potentially harmful situations
        - 'ERROR': Error messages for serious problems
        - 'CRITICAL': Critical messages for very serious errors

    Note
    ----
    This function recreates all log handlers with the new level.
    The file handler maintains DEBUG level to preserve comprehensive logging.
    """
    # Remove existing handlers
    logger.remove()

    # Recreate console handler with new level
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Recreate file handler - always keep DEBUG level for comprehensive file logging
    # This ensures that even when console level is raised, file logs remain detailed
    logger.add(
        _log_file,
        level="DEBUG",  # File logging always captures debug info
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True,
    )


def get_log_file_path() -> Path:
    """
    Get the current log file path.

    Returns
    -------
    Path
        Path to the current log file
    """
    return _log_file


def get_log_directory() -> Path:
    """
    Get the log directory path.

    Returns
    -------
    Path
        Path to the log directory
    """
    return _log_dir


# Export public interface
__all__ = ["logger", "set_logger_level", "get_log_file_path", "get_log_directory"]
