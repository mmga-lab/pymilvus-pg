#!/usr/bin/env python
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Default logger configuration
# Resolve execution directory (cwd) and allow override via env var. This makes
# logs follow the actual run location, convenient for container or script execution.
_log_dir = Path(os.getenv("PYMILVUS_PG_LOG_DIR", Path.cwd() / "logs"))
_log_dir.mkdir(parents=True, exist_ok=True)
# Generate a unique log file name for each run based on the current timestamp
_log_file = _log_dir / f"pymilvus_pg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",  # Default log level
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# Add a file sink to persist logs. This keeps DEBUG and above messages in a rotated log file.
# Each run will have a unique log file based on timestamp.
logger.add(
    _log_file,
    level="DEBUG",
    rotation="10 MB",  # Rotate when file reaches 10 MB
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress rotated files
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
)


def set_logger_level(level: str):
    """
    Dynamically set the log level for the logger.
    Args:
        level (str): Log level, e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    logger.remove()
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

    # Re-add the file sink with the new log level and unique file name for this run
    logger.add(
        _log_file,
        level=level.upper(),
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True,
    )


# Export the configured logger instance and the set_logger_level function
__all__ = ["logger", "set_logger_level"]
