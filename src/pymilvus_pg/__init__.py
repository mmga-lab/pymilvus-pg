"""
PyMilvus-PG: Milvus Client with PostgreSQL Synchronization

This package provides a Milvus client that synchronizes write operations to PostgreSQL
for validation and comparison purposes. It extends the standard pymilvus client with
additional functionality for maintaining shadow copies of vector data in PostgreSQL.

Main Components:
    MilvusPGClient: Extended Milvus client with PostgreSQL synchronization
    logger: Configured logger instance for the package
    set_logger_level: Function to adjust logging levels dynamically

Example:
    >>> from pymilvus_pg import MilvusPGClient, logger, set_logger_level
    >>>
    >>> # Configure logging level
    >>> set_logger_level("INFO")
    >>>
    >>> # Initialize client
    >>> client = MilvusPGClient(
    ...     uri="http://localhost:19530",
    ...     pg_conn_str="postgresql://user:pass@localhost/db"
    ... )
"""

from .logger_config import logger, set_logger_level
from .milvus_pg_client import MilvusPGClient

__all__ = [
    "MilvusPGClient",
    "logger",
    "set_logger_level",
]
