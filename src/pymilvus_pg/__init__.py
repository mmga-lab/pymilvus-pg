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

from ._version import __version__
from .exceptions import (
    CollectionNotFoundError,
    ConnectionError,
    DataTypeMismatchError,
    FilterConversionError,
    MilvusPGError,
    SchemaError,
    SyncError,
    TransactionError,
    ValidationError,
)
from .lmdb_manager import LMDBManager, PKOperation, PKStatus
from .logger_config import logger, set_logger_level
from .milvus_pg_client import MilvusPGClient
from .types import (
    ComparisonResult,
    DeleteResult,
    EntityData,
    FieldValue,
    FilterExpression,
    InsertResult,
    OutputFields,
    PrimaryKeyComparisonResult,
    PrimaryKeyList,
    PrimaryKeyType,
    QueryResult,
    UpsertResult,
)

__all__ = [
    # Version
    "__version__",
    # Main client
    "MilvusPGClient",
    # LMDB Manager
    "LMDBManager",
    "PKStatus",
    "PKOperation",
    # Logging
    "logger",
    "set_logger_level",
    # Exceptions
    "MilvusPGError",
    "ConnectionError",
    "SchemaError",
    "CollectionNotFoundError",
    "SyncError",
    "ValidationError",
    "FilterConversionError",
    "TransactionError",
    "DataTypeMismatchError",
    # Types
    "PrimaryKeyType",
    "PrimaryKeyList",
    "EntityData",
    "FieldValue",
    "FilterExpression",
    "OutputFields",
    "QueryResult",
    "InsertResult",
    "UpsertResult",
    "DeleteResult",
    "ComparisonResult",
    "PrimaryKeyComparisonResult",
]
