"""Custom exceptions for pymilvus_pg package."""

from typing import Any


class MilvusPGError(Exception):
    """Base exception for all pymilvus_pg errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConnectionError(MilvusPGError):
    """Raised when there are connection issues with Milvus or PostgreSQL."""

    pass


class SchemaError(MilvusPGError):
    """Raised when there are schema-related errors."""

    pass


class CollectionNotFoundError(SchemaError):
    """Raised when a collection is not found."""

    def __init__(self, collection_name: str) -> None:
        super().__init__(f"Collection '{collection_name}' not found")
        self.collection_name = collection_name


class SyncError(MilvusPGError):
    """Raised when data synchronization between Milvus and PostgreSQL fails."""

    pass


class ValidationError(MilvusPGError):
    """Raised when data validation fails."""

    pass


class FilterConversionError(MilvusPGError):
    """Raised when Milvus filter cannot be converted to SQL."""

    def __init__(self, filter_expr: str, reason: str) -> None:
        super().__init__(f"Cannot convert filter '{filter_expr}': {reason}")
        self.filter_expr = filter_expr
        self.reason = reason


class TransactionError(MilvusPGError):
    """Raised when a transaction fails."""

    pass


class DataTypeMismatchError(SchemaError):
    """Raised when data types don't match between Milvus and PostgreSQL."""

    def __init__(self, field_name: str, milvus_type: str, pg_type: str) -> None:
        super().__init__(
            f"Data type mismatch for field '{field_name}': Milvus type {milvus_type} vs PostgreSQL type {pg_type}"
        )
        self.field_name = field_name
        self.milvus_type = milvus_type
        self.pg_type = pg_type
