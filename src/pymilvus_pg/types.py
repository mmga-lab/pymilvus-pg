"""Type definitions for pymilvus_pg package."""

from typing import Any, TypeAlias, TypedDict

# Primary key types
PrimaryKeyType: TypeAlias = int | str
PrimaryKeyList: TypeAlias = list[PrimaryKeyType]

# Data types
EntityData: TypeAlias = list[dict[str, Any]]
FieldValue: TypeAlias = bool | int | float | str | list[Any] | dict[str, Any] | None
FilterExpression: TypeAlias = str
OutputFields: TypeAlias = list[str]

# Result types
QueryResult: TypeAlias = list[dict[str, Any]]


class InsertResult(TypedDict):
    insert_count: int
    ids: list[PrimaryKeyType]


class UpsertResult(TypedDict):
    upsert_count: int


class DeleteResult(TypedDict):
    delete_count: int


# Comparison types
class ComparisonResult(TypedDict):
    is_equal: bool
    row_count_milvus: int
    row_count_pg: int
    differences: dict[str, Any]


class PrimaryKeyComparisonResult(TypedDict):
    total_milvus: int
    total_pg: int
    common: int
    only_in_milvus: int
    only_in_pg: int
    milvus_only_keys: list[PrimaryKeyType]
    pg_only_keys: list[PrimaryKeyType]


# Schema types
class FieldSchema(TypedDict, total=False):
    name: str
    dtype: str
    is_primary: bool
    auto_id: bool
    dim: int
    max_length: int
