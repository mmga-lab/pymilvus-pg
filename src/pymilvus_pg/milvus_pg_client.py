"""milvus_pg_client.py
A Milvus client wrapper that synchronizes write operations to PostgreSQL for
validation purposes, mirroring the behaviour of the original DuckDB version.

This module provides a MilvusPGClient class that extends the standard MilvusClient
to provide synchronized write operations between Milvus and PostgreSQL databases.
It ensures data consistency by maintaining shadow copies in PostgreSQL for
validation and comparison purposes.

Key Features:
- Synchronized insert/upsert/delete operations
- Automatic schema mapping between Milvus and PostgreSQL
- Entity comparison for data validation
- Thread-safe operations with locking
- Optional vector field handling
"""

from __future__ import annotations

import ast
import json
import re
import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import wraps
from multiprocessing import cpu_count
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from deepdiff import DeepDiff
from psycopg2 import pool
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values
from pymilvus import CollectionSchema, DataType, MilvusClient, connections

from .exceptions import ConnectionError as MilvusPGConnectionError
from .exceptions import FilterConversionError, SyncError
from .lmdb_manager import LMDBManager, PKOperation, PKStatus
from .logger_config import logger
from .types import (
    EntityData,
    FilterExpression,
    OutputFields,
    PrimaryKeyList,
)

__all__ = ["MilvusPGClient"]


def _generate_detailed_diff_info(milvus_df: pd.DataFrame, pg_df: pd.DataFrame) -> list[str]:
    """
    Generate detailed difference information for debugging.

    This function analyzes differences between aligned DataFrames and generates
    detailed row-level and column-level difference information.

    Parameters
    ----------
    milvus_df : pd.DataFrame
        Aligned DataFrame from Milvus
    pg_df : pd.DataFrame
        Aligned DataFrame from PostgreSQL

    Returns
    -------
    list[str]
        List of detailed difference messages
    """
    diff_info = []

    # Row-by-row differences
    diff_info.append("--- ROW-BY-ROW DIFFERENCES ---")
    # Identify rows where any column differs
    diff_mask = ~(milvus_df.eq(pg_df))
    for pk, row_mask in diff_mask.iterrows():
        if row_mask.any():
            diff_info.append(f"Primary Key: {pk} has row-level differences:")
            for col, is_diff in row_mask.items():
                if is_diff:
                    m_val = milvus_df.at[pk, col]
                    p_val = pg_df.at[pk, col]
                    diff_info.append(f"  Column '{col}': milvus={m_val}, pg={p_val}")
    diff_info.append("=== END ROW-BY-ROW DIFFERENCES ===")

    return diff_info


def _compare_batch_worker(
    batch_start: int,
    batch_pks: list,
    batch_size: int,
    total_pks: int,
    collection_name: str,
    primary_field: str,
    ignore_vector: bool,
    float_vector_fields: list,
    json_fields: list,
    array_fields: list,
    varchar_fields: list,
    pg_conn_str: str,
    milvus_uri: str,
    milvus_token: str,
    sample_vector: bool = False,
    vector_sample_size: int = 8,
    vector_precision_decimals: int = 1,
    vector_comparison_significant_digits: int = 1,
) -> tuple[int, bool, list[str]]:
    """
    Worker function for multiprocessing batch comparison.

    This function is separate from the class to make it picklable for multiprocessing.
    """
    import pandas as pd
    import psycopg2
    from pymilvus import MilvusClient

    batch_num = batch_start // batch_size + 1

    try:
        # Create Milvus client for this process
        milvus_client = MilvusClient(uri=milvus_uri, token=milvus_token)

        # Milvus fetch
        milvus_filter = f"{primary_field} in {list(batch_pks)}"
        milvus_data = milvus_client.query(collection_name, filter=milvus_filter, output_fields=["*"])
        milvus_df = pd.DataFrame(milvus_data)

        # Drop vector columns for comparison if ignoring vectors
        if ignore_vector and float_vector_fields:
            milvus_df.drop(
                columns=[c for c in float_vector_fields if c in milvus_df.columns],
                inplace=True,
                errors="ignore",
            )

        # PG fetch - use a new connection for process safety
        pg_conn = psycopg2.connect(pg_conn_str)
        try:
            placeholder = ", ".join(["%s"] * len(batch_pks))
            with pg_conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {collection_name} WHERE {primary_field} IN ({placeholder});",
                    batch_pks,
                )
                pg_rows = cursor.fetchall()
                colnames = [desc[0] for desc in cursor.description] if cursor.description else []
            pg_df = pd.DataFrame(pg_rows, columns=colnames or [])
        finally:
            pg_conn.close()

        # Compare data - create minimal comparison logic
        # Align DataFrames for comparison
        if primary_field and primary_field in milvus_df.columns and primary_field not in milvus_df.index.names:
            milvus_df.set_index(primary_field, inplace=True)
        if primary_field and primary_field in pg_df.columns and primary_field not in pg_df.index.names:
            pg_df.set_index(primary_field, inplace=True)

        common_cols = [c for c in milvus_df.columns if c in pg_df.columns]
        if common_cols:
            milvus_df = milvus_df.loc[:, common_cols]
            pg_df = pg_df.loc[:, common_cols]

        # JSON fields normalization
        for field in json_fields:
            if field in pg_df.columns:
                pg_df[field] = pg_df[field].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x and x[0] in ["{", "[", '"'] else x
                )
            if field in milvus_df.columns:
                milvus_df[field] = milvus_df[field].apply(
                    lambda x: x
                    if isinstance(x, dict)
                    else json.loads(x)
                    if isinstance(x, str) and x and x[0] in ["{", "[", '"']
                    else x
                )

        # Array and vector fields normalization
        def _to_py_list(val: Any, round_floats: bool = False, precision: int = vector_precision_decimals) -> Any:
            """Ensure value is list of Python scalars (convert numpy types)."""
            if val is None:
                return val
            lst = list(val) if not isinstance(val, list) else val
            cleaned = []
            for item in lst:
                import numpy as np

                if isinstance(item, (np.floating | float)):
                    f_item = float(item)
                    if round_floats:
                        cleaned.append(round(f_item, precision))
                    else:
                        cleaned.append(f_item)
                elif isinstance(item, np.integer):
                    cleaned.append(int(item))
                else:
                    cleaned.append(item)
            return cleaned

        for field in array_fields:
            if field in milvus_df.columns:
                milvus_df[field] = milvus_df[field].apply(_to_py_list)
            if field in pg_df.columns:
                pg_df[field] = pg_df[field].apply(_to_py_list)

        for field in float_vector_fields:
            if field in milvus_df.columns:
                if sample_vector and not ignore_vector:
                    # Apply sampling to Milvus vectors for consistent comparison
                    def sample_vector_func(vector: list) -> list:
                        if not isinstance(vector, list) or len(vector) == 0:
                            return vector
                        vector_len = len(vector)
                        if vector_len <= vector_sample_size:
                            return vector

                        indices = [0]  # Always include first
                        step = (vector_len - 1) / (vector_sample_size - 1)
                        for i in range(1, vector_sample_size - 1):
                            idx = int(round(i * step))
                            indices.append(idx)
                        indices.append(vector_len - 1)  # Always include last

                        return [vector[i] for i in indices]

                    milvus_df[field] = milvus_df[field].apply(
                        lambda x: sample_vector_func(_to_py_list(x, round_floats=True))
                    )
                else:
                    milvus_df[field] = milvus_df[field].apply(_to_py_list, round_floats=True)
            if field in pg_df.columns:
                pg_df[field] = pg_df[field].apply(_to_py_list, round_floats=True)

        # Remove vector columns if ignoring them
        if ignore_vector and float_vector_fields:
            milvus_df.drop(
                columns=[c for c in float_vector_fields if c in milvus_df.columns], inplace=True, errors="ignore"
            )
            pg_df.drop(columns=[c for c in float_vector_fields if c in pg_df.columns], inplace=True, errors="ignore")

        shared_idx = milvus_df.index.intersection(pg_df.index)
        milvus_aligned = milvus_df.loc[shared_idx].sort_index()
        pg_aligned = pg_df.loc[shared_idx].sort_index()

        milvus_aligned = milvus_aligned.reindex(columns=pg_aligned.columns)
        pg_aligned = pg_aligned.reindex(columns=milvus_aligned.columns)

        # Direct comparison using DeepDiff with configurable precision
        t0 = time.time()
        milvus_dict = milvus_aligned.to_dict(orient="records")
        pg_dict = pg_aligned.to_dict(orient="records")

        # Use DeepDiff with configurable significant_digits for tolerance control
        # Lower significant_digits = higher tolerance (e.g., 1 = compare only 1st decimal place)
        diff = DeepDiff(
            milvus_dict,
            pg_dict,
            ignore_order=True,
            significant_digits=vector_comparison_significant_digits,
            view="tree",
        )
        has_differences = bool(diff)
        tt = time.time() - t0

        if has_differences:
            logger.debug(f"Differences found in batch {batch_num}")
            logger.info(f"Differences found in batch {batch_num}: {diff}")

            # Generate detailed diff information
            detailed_diff = _generate_detailed_diff_info(milvus_aligned, pg_aligned)

        else:
            logger.debug(f"No differences found in batch {batch_num}")
            detailed_diff = []

        logger.debug(f"Time taken to compare batch {batch_num} using DeepDiff: {tt} seconds")
        return len(batch_pks), has_differences, detailed_diff

    except Exception as e:
        # Use print instead of logger since we're in a separate process
        print(f"Error comparing batch {batch_num}: {e}")
        return len(batch_pks), True, []


class MilvusPGClient(MilvusClient):
    """Milvus client with synchronous PostgreSQL shadow writes for validation.

    This client extends the standard MilvusClient to provide synchronized operations
    between Milvus and PostgreSQL databases. All write operations are performed
    on both systems within transactions to ensure consistency.

    Parameters
    ----------
    pg_conn_str: str
        PostgreSQL connection string in libpq URI or keyword format.
    uri: str, optional
        Milvus server uri, passed through to :class:`pymilvus.MilvusClient`.
    token: str, optional
        Auth token for Milvus authentication.
    ignore_vector: bool, optional
        If True, skip handling FLOAT_VECTOR fields in PostgreSQL operations and comparisons.
        This is useful when PostgreSQL is only used for metadata validation.
    sample_vector: bool, optional
        If True, sample only a subset of values from each vector for PostgreSQL storage.
        This improves performance while still allowing validation. Default is False.
    vector_sample_size: int, optional
        Number of values to sample from each vector when sample_vector is True.
        Default is 8.
    vector_comparison_significant_digits: int, optional
        Number of significant digits for vector comparisons. Lower values = higher tolerance.
        Default is 1 (compare only 1st decimal place).

    Attributes
    ----------
    pg_pool : psycopg2.pool.ThreadedConnectionPool
        PostgreSQL connection pool
    ignore_vector : bool
        Whether to ignore vector fields in PostgreSQL operations
    primary_field : str
        Name of the primary key field
    fields_name_list : list[str]
        List of all field names in the collection
    json_fields : list[str]
        List of JSON field names
    array_fields : list[str]
        List of ARRAY field names
    varchar_fields : list[str]
        List of VARCHAR field names
    float_vector_fields : list[str]
        List of FLOAT_VECTOR field names
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Extract custom parameters before calling parent constructor
        self.ignore_vector: bool = kwargs.pop("ignore_vector", False)
        self.sample_vector: bool = kwargs.pop("sample_vector", False)
        self.vector_sample_size: int = kwargs.pop("vector_sample_size", 8)
        self.vector_precision_decimals: int = kwargs.pop("vector_precision_decimals", 3)
        self.vector_comparison_significant_digits: int = kwargs.pop("vector_comparison_significant_digits", 1)
        self.pg_conn_str: str = kwargs.pop("pg_conn_str")
        self.enable_lmdb: bool = kwargs.pop("enable_lmdb", True)
        self.lmdb_path: str | None = kwargs.pop("lmdb_path", None)
        self.lmdb_map_size: int = kwargs.pop("lmdb_map_size", 10 * 1024 * 1024 * 1024)
        uri = kwargs.get("uri", "")
        token = kwargs.get("token", "")

        # Initialize parent MilvusClient
        super().__init__(*args, **kwargs)
        self.uri = uri
        self.token = token

        # Connect to Milvus
        logger.debug(f"Connecting to Milvus with URI: {uri}")
        logger.debug(f"Connecting to PostgreSQL with connection string: {self.pg_conn_str}")
        connections.connect(uri=uri, token=token)

        # Connect to PostgreSQL with connection pooling
        logger.info("Initializing PostgreSQL connection pool")
        try:
            self.pg_pool = pool.ThreadedConnectionPool(minconn=2, maxconn=20, dsn=self.pg_conn_str)
            logger.debug("PostgreSQL connection pool established successfully")

            # Test connection
            test_conn = self.pg_pool.getconn()
            test_conn.autocommit = False
            self.pg_pool.putconn(test_conn)

        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise MilvusPGConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

        # Initialize schema-related caches
        self.primary_field: str = ""
        self.fields_name_list: list[str] = []
        self.json_fields: list[str] = []
        self.array_fields: list[str] = []
        self.varchar_fields: list[str] = []
        self.float_vector_fields: list[str] = []

        # Schema cache to avoid repeated schema fetching
        self._schema_cache: dict[str, CollectionSchema] = {}
        self._current_collection: str = ""

        # Thread synchronization lock for write operations
        self._lock: threading.Lock = threading.Lock()

        # Initialize LMDB manager if enabled
        self.lmdb_manager: LMDBManager | None = None
        if self.enable_lmdb:
            logger.info("Initializing LMDB manager for primary key tracking")
            self.lmdb_manager = LMDBManager(db_path=self.lmdb_path, map_size=self.lmdb_map_size)
            self.lmdb_manager.connect()

    @contextmanager
    def _get_pg_connection(self) -> Generator[PGConnection, None, None]:
        """Context manager for getting PostgreSQL connections from pool."""
        conn = None
        try:
            conn = self.pg_pool.getconn()
            conn.autocommit = False
            yield conn
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.pg_pool.putconn(conn)

    def _sample_vector(self, vector: list | np.ndarray) -> list:
        """
        Sample a subset of values from a vector for PostgreSQL storage.

        This method implements a deterministic sampling strategy that:
        - Always includes the first and last values
        - Evenly samples the remaining values across the vector

        Parameters
        ----------
        vector : list | np.ndarray
            The input vector to sample

        Returns
        -------
        list
            Sampled vector with self.vector_sample_size values
        """
        if not isinstance(vector, list | np.ndarray) or len(vector) == 0:
            return list(vector) if isinstance(vector, np.ndarray) else vector

        vector_len = len(vector)
        sample_size = self.vector_sample_size

        # If vector is smaller than sample size, return as is
        if vector_len <= sample_size:
            return list(vector)

        # Deterministic sampling: always include first and last, evenly sample the rest
        indices = [0]  # Always include first

        # Calculate step size for even sampling
        step = (vector_len - 1) / (sample_size - 1)

        # Sample intermediate values
        for i in range(1, sample_size - 1):
            idx = int(round(i * step))
            indices.append(idx)

        indices.append(vector_len - 1)  # Always include last

        # Extract sampled values
        if isinstance(vector, np.ndarray):
            sampled = vector[indices].tolist()
        else:
            sampled = [vector[i] for i in indices]

        return sampled  # type: ignore[no-any-return]

    def _round_vector_precision(self, vector: list | np.ndarray) -> list:
        """
        Round vector values to specified decimal places to ensure consistent precision.

        This method helps eliminate floating-point precision inconsistencies between
        Milvus and PostgreSQL by rounding all vector values to a fixed number of
        decimal places during insertion.

        Parameters
        ----------
        vector : list | np.ndarray
            The input vector to round

        Returns
        -------
        list
            Vector with rounded values
        """
        if not isinstance(vector, list | np.ndarray) or len(vector) == 0:
            return list(vector) if isinstance(vector, np.ndarray) else vector

        # Round each value to specified decimal places
        if isinstance(vector, np.ndarray):
            rounded = [round(float(val), self.vector_precision_decimals) for val in vector]
        else:
            rounded = [round(float(val), self.vector_precision_decimals) for val in vector]

        return rounded

    def _serialize_special_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Serialize JSON and ARRAY fields to string format for PostgreSQL storage.
        Also applies vector sampling if enabled.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data to serialize

        Returns
        -------
        pd.DataFrame
            DataFrame with JSON and ARRAY fields serialized to strings and vectors sampled
        """
        # Create a copy to avoid modifying the original DataFrame
        df_copy = df.copy()

        # Serialize JSON fields
        for field in self.json_fields:
            if field in df_copy.columns:
                df_copy[field] = df_copy[field].apply(json.dumps)
                logger.debug(f"Serialized JSON field: {field}")

        # Serialize ARRAY fields
        for field in self.array_fields:
            if field in df_copy.columns:
                df_copy[field] = df_copy[field].apply(json.dumps)
                logger.debug(f"Serialized ARRAY field: {field}")

        # Process vector fields if not ignored
        if not self.ignore_vector:
            for field in self.float_vector_fields:
                if field in df_copy.columns:
                    # Apply precision control to vector values
                    df_copy[field] = df_copy[field].apply(self._round_vector_precision)
                    logger.debug(
                        f"Applied precision control to vector field: {field} (decimals: {self.vector_precision_decimals})"
                    )

                    # Sample vector fields if enabled
                    if self.sample_vector:
                        df_copy[field] = df_copy[field].apply(self._sample_vector)
                        logger.debug(f"Sampled vector field: {field} to {self.vector_sample_size} values")

        return df_copy

    def _remove_vector_fields_if_ignored(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove vector columns from DataFrame if vector fields are being ignored.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to process

        Returns
        -------
        pd.DataFrame
            DataFrame with vector fields removed if ignore_vector is True
        """
        if self.ignore_vector and self.float_vector_fields:
            original_cols = df.columns.tolist()
            df = df.drop(columns=[c for c in self.float_vector_fields if c in df.columns], errors="ignore")
            if len(df.columns) < len(original_cols):
                dropped_fields = set(original_cols) - set(df.columns)
                logger.debug(f"Dropped vector fields from PostgreSQL operation: {dropped_fields}")
        return df

    def _fast_preprocess_data(self, data: list[dict]) -> tuple[list[str], list[tuple]]:
        """
        Fast data preprocessing without DataFrame conversion for small to medium datasets.

        This method processes data directly without creating intermediate DataFrames,
        significantly reducing memory usage and processing time.

        Parameters
        ----------
        data : list[dict]
            List of records to process

        Returns
        -------
        tuple[list[str], list[tuple]]
            Tuple of (column_names, processed_values)
        """
        if not data:
            return [], []

        # Get columns from first record, filter out vector fields if needed
        all_columns = list(data[0].keys())
        if self.ignore_vector and self.float_vector_fields:
            columns = [col for col in all_columns if col not in self.float_vector_fields]
        else:
            columns = all_columns

        # Process records directly
        values = []
        for record in data:
            processed_record = []
            for col in columns:
                value = record.get(col)

                # Serialize JSON fields
                if col in self.json_fields and value is not None:
                    if not isinstance(value, str):
                        value = json.dumps(value)

                # Serialize ARRAY fields
                elif col in self.array_fields and value is not None:
                    if not isinstance(value, str):
                        value = json.dumps(value)

                # Process vector fields if not ignored
                elif col in self.float_vector_fields and value is not None and not self.ignore_vector:
                    # Apply precision control to vector values
                    value = self._round_vector_precision(value)
                    # Sample vector fields if enabled
                    if self.sample_vector:
                        value = self._sample_vector(value)

                processed_record.append(value)
            values.append(tuple(processed_record))

        logger.debug(f"Fast preprocessed {len(values)} records with {len(columns)} columns")
        return columns, values

    def _stream_process_large_data(
        self, data: list[dict], batch_size: int = 1000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Process large datasets in streaming fashion to reduce memory usage.

        This method processes data in batches to avoid loading the entire dataset
        into memory at once, which is crucial for large-scale operations.

        Parameters
        ----------
        data : list[dict]
            List of records to process
        batch_size : int, optional
            Number of records to process per batch, by default 1000

        Yields
        ------
        pd.DataFrame
            Processed DataFrame batches with serialized fields
        """
        logger.debug(f"Starting streaming processing of {len(data)} records with batch size {batch_size}")

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_df = pd.DataFrame(batch)

            # Process the batch with serialization and field removal
            processed_df = self._serialize_special_fields(batch_df)
            processed_df = self._remove_vector_fields_if_ignored(processed_df)

            logger.debug(f"Processed batch {i // batch_size + 1}: {len(batch)} records")
            yield processed_df

    def _calculate_optimal_batch_size(self, data_size_estimate: int, record_count: int) -> int:
        """
        Calculate optimal batch size based on data characteristics.

        Parameters
        ----------
        data_size_estimate : int
            Estimated total data size in bytes
        record_count : int
            Number of records

        Returns
        -------
        int
            Optimal batch size for the given data characteristics
        """
        if record_count == 0:
            return 1000

        # Target batch size: ~1MB per batch for PostgreSQL optimal performance
        target_batch_bytes = 1024 * 1024
        avg_record_size = data_size_estimate / record_count
        optimal_size = max(100, min(10000, int(target_batch_bytes / avg_record_size)))

        logger.debug(f"Calculated optimal batch size: {optimal_size} (avg record size: {avg_record_size:.0f} bytes)")
        return optimal_size

    # ---------------------------------------------------------------------
    # Schema and utility methods
    # ---------------------------------------------------------------------
    def _get_schema(self, collection_name: str) -> CollectionSchema:
        """
        Retrieve and cache schema information from Milvus for a collection.

        This method fetches the collection schema from Milvus and populates
        internal field lists for efficient field type checking during operations.
        Uses caching to avoid repeated schema fetching for the same collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection to get schema for

        Returns
        -------
        CollectionSchema
            The collection schema from Milvus
        """
        # Check if schema is already cached and collection hasn't changed
        if collection_name in self._schema_cache and self._current_collection == collection_name:
            logger.debug(f"Using cached schema for collection: {collection_name}")
            return self._schema_cache[collection_name]

        logger.debug(f"Fetching schema for collection: {collection_name}")
        temp_client = MilvusClient(uri=self.uri, token=self.token)
        schema_info = temp_client.describe_collection(collection_name)
        schema = CollectionSchema.construct_from_dict(schema_info)

        # Cache the schema
        self._schema_cache[collection_name] = schema
        self._current_collection = collection_name

        # Update field caches for the current collection
        self._update_field_caches(schema)

        logger.debug(
            f"Schema cached - Primary field: {self.primary_field}, "
            f"Fields: {len(self.fields_name_list)}, "
            f"JSON: {len(self.json_fields)}, "
            f"Arrays: {len(self.array_fields)}, "
            f"Vectors: {len(self.float_vector_fields)}"
        )
        return schema

    def _update_field_caches(self, schema: CollectionSchema) -> None:
        """
        Update internal field type caches based on schema.

        Parameters
        ----------
        schema : CollectionSchema
            The collection schema to process
        """
        # Reset field caches
        self.primary_field = ""
        self.fields_name_list.clear()
        self.json_fields.clear()
        self.array_fields.clear()
        self.varchar_fields.clear()
        self.float_vector_fields.clear()

        # Populate field type lists for efficient lookup
        for field in schema.fields:
            self.fields_name_list.append(field.name)
            if field.is_primary:
                self.primary_field = field.name
            if field.dtype == DataType.FLOAT_VECTOR:
                self.float_vector_fields.append(field.name)
            if field.dtype == DataType.ARRAY:
                self.array_fields.append(field.name)
            if field.dtype == DataType.JSON:
                self.json_fields.append(field.name)
            if field.dtype == DataType.VARCHAR:
                self.varchar_fields.append(field.name)

    @staticmethod
    def _milvus_dtype_to_pg(milvus_type: DataType) -> str:
        """
        Map Milvus DataType to equivalent PostgreSQL type.

        Parameters
        ----------
        milvus_type : DataType
            Milvus field data type

        Returns
        -------
        str
            Corresponding PostgreSQL data type string
        """
        mapping = {
            DataType.BOOL: "BOOLEAN",
            DataType.INT8: "SMALLINT",
            DataType.INT16: "SMALLINT",
            DataType.INT32: "INTEGER",
            DataType.INT64: "BIGINT",
            DataType.FLOAT: "REAL",
            DataType.DOUBLE: "DOUBLE PRECISION",
            DataType.VARCHAR: "VARCHAR",
            DataType.JSON: "JSONB",
            DataType.FLOAT_VECTOR: "DOUBLE PRECISION[]",
            DataType.ARRAY: "JSONB",  # Fallback – store as JSON if unknown element type
        }
        return mapping.get(milvus_type, "TEXT")

    # ------------------------------------------------------------------
    # Collection DDL operations
    # ------------------------------------------------------------------
    def create_collection(self, collection_name: str, schema: CollectionSchema, **kwargs: Any) -> Any:
        """
        Create a collection in both PostgreSQL and Milvus.

        This method creates the collection schema in PostgreSQL first, then creates
        the corresponding collection in Milvus. If vector fields are ignored, they
        are excluded from the PostgreSQL table.

        Parameters
        ----------
        collection_name : str
            Name of the collection to create
        schema : CollectionSchema
            Milvus collection schema
        **kwargs : Any
            Additional arguments passed to Milvus create_collection

        Returns
        -------
        Any
            Result from Milvus create_collection operation
        """
        logger.info(f"Creating collection '{collection_name}' in PostgreSQL and Milvus")

        # Build PostgreSQL CREATE TABLE SQL based on Milvus schema
        cols_sql = []
        for f in schema.fields:
            # Skip vector fields in PostgreSQL if requested
            if self.ignore_vector and f.dtype == DataType.FLOAT_VECTOR:
                logger.debug(f"Skipping vector field '{f.name}' in PostgreSQL table")
                continue
            pg_type = self._milvus_dtype_to_pg(f.dtype)
            col_def = f"{f.name} {pg_type}"
            if f.is_primary:
                col_def += " PRIMARY KEY"
            cols_sql.append(col_def)

        create_sql = f"CREATE TABLE IF NOT EXISTS {collection_name} ({', '.join(cols_sql)});"
        logger.debug(f"PostgreSQL CREATE TABLE SQL: {create_sql}")

        # Create PostgreSQL table
        try:
            pg_start = time.time()
            with self._get_pg_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_sql)
                conn.commit()
            logger.debug(f"PostgreSQL table created in {time.time() - pg_start:.3f}s")
        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            logger.error(f"Failed to create PostgreSQL table: {e}")
            raise SyncError(f"Failed to create PostgreSQL table: {e}") from e

        # Create collection in Milvus
        # Pass schema as keyword argument to align with MilvusClient signature
        milvus_start = time.time()
        try:
            result = super().create_collection(collection_name, schema=schema, consistency_level="Strong", **kwargs)
            logger.debug(f"Milvus collection created in {time.time() - milvus_start:.3f}s")
            return result
        except Exception as e:
            logger.error(f"Failed to create Milvus collection: {e}")
            raise

    def drop_collection(self, collection_name: str) -> Any:
        """
        Drop a collection from both PostgreSQL and Milvus.

        Parameters
        ----------
        collection_name : str
            Name of the collection to drop

        Returns
        -------
        Any
            Result from Milvus drop_collection operation
        """
        logger.info(f"Dropping collection '{collection_name}' from PostgreSQL and Milvus")

        # Drop PostgreSQL table
        try:
            start = time.time()
            with self._get_pg_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"DROP TABLE IF EXISTS {collection_name};")
                conn.commit()
            logger.debug(f"PostgreSQL table dropped in {time.time() - start:.3f}s")
        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            logger.error(f"Failed to drop PostgreSQL table: {e}")
            raise SyncError(f"Failed to drop PostgreSQL table: {e}") from e

        # Clear LMDB data for this collection if enabled
        if self.lmdb_manager:
            try:
                count = self.lmdb_manager.clear_collection(collection_name)
                logger.debug(f"Cleared {count} LMDB entries for collection '{collection_name}'")
            except Exception as e:
                logger.error(f"Failed to clear LMDB data: {e}")
                # Continue with Milvus drop even if LMDB clear fails

        # Drop Milvus collection
        milvus_start = time.time()
        try:
            result = super().drop_collection(collection_name)
            logger.debug(f"Milvus collection dropped in {time.time() - milvus_start:.3f}s")
            return result
        except Exception as e:
            logger.error(f"Failed to drop Milvus collection: {e}")
            raise

    # ------------------------------------------------------------------
    # Write operations with transactional shadow writes
    # ------------------------------------------------------------------
    @staticmethod
    def _synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to run method under instance-level lock for thread safety."""

        @wraps(method)
        def _wrapper(self: MilvusPGClient, *args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return method(self, *args, **kwargs)

        return _wrapper

    @_synchronized
    def insert(self, collection_name: str, data: EntityData, **kwargs: Any) -> Any:
        """
        Insert data into both Milvus and PostgreSQL within a transaction.

        This method performs synchronized insert operations on both databases.
        PostgreSQL operations are executed first within a transaction, and if
        successful, the Milvus insert is performed. If the Milvus operation fails,
        the PostgreSQL transaction is rolled back.

        Parameters
        ----------
        collection_name : str
            Name of the target collection
        data : list[dict[str, Any]]
            List of records to insert
        **kwargs : Any
            Additional arguments passed to Milvus insert

        Returns
        -------
        Any
            Result from Milvus insert operation
        """
        self._get_schema(collection_name)
        logger.info(f"Inserting {len(data)} records into collection '{collection_name}'")
        logger.debug(f"Insert data sample: {data[0] if data else 'empty'}")

        # Use fast preprocessing for better performance
        # For datasets under 10K records, use direct processing without DataFrame
        # For larger datasets, fallback to streaming processing
        large_dataset_threshold = 10000

        if len(data) <= large_dataset_threshold:
            # Fast direct processing without DataFrame conversion
            logger.debug(f"Using fast preprocessing for {len(data)} records")
            columns, values = self._fast_preprocess_data(data)
        else:
            # Fallback to streaming processing for very large datasets
            logger.debug(f"Using streaming processing for large dataset ({len(data)} records)")
            import sys

            data_size_estimate = sys.getsizeof(data) + sum(sys.getsizeof(record) for record in data[:100])
            optimal_batch_size = self._calculate_optimal_batch_size(data_size_estimate, len(data))

            # Process data in streaming fashion
            all_values = []
            columns = None

            for batch_df in self._stream_process_large_data(data, optimal_batch_size):
                if columns is None:
                    columns = list(batch_df.columns)
                batch_values = [tuple(row) for row in batch_df.itertuples(index=False, name=None)]
                all_values.extend(batch_values)

            values = all_values

        # Build efficient batch INSERT SQL
        if columns is None:
            raise SyncError("No columns available for insert operation")
        insert_sql = f"INSERT INTO {collection_name} ({', '.join(columns)}) VALUES %s"
        logger.debug(f"Prepared {len(values)} rows for PostgreSQL batch insert")

        # Execute synchronized insert operations
        with self._get_pg_connection() as conn:
            try:
                # Execute PostgreSQL batch insert using execute_values for performance
                logger.debug("Starting PostgreSQL batch INSERT operation")
                t0 = time.time()
                with conn.cursor() as cursor:
                    # Use dynamic page_size based on data size for better performance
                    page_size = min(2000, max(500, len(values) // 10))
                    execute_values(cursor, insert_sql, values, page_size=page_size)
                    pg_duration = time.time() - t0
                    logger.debug(f"PostgreSQL batch INSERT completed: {cursor.rowcount} rows in {pg_duration:.3f}s")

                # Execute Milvus insert operation
                logger.debug("Starting Milvus insert operation")
                t0 = time.time()
                result = super().insert(collection_name, data, **kwargs)
                milvus_duration = time.time() - t0
                logger.debug(f"Milvus insert completed in {milvus_duration:.3f}s")

                # Sync to LMDB if enabled
                if self.lmdb_manager:
                    logger.debug("Syncing primary keys to LMDB")
                    pk_states = []
                    for record in data:
                        if self.primary_field in record:
                            pk_states.append((record[self.primary_field], PKStatus.EXISTS, PKOperation.INSERT))
                    if pk_states:
                        self.lmdb_manager.batch_record_pk_states(collection_name, pk_states)
                        logger.debug(f"LMDB sync completed for {len(pk_states)} primary keys")

                # Commit PostgreSQL transaction on successful Milvus insert
                conn.commit()
                logger.debug("PostgreSQL transaction committed successfully")
                return result

            except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
                logger.error(f"PostgreSQL insert failed for collection '{collection_name}': {e}")
                raise SyncError(f"PostgreSQL insert failed: {e}") from e
            except Exception as e:
                logger.error(f"Milvus insert failed for collection '{collection_name}', PostgreSQL rolled back: {e}")
                raise SyncError(f"Milvus insert failed, PostgreSQL rolled back: {e}") from e

    @_synchronized
    def upsert(self, collection_name: str, data: EntityData, **kwargs: Any) -> Any:
        """
        Upsert data into both Milvus and PostgreSQL within a transaction.

        This method performs synchronized upsert (insert or update) operations on both databases.
        Uses PostgreSQL's ON CONFLICT clause for efficient upsert operations.

        Parameters
        ----------
        collection_name : str
            Name of the target collection
        data : list[dict[str, Any]]
            List of records to upsert
        **kwargs : Any
            Additional arguments passed to Milvus upsert

        Returns
        -------
        Any
            Result from Milvus upsert operation
        """
        self._get_schema(collection_name)
        logger.info(f"Upserting {len(data)} records into collection '{collection_name}'")
        logger.debug(f"Upsert data sample: {data[0] if data else 'empty'}")

        # Use fast preprocessing for upsert as well
        columns, values = self._fast_preprocess_data(data)

        # Build PostgreSQL UPSERT SQL with ON CONFLICT clause
        updates = ", ".join([f"{col}=EXCLUDED.{col}" for col in columns])
        insert_sql = (
            f"INSERT INTO {collection_name} ({', '.join(columns)}) VALUES %s "
            f"ON CONFLICT ({self.primary_field}) DO UPDATE SET {updates}"
        )
        logger.debug(f"Prepared PostgreSQL upsert with conflict resolution on '{self.primary_field}'")

        # Execute synchronized upsert operations
        with self._get_pg_connection() as conn:
            try:
                # Execute PostgreSQL batch upsert
                logger.debug("Starting PostgreSQL batch UPSERT operation")
                t0 = time.time()
                with conn.cursor() as cursor:
                    # Use dynamic page_size based on data size for better performance
                    page_size = min(2000, max(500, len(values) // 10))
                    execute_values(cursor, insert_sql, values, page_size=page_size)
                    pg_duration = time.time() - t0
                    logger.debug(
                        f"PostgreSQL batch UPSERT completed: {cursor.rowcount} rows affected in {pg_duration:.3f}s"
                    )

                # Execute Milvus upsert operation
                logger.debug("Starting Milvus upsert operation")
                t0 = time.time()
                result = super().upsert(collection_name, data, **kwargs)
                milvus_duration = time.time() - t0
                logger.debug(f"Milvus upsert completed in {milvus_duration:.3f}s")

                # Sync to LMDB if enabled
                if self.lmdb_manager:
                    logger.debug("Syncing primary keys to LMDB for upsert")
                    pk_states = []
                    for record in data:
                        if self.primary_field in record:
                            pk_states.append((record[self.primary_field], PKStatus.EXISTS, PKOperation.UPSERT))
                    if pk_states:
                        self.lmdb_manager.batch_record_pk_states(collection_name, pk_states)
                        logger.debug(f"LMDB sync completed for {len(pk_states)} primary keys")

                # Commit PostgreSQL transaction on successful Milvus upsert
                conn.commit()
                logger.debug("PostgreSQL transaction committed successfully")
                return result

            except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
                logger.error(f"PostgreSQL upsert failed for collection '{collection_name}': {e}")
                raise SyncError(f"PostgreSQL upsert failed: {e}") from e
            except Exception as e:
                logger.error(f"Milvus upsert failed for collection '{collection_name}', PostgreSQL rolled back: {e}")
                raise SyncError(f"Milvus upsert failed, PostgreSQL rolled back: {e}") from e

    @_synchronized
    def delete(self, collection_name: str, ids: PrimaryKeyList, **kwargs: Any) -> Any:
        """
        Delete records from both Milvus and PostgreSQL within a transaction.

        Parameters
        ----------
        collection_name : str
            Name of the target collection
        ids : list[int | str]
            List of primary key values to delete
        **kwargs : Any
            Additional arguments passed to Milvus delete

        Returns
        -------
        Any
            Result from Milvus delete operation
        """
        self._get_schema(collection_name)
        logger.info(f"Deleting {len(ids)} records from collection '{collection_name}'")
        logger.debug(f"Delete IDs sample: {ids[:5] if len(ids) > 5 else ids}")

        # Build PostgreSQL DELETE SQL with IN clause
        placeholder = ", ".join(["%s"] * len(ids))
        delete_sql = f"DELETE FROM {collection_name} WHERE {self.primary_field} IN ({placeholder});"

        # Execute synchronized delete operations
        with self._get_pg_connection() as conn:
            try:
                # Execute PostgreSQL delete
                logger.debug("Starting PostgreSQL DELETE operation")
                t0 = time.time()
                with conn.cursor() as cursor:
                    cursor.execute(delete_sql, ids)
                    pg_duration = time.time() - t0
                    logger.debug(f"PostgreSQL DELETE completed: {cursor.rowcount} rows deleted in {pg_duration:.3f}s")

                # Execute Milvus delete
                logger.debug("Starting Milvus delete operation")
                t0 = time.time()
                result = super().delete(collection_name, ids=ids, **kwargs)
                milvus_duration = time.time() - t0
                logger.debug(f"Milvus delete completed in {milvus_duration:.3f}s")

                # Sync to LMDB if enabled
                if self.lmdb_manager:
                    logger.debug("Syncing primary keys to LMDB for delete")
                    pk_states = [(pk, PKStatus.DELETED, PKOperation.DELETE) for pk in ids]
                    if pk_states:
                        self.lmdb_manager.batch_record_pk_states(collection_name, pk_states)
                        logger.debug(f"LMDB sync completed for {len(pk_states)} primary keys")

                # Commit PostgreSQL transaction on successful Milvus delete
                conn.commit()
                logger.debug("PostgreSQL transaction committed successfully")
                return result

            except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
                logger.error(f"PostgreSQL delete failed for collection '{collection_name}': {e}")
                raise SyncError(f"PostgreSQL delete failed: {e}") from e
            except Exception as e:
                logger.error(f"Milvus delete failed for collection '{collection_name}', PostgreSQL rolled back: {e}")
                raise SyncError(f"Milvus delete failed, PostgreSQL rolled back: {e}") from e

    # ------------------------------------------------------------------
    # Read operations and validation helpers
    # ------------------------------------------------------------------
    @_synchronized
    def query(
        self,
        collection_name: str,
        filter: FilterExpression = "",
        output_fields: OutputFields | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query data from both Milvus and PostgreSQL for comparison.

        This method performs parallel queries on both databases and returns
        aligned DataFrames for comparison purposes.

        Parameters
        ----------
        collection_name : str
            Name of the collection to query
        filter : str, optional
            Filter expression in Milvus syntax (converted to SQL for PostgreSQL)
        output_fields : list[str] | None, optional
            List of fields to return. If None, returns all fields.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple of (milvus_dataframe, postgresql_dataframe) with aligned data
        """
        # Handle mutable default argument
        if output_fields is None:
            output_fields = ["*"]

        logger.debug(f"Querying collection '{collection_name}' with filter: '{filter}'")
        logger.debug(f"Output fields: {output_fields}")

        # Fetch from Milvus
        logger.debug("Executing Milvus query")
        t0 = time.time()
        milvus_res = super().query(collection_name, filter=filter, output_fields=output_fields)
        milvus_df = pd.DataFrame(milvus_res)
        milvus_duration = time.time() - t0
        logger.debug(f"Milvus query completed: {len(milvus_df)} rows in {milvus_duration:.3f}s")

        # Convert Milvus filter to PostgreSQL SQL
        sql_filter = self._milvus_filter_to_sql(filter) if filter else "TRUE"
        cols = ", ".join(output_fields)
        pg_sql = f"SELECT {cols} FROM {collection_name} WHERE {sql_filter};"
        logger.debug(f"PostgreSQL query SQL: {pg_sql}")

        # Fetch from PostgreSQL
        logger.debug("Executing PostgreSQL query")
        t0 = time.time()
        with self._get_pg_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(pg_sql)
                pg_rows = cursor.fetchall()
                # Handle column names for DataFrame construction
                if cursor.description:
                    colnames = [str(desc[0]) for desc in cursor.description]
                    pg_df = pd.DataFrame(pg_rows, columns=colnames)
                else:
                    pg_df = pd.DataFrame(pg_rows)
        pg_duration = time.time() - t0
        logger.debug(f"PostgreSQL query completed: {len(pg_df)} rows in {pg_duration:.3f}s")

        # Align DataFrames for comparison
        milvus_aligned, pg_aligned = self._align_df(milvus_df, pg_df)
        return milvus_aligned, pg_aligned

    @_synchronized
    def export(self, collection_name: str) -> pd.DataFrame:
        """
        Export all data from PostgreSQL table as DataFrame.

        Parameters
        ----------
        collection_name : str
            Name of the collection to export

        Returns
        -------
        pd.DataFrame
            DataFrame containing all records from PostgreSQL table
        """
        logger.debug(f"Exporting all data from PostgreSQL table '{collection_name}'")
        t0 = time.time()

        with self._get_pg_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {collection_name};")
                rows = cursor.fetchall()
                colnames = [desc[0] for desc in cursor.description] if cursor.description else []

        result = [dict(zip(colnames, r, strict=False)) for r in rows]
        df = pd.DataFrame(result)

        duration = time.time() - t0
        logger.debug(f"Export completed: {len(df)} rows in {duration:.3f}s")
        return df

    @_synchronized
    def count(self, collection_name: str) -> dict[str, int]:
        """
        Get record counts from both Milvus and PostgreSQL.

        Parameters
        ----------
        collection_name : str
            Name of the collection to count

        Returns
        -------
        dict
            Dictionary with 'milvus_count' and 'pg_count' keys
        """
        logger.debug(f"Getting record counts for collection '{collection_name}'")

        # Get Milvus count
        try:
            logger.debug("Querying Milvus count")
            milvus_count_res = super().query(collection_name, filter="", output_fields=["count(*)"])
            milvus_count = milvus_count_res[0]["count(*)"] if milvus_count_res else 0
            logger.debug(f"Milvus count: {milvus_count}")
        except Exception as e:
            logger.error(f"Failed to query Milvus count for collection '{collection_name}': {e}")
            milvus_count = 0

        # Get PostgreSQL count
        try:
            logger.debug("Querying PostgreSQL count")
            # Check if table exists first
            with self._get_pg_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s;",
                        (collection_name,),
                    )
                    result = cursor.fetchone()
                    table_exists = result[0] > 0 if result else False

                    if table_exists:
                        cursor.execute(f"SELECT COUNT(*) FROM {collection_name};")
                        result = cursor.fetchone()
                        pg_count = int(result[0]) if result else 0
                        logger.debug(f"PostgreSQL count: {pg_count}")
                    else:
                        logger.error(f"PostgreSQL table '{collection_name}' does not exist")
                        pg_count = 0
        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            logger.error(f"Failed to query PostgreSQL count for collection '{collection_name}': {e}")
            pg_count = 0

        return {"milvus_count": milvus_count, "pg_count": pg_count}

    # ------------------------------------------------------------------
    # Data comparison and validation methods
    # ------------------------------------------------------------------
    def _compare_df(self, milvus_df: pd.DataFrame, pg_df: pd.DataFrame) -> DeepDiff:
        """
        Compare two DataFrames using DeepDiff for detailed difference detection.

        This method aligns the DataFrames first to ensure identical structure,
        then uses DeepDiff to detect any differences with tolerance for floating
        point precision.

        Parameters
        ----------
        milvus_df : pd.DataFrame
            DataFrame from Milvus query
        pg_df : pd.DataFrame
            DataFrame from PostgreSQL query

        Returns
        -------
        DeepDiff
            Difference object containing detected differences, empty if identical
        """
        # Align DataFrames to ensure identical structure
        milvus_aligned, pg_aligned = self._align_df(milvus_df, pg_df)

        # First try pandas DataFrame.equals() for fast comparison
        if milvus_aligned.equals(pg_aligned):
            # Return empty DeepDiff object if DataFrames are equal
            return DeepDiff({}, {})

        # If not equal, use DeepDiff for detailed comparison
        # Convert to dictionaries for DeepDiff comparison
        milvus_dict = milvus_aligned.to_dict("list")
        pg_dict = pg_aligned.to_dict("list")

        # Use DeepDiff with tolerance for floating point differences
        diff = DeepDiff(
            milvus_dict,
            pg_dict,
            ignore_order=True,  # Ignore row order differences
            significant_digits=1,  # Tolerance for floating point precision
        )

        # Print detailed differences for debugging if differences found
        if diff:
            self._print_detailed_diff(milvus_aligned, pg_aligned)

        return diff

    def _print_detailed_diff(self, milvus_df: pd.DataFrame, pg_df: pd.DataFrame) -> None:
        """
        Print detailed differences with primary key information for debugging.

        This method analyzes the DeepDiff result and prints specific row-level
        differences with primary key information to aid in debugging.

        Parameters
        ----------
        milvus_df : pd.DataFrame
            Aligned DataFrame from Milvus
        pg_df : pd.DataFrame
            Aligned DataFrame from PostgreSQL
        """
        # Use the shared helper function to generate detailed diff information
        detailed_diff = _generate_detailed_diff_info(milvus_df, pg_df)

        # Log each line of detailed diff information
        for diff_line in detailed_diff:
            logger.error(diff_line)

    def query_result_compare(
        self,
        collection_name: str,
        filter: FilterExpression = "",
        output_fields: OutputFields | None = None,
    ) -> DeepDiff:
        """
        Compare query results between Milvus and PostgreSQL.

        This method executes the same query on both databases and compares
        the results, logging any differences found.

        Parameters
        ----------
        collection_name : str
            Name of the collection to query
        filter : str, optional
            Filter expression to apply
        output_fields : list[str] | None, optional
            Fields to include in the query

        Returns
        -------
        DeepDiff
            Difference object, empty if results match
        """
        logger.debug(f"Comparing query results for collection '{collection_name}'")

        # Execute queries on both databases
        milvus_df, pg_df = self.query(collection_name, filter=filter, output_fields=output_fields)

        # Log query results for debugging (loguru handles level checking efficiently)
        logger.debug(f"Milvus query result:\n{milvus_df}")
        logger.debug(f"PostgreSQL query result:\n{pg_df}")

        # Compare the results
        diff = self._compare_df(milvus_df, pg_df)

        if diff:
            logger.error(
                f"Query result mismatch for collection '{collection_name}' "
                f"with filter '{filter}' and output fields '{output_fields}'"
            )
            logger.error(f"Differences detected: {diff}")
        else:
            logger.debug(
                f"Query results match for collection '{collection_name}' "
                f"with filter '{filter}' and output fields '{output_fields}'"
            )
        return diff

    # ------------------------------------------------------------------
    # Internal helpers for query alignment
    # ------------------------------------------------------------------
    def _milvus_filter_to_sql(self, filter_expr: FilterExpression) -> str:
        """Convert simple Milvus filter expressions to PostgreSQL SQL WHERE clauses.

        This method provides basic filter conversion for common Milvus filter patterns.
        It's designed to handle filters generated by the sampling and testing utilities
        in this project, not arbitrary complex expressions.

        Supported Conversions:
        ----------------------
        1. Logical operators: 'and' -> 'AND', 'or' -> 'OR', 'not' -> 'NOT'
        2. Equality operators: '==' -> '='
        3. List membership: 'field in [1,2,3]' -> 'field IN (1,2,3)'
        4. String patterns: 'field LIKE "pattern"' -> 'field LIKE \'pattern\''
        5. Null checks: 'field is null' -> 'field IS NULL'
        6. JSON access: 'field["key"]' -> 'field->>\'key\''
        7. String literals: '"text"' -> '\'text\''

        Limitations:
        ------------
        - No support for complex nested expressions
        - No arithmetic operations
        - No subqueries or joins
        - Limited function support

        Parameters
        ----------
        filter_expr : FilterExpression
            Milvus filter expression string

        Returns
        -------
        str
            PostgreSQL-compatible SQL WHERE clause

        Raises
        ------
        FilterConversionError
            If the filter contains unsupported syntax or invalid list formats

        Examples
        --------
        >>> client._milvus_filter_to_sql('age > 18 and name == "John"')
        "age > 18 AND name = 'John'"

        >>> client._milvus_filter_to_sql('id in [1,2,3]')
        'id IN (1,2,3)'

        >>> client._milvus_filter_to_sql('metadata["category"] == "books"')
        "metadata->>'category' = 'books'"
        """
        import re

        if not filter_expr or filter_expr.strip() == "":
            return "TRUE"  # No filter

        expr = filter_expr

        # 1. Logical operators to uppercase
        expr = re.sub(r"\b(and)\b", "AND", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\b(or)\b", "OR", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\b(not)\b", "NOT", expr, flags=re.IGNORECASE)

        # 2. Equality '==' -> '='
        expr = re.sub(r"(?<![!<>])==", "=", expr)

        # 3. IN list: field in [1,2] -> field IN (1,2)
        def _in_repl(match: re.Match[str]) -> str:
            """Convert Python list syntax to SQL IN clause."""
            field = match.group(1)
            values = match.group(2)
            try:
                py_list = ast.literal_eval(values)
                if not isinstance(py_list, list):
                    raise ValueError("Expected list format")
            except (ValueError, SyntaxError) as e:
                raise FilterConversionError(filter_expr, f"Invalid list format: {values}") from e
            sql_list = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in py_list])
            return f"{field} IN ({sql_list})"

        expr = re.sub(r"(\w+)\s+in\s+(\[[^\]]*\])", _in_repl, expr, flags=re.IGNORECASE)

        # 4. LIKE ensure single quotes
        expr = re.sub(r'LIKE\s+"([^"]*)"', lambda m: f"LIKE '{m.group(1)}'", expr)

        # 5. IS NULL variants
        expr = re.sub(r"is\s+null", "IS NULL", expr, flags=re.IGNORECASE)
        expr = re.sub(r"is\s+not\s+null", "IS NOT NULL", expr, flags=re.IGNORECASE)

        # 6. JSON key access: field["key"] -> field->>'key'
        expr = re.sub(r"(\w+)\[\"([\w_]+)\"\]", r"\1->>'\2'", expr)

        # 7. Strings: "abc" -> 'abc'
        expr = re.sub(r'"([^"]*)"', lambda m: f"'{m.group(1)}'", expr)

        # 8. Collapse spaces
        expr = re.sub(r"\s+", " ", expr).strip()

        return expr

    def _align_df(self, milvus_df: pd.DataFrame, pg_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align two DataFrames on primary key and common columns, normalise JSON/ARRAY types."""
        if (
            self.primary_field
            and self.primary_field in milvus_df.columns
            and self.primary_field not in milvus_df.index.names
        ):
            milvus_df.set_index(self.primary_field, inplace=True)
        if self.primary_field and self.primary_field in pg_df.columns and self.primary_field not in pg_df.index.names:
            pg_df.set_index(self.primary_field, inplace=True)

        common_cols = [c for c in milvus_df.columns if c in pg_df.columns]
        if common_cols:
            milvus_df = milvus_df.loc[:, common_cols]
            pg_df = pg_df.loc[:, common_cols]

        # JSON fields normalisation
        for field in self.json_fields:
            if field in pg_df.columns:
                pg_df[field] = pg_df[field].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x and x[0] in ["{", "[", '"'] else x
                )
            if field in milvus_df.columns:
                milvus_df[field] = milvus_df[field].apply(
                    lambda x: x
                    if isinstance(x, dict)
                    else json.loads(x)
                    if isinstance(x, str) and x and x[0] in ["{", "[", '"']
                    else x
                )

        def _to_py_list(val: Any, round_floats: bool = False) -> Any:
            """Ensure value is list of Python scalars (convert numpy types)."""
            if val is None:
                return val
            lst = list(val) if not isinstance(val, list) else val
            cleaned = []
            for item in lst:
                if isinstance(item, (np.floating | float)):
                    f_item = float(item)
                    if round_floats:
                        cleaned.append(round(f_item, self.vector_precision_decimals))
                    else:
                        cleaned.append(f_item)
                elif isinstance(item, np.integer):
                    cleaned.append(int(item))
                else:
                    cleaned.append(item)
            return cleaned

        for field in self.array_fields:
            if field in milvus_df.columns:
                milvus_df[field] = milvus_df[field].apply(_to_py_list)
            if field in pg_df.columns:
                pg_df[field] = pg_df[field].apply(_to_py_list)

        for field in self.float_vector_fields:
            if field in milvus_df.columns:
                if self.sample_vector and not self.ignore_vector:
                    # Apply sampling to Milvus vectors for consistent comparison
                    milvus_df[field] = milvus_df[field].apply(
                        lambda x: self._sample_vector(_to_py_list(x, round_floats=True))
                    )
                else:
                    milvus_df[field] = milvus_df[field].apply(_to_py_list, round_floats=True)
            if field in pg_df.columns:
                # PostgreSQL vectors are already sampled during storage if sample_vector is enabled
                pg_df[field] = pg_df[field].apply(_to_py_list, round_floats=True)

        # Remove vector columns if ignoring them
        if self.ignore_vector and self.float_vector_fields:
            milvus_df.drop(
                columns=[c for c in self.float_vector_fields if c in milvus_df.columns], inplace=True, errors="ignore"
            )
            pg_df.drop(
                columns=[c for c in self.float_vector_fields if c in pg_df.columns], inplace=True, errors="ignore"
            )

        shared_idx = milvus_df.index.intersection(pg_df.index)
        milvus_aligned = milvus_df.loc[shared_idx].sort_index()
        pg_aligned = pg_df.loc[shared_idx].sort_index()

        milvus_aligned = milvus_aligned.reindex(columns=pg_aligned.columns)
        pg_aligned = pg_aligned.reindex(columns=milvus_aligned.columns)
        return milvus_aligned, pg_aligned

    # ------------------------------------------------------------------
    # Sampling & filter generation helpers (port from DuckDB client)
    # ------------------------------------------------------------------
    @_synchronized
    def sample_data(self, collection_name: str, num_samples: int = 100) -> pd.DataFrame:
        """Sample rows from PostgreSQL table for the given collection."""
        self._get_schema(collection_name)
        with self._get_pg_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {collection_name} ORDER BY random() LIMIT %s;",
                    (num_samples,),
                )
                rows = cursor.fetchall()
                colnames = [desc[0] for desc in cursor.description] if cursor.description else []
        return pd.DataFrame(rows, columns=colnames)

    def generate_milvus_filter(self, collection_name: str, num_samples: int = 100) -> list[str]:
        """Generate diverse Milvus filter expressions from sampled data (scalar fields only)."""
        df = self.sample_data(collection_name, num_samples)
        schema = self._get_schema(collection_name)

        scalar_types = {"BOOL", "INT8", "INT16", "INT32", "INT64", "FLOAT", "DOUBLE", "VARCHAR"}
        exprs: list[str] = []
        for field in [f for f in schema.fields if f.dtype.name in scalar_types and f.name in df.columns]:
            series = df[field.name]
            # IS NULL / IS NOT NULL
            if series.isnull().sum() > 0:
                exprs.append(f"{field.name} IS NULL")
                exprs.append(f"{field.name} IS NOT NULL")

            values = series.dropna().unique()
            dtype_name = field.dtype.name
            if len(values) == 0:
                continue

            if len(values) == 1:
                val = values[0]
                if dtype_name == "VARCHAR":
                    exprs.extend([f"{field.name} == '{val}'", f"{field.name} != '{val}'"])
                    if len(val) > 2:
                        exprs.extend(
                            [
                                f"{field.name} LIKE '{val[:2]}%'",
                                f"{field.name} LIKE '%{val[-2:]}'",
                                f"{field.name} LIKE '%{val[1:-1]}%'",
                            ]
                        )
                else:
                    exprs.extend([f"{field.name} == {val}", f"{field.name} != {val}"])
            else:
                # Numeric fields
                try:
                    # Try to convert to numpy array to handle min/max safely
                    values_array = np.array(values)
                    if np.issubdtype(values_array.dtype, np.number):
                        is_integer_field = "INT" in dtype_name.upper()
                        min_val, max_val = np.min(values_array), np.max(values_array)

                        if is_integer_field:
                            minv, maxv = int(min_val), int(max_val)
                        else:
                            minv, maxv = int(float(min_val)), int(float(max_val))

                        exprs.extend(
                            [
                                f"{field.name} > {minv}",
                                f"{field.name} < {maxv}",
                                f"{field.name} >= {minv}",
                                f"{field.name} <= {maxv}",
                                f"{field.name} >= {minv} AND {field.name} <= {maxv}",
                            ]
                        )
                        # IN / NOT IN (first 5 vals)
                        vals_str = ", ".join(str(v) for v in values[:5])
                        exprs.extend([f"{field.name} in [{vals_str}]", f"{field.name} not in [{vals_str}]"])
                        # Extra numeric examples
                        if is_integer_field:
                            exprs.append(f"{field.name} % 2 == 0")
                except (ValueError, TypeError):
                    # If numeric conversion fails, treat as non-numeric
                    pass

                # String fields
                if dtype_name == "VARCHAR":
                    vals_str = ", ".join(f"'{v}'" for v in values[:5])
                    exprs.extend([f"{field.name} in [{vals_str}]", f"{field.name} not in [{vals_str}]"])
                    for v in values[:3]:
                        if len(str(v)) > 2:
                            v_str = str(v)
                            exprs.extend(
                                [
                                    f"{field.name} LIKE '{v_str[:2]}%'",
                                    f"{field.name} LIKE '%{v_str[-2:]}'",
                                    f"{field.name} LIKE '%{v_str[1:-1]}%'",
                                ]
                            )
                # Bool fields
                elif dtype_name == "BOOL":
                    for v in values:
                        exprs.extend(
                            [
                                f"{field.name} == {str(v).lower()}",
                                f"{field.name} != {str(v).lower()}",
                            ]
                        )
        return exprs

    # ------------------------------------------------------------------
    # Entity comparison and data validation methods
    # ------------------------------------------------------------------
    def _get_all_primary_keys_milvus(self, collection_name: str, batch_size: int = 1000) -> list:
        """Internal method to get all primary keys from Milvus."""
        return self.get_all_primary_keys_from_milvus(collection_name, batch_size)

    def _get_all_primary_keys_pg(self, collection_name: str) -> list:
        """Internal method to get all primary keys from PostgreSQL."""
        with self._get_pg_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT {self.primary_field} FROM {collection_name};")
                rows = cursor.fetchall()
        return [r[0] for r in rows]

    def get_all_primary_keys_from_milvus(self, collection_name: str, batch_size: int = 1000) -> list:
        """
        Retrieve all primary key values from Milvus collection using query_iterator.

        This method efficiently retrieves all primary keys from a Milvus collection
        using batched iteration to handle large datasets without memory issues.

        Parameters
        ----------
        collection_name : str
            Name of the collection to extract primary keys from
        batch_size : int, optional
            Number of records to process per batch, by default 1000

        Returns
        -------
        list
            List of all primary key values from the collection
        """
        self._get_schema(collection_name)

        logger.info(f"Retrieving all primary keys from Milvus collection '{collection_name}'")
        logger.debug(f"Using batch size: {batch_size}")
        t0 = time.time()

        all_pks = []

        try:
            # Use query_iterator for efficient batch processing
            iterator = super().query_iterator(
                collection_name=collection_name, batch_size=batch_size, filter="", output_fields=[self.primary_field]
            )

            batch_count = 0
            while True:
                result = iterator.next()
                if not result:
                    iterator.close()
                    break

                # Extract primary key values from batch
                pks_in_batch = [row[self.primary_field] for row in result]
                all_pks.extend(pks_in_batch)
                batch_count += 1

                # Log progress every 10 batches to avoid log spam
                if batch_count % 10 == 0:
                    logger.debug(f"Processed {batch_count} batches, collected {len(all_pks)} primary keys")

        except Exception as e:
            logger.error(f"Error retrieving primary keys from Milvus collection '{collection_name}': {e}")
            raise

        duration = time.time() - t0
        logger.info(f"Retrieved {len(all_pks)} primary keys from Milvus in {duration:.3f}s")
        return all_pks

    @_synchronized
    def compare_primary_keys(self, collection_name: str) -> dict:
        """
        Compare primary keys between Milvus and PostgreSQL to detect inconsistencies.

        This method retrieves all primary keys from both databases and compares them
        to identify missing or extra records in either system.

        Parameters
        ----------
        collection_name : str
            Name of the collection to compare

        Returns
        -------
        dict
            Dictionary containing comparison results:
            - milvus_count: Number of records in Milvus
            - pg_count: Number of records in PostgreSQL
            - common_count: Number of common primary keys
            - only_in_milvus: List of keys only in Milvus
            - only_in_pg: List of keys only in PostgreSQL
            - has_differences: Boolean indicating if differences were found
        """
        self._get_schema(collection_name)

        logger.info(f"Comparing primary keys for collection '{collection_name}'")

        # Get all primary keys from Milvus
        logger.debug("Retrieving primary keys from Milvus")
        milvus_pks = set(self.get_all_primary_keys_from_milvus(collection_name))

        # Get all primary keys from PostgreSQL
        logger.debug("Retrieving primary keys from PostgreSQL")
        t0 = time.time()
        with self._get_pg_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT {self.primary_field} FROM {collection_name};")
                pg_pks_rows = cursor.fetchall()
        pg_pks = set(r[0] for r in pg_pks_rows)
        pg_duration = time.time() - t0
        logger.debug(f"Retrieved {len(pg_pks)} primary keys from PostgreSQL in {pg_duration:.3f}s")

        # Calculate differences
        only_in_milvus = milvus_pks - pg_pks
        only_in_pg = pg_pks - milvus_pks
        common_pks = milvus_pks & pg_pks

        result = {
            "milvus_count": len(milvus_pks),
            "pg_count": len(pg_pks),
            "common_count": len(common_pks),
            "only_in_milvus": sorted(list(only_in_milvus)),
            "only_in_pg": sorted(list(only_in_pg)),
            "has_differences": bool(only_in_milvus or only_in_pg),
        }

        # Log comparison results
        if result["has_differences"]:
            logger.error(f"Primary key comparison found differences for collection '{collection_name}':")
            logger.error(f"  Milvus count: {result['milvus_count']}")
            logger.error(f"  PostgreSQL count: {result['pg_count']}")
            logger.error(f"  Common keys: {result['common_count']}")
            logger.error(f"  Only in Milvus: {len(only_in_milvus)} keys")
            logger.error(f"  Only in PostgreSQL: {len(only_in_pg)} keys")

            # Show sample differences to avoid log overflow
            if only_in_milvus:
                sample_milvus = list(only_in_milvus)[:10]
                logger.error(f"  Sample Milvus-only keys: {sample_milvus}")
            if only_in_pg:
                sample_pg = list(only_in_pg)[:10]
                logger.error(f"  Sample PostgreSQL-only keys: {sample_pg}")
        else:
            logger.info("Primary key comparison passed:")
            logger.info(f"  Both Milvus and PostgreSQL have {result['common_count']} matching primary keys")

        return result

    def _validate_comparison_parameters(
        self,
        collection_name: str,
        batch_size: int,
        retry: int,
        retry_interval: float,
        full_scan: bool,
        compare_pks_first: bool,
    ) -> None:
        """Validate parameters for entity comparison."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if retry < 0:
            raise ValueError("retry must be non-negative")
        if retry_interval < 0:
            raise ValueError("retry_interval must be non-negative")
        logger.debug(
            f"Comparison parameters: batch_size={batch_size}, retry={retry}, "
            f"full_scan={full_scan}, compare_pks_first={compare_pks_first}"
        )

    def _perform_primary_key_comparison(self, collection_name: str) -> bool:
        """
        Perform primary key comparison between Milvus and PostgreSQL.

        Returns
        -------
        bool
            True if primary keys match, False otherwise
        """
        logger.debug("Stage 1: Primary key comparison")
        pk_comparison = self.compare_primary_keys(collection_name)
        if pk_comparison["has_differences"]:
            logger.error("Primary key comparison failed - data comparison may be inaccurate")

            # If LMDB is enabled, perform three-way validation to identify the source of error
            if self.enable_lmdb and self.lmdb_manager:
                logger.info("Performing three-way validation with LMDB to identify error source...")
                validation_result = self.three_way_pk_validation(collection_name)

                if validation_result["inconsistent_pks"] > 0:
                    logger.error(
                        f"Three-way validation found {validation_result['inconsistent_pks']} inconsistent PKs:"
                    )
                    logger.error(f"  - Milvus errors: {len(validation_result['milvus_errors'])} PKs")
                    logger.error(f"  - PostgreSQL errors: {len(validation_result['pg_errors'])} PKs")
                    logger.error(f"  - LMDB errors: {len(validation_result['lmdb_errors'])} PKs")

                    # Log sample of errors for each database
                    if validation_result["milvus_errors"]:
                        logger.error(f"  Sample Milvus errors: {validation_result['milvus_errors'][:5]}")
                    if validation_result["pg_errors"]:
                        logger.error(f"  Sample PostgreSQL errors: {validation_result['pg_errors'][:5]}")
                    if validation_result["lmdb_errors"]:
                        logger.error(f"  Sample LMDB errors: {validation_result['lmdb_errors'][:5]}")

            return False
        else:
            logger.debug("Primary key comparison passed")
            return True

    def _perform_count_comparison(
        self, collection_name: str, retry: int, retry_interval: float
    ) -> tuple[bool, int, int]:
        """
        Perform count comparison with retry logic.

        Returns
        -------
        tuple[bool, int, int]
            Tuple of (count_match, milvus_total, pg_total)
        """
        logger.debug("Stage 2: Record count comparison with retry logic")
        milvus_total = 0
        pg_total = 0
        for attempt in range(retry):
            count_res = self.count(collection_name)
            milvus_total = count_res["milvus_count"]
            pg_total = count_res["pg_count"]

            if milvus_total == pg_total:
                logger.debug(f"Count comparison passed on attempt {attempt + 1}")
                break

            logger.warning(
                f"Count mismatch on attempt {attempt + 1}/{retry}: "
                f"Milvus ({milvus_total}) vs PostgreSQL ({pg_total}). "
                f"Retrying in {retry_interval}s..."
            )
            if attempt < retry - 1:
                time.sleep(retry_interval)

        count_match = milvus_total == pg_total
        if not count_match:
            logger.error(f"Count mismatch after {retry} attempts: Milvus ({milvus_total}) vs PostgreSQL ({pg_total})")
        return count_match, milvus_total, pg_total

    def _perform_full_data_comparison(self, collection_name: str, batch_size: int) -> bool:
        """
        Perform full data comparison using concurrent batch processing.

        Returns
        -------
        bool
            True if all data matches, False otherwise
        """
        logger.debug("Stage 3: Full data comparison")
        t0 = time.time()

        # Get primary keys for batch processing
        with self._get_pg_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT {self.primary_field} FROM {collection_name};")
                pks_rows = cursor.fetchall()
        pks = [r[0] for r in pks_rows]

        total_pks = len(pks)
        if total_pks == 0:
            logger.info(f"No entities to compare for collection '{collection_name}'")
            return True

        logger.info(f"Starting full data comparison for {total_pks} entities using {batch_size} batch size")

        return self._execute_concurrent_comparison(collection_name, pks, batch_size, t0)

    def _execute_concurrent_comparison(
        self, collection_name: str, pks: list, batch_size: int, start_time: float
    ) -> bool:
        """Execute concurrent batch comparison of entities using multiprocessing."""
        total_pks = len(pks)
        # Use fewer processes than CPU cores to avoid overwhelming the system
        max_workers = min(cpu_count() - 1, (total_pks + batch_size - 1) // batch_size, 16)
        max_workers = max(1, max_workers)  # Ensure at least 1 worker
        compared = 0

        # Create batch jobs for multiprocessing
        batch_jobs = []
        for batch_start in range(0, total_pks, batch_size):
            batch_pks = pks[batch_start : batch_start + batch_size]
            batch_job = (
                batch_start,
                batch_pks,
                batch_size,
                total_pks,
                collection_name,
                self.primary_field,
                self.ignore_vector,
                self.float_vector_fields,
                self.json_fields,
                self.array_fields,
                self.varchar_fields,
                self.pg_conn_str,
                self.uri,
                self.token,
                self.sample_vector,
                self.vector_sample_size,
                self.vector_precision_decimals,
                self.vector_comparison_significant_digits,
            )
            batch_jobs.append(batch_job)

        has_any_differences = False
        milestones = {max(1, total_pks // 4), max(1, total_pks // 2), max(1, (total_pks * 3) // 4), total_pks}

        logger.info(f"Starting concurrent comparison with {max_workers} processes for {len(batch_jobs)} batches")

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batch jobs
                future_to_batch = {
                    executor.submit(_compare_batch_worker, *batch_job): batch_job for batch_job in batch_jobs
                }

                # Process completed batches
                for future in as_completed(future_to_batch):
                    try:
                        batch_size_actual, has_differences, detailed_diff = future.result()
                        if has_differences:
                            has_any_differences = True
                            batch_job = future_to_batch[future]
                            batch_start = batch_job[0]
                            batch_num = batch_start // batch_size + 1
                            logger.error(f"Differences detected in batch {batch_num}")

                            # Log detailed diff information
                            for diff_line in detailed_diff:
                                logger.error(diff_line)

                        compared += batch_size_actual
                        if compared in milestones:
                            logger.info(
                                f"Comparison progress: {compared}/{total_pks} ({(compared * 100) // total_pks}%) done."
                            )

                    except Exception as e:
                        logger.error(f"Batch comparison failed: {e}")
                        has_any_differences = True

        except Exception as e:
            logger.error(f"Process pool execution failed: {e}")
            # Fallback to single-threaded comparison
            logger.info("Falling back to single-threaded comparison")
            return self._execute_single_threaded_comparison(collection_name, pks, batch_size)

        logger.info(f"Entity comparison completed for collection '{collection_name}'.")
        logger.info(f"Entity comparison completed in {time.time() - start_time:.3f} s.")

        if has_any_differences:
            logger.error(f"Entity comparison found differences for collection '{collection_name}'.")
            return False
        else:
            logger.info(f"Entity comparison successful - no differences found for collection '{collection_name}'.")
            return True

    def _execute_single_threaded_comparison(self, collection_name: str, pks: list, batch_size: int) -> bool:
        """Fallback single-threaded comparison when multiprocessing fails."""
        total_pks = len(pks)
        has_any_differences = False

        logger.info(f"Running single-threaded comparison for {total_pks} entities")

        for batch_start in range(0, total_pks, batch_size):
            batch_pks = pks[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (total_pks + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches}")

            try:
                # Use original comparison logic
                milvus_filter = f"{self.primary_field} in {list(batch_pks)}"
                milvus_data = super().query(collection_name, filter=milvus_filter, output_fields=["*"])
                milvus_df = pd.DataFrame(milvus_data)

                # PG fetch
                with self._get_pg_connection() as conn:
                    placeholder = ", ".join(["%s"] * len(batch_pks))
                    with conn.cursor() as cursor:
                        cursor.execute(
                            f"SELECT * FROM {collection_name} WHERE {self.primary_field} IN ({placeholder});",
                            batch_pks,
                        )
                        pg_rows = cursor.fetchall()
                        colnames = [desc[0] for desc in cursor.description] if cursor.description else []
                    pg_df = pd.DataFrame(pg_rows, columns=colnames or [])

                # Compare using existing method
                diff = self._compare_df(milvus_df, pg_df)
                if diff:
                    has_any_differences = True
                    logger.error(f"Differences detected in batch {batch_num}")

            except Exception as e:
                logger.error(f"Error in single-threaded comparison batch {batch_num}: {e}")
                has_any_differences = True

        return not has_any_differences

    def three_way_pk_validation(self, collection_name: str, sample_size: int | None = None) -> dict[str, Any]:
        """
        Perform three-way primary key validation between Milvus, PostgreSQL, and LMDB.

        This method uses LMDB as a tiebreaker when Milvus and PostgreSQL disagree.
        The "majority vote" approach helps identify which database has incorrect data.

        Parameters
        ----------
        collection_name : str
            Name of the collection to validate
        sample_size : int | None, optional
            If provided, only validate a random sample of primary keys

        Returns
        -------
        dict
            Validation results including:
            - total_pks: Total number of primary keys checked
            - consistent_pks: Number of PKs consistent across all 3 databases
            - inconsistent_pks: Number of PKs with inconsistencies
            - milvus_errors: PKs where Milvus appears to be wrong
            - pg_errors: PKs where PostgreSQL appears to be wrong
            - lmdb_errors: PKs where LMDB appears to be wrong
            - details: Detailed information about each inconsistency
        """
        if not self.lmdb_manager:
            raise ValueError("LMDB is not enabled. Cannot perform three-way validation.")

        self._get_schema(collection_name)
        logger.info(f"Starting three-way PK validation for collection '{collection_name}'")

        # Get primary keys from all three sources
        logger.info("Fetching primary keys from Milvus")
        milvus_pks = set(self._get_all_primary_keys_milvus(collection_name))
        logger.info(f"Milvus PKs: {len(milvus_pks)}")
        logger.info("Fetching primary keys from PostgreSQL")
        pg_pks = set(self._get_all_primary_keys_pg(collection_name))
        logger.info(f"PostgreSQL PKs: {len(pg_pks)}")

        logger.info("Fetching primary keys from LMDB")
        lmdb_pks = set(self.lmdb_manager.get_collection_pks(collection_name, PKStatus.EXISTS))
        logger.info(f"LMDB PKs: {len(lmdb_pks)}")
        # If sample size is specified, randomly sample PKs
        all_pks = milvus_pks | pg_pks | lmdb_pks
        if sample_size and len(all_pks) > sample_size:
            import random

            all_pks = set(random.sample(list(all_pks), sample_size))
            milvus_pks = milvus_pks & all_pks
            pg_pks = pg_pks & all_pks
            lmdb_pks = lmdb_pks & all_pks

        # Initialize result tracking
        result: dict[str, Any] = {
            "total_pks": len(all_pks),
            "consistent_pks": 0,
            "inconsistent_pks": 0,
            "milvus_errors": [],
            "pg_errors": [],
            "lmdb_errors": [],
            "details": [],
        }

        # Check each PK
        for pk in all_pks:
            in_milvus = pk in milvus_pks
            in_pg = pk in pg_pks
            in_lmdb = pk in lmdb_pks

            # All three agree - consistent
            if in_milvus == in_pg == in_lmdb:
                result["consistent_pks"] += 1
            else:
                result["inconsistent_pks"] += 1

                # Determine which database(s) are wrong using majority vote
                vote_exists = sum([in_milvus, in_pg, in_lmdb])

                # Majority says exists (2 or 3 votes)
                if vote_exists >= 2:
                    correct_state = "exists"
                    if not in_milvus:
                        result["milvus_errors"].append(pk)
                    if not in_pg:
                        result["pg_errors"].append(pk)
                    if not in_lmdb:
                        result["lmdb_errors"].append(pk)
                # Majority says deleted (0 or 1 vote)
                else:
                    correct_state = "deleted"
                    if in_milvus:
                        result["milvus_errors"].append(pk)
                    if in_pg:
                        result["pg_errors"].append(pk)
                    if in_lmdb:
                        result["lmdb_errors"].append(pk)

                # Add detailed information
                detail = {
                    "pk": pk,
                    "in_milvus": in_milvus,
                    "in_pg": in_pg,
                    "in_lmdb": in_lmdb,
                    "correct_state": correct_state,
                    "vote_count": vote_exists,
                }

                # Get LMDB metadata if available
                if self.lmdb_manager:
                    lmdb_state = self.lmdb_manager.get_pk_state(collection_name, pk)
                    if lmdb_state:
                        detail["lmdb_metadata"] = lmdb_state

                result["details"].append(detail)

        # Log summary
        logger.info(f"Three-way validation complete: {result['consistent_pks']}/{result['total_pks']} PKs consistent")
        if result["inconsistent_pks"] > 0:
            logger.warning(f"Found {result['inconsistent_pks']} inconsistent PKs")
            logger.warning(f"Milvus errors: {len(result['milvus_errors'])}")
            logger.warning(f"PostgreSQL errors: {len(result['pg_errors'])}")
            logger.warning(f"LMDB errors: {len(result['lmdb_errors'])}")

        return result

    def entity_compare(
        self,
        collection_name: str,
        batch_size: int = 10000,
        *,
        retry: int = 5,
        retry_interval: float = 5.0,
        full_scan: bool = False,
        compare_pks_first: bool = True,
    ) -> bool:
        """
        Perform comprehensive comparison of entity data between Milvus and PostgreSQL.

        This method performs a multi-stage comparison process:
        1. Optional primary key comparison to identify missing records
        2. Record count comparison with retry logic for eventual consistency
        3. Optional full data comparison using concurrent batch processing

        Parameters
        ----------
        collection_name : str
            Name of the collection to compare
        batch_size : int, optional
            Number of records to process per batch, by default 1000
        retry : int, optional
            Number of retry attempts for count comparison, by default 5
        retry_interval : float, optional
            Seconds to wait between retry attempts, by default 5.0
        full_scan : bool, optional
            Whether to perform full data comparison or just count check, by default False
        compare_pks_first : bool, optional
            Whether to compare primary keys before data comparison, by default True

        Returns
        -------
        bool
            True if comparison passed, False if differences were found
        """
        self._get_schema(collection_name)
        logger.info(f"Starting entity comparison for collection '{collection_name}'")

        # Validate parameters
        self._validate_comparison_parameters(
            collection_name, batch_size, retry, retry_interval, full_scan, compare_pks_first
        )

        # Stage 1: Primary key comparison (if enabled)
        if compare_pks_first:
            if not self._perform_primary_key_comparison(collection_name):
                if not full_scan:
                    return False

        # Stage 2: Count comparison with retry logic
        count_match, milvus_total, pg_total = self._perform_count_comparison(collection_name, retry, retry_interval)

        # Stage 3: Full data comparison (if requested)
        if not full_scan:
            if count_match:
                logger.info(f"Count check passed: {milvus_total} records in both systems")
            return count_match

        # Perform detailed entity comparison
        return self._perform_full_data_comparison(collection_name, batch_size)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> MilvusPGClient:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and cleanup resources."""
        self._cleanup_resources()

    def __del__(self) -> None:
        """Cleanup resources when object is garbage collected."""
        self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Internal method to cleanup resources, avoiding duplicate cleanup."""
        # Close LMDB if enabled
        try:
            if hasattr(self, "lmdb_manager") and self.lmdb_manager:
                self.lmdb_manager.close()
                logger.debug("LMDB connection closed successfully")
                self.lmdb_manager = None
        except Exception as e:
            logger.warning(f"Error closing LMDB connection: {e}")

        # Close PostgreSQL pool
        try:
            if hasattr(self, "pg_pool") and self.pg_pool:
                # Check if pool is already closed to avoid double-close warnings
                if not self.pg_pool.closed:
                    self.pg_pool.closeall()
                    logger.debug("PostgreSQL connection pool closed successfully")
                # Mark pool as None to prevent duplicate cleanup
                self.pg_pool = None
        except Exception as e:
            # Only log if it's not a "pool already closed" error
            if "closed" not in str(e).lower():
                logger.warning(f"Error closing PostgreSQL connection pool: {e}")
