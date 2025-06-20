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

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from deepdiff import DeepDiff
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values
from pymilvus import CollectionSchema, DataType, MilvusClient, connections

from .logger_config import logger

__all__ = ["MilvusPGClient"]


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

    Attributes
    ----------
    pg_conn : PGConnection
        PostgreSQL database connection
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

    def __init__(self, *args: Any, **kwargs: Any):
        # Extract custom parameters before calling parent constructor
        self.ignore_vector: bool = kwargs.pop("ignore_vector", False)
        self.pg_conn_str: str = kwargs.pop("pg_conn_str")
        uri = kwargs.get("uri", "")
        token = kwargs.get("token", "")

        # Initialize parent MilvusClient
        super().__init__(*args, **kwargs)
        self.uri = uri
        self.token = token

        # Connect to Milvus
        logger.debug(f"Connecting to Milvus with URI: {uri}")
        connections.connect(uri=uri, token=token)

        # Connect to PostgreSQL
        logger.info("Connecting to PostgreSQL database")
        try:
            self.pg_conn: PGConnection = psycopg2.connect(self.pg_conn_str)
            self.pg_conn.autocommit = False  # We'll manage transactions manually
            logger.debug("PostgreSQL connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

        # Initialize schema-related caches
        self.primary_field: str = ""
        self.fields_name_list: list[str] = []
        self.json_fields: list[str] = []
        self.array_fields: list[str] = []
        self.varchar_fields: list[str] = []
        self.float_vector_fields: list[str] = []

        # Thread synchronization lock for write operations
        self._lock: threading.Lock = threading.Lock()

    # ---------------------------------------------------------------------
    # Schema and utility methods
    # ---------------------------------------------------------------------
    def _get_schema(self, collection_name: str) -> CollectionSchema:
        """
        Retrieve and cache schema information from Milvus for a collection.

        This method fetches the collection schema from Milvus and populates
        internal field lists for efficient field type checking during operations.

        Parameters
        ----------
        collection_name : str
            Name of the collection to get schema for

        Returns
        -------
        CollectionSchema
            The collection schema from Milvus
        """
        logger.debug(f"Retrieving schema for collection: {collection_name}")
        temp_client = MilvusClient(uri=self.uri, token=self.token)
        schema_info = temp_client.describe_collection(collection_name)
        schema = CollectionSchema.construct_from_dict(schema_info)

        # Reset field caches for the new collection
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

        logger.debug(
            f"Schema cached - Primary field: {self.primary_field}, "
            f"Fields: {len(self.fields_name_list)}, "
            f"JSON: {len(self.json_fields)}, "
            f"Arrays: {len(self.array_fields)}, "
            f"Vectors: {len(self.float_vector_fields)}"
        )
        return schema

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
    def create_collection(self, collection_name: str, schema: CollectionSchema, **kwargs: Any):
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
            with self.pg_conn.cursor() as cursor:
                cursor.execute(create_sql)
            self.pg_conn.commit()
            logger.debug(f"PostgreSQL table created in {time.time() - pg_start:.3f}s")
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Failed to create PostgreSQL table: {e}")
            raise RuntimeError(f"Failed to create PG table: {e}") from e

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

    def drop_collection(self, collection_name: str):
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
            with self.pg_conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {collection_name};")
            self.pg_conn.commit()
            logger.debug(f"PostgreSQL table dropped in {time.time() - start:.3f}s")
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Failed to drop PostgreSQL table: {e}")
            raise RuntimeError(f"Failed to drop PG table: {e}") from e

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
    def _synchronized(method):
        """Decorator to run method under instance-level lock for thread safety."""
        from functools import wraps

        @wraps(method)
        def _wrapper(self, *args, **kwargs):
            with self._lock:
                return method(self, *args, **kwargs)

        return _wrapper

    @_synchronized
    def insert(self, collection_name: str, data: list[dict[str, Any]], **kwargs: Any):
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

        # Prepare DataFrame for JSON/ARRAY serialization
        df = pd.DataFrame(data)

        # Serialize JSON and ARRAY fields to string format for PostgreSQL
        for field in self.json_fields:
            if field in df.columns:
                df[field] = df[field].apply(json.dumps)
                logger.debug(f"Serialized JSON field: {field}")

        for field in self.array_fields:
            if field in df.columns:
                df[field] = df[field].apply(json.dumps)
                logger.debug(f"Serialized ARRAY field: {field}")

        # Remove vector columns from PostgreSQL insert if ignoring vectors
        if self.ignore_vector and self.float_vector_fields:
            original_cols = df.columns.tolist()
            df.drop(columns=[c for c in self.float_vector_fields if c in df.columns], inplace=True, errors="ignore")
            if len(df.columns) < len(original_cols):
                logger.debug(f"Dropped vector fields from PostgreSQL insert: {set(original_cols) - set(df.columns)}")

        # Build efficient batch INSERT SQL
        columns = list(df.columns)
        insert_sql = f"INSERT INTO {collection_name} ({', '.join(columns)}) VALUES %s"
        values = [tuple(row) for row in df.itertuples(index=False, name=None)]
        logger.debug(f"Prepared {len(values)} rows for PostgreSQL batch insert")

        try:
            # Execute PostgreSQL batch insert using execute_values for performance
            logger.debug("Starting PostgreSQL batch INSERT operation")
            t0 = time.time()
            with self.pg_conn.cursor() as cursor:
                execute_values(cursor, insert_sql, values, page_size=1000)
                pg_duration = time.time() - t0
                logger.debug(f"PostgreSQL batch INSERT completed: {cursor.rowcount} rows in {pg_duration:.3f}s")
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"PostgreSQL insert failed for collection '{collection_name}': {e}")
            raise RuntimeError(f"PostgreSQL insert failed: {e}") from e

        try:
            # Execute Milvus insert operation
            logger.debug("Starting Milvus insert operation")
            t0 = time.time()
            result = super().insert(collection_name, data, **kwargs)
            milvus_duration = time.time() - t0
            logger.debug(f"Milvus insert completed in {milvus_duration:.3f}s")

            # Commit PostgreSQL transaction on successful Milvus insert
            self.pg_conn.commit()
            logger.debug("PostgreSQL transaction committed successfully")
            return result

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Milvus insert failed for collection '{collection_name}', PostgreSQL rolled back: {e}")
            raise RuntimeError(f"Milvus insert failed, PG rolled back: {e}") from e

    @_synchronized
    def upsert(self, collection_name: str, data: list[dict[str, Any]], **kwargs: Any):
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

        # Prepare data similar to insert
        df = pd.DataFrame(data)
        for field in self.json_fields:
            if field in df.columns:
                df[field] = df[field].apply(json.dumps)
        for field in self.array_fields:
            if field in df.columns:
                df[field] = df[field].apply(json.dumps)

        # Remove vector columns if ignoring them
        if self.ignore_vector and self.float_vector_fields:
            df.drop(columns=[c for c in self.float_vector_fields if c in df.columns], inplace=True, errors="ignore")

        # Build PostgreSQL UPSERT SQL with ON CONFLICT clause
        cols = list(df.columns)
        updates = ", ".join([f"{col}=EXCLUDED.{col}" for col in cols])
        insert_sql = (
            f"INSERT INTO {collection_name} ({', '.join(cols)}) VALUES %s "
            f"ON CONFLICT ({self.primary_field}) DO UPDATE SET {updates}"
        )
        values = [tuple(row) for row in df.itertuples(index=False, name=None)]
        logger.debug(f"Prepared PostgreSQL upsert with conflict resolution on '{self.primary_field}'")

        try:
            # Execute PostgreSQL batch upsert
            logger.debug("Starting PostgreSQL batch UPSERT operation")
            t0 = time.time()
            with self.pg_conn.cursor() as cursor:
                execute_values(cursor, insert_sql, values, page_size=1000)
                pg_duration = time.time() - t0
                logger.debug(
                    f"PostgreSQL batch UPSERT completed: {cursor.rowcount} rows affected in {pg_duration:.3f}s"
                )
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"PostgreSQL upsert failed for collection '{collection_name}': {e}")
            raise RuntimeError(f"PostgreSQL upsert failed: {e}") from e

        try:
            # Execute Milvus upsert operation
            logger.debug("Starting Milvus upsert operation")
            t0 = time.time()
            result = super().upsert(collection_name, data, **kwargs)
            milvus_duration = time.time() - t0
            logger.debug(f"Milvus upsert completed in {milvus_duration:.3f}s")

            # Commit PostgreSQL transaction on successful Milvus upsert
            self.pg_conn.commit()
            logger.debug("PostgreSQL transaction committed successfully")
            return result

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Milvus upsert failed for collection '{collection_name}', PostgreSQL rolled back: {e}")
            raise RuntimeError(f"Milvus upsert failed, PG rolled back: {e}") from e

    @_synchronized
    def delete(self, collection_name: str, ids: list[int | str], **kwargs: Any):
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

        try:
            # Execute PostgreSQL delete
            logger.debug("Starting PostgreSQL DELETE operation")
            t0 = time.time()
            with self.pg_conn.cursor() as cursor:
                cursor.execute(delete_sql, ids)
                pg_duration = time.time() - t0
                logger.debug(f"PostgreSQL DELETE completed: {cursor.rowcount} rows deleted in {pg_duration:.3f}s")
        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"PostgreSQL delete failed for collection '{collection_name}': {e}")
            raise RuntimeError(f"PostgreSQL delete failed: {e}") from e

        try:
            # Execute Milvus delete
            logger.debug("Starting Milvus delete operation")
            t0 = time.time()
            result = super().delete(collection_name, ids=ids, **kwargs)
            milvus_duration = time.time() - t0
            logger.debug(f"Milvus delete completed in {milvus_duration:.3f}s")

            # Commit PostgreSQL transaction on successful Milvus delete
            self.pg_conn.commit()
            logger.debug("PostgreSQL transaction committed successfully")
            return result

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Milvus delete failed for collection '{collection_name}', PostgreSQL rolled back: {e}")
            raise RuntimeError(f"Milvus delete failed, PG rolled back: {e}") from e

    # ------------------------------------------------------------------
    # Read operations and validation helpers
    # ------------------------------------------------------------------
    @_synchronized
    def query(self, collection_name: str, filter: str = "", output_fields: list[str] | None = None):
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
        with self.pg_conn.cursor() as cursor:
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
    def export(self, collection_name: str):
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

        with self.pg_conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {collection_name};")
            rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description] if cursor.description else []

        result = [dict(zip(colnames, r, strict=False)) for r in rows]
        df = pd.DataFrame(result)

        duration = time.time() - t0
        logger.debug(f"Export completed: {len(df)} rows in {duration:.3f}s")
        return df

    @_synchronized
    def count(self, collection_name: str):
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
            with self.pg_conn.cursor() as cursor:
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
        except Exception as e:
            logger.error(f"Failed to query PostgreSQL count for collection '{collection_name}': {e}")
            pg_count = 0

        return {"milvus_count": milvus_count, "pg_count": pg_count}

    # ------------------------------------------------------------------
    # Data comparison and validation methods
    # ------------------------------------------------------------------
    def _compare_df(self, milvus_df: pd.DataFrame, pg_df: pd.DataFrame):
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

        # Convert to dictionaries for DeepDiff comparison
        milvus_dict = milvus_aligned.to_dict("list")
        pg_dict = pg_aligned.to_dict("list")

        # Use DeepDiff with tolerance for floating point differences
        diff = DeepDiff(
            milvus_dict,
            pg_dict,
            ignore_order=True,  # Ignore row order differences
            significant_digits=3,  # Tolerance for floating point precision
        )

        # Print detailed differences for debugging if differences found
        if diff:
            self._print_detailed_diff(milvus_aligned, pg_aligned)

        return diff

    def _print_detailed_diff(self, milvus_df: pd.DataFrame, pg_df: pd.DataFrame):
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
        # Row-by-row differences
        logger.error("--- ROW-BY-ROW DIFFERENCES ---")
        # Identify rows where any column differs
        diff_mask = ~(milvus_df.eq(pg_df))
        for pk, row_mask in diff_mask.iterrows():
            if row_mask.any():
                logger.error(f"Primary Key: {pk} has row-level differences:")
                for col, is_diff in row_mask.items():
                    if is_diff:
                        m_val = milvus_df.at[pk, col]
                        p_val = pg_df.at[pk, col]
                        logger.error(f"  Column '{col}': milvus={m_val}, pg={p_val}")
        logger.error("=== END ROW-BY-ROW DIFFERENCES ===")

    def query_result_compare(self, collection_name: str, filter: str = "", output_fields: list[str] | None = None):
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
    def _milvus_filter_to_sql(self, filter_expr: str) -> str:  # noqa: D401
        """Convert simple Milvus filter to PostgreSQL SQL.

        Current support:
        1. Logical operators to uppercase
        2. Equality '==' -> '='
        3. IN list: field in [1,2] -> field IN (1,2)
        4. LIKE ensure single quotes
        5. IS NULL variants
        6. JSON key access: field["key"] -> field->>'key'
        7. Strings: "abc" -> 'abc'
        8. Collapse spaces

        Note: This is *not* a full parser – it handles common simple filters generated
        by helper utilities in this project. Extend if you need more complex syntax.
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
        def _in_repl(match):
            field = match.group(1)
            values = match.group(2)
            try:
                py_list = eval(values)
            except Exception:
                py_list = []
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

    def _align_df(self, milvus_df: pd.DataFrame, pg_df: pd.DataFrame):
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

        def _to_py_list(val, round_floats=False):
            """Ensure value is list of Python scalars (convert numpy types)."""
            if val is None:
                return val
            lst = list(val) if not isinstance(val, list) else val
            cleaned = []
            for item in lst:
                if isinstance(item, (np.floating | float)):
                    f_item = float(item)
                    if round_floats:
                        cleaned.append(round(f_item, 3))
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
                milvus_df[field] = milvus_df[field].apply(_to_py_list, round_floats=True)
            if field in pg_df.columns:
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
    def sample_data(self, collection_name: str, num_samples: int = 100):
        """Sample rows from PostgreSQL table for the given collection."""
        self._get_schema(collection_name)
        with self.pg_conn.cursor() as cursor:
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
                            minv, maxv = float(min_val), float(max_val)

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
        with self.pg_conn.cursor() as cursor:
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

    def entity_compare(
        self,
        collection_name: str,
        batch_size: int = 1000,
        *,
        retry: int = 5,
        retry_interval: float = 5.0,
        full_scan: bool = False,
        compare_pks_first: bool = True,
    ):
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
        logger.debug(
            f"Parameters: batch_size={batch_size}, retry={retry}, "
            f"full_scan={full_scan}, compare_pks_first={compare_pks_first}"
        )

        # Stage 1: Primary key comparison (if enabled)
        if compare_pks_first:
            logger.debug("Stage 1: Primary key comparison")
            pk_comparison = self.compare_primary_keys(collection_name)
            if pk_comparison["has_differences"]:
                logger.error("Primary key comparison failed - data comparison may be inaccurate")
                if not full_scan:
                    return False
            else:
                logger.debug("Primary key comparison passed")

        # Stage 2: Count comparison with retry logic for eventual consistency
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

        # Validate final count results
        count_match = milvus_total == pg_total
        if not count_match:
            logger.error(f"Count mismatch after {retry} attempts: Milvus ({milvus_total}) vs PostgreSQL ({pg_total})")

        # Stage 3: Full data comparison (if requested)
        if not full_scan:
            if count_match:
                logger.info(f"Count check passed: {milvus_total} records in both systems")
            return count_match

        # Perform detailed entity comparison
        logger.debug("Stage 3: Full data comparison")
        t0 = time.time()

        # Get primary keys for batch processing
        # Use PostgreSQL keys to ensure we're comparing existing records
        with self.pg_conn.cursor() as cursor:
            cursor.execute(f"SELECT {self.primary_field} FROM {collection_name};")
            pks_rows = cursor.fetchall()
        pks = [r[0] for r in pks_rows]

        total_pks = len(pks)
        if total_pks == 0:
            logger.info(f"No entities to compare for collection '{collection_name}'")
            return True

        logger.info(f"Starting full data comparison for {total_pks} entities using {batch_size} batch size")

        # Use concurrent processing for better performance
        max_workers = min(16, (total_pks + batch_size - 1) // batch_size)  # Limit concurrent threads
        compared = 0
        # Note: Using atomic counter without lock for progress tracking (minor race conditions acceptable)

        def compare_batch(
            batch_start: int,
            batch_pks: list,
            primary_field: str,
            ignore_vector: bool,
            float_vector_fields: list,
            pg_conn_str: str,
        ) -> tuple[int, bool]:
            """Compare a single batch between Milvus and PostgreSQL."""
            batch_end = min(batch_start + batch_size, total_pks)
            batch_num = batch_start // batch_size + 1
            total_batches = (total_pks + batch_size - 1) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches}: entities {batch_start + 1}-{batch_end}/{total_pks}"
            )

            try:
                # Milvus fetch
                milvus_filter = f"{primary_field} in {list(batch_pks)}"
                milvus_data = MilvusClient.query(self, collection_name, filter=milvus_filter, output_fields=["*"])
                milvus_df = pd.DataFrame(milvus_data)
                # Drop vector columns for comparison if ignoring vectors
                if ignore_vector and float_vector_fields:
                    milvus_df.drop(
                        columns=[c for c in float_vector_fields if c in milvus_df.columns],
                        inplace=True,
                        errors="ignore",
                    )

                # PG fetch - create a new connection for thread safety
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
                    pg_df = pd.DataFrame(pg_rows, columns=list(colnames) if colnames else None)
                finally:
                    pg_conn.close()

                # Compare data - create temporary instance for comparison
                temp_client = MilvusPGClient.__new__(MilvusPGClient)
                temp_client.primary_field = primary_field
                temp_client.ignore_vector = ignore_vector
                temp_client.float_vector_fields = float_vector_fields
                temp_client.json_fields = self.json_fields
                temp_client.array_fields = self.array_fields
                temp_client.varchar_fields = self.varchar_fields

                diff = temp_client._compare_df(milvus_df, pg_df)
                has_differences = bool(diff)
                if has_differences:
                    logger.error(f"Differences detected between Milvus and PostgreSQL for batch {batch_num}:\n{diff}")

                return len(batch_pks), has_differences

            except Exception as e:
                logger.error(f"Error comparing batch {batch_num}: {e}")
                return len(batch_pks), True  # Treat errors as differences

        # Create batch jobs
        batch_jobs = []
        for batch_start in range(0, total_pks, batch_size):
            batch_pks = pks[batch_start : batch_start + batch_size]
            batch_jobs.append((batch_start, batch_pks))

        # Execute batches concurrently
        has_any_differences = False
        milestones = {max(1, total_pks // 4), max(1, total_pks // 2), max(1, (total_pks * 3) // 4), total_pks}

        logger.info(f"Starting concurrent comparison with {max_workers} threads for {len(batch_jobs)} batches")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(
                    compare_batch,
                    batch_start,
                    batch_pks,
                    self.primary_field,
                    self.ignore_vector,
                    self.float_vector_fields,
                    self.pg_conn_str,
                ): (batch_start, batch_pks)
                for batch_start, batch_pks in batch_jobs
            }

            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_start, batch_pks = future_to_batch[future]
                try:
                    batch_size_actual, has_differences = future.result()
                    if has_differences:
                        has_any_differences = True

                    # Update progress counter (minor race conditions in logging acceptable)
                    compared += batch_size_actual
                    if compared in milestones:
                        logger.info(
                            f"Comparison progress: {compared}/{total_pks} ({(compared * 100) // total_pks}%) done."
                        )

                except Exception as e:
                    logger.error(f"Batch comparison failed for batch starting at {batch_start}: {e}")
                    has_any_differences = True

        logger.info(f"Entity comparison completed for collection '{collection_name}'.")
        logger.info(f"Entity comparison completed in {time.time() - t0:.3f} s.")

        if has_any_differences:
            logger.error(f"Entity comparison found differences for collection '{collection_name}'.")
            return False
        else:
            logger.info(f"Entity comparison successful - no differences found for collection '{collection_name}'.")
            return True

    # ------------------------------------------------------------------
    def __del__(self):
        try:
            if hasattr(self, "pg_conn"):
                self.pg_conn.close()
        except Exception as e:
            logger.warning(f"Error closing PostgreSQL connection: {e}")
