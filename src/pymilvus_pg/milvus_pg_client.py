"""milvus_pg_client.py
A Milvus client wrapper that synchronizes write operations to PostgreSQL for
validation purposes, mirroring the behaviour of the original DuckDB version.

NOTE: This is a **minimum-viable** refactor focusing on replacing DuckDB with
PostgreSQL.  Further optimisation (pooling, async, SQL injection safety, etc.)
can be added incrementally.
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
from psycopg2.extensions import cursor as PGCursor
from psycopg2.extras import execute_values
from pymilvus import Collection, CollectionSchema, DataType, MilvusClient, connections

from .logger_config import logger

__all__ = ["MilvusPGClient"]


class MilvusPGClient(MilvusClient):
    """Milvus client with synchronous PostgreSQL shadow writes for validation.

    Parameters
    ----------
    pg_conn_str: str
        PostgreSQL connection string in libpq URI or keyword format.
    uri: str, optional
        Milvus server uri, passed through to :class:`pymilvus.MilvusClient`.
    token: str, optional
        Auth token for Milvus.
    ignore_vector: bool, optional
        If True, skip handling FLOAT_VECTOR fields in PostgreSQL operations and comparisons.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.ignore_vector: bool = kwargs.pop("ignore_vector", False)  # noqa: D401
        self.pg_conn_str: str = kwargs.pop("pg_conn_str")
        uri = kwargs.get("uri", "")
        token = kwargs.get("token", "")

        super().__init__(*args, **kwargs)
        # Connect to Milvus
        connections.connect(uri=uri, token=token)

        # Connect to PostgreSQL
        logger.info("Connecting to PostgreSQL …")
        self.pg_conn: PGConnection = psycopg2.connect(self.pg_conn_str)
        self.pg_conn.autocommit = False  # We'll manage transactions manually

        # Cache schema related information
        self.primary_field: str = ""
        self.fields_name_list: list[str] = []
        self.json_fields: list[str] = []
        self.array_fields: list[str] = []
        self.varchar_fields: list[str] = []
        self.float_vector_fields: list[str] = []
        # Lock to synchronize write operations across threads
        self._lock: threading.Lock = threading.Lock()

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _get_schema(self, collection_name: str) -> CollectionSchema:
        """Cache schema information from Milvus for a collection."""
        c = Collection(collection_name)
        schema = c.schema
        # reset caches for each collection
        self.primary_field = ""
        self.fields_name_list.clear()
        self.json_fields.clear()
        self.array_fields.clear()
        self.varchar_fields.clear()
        self.float_vector_fields.clear()

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
        return schema

    @staticmethod
    def _milvus_dtype_to_pg(milvus_type: DataType) -> str:
        """Map Milvus DataType to PostgreSQL type."""
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
    # Collection DDL
    # ------------------------------------------------------------------
    def create_collection(self, collection_name: str, schema: CollectionSchema, **kwargs: Any):
        logger.info(f"Creating collection '{collection_name}' in PostgreSQL & Milvus …")
        # Build PG CREATE TABLE SQL based on schema
        cols_sql = []
        for f in schema.fields:
            # Skip vector fields in PostgreSQL if requested
            if self.ignore_vector and f.dtype == DataType.FLOAT_VECTOR:
                continue
            pg_type = self._milvus_dtype_to_pg(f.dtype)
            col_def = f"{f.name} {pg_type}"
            if f.is_primary:
                col_def += " PRIMARY KEY"
            cols_sql.append(col_def)
        create_sql = f"CREATE TABLE IF NOT EXISTS {collection_name} ({', '.join(cols_sql)});"

        try:
            pg_start = time.time()
            with self.pg_conn.cursor() as cursor:
                cursor.execute(create_sql)
            self.pg_conn.commit()
            logger.info(f"PG CREATE TABLE completed in {time.time() - pg_start:.3f} s")
        except Exception as e:
            self.pg_conn.rollback()
            raise RuntimeError(f"Failed to create PG table: {e}") from e

        # Create collection in Milvus (may raise if exists)
        # Pass schema as a keyword argument to align with MilvusClient signature where the
        # second positional parameter is `dimension` (int). Using a keyword eliminates the
        # type mismatch error that was raised earlier.
        milvus_start = time.time()
        res = super().create_collection(collection_name, schema=schema, consistency_level="Strong", **kwargs)
        logger.info(f"Milvus create_collection completed in {time.time() - milvus_start:.3f} s.")
        return res

    def drop_collection(self, collection_name: str):
        logger.info(f"Dropping collection '{collection_name}' …")
        try:
            start = time.time()
            with self.pg_conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {collection_name};")
            self.pg_conn.commit()
            logger.info("PG DROP TABLE completed in %.3f s", time.time() - start)
        except Exception as e:
            self.pg_conn.rollback()
            raise RuntimeError(f"Failed to drop PG table: {e}") from e
        milvus_start = time.time()
        res = super().drop_collection(collection_name)
        logger.info("Milvus drop_collection completed in %.3f s", time.time() - milvus_start)
        return res

    # ------------------------------------------------------------------
    # Write ops with transactional shadow writes
    # ------------------------------------------------------------------
    @staticmethod
    def _synchronized(method):
        """Decorator to run method under instance-level lock."""
        from functools import wraps

        @wraps(method)
        def _wrapper(self, *args, **kwargs):
            with self._lock:
                return method(self, *args, **kwargs)

        return _wrapper

    @_synchronized
    def insert(self, collection_name: str, data: list[dict[str, Any]], **kwargs: Any):
        self._get_schema(collection_name)
        logger.info(f"Insert {len(data)} rows into '{collection_name}' …")

        # Prepare DataFrame for JSON/ARRAY serialisation
        df = pd.DataFrame(data)
        for field in self.json_fields:
            df[field] = df[field].apply(json.dumps)
        for field in self.array_fields:
            df[field] = df[field].apply(json.dumps)
        # Drop vector columns for PG if ignoring vectors
        if self.ignore_vector and self.float_vector_fields:
            df.drop(columns=[c for c in self.float_vector_fields if c in df.columns], inplace=True, errors="ignore")

        # Build INSERT SQL using efficient batch insert
        columns = list(df.columns)
        insert_sql = f"INSERT INTO {collection_name} ({', '.join(columns)}) VALUES %s"
        values = [tuple(row) for row in df.itertuples(index=False, name=None)]

        try:
            # --- PostgreSQL insert using execute_values for better performance ---
            logger.info(f"Executing PostgreSQL batch INSERT (rows={len(values)})…")
            t0 = time.time()
            with self.pg_conn.cursor() as cursor:
                execute_values(cursor, insert_sql, values, page_size=1000)
                logger.info(f"PostgreSQL batch INSERT executed (rowcount={cursor.rowcount}).")
            logger.info(f"PostgreSQL batch INSERT completed in {time.time() - t0:.3f} s.")
        except Exception as e:
            self.pg_conn.rollback()
            raise RuntimeError(f"PostgreSQL insert failed: {e}") from e

        try:
            # --- Milvus insert ---
            logger.info("Calling Milvus insert …")
            t0 = time.time()
            result = super().insert(collection_name, data, **kwargs)
            logger.info(f"Milvus insert completed in {time.time() - t0:.3f} s.")
            self.pg_conn.commit()
            logger.info("PostgreSQL committed. Milvus insert succeeded.")
            return result
        except Exception as e:
            self.pg_conn.rollback()
            raise RuntimeError(f"Milvus insert failed, PG rolled back: {e}") from e

    @_synchronized
    def upsert(self, collection_name: str, data: list[dict[str, Any]], **kwargs: Any):
        self._get_schema(collection_name)
        df = pd.DataFrame(data)
        for field in self.json_fields:
            df[field] = df[field].apply(json.dumps)
        for field in self.array_fields:
            df[field] = df[field].apply(json.dumps)
        # Drop vector columns for PG if ignoring vectors
        if self.ignore_vector and self.float_vector_fields:
            df.drop(columns=[c for c in self.float_vector_fields if c in df.columns], inplace=True, errors="ignore")
        cols = list(df.columns)
        updates = ", ".join([f"{col}=EXCLUDED.{col}" for col in cols])
        insert_sql = (
            f"INSERT INTO {collection_name} ({', '.join(cols)}) VALUES %s "
            f"ON CONFLICT ({self.primary_field}) DO UPDATE SET {updates}"
        )
        values = [tuple(row) for row in df.itertuples(index=False, name=None)]
        try:
            # --- PostgreSQL upsert using execute_values for better performance ---
            logger.info(f"Executing PostgreSQL batch UPSERT (rows={len(values)})…")
            t0 = time.time()
            with self.pg_conn.cursor() as cursor:
                execute_values(cursor, insert_sql, values, page_size=1000)
                logger.info(f"PostgreSQL batch UPSERT executed (rowcount={cursor.rowcount}).")
            logger.info(f"PostgreSQL batch UPSERT completed in {time.time() - t0:.3f} s.")
        except Exception as e:
            self.pg_conn.rollback()
            raise RuntimeError(f"PostgreSQL upsert failed: {e}") from e

        try:
            # --- Milvus upsert (insert with conflict) ---
            logger.info("Calling Milvus upsert …")
            t0 = time.time()
            result = super().upsert(collection_name, data, **kwargs)
            logger.info(f"Milvus upsert completed in {time.time() - t0:.3f} s.")
            self.pg_conn.commit()
            logger.info("PostgreSQL committed. Milvus upsert succeeded.")
            return result
        except Exception as e:
            self.pg_conn.rollback()
            raise RuntimeError(f"Milvus upsert failed, PG rolled back: {e}") from e

    @_synchronized
    def delete(self, collection_name: str, ids: list[int | str], **kwargs: Any):
        self._get_schema(collection_name)
        placeholder = ", ".join(["%s"] * len(ids))
        delete_sql = f"DELETE FROM {collection_name} WHERE {self.primary_field} IN ({placeholder});"
        try:
            with self.pg_conn.cursor() as cursor:
                cursor.execute(delete_sql, ids)
        except Exception as e:
            self.pg_conn.rollback()
            raise RuntimeError(f"PostgreSQL delete failed: {e}") from e

        try:
            result = super().delete(collection_name, ids=ids, **kwargs)
            self.pg_conn.commit()
            return result
        except Exception as e:
            self.pg_conn.rollback()
            raise RuntimeError(f"Milvus delete failed, PG rolled back: {e}") from e

    # ------------------------------------------------------------------
    # Read utilities (validation helpers)
    # ------------------------------------------------------------------
    @_synchronized
    def query(self, collection_name: str, filter: str = "", output_fields: list[str] | None = None):
        # Avoid mutable default argument
        if output_fields is None:
            output_fields = ["*"]

        # Fetch from Milvus
        milvus_res = super().query(collection_name, filter=filter, output_fields=output_fields)
        milvus_df = pd.DataFrame(milvus_res)

        # Fetch from PostgreSQL with equivalent filter
        sql_filter = self._milvus_filter_to_sql(filter) if filter else "TRUE"
        cols = ", ".join(output_fields)
        with self.pg_conn.cursor() as cursor:
            cursor.execute(f"SELECT {cols} FROM {collection_name} WHERE {sql_filter};")
            pg_rows = cursor.fetchall()
            # Handle column names for DataFrame
            if cursor.description:
                colnames = [str(desc[0]) for desc in cursor.description]
                pg_df = pd.DataFrame(pg_rows, columns=colnames)
            else:
                pg_df = pd.DataFrame(pg_rows)

        milvus_aligned, pg_aligned = self._align_df(milvus_df, pg_df)
        return milvus_aligned, pg_aligned

    @_synchronized
    def export(self, collection_name: str):
        with self.pg_conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {collection_name};")
            rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description] if cursor.description else []
        res = [dict(zip(colnames, r, strict=False)) for r in rows]
        df = pd.DataFrame(res)
        return df

    @_synchronized
    def count(self, collection_name: str):
        """Return counts from Milvus and PostgreSQL for the given collection."""
        try:
            milvus_count_res = super().query(collection_name, filter="", output_fields=["count(*)"])
            milvus_count = milvus_count_res[0]["count(*)"] if milvus_count_res else 0
        except Exception as e:
            logger.error(f"Failed to query Milvus count for collection '{collection_name}': {e}")
            milvus_count = 0

        try:
            # Check table existence
            with self.pg_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s;",
                    (collection_name,),
                )
                result = cursor.fetchone()
                exists = result[0] > 0 if result else False
                if exists:
                    cursor.execute(f"SELECT COUNT(*) FROM {collection_name};")
                    result = cursor.fetchone()
                    pg_count = int(result[0]) if result else 0
                else:
                    logger.error(f"PostgreSQL table '{collection_name}' does not exist.")
                    pg_count = 0
        except Exception as e:
            logger.error(f"Failed to query PostgreSQL count for collection '{collection_name}': {e}")
            pg_count = 0

        return {"milvus_count": milvus_count, "pg_count": pg_count}

    # ------------------------------------------------------------------
    # Comparison helpers (optional)
    # ------------------------------------------------------------------
    def _compare_df(self, milvus_df: pd.DataFrame, pg_df: pd.DataFrame):
        # Align first (primary key & column order) to ensure identical structure
        milvus_df, pg_df_aligned = self._align_df(milvus_df, pg_df)

        # Use tolerance on float comparison to avoid insignificant decimal differences
        milvus_dict = milvus_df.to_dict("list")
        pg_dict = pg_df_aligned.to_dict("list")
        diff = DeepDiff(
            milvus_dict,
            pg_dict,
            ignore_order=True,
            significant_digits=1,
        )
        return diff

    def query_result_compare(self, collection_name: str, filter: str = "", output_fields: list[str] | None = None):
        milvus_df, pg_df = self.query(collection_name, filter=filter, output_fields=output_fields)
        logger.info(f"Milvus query result:\n{milvus_df}")
        logger.info(f"PostgreSQL query result:\n{pg_df}")
        diff = self._compare_df(milvus_df, pg_df)
        if diff:
            logger.error(
                "Query result mismatch for collection '%s' with filter '%s' and output fields '%s'.",
                collection_name,
                filter,
                output_fields,
            )
        else:
            logger.info(
                "Query result match for collection '%s' with filter '%s' and output fields '%s'.",
                collection_name,
                filter,
                output_fields,
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

        def _to_py_list(val):
            """Ensure value is list of Python scalars (convert numpy types)."""
            if val is None:
                return val
            lst = list(val) if not isinstance(val, list) else val
            cleaned = []
            for item in lst:
                if isinstance(item, np.floating):
                    cleaned.append(float(item))
                elif isinstance(item, np.integer):
                    cleaned.append(int(item))
                else:
                    cleaned.append(item)
            return cleaned

        for field in self.array_fields + self.float_vector_fields:
            if field in milvus_df.columns:
                milvus_df[field] = milvus_df[field].apply(_to_py_list)
            if field in pg_df.columns:
                pg_df[field] = pg_df[field].apply(_to_py_list)

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
                        minv, maxv = float(np.min(values_array)), float(np.max(values_array))
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
                        if np.issubdtype(values_array.dtype, np.integer):
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
    # Entity comparison (Milvus vs PostgreSQL)
    # ------------------------------------------------------------------
    def get_all_primary_keys_from_milvus(self, collection_name: str, batch_size: int = 1000) -> list:
        """使用 query_iterator 遍历获取 Milvus 中的所有主键值."""
        self._get_schema(collection_name)
        
        logger.info(f"使用 query_iterator 获取集合 '{collection_name}' 的所有主键...")
        t0 = time.time()
        
        all_pks = []
        
        try:
            # 使用 query_iterator 遍历所有数据
            iterator = super().query_iterator(
                collection_name=collection_name,
                batch_size=batch_size,
                filter="",
                output_fields=[self.primary_field]
            )
            
            while True:
                result = iterator.next()
                if not result:
                    iterator.close()
                    break
                
                # 提取主键值
                pks_in_batch = [row[self.primary_field] for row in result]
                all_pks.extend(pks_in_batch)
                
                if len(all_pks) % (batch_size * 10) == 0:  # 每 10 个批次记录一次进度
                    logger.info(f"已获取 {len(all_pks)} 个主键...")
        
        except Exception as e:
            logger.error(f"使用 query_iterator 获取主键时出错: {e}")
            raise
        
        logger.info(f"从 Milvus 获取了 {len(all_pks)} 个主键，耗时 {time.time() - t0:.3f} 秒")
        return all_pks

    @_synchronized  
    def compare_primary_keys(self, collection_name: str) -> dict:
        """比较 Milvus 和 PostgreSQL 中的主键差异."""
        self._get_schema(collection_name)
        
        logger.info(f"开始比较集合 '{collection_name}' 的主键...")
        
        # 获取 Milvus 的所有主键
        milvus_pks = set(self.get_all_primary_keys_from_milvus(collection_name))
        
        # 获取 PostgreSQL 的所有主键
        logger.info("获取 PostgreSQL 中的所有主键...")
        t0 = time.time()
        with self.pg_conn.cursor() as cursor:
            cursor.execute(f"SELECT {self.primary_field} FROM {collection_name};")
            pg_pks_rows = cursor.fetchall()
        pg_pks = set(r[0] for r in pg_pks_rows)
        logger.info(f"从 PostgreSQL 获取了 {len(pg_pks)} 个主键，耗时 {time.time() - t0:.3f} 秒")
        
        # 计算差异
        only_in_milvus = milvus_pks - pg_pks
        only_in_pg = pg_pks - milvus_pks
        common_pks = milvus_pks & pg_pks
        
        result = {
            "milvus_count": len(milvus_pks),
            "pg_count": len(pg_pks),
            "common_count": len(common_pks),
            "only_in_milvus": sorted(list(only_in_milvus)),
            "only_in_pg": sorted(list(only_in_pg)),
            "has_differences": bool(only_in_milvus or only_in_pg)
        }
        
        if result["has_differences"]:
            logger.error(f"主键比较发现差异:")
            logger.error(f"  Milvus 总数: {result['milvus_count']}")
            logger.error(f"  PostgreSQL 总数: {result['pg_count']}")
            logger.error(f"  共同主键数: {result['common_count']}")
            logger.error(f"  仅在 Milvus 中: {len(only_in_milvus)} 个")
            logger.error(f"  仅在 PostgreSQL 中: {len(only_in_pg)} 个")
            
            # 显示部分差异示例（避免日志过长）
            if only_in_milvus:
                sample_milvus = list(only_in_milvus)[:10]
                logger.error(f"  仅在 Milvus 中的主键示例: {sample_milvus}")
            if only_in_pg:
                sample_pg = list(only_in_pg)[:10]
                logger.error(f"  仅在 PostgreSQL 中的主键示例: {sample_pg}")
        else:
            logger.info(f"主键比较通过:")
            logger.info(f"  Milvus 和 PostgreSQL 都有 {result['common_count']} 个相同的主键")
        
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
    ):  # full_scan controls whether to compare all data or just count
        """Compare entire collection data between Milvus and PostgreSQL in batches."""
        self._get_schema(collection_name)

        # 首先进行主键比较（如果启用）
        if compare_pks_first:
            logger.info("开始主键比较...")
            pk_comparison = self.compare_primary_keys(collection_name)
            if pk_comparison["has_differences"]:
                logger.error("主键比较发现差异，详细对比可能不准确")
                # 可以选择是否继续进行详细比较
                if not full_scan:
                    return False
            else:
                logger.info("主键比较通过，继续进行详细对比...")

        # Count check with retry for eventual consistency
        milvus_total = 0
        pg_total = 0
        for attempt in range(retry):
            count_res = self.count(collection_name)
            milvus_total = count_res["milvus_count"]
            pg_total = count_res["pg_count"]
            if milvus_total == pg_total:
                break

            logger.warning(
                f"Count mismatch for collection '{collection_name}' (attempt {attempt + 1}/{retry}): Milvus ({milvus_total}) vs PostgreSQL ({pg_total}). Retrying after {retry_interval}s…",
            )
            if attempt < retry - 1:
                time.sleep(retry_interval)

        # Final validation after retries
        check_res = True
        if milvus_total != pg_total:
            logger.error(
                f"Count mismatch for collection '{collection_name}' after {retry} attempts: Milvus ({milvus_total}) vs PostgreSQL ({pg_total}).",
            )
            check_res = False

        # If not full_scan, only do count check
        if not full_scan:
            if check_res:
                logger.info(
                    f"Count check passed for collection '{collection_name}': Milvus ({milvus_total}) vs PostgreSQL ({pg_total}). Skipping full data compare (full_scan=False)."
                )
            return check_res

        t0 = time.time()
        # 如果已经进行了主键比较，使用共同的主键进行详细对比
        if compare_pks_first and 'pk_comparison' in locals():
            # 使用共同的主键，避免由于主键不匹配导致的比较问题
            if pk_comparison["common_count"] > 0:
                # 从 PostgreSQL 获取共同主键（保持原有逻辑的一致性）
                with self.pg_conn.cursor() as cursor:
                    cursor.execute(f"SELECT {self.primary_field} FROM {collection_name};")
                    pks_rows = cursor.fetchall()
                pks = [r[0] for r in pks_rows]
                # 过滤出共同的主键
                common_pks_set = set(pk_comparison["only_in_milvus"] + pk_comparison["only_in_pg"])
                if common_pks_set:
                    # 如果有差异，只比较共同部分
                    pks = [pk for pk in pks if pk not in common_pks_set]
            else:
                pks = []
        else:
            # Retrieve primary keys from PG (原有逻辑)
            with self.pg_conn.cursor() as cursor:
                cursor.execute(f"SELECT {self.primary_field} FROM {collection_name};")
                pks_rows = cursor.fetchall()
            pks = [r[0] for r in pks_rows]

        total_pks = len(pks)
        if total_pks == 0:
            logger.info(f"No entities to compare for collection '{collection_name}'.")
            return True

        # Use concurrent processing for better performance
        max_workers = min(16, (total_pks + batch_size - 1) // batch_size)  # Limit concurrent threads
        compared = 0
        comparison_lock = threading.Lock()
        
        def compare_batch(batch_start: int, batch_pks: list, primary_field: str, ignore_vector: bool, 
                         float_vector_fields: list, pg_conn_str: str) -> tuple[int, bool]:
            """Compare a single batch between Milvus and PostgreSQL."""
            batch_end = min(batch_start + batch_size, total_pks)
            batch_num = batch_start // batch_size + 1
            total_batches = (total_pks + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: entities {batch_start + 1}-{batch_end}/{total_pks}")
            
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
                executor.submit(compare_batch, batch_start, batch_pks, self.primary_field, 
                              self.ignore_vector, self.float_vector_fields, self.pg_conn_str): (batch_start, batch_pks)
                for batch_start, batch_pks in batch_jobs
            }
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_start, batch_pks = future_to_batch[future]
                try:
                    batch_size_actual, has_differences = future.result()
                    if has_differences:
                        has_any_differences = True
                    
                    with comparison_lock:
                        compared += batch_size_actual
                        if compared in milestones:
                            logger.info(f"Comparison progress: {compared}/{total_pks} ({(compared * 100) // total_pks}%) done.")
                            
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
            if hasattr(self, 'pg_conn'):
                self.pg_conn.close()
        except Exception as e:
            logger.warning(f"Error closing PostgreSQL connection: {e}")
