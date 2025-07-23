"""lmdb_manager.py
LMDB manager for tracking primary key states as a third validation source.

This module provides LMDBManager class for maintaining a lightweight key-value
store that tracks the state of primary keys across write operations. It serves
as a tiebreaker when Milvus and PostgreSQL data diverge.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

import lmdb

from .exceptions import ConnectionError as MilvusPGConnectionError
from .logger_config import logger


class PKStatus(str, Enum):
    """Primary key status in the database."""

    EXISTS = "exists"
    DELETED = "deleted"


class PKOperation(str, Enum):
    """Type of operation performed on the primary key."""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"


class LMDBManager:
    """
    Manages LMDB database for primary key state tracking.

    This class provides a lightweight key-value store for tracking the existence
    and modification history of primary keys, serving as a third source of truth
    for data validation between Milvus and PostgreSQL.
    """

    def __init__(self, db_path: str | None = None, map_size: int = 100 * 1024 * 1024 * 1024):
        """
        Initialize LMDB manager.

        Parameters
        ----------
        db_path : str | None, optional
            Path to LMDB database directory. If None, uses default location.
        map_size : int, optional
            Maximum size of the database in bytes, default 100GB.
        """
        if db_path is None:
            db_path = os.path.join(os.getcwd(), ".pymilvus_pg_lmdb")

        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.map_size = map_size
        self._env: lmdb.Environment | None = None

        logger.info(f"Initializing LMDB at {self.db_path} with map_size={map_size / 1024 / 1024 / 1024:.2f}GB")

    def connect(self) -> None:
        """Open LMDB environment."""
        if self._env is None:
            try:
                self._env = lmdb.open(
                    str(self.db_path),
                    map_size=self.map_size,
                    max_dbs=10,
                    writemap=True,
                    metasync=False,
                    sync=False,
                )
                logger.info("LMDB connection established")
            except Exception as e:
                raise MilvusPGConnectionError(f"Failed to open LMDB: {e}") from e

    def close(self) -> None:
        """Close LMDB environment."""
        if self._env:
            self._env.close()
            self._env = None
            logger.info("LMDB connection closed")

    @contextmanager
    def transaction(self, write: bool = False) -> Generator[Any, None, None]:
        """
        Context manager for LMDB transactions.

        Parameters
        ----------
        write : bool, optional
            Whether this is a write transaction, default False.
        """
        if not self._env:
            self.connect()

        assert self._env is not None
        txn = self._env.begin(write=write)
        try:
            yield txn
            if write:
                txn.commit()
        except Exception:
            txn.abort()
            raise

    def record_pk_state(
        self,
        collection_name: str,
        pk: Any,
        status: PKStatus,
        operation: PKOperation,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record the state of a primary key.

        Parameters
        ----------
        collection_name : str
            Name of the collection
        pk : Any
            Primary key value
        status : PKStatus
            Current status of the key (EXISTS or DELETED)
        operation : PKOperation
            Operation that led to this state
        metadata : dict, optional
            Additional metadata to store
        """
        key = f"{collection_name}:{pk}".encode()
        value = {
            "status": status.value,
            "operation": operation.value,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        with self.transaction(write=True) as txn:
            txn.put(key, json.dumps(value).encode())

    def get_pk_state(self, collection_name: str, pk: Any) -> dict[str, Any] | None:
        """
        Get the state of a primary key.

        Parameters
        ----------
        collection_name : str
            Name of the collection
        pk : Any
            Primary key value

        Returns
        -------
        dict | None
            State information if found, None otherwise
        """
        key = f"{collection_name}:{pk}".encode()

        with self.transaction() as txn:
            value = txn.get(key)
            if value:
                return json.loads(value.decode())  # type: ignore[no-any-return]
        return None

    def batch_record_pk_states(
        self,
        collection_name: str,
        pk_states: list[tuple[Any, PKStatus, PKOperation]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record states for multiple primary keys in a single transaction.

        Parameters
        ----------
        collection_name : str
            Name of the collection
        pk_states : list[tuple]
            List of (pk, status, operation) tuples
        metadata : dict, optional
            Additional metadata to store for all keys
        """
        timestamp = time.time()

        with self.transaction(write=True) as txn:
            for pk, status, operation in pk_states:
                key = f"{collection_name}:{pk}".encode()
                value = {
                    "status": status.value,
                    "operation": operation.value,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                }
                txn.put(key, json.dumps(value).encode())

    def batch_record_pk_states_in_transaction(
        self,
        txn: Any,
        collection_name: str,
        pk_states: list[tuple[Any, PKStatus, PKOperation]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record states for multiple primary keys using an existing transaction.
        
        This method does not commit the transaction, allowing the caller to
        coordinate commits across multiple operations.
        
        Parameters
        ----------
        txn : lmdb.Transaction
            Active LMDB write transaction
        collection_name : str
            Name of the collection
        pk_states : list[tuple]
            List of (pk, status, operation) tuples
        metadata : dict, optional
            Additional metadata to store for all keys
        """
        timestamp = time.time()
        
        for pk, status, operation in pk_states:
            key = f"{collection_name}:{pk}".encode()
            value = {
                "status": status.value,
                "operation": operation.value,
                "timestamp": timestamp,
                "metadata": metadata or {},
            }
            txn.put(key, json.dumps(value).encode())

    def get_collection_pks(self, collection_name: str, status: PKStatus | None = None) -> list[Any]:
        """
        Get all primary keys for a collection, optionally filtered by status.

        Parameters
        ----------
        collection_name : str
            Name of the collection
        status : PKStatus, optional
            Filter by status if provided

        Returns
        -------
        list
            List of primary keys
        """
        prefix = f"{collection_name}:".encode()
        pks = []

        with self.transaction() as txn:
            cursor = txn.cursor()
            cursor.set_range(prefix)

            for key, value in cursor:
                if not key.startswith(prefix):
                    break

                if status is not None:
                    state = json.loads(value.decode())
                    if state["status"] != status.value:
                        continue

                # Extract PK from key
                pk_str = key.decode().split(":", 1)[1]
                # Try to convert to int if possible
                try:
                    pk = int(pk_str)
                except ValueError:
                    pk = pk_str
                pks.append(pk)

        return pks

    def clear_collection(self, collection_name: str) -> int:
        """
        Clear all data for a collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection to clear

        Returns
        -------
        int
            Number of keys deleted
        """
        prefix = f"{collection_name}:".encode()
        count = 0

        with self.transaction(write=True) as txn:
            cursor = txn.cursor()
            cursor.set_range(prefix)

            keys_to_delete = []
            for key, _ in cursor:
                if not key.startswith(prefix):
                    break
                keys_to_delete.append(key)

            for key in keys_to_delete:
                if txn.delete(key):
                    count += 1

        logger.info(f"Cleared {count} keys for collection '{collection_name}'")
        return count

    def get_stats(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns
        -------
        dict
            Database statistics including size and entry count
        """
        if not self._env:
            self.connect()

        assert self._env is not None
        stat = self._env.stat()
        info = self._env.info()

        return {
            "db_path": str(self.db_path),
            "map_size": info["map_size"],
            "used_size": stat["psize"] * (stat["branch_pages"] + stat["leaf_pages"] + stat["overflow_pages"]),
            "entries": stat["entries"],
            "depth": stat["depth"],
        }
