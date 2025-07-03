"""test_lmdb_validation.py
Test cases for three-way validation with LMDB integration.
"""

import os
import random
import time

import pytest
from dotenv import load_dotenv
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient, PKStatus
from pymilvus_pg.lmdb_manager import PKOperation

load_dotenv()


@pytest.fixture
def pg_conn_str() -> str:
    """PostgreSQL connection string fixture."""
    return os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/postgres")


@pytest.fixture
def milvus_uri() -> str:
    """Milvus URI fixture."""
    return os.getenv("MILVUS_URI", "http://localhost:19530")


@pytest.fixture
def client_with_lmdb(pg_conn_str: str, milvus_uri: str) -> MilvusPGClient:
    """Create MilvusPGClient instance with LMDB enabled."""
    client = MilvusPGClient(
        uri=milvus_uri,
        pg_conn_str=pg_conn_str,
        enable_lmdb=True,
        lmdb_path=".test_lmdb",
    )
    yield client
    client.close()


@pytest.fixture
def test_collection_lmdb(client_with_lmdb: MilvusPGClient) -> str:
    """Create a test collection for LMDB tests."""
    collection_name = f"test_lmdb_{int(time.time())}"

    # Create schema
    schema = client_with_lmdb.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("text", DataType.VARCHAR, max_length=200)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)

    # Create collection
    client_with_lmdb.create_collection(collection_name, schema)

    # Create index on vector field
    index_params = IndexParams()
    index_params.add_index("vector", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
    client_with_lmdb.create_index(collection_name, index_params)

    # Load collection
    client_with_lmdb.load_collection(collection_name)

    yield collection_name

    # Cleanup
    try:
        client_with_lmdb.drop_collection(collection_name)
    except Exception:
        pass


def test_lmdb_basic_operations(client_with_lmdb: MilvusPGClient, test_collection_lmdb: str) -> None:
    """Test basic LMDB operations with insert, upsert, and delete."""
    collection_name = test_collection_lmdb

    # Insert data
    data = [{"id": i, "text": f"text_{i}", "vector": [random.random() for _ in range(4)]} for i in range(10)]
    client_with_lmdb.insert(collection_name, data)

    # Check LMDB has the primary keys
    lmdb_pks = client_with_lmdb.lmdb_manager.get_collection_pks(collection_name, PKStatus.EXISTS)
    assert len(lmdb_pks) == 10
    assert set(lmdb_pks) == set(range(10))

    # Delete some records
    client_with_lmdb.delete(collection_name, ids=[0, 1, 2])

    # Check LMDB shows deleted status
    for pk in [0, 1, 2]:
        state = client_with_lmdb.lmdb_manager.get_pk_state(collection_name, pk)
        assert state is not None
        assert state["status"] == PKStatus.DELETED.value
        assert state["operation"] == "delete"

    # Upsert data (update existing and add new)
    upsert_data = [
        {"id": 3, "text": "updated_3", "vector": [0.1, 0.2, 0.3, 0.4]},  # Update existing
        {"id": 10, "text": "new_10", "vector": [0.5, 0.6, 0.7, 0.8]},  # Add new
    ]
    client_with_lmdb.upsert(collection_name, upsert_data)

    # Check LMDB reflects upsert
    state_3 = client_with_lmdb.lmdb_manager.get_pk_state(collection_name, 3)
    assert state_3["operation"] == "upsert"
    assert state_3["status"] == PKStatus.EXISTS.value

    state_10 = client_with_lmdb.lmdb_manager.get_pk_state(collection_name, 10)
    assert state_10["operation"] == "upsert"
    assert state_10["status"] == PKStatus.EXISTS.value


def test_three_way_validation_consistent(client_with_lmdb: MilvusPGClient, test_collection_lmdb: str) -> None:
    """Test three-way validation when all databases are consistent."""
    collection_name = test_collection_lmdb

    # Insert data
    data = [{"id": i, "text": f"text_{i}", "vector": [random.random() for _ in range(4)]} for i in range(20)]
    client_with_lmdb.insert(collection_name, data)

    # Wait for sync
    time.sleep(1)

    # Perform three-way validation
    result = client_with_lmdb.three_way_pk_validation(collection_name)

    # All should be consistent
    assert result["total_pks"] == 20
    assert result["consistent_pks"] == 20
    assert result["inconsistent_pks"] == 0
    assert len(result["milvus_errors"]) == 0
    assert len(result["pg_errors"]) == 0
    assert len(result["lmdb_errors"]) == 0


def test_three_way_validation_with_inconsistencies(
    client_with_lmdb: MilvusPGClient, test_collection_lmdb: str, pg_conn_str: str
) -> None:
    """Test three-way validation with simulated inconsistencies."""
    collection_name = test_collection_lmdb

    # Insert initial data
    data = [{"id": i, "text": f"text_{i}", "vector": [random.random() for _ in range(4)]} for i in range(10)]
    client_with_lmdb.insert(collection_name, data)

    # Wait for sync
    time.sleep(1)

    # Simulate inconsistency: Manually delete from PostgreSQL only
    import psycopg2

    with psycopg2.connect(pg_conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {collection_name} WHERE id = 5")
        conn.commit()

    # Simulate inconsistency: Manually update LMDB to show a PK as deleted
    client_with_lmdb.lmdb_manager.record_pk_state(collection_name, 7, PKStatus.DELETED, PKOperation.DELETE)

    # Perform three-way validation
    result = client_with_lmdb.three_way_pk_validation(collection_name)

    # Should detect inconsistencies
    assert result["inconsistent_pks"] > 0

    # PK 5: exists in Milvus and LMDB, missing in PG (majority says exists)
    # So PG should be in error
    assert 5 in result["pg_errors"]

    # PK 7: exists in Milvus and PG, deleted in LMDB (majority says exists)
    # So LMDB should be in error
    assert 7 in result["lmdb_errors"]

    # Check details
    for detail in result["details"]:
        if detail["pk"] == 5:
            assert detail["in_milvus"] is True
            assert detail["in_pg"] is False
            assert detail["in_lmdb"] is True
            assert detail["correct_state"] == "exists"
        elif detail["pk"] == 7:
            assert detail["in_milvus"] is True
            assert detail["in_pg"] is True
            assert detail["in_lmdb"] is False
            assert detail["correct_state"] == "exists"


def test_lmdb_performance(client_with_lmdb: MilvusPGClient, test_collection_lmdb: str) -> None:
    """Test LMDB performance with larger dataset."""
    collection_name = test_collection_lmdb

    # Insert larger dataset
    batch_size = 1000
    num_batches = 5

    start_time = time.time()
    for batch in range(num_batches):
        data = [
            {"id": batch * batch_size + i, "text": f"text_{batch}_{i}", "vector": [random.random() for _ in range(4)]}
            for i in range(batch_size)
        ]
        client_with_lmdb.insert(collection_name, data)

    insert_time = time.time() - start_time
    total_records = batch_size * num_batches

    print(f"\nInserted {total_records} records in {insert_time:.2f}s")
    print(f"Rate: {total_records / insert_time:.0f} records/sec")

    # Test PK lookup performance
    start_time = time.time()
    all_pks = client_with_lmdb.lmdb_manager.get_collection_pks(collection_name, PKStatus.EXISTS)
    lookup_time = time.time() - start_time

    assert len(all_pks) == total_records
    print(f"Retrieved {len(all_pks)} PKs from LMDB in {lookup_time:.3f}s")

    # Test three-way validation performance on sample
    start_time = time.time()
    result = client_with_lmdb.three_way_pk_validation(collection_name, sample_size=100)
    validation_time = time.time() - start_time

    print(f"Three-way validation of 100 samples completed in {validation_time:.3f}s")
    assert result["total_pks"] == 100
    assert result["consistent_pks"] == 100


def test_lmdb_metadata_tracking(client_with_lmdb: MilvusPGClient, test_collection_lmdb: str) -> None:
    """Test LMDB metadata tracking for operations."""
    collection_name = test_collection_lmdb

    # Insert with metadata
    pk = 999
    data = [{"id": pk, "text": "test_metadata", "vector": [0.1, 0.2, 0.3, 0.4]}]
    client_with_lmdb.insert(collection_name, data)

    # Check metadata
    state = client_with_lmdb.lmdb_manager.get_pk_state(collection_name, pk)
    assert state is not None
    assert state["status"] == PKStatus.EXISTS.value
    assert state["operation"] == "insert"
    assert "timestamp" in state

    # Delete and check updated metadata
    client_with_lmdb.delete(collection_name, ids=[pk])

    state = client_with_lmdb.lmdb_manager.get_pk_state(collection_name, pk)
    assert state["status"] == PKStatus.DELETED.value
    assert state["operation"] == "delete"

    # Upsert and check
    client_with_lmdb.upsert(collection_name, data)

    state = client_with_lmdb.lmdb_manager.get_pk_state(collection_name, pk)
    assert state["status"] == PKStatus.EXISTS.value
    assert state["operation"] == "upsert"


def test_lmdb_stats(client_with_lmdb: MilvusPGClient, test_collection_lmdb: str) -> None:
    """Test LMDB statistics reporting."""
    # Get initial stats
    stats = client_with_lmdb.lmdb_manager.get_stats()
    assert "db_path" in stats
    assert "map_size" in stats
    assert "entries" in stats
    initial_entries = stats["entries"]

    # Add some data
    collection_name = test_collection_lmdb
    data = [{"id": i, "text": f"text_{i}", "vector": [random.random() for _ in range(4)]} for i in range(100)]
    client_with_lmdb.insert(collection_name, data)

    # Check stats again
    stats = client_with_lmdb.lmdb_manager.get_stats()
    assert stats["entries"] == initial_entries + 100

    print("\nLMDB Stats:")
    print(f"  Database: {stats['db_path']}")
    print(f"  Entries: {stats['entries']}")
    print(f"  Used size: {stats['used_size'] / 1024 / 1024:.2f} MB")
    print(f"  Map size: {stats['map_size'] / 1024 / 1024 / 1024:.2f} GB")
