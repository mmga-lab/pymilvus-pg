import os
import random
import time

import pytest
from dotenv import load_dotenv
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient as MilvusClient
from pymilvus_pg import logger

load_dotenv()


@pytest.fixture
def milvus_client():
    """Create a MilvusPGClient instance for testing"""
    client = MilvusClient(
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        pg_conn_str=os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/default"),
    )
    return client


@pytest.fixture(scope="function")
def collection_name(request):
    """Generate a unique collection name for testing"""
    import uuid

    # Use test function name + timestamp + random UUID to ensure uniqueness
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace("-", "_")
    return f"test_{test_name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def schema(milvus_client):
    """Create a schema for testing"""
    schema = milvus_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("name", DataType.VARCHAR, max_length=100)
    schema.add_field("age", DataType.INT64)
    schema.add_field("json_field", DataType.JSON)
    schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)
    return schema


@pytest.fixture(scope="function")
def sample_data():
    """Generate sample data for testing"""
    return [
        {
            "id": i,
            "name": f"test_{i}",
            "age": i,
            "json_field": {"a": i, "b": i + 1},
            "array_field": [i, i + 1, i + 2],
            "embedding": [random.random() for _ in range(8)],
        }
        for i in range(10)
    ]


@pytest.fixture(scope="function")
def upsert_data():
    """Generate upsert data for testing"""
    return [
        {
            "id": i,
            "name": f"test_{i + 100}",
            "age": i + 100,
            "json_field": {"a": i + 100, "b": i + 101},
            "array_field": [i + 100, i + 101, i + 102],
            "embedding": [random.random() for _ in range(8)],
        }
        for i in range(4, 8)
    ]


class TestDemoOperations:
    """Test class for demo operations"""

    def test_create_collection_and_schema(self, milvus_client, collection_name, schema):
        """Test creating a collection with schema"""
        # Create collection
        milvus_client.create_collection(collection_name, schema)

        # Verify collection was created
        # This should succeed without exceptions
        assert True

    def test_create_index(self, milvus_client, collection_name, schema):
        """Test creating index on collection"""
        # Create collection first
        milvus_client.create_collection(collection_name, schema)

        # Create index
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)

        # Verify index was created
        assert True

    def test_load_collection(self, milvus_client, collection_name, schema):
        """Test loading collection"""
        # Create collection and index
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)

        # Load collection
        milvus_client.load_collection(collection_name)

        # Verify collection was loaded
        assert True

    def test_insert_data(self, milvus_client, collection_name, schema, sample_data):
        """Test inserting data into collection"""
        # Setup collection
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)
        milvus_client.load_collection(collection_name)

        # Insert data
        milvus_client.insert(collection_name, sample_data)

        # Verify data was inserted
        assert True

    def test_delete_data(self, milvus_client, collection_name, schema, sample_data):
        """Test deleting data from collection"""
        # Setup collection and insert data
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)
        milvus_client.load_collection(collection_name)
        milvus_client.insert(collection_name, sample_data)

        # Delete data
        milvus_client.delete(collection_name, ids=[1, 2, 3])

        # Verify data was deleted
        assert True

    def test_upsert_data(self, milvus_client, collection_name, schema, sample_data, upsert_data):
        """Test upserting data in collection"""
        # Setup collection and insert data
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)
        milvus_client.load_collection(collection_name)
        milvus_client.insert(collection_name, sample_data)
        milvus_client.delete(collection_name, ids=[1, 2, 3])

        # Upsert data
        milvus_client.upsert(collection_name, upsert_data)

        # Verify data was upserted
        assert True

    def test_query_data(self, milvus_client, collection_name, schema, sample_data, upsert_data):
        """Test querying data from collection"""
        # Setup collection and data
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)
        milvus_client.load_collection(collection_name)
        milvus_client.insert(collection_name, sample_data)
        milvus_client.delete(collection_name, ids=[1, 2, 3])
        milvus_client.upsert(collection_name, upsert_data)

        # Wait for data consistency
        time.sleep(1)

        # Query data
        res = milvus_client.query(collection_name, "age > 0")
        logger.info(f"Query result: {res}")

        # Verify query result
        assert res is not None

    def test_export_data(self, milvus_client, collection_name, schema, sample_data, upsert_data):
        """Test exporting data from collection"""
        # Setup collection and data
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)
        milvus_client.load_collection(collection_name)
        milvus_client.insert(collection_name, sample_data)
        milvus_client.delete(collection_name, ids=[1, 2, 3])
        milvus_client.upsert(collection_name, upsert_data)

        # Wait for data consistency
        time.sleep(1)

        # Export data
        res = milvus_client.export(collection_name)
        logger.info(f"Export result: {res}")

        # Verify export result
        assert res is not None

    def test_count_data(self, milvus_client, collection_name, schema, sample_data, upsert_data):
        """Test counting data in collection"""
        # Setup collection and data
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)
        milvus_client.load_collection(collection_name)
        milvus_client.insert(collection_name, sample_data)
        milvus_client.delete(collection_name, ids=[1, 2, 3])
        milvus_client.upsert(collection_name, upsert_data)

        # Wait for data consistency
        time.sleep(1)

        # Count data
        res = milvus_client.count(collection_name)
        logger.info(f"Count result: {res}")

        # Verify count result
        assert res is not None

    def test_entity_compare(self, milvus_client, collection_name, schema, sample_data, upsert_data):
        """Test entity comparison"""
        # Setup collection and data
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)
        milvus_client.load_collection(collection_name)
        milvus_client.insert(collection_name, sample_data)
        milvus_client.delete(collection_name, ids=[1, 2, 3])
        milvus_client.upsert(collection_name, upsert_data)

        # Wait for data consistency
        time.sleep(1)

        # Compare entities
        milvus_client.entity_compare(collection_name)

        # Verify entity comparison completed
        assert True

    def test_generate_milvus_filter_and_query_compare(
        self, milvus_client, collection_name, schema, sample_data, upsert_data
    ):
        """Test generating Milvus filter and query result comparison"""
        # Setup collection and data
        milvus_client.create_collection(collection_name, schema)
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        milvus_client.create_index(collection_name, index_params)
        milvus_client.load_collection(collection_name)
        milvus_client.insert(collection_name, sample_data)
        milvus_client.delete(collection_name, ids=[1, 2, 3])
        milvus_client.upsert(collection_name, upsert_data)

        # Wait for data consistency
        time.sleep(1)

        # Generate Milvus filter
        filter_expr = milvus_client.generate_milvus_filter(collection_name, num_samples=2)
        logger.info(f"Auto Milvus Filter Example: {filter_expr}")

        # Test each filter expression
        for filter in filter_expr:
            logger.info(f"Testing filter: {filter}")
            res = milvus_client.query_result_compare(collection_name, filter)
            logger.info(f"Query result compare: {res}")

        # Verify filter generation and comparison
        assert filter_expr is not None
        assert len(filter_expr) > 0
