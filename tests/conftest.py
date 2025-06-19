"""
Pytest configuration and shared fixtures for pymilvus-pg tests.

This module provides common test fixtures and configuration for testing
the pymilvus-pg library functionality.
"""

import os
import time
import pytest
import psycopg2
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient


@pytest.fixture(scope="session")
def milvus_uri():
    """Milvus server URI for testing."""
    return os.getenv("MILVUS_URI", "http://localhost:19530")


@pytest.fixture(scope="session")
def pg_conn_str():
    """PostgreSQL connection string for testing."""
    return os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/test_db")


@pytest.fixture(scope="session")
def test_client(milvus_uri, pg_conn_str):
    """Create a test client instance."""
    client = MilvusPGClient(
        uri=milvus_uri,
        pg_conn_str=pg_conn_str,
        ignore_vector=False
    )
    yield client
    # Cleanup after tests
    if hasattr(client, 'pg_conn') and client.pg_conn:
        client.pg_conn.close()


@pytest.fixture
def test_collection_name():
    """Generate unique collection name for each test."""
    return f"test_collection_{int(time.time() * 1000)}"


@pytest.fixture
def basic_schema(test_client):
    """Create a basic schema for testing."""
    schema = test_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("name", DataType.VARCHAR, max_length=100)
    schema.add_field("age", DataType.INT64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)
    return schema


@pytest.fixture
def complex_schema(test_client):
    """Create a complex schema with various field types for testing."""
    schema = test_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("name", DataType.VARCHAR, max_length=100)
    schema.add_field("age", DataType.INT64)
    schema.add_field("score", DataType.DOUBLE)
    schema.add_field("is_active", DataType.BOOL)
    schema.add_field("json_field", DataType.JSON)
    schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)
    return schema


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    import random
    return [
        {
            "id": i,
            "name": f"test_entity_{i}",
            "age": 20 + i,
            "score": 85.5 + i * 0.5,
            "is_active": i % 2 == 0,
            "json_field": {"category": f"cat_{i%3}", "value": i * 10},
            "array_field": [i, i + 1, i + 2],
            "embedding": [random.random() for _ in range(8)]
        }
        for i in range(10)
    ]


@pytest.fixture
def created_collection(test_client, test_collection_name, complex_schema):
    """Create a test collection and clean it up after the test."""
    # Create collection
    test_client.create_collection(test_collection_name, complex_schema)
    
    # Create index
    index_params = IndexParams()
    index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
    test_client.create_index(test_collection_name, index_params)
    
    # Load collection
    test_client.load_collection(test_collection_name)
    
    yield test_collection_name
    
    # Cleanup
    try:
        test_client.drop_collection(test_collection_name)
    except Exception:
        pass  # Collection might already be dropped


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment before each test."""
    # Add any setup needed before each test
    yield
    # Add any cleanup needed after each test


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring external services"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    ) 