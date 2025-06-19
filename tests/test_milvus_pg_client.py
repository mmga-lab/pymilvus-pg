"""
Test cases for MilvusPGClient functionality.

This module contains comprehensive tests for the MilvusPGClient class,
including basic operations, data synchronization, and validation features.
"""

import pytest
import time
import random
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient


class TestMilvusPGClientBasic:
    """Test basic MilvusPGClient functionality."""

    def test_client_initialization(self, milvus_uri, pg_conn_str):
        """Test client initialization with valid parameters."""
        client = MilvusPGClient(
            uri=milvus_uri,
            pg_conn_str=pg_conn_str
        )
        assert client is not None
        assert hasattr(client, 'pg_conn')
        assert hasattr(client, 'ignore_vector')
        client.pg_conn.close()

    def test_client_initialization_with_ignore_vector(self, milvus_uri, pg_conn_str):
        """Test client initialization with ignore_vector option."""
        client = MilvusPGClient(
            uri=milvus_uri,
            pg_conn_str=pg_conn_str,
            ignore_vector=True
        )
        assert client.ignore_vector is True
        client.pg_conn.close()

    def test_schema_creation(self, test_client):
        """Test schema creation functionality."""
        schema = test_client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("name", DataType.VARCHAR, max_length=100)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)
        
        assert schema is not None
        assert len(schema.fields) == 3

    def test_dtype_mapping(self):
        """Test Milvus to PostgreSQL data type mapping."""
        mapping_tests = [
            (DataType.BOOL, "BOOLEAN"),
            (DataType.INT64, "BIGINT"),
            (DataType.FLOAT, "REAL"),
            (DataType.DOUBLE, "DOUBLE PRECISION"),
            (DataType.VARCHAR, "VARCHAR"),
            (DataType.JSON, "JSONB"),
            (DataType.FLOAT_VECTOR, "DOUBLE PRECISION[]"),
        ]
        
        for milvus_type, expected_pg_type in mapping_tests:
            result = MilvusPGClient._milvus_dtype_to_pg(milvus_type)
            assert result == expected_pg_type


@pytest.mark.integration
class TestMilvusPGClientOperations:
    """Test CRUD operations with MilvusPGClient."""

    def test_create_collection(self, test_client, test_collection_name, basic_schema):
        """Test collection creation."""
        test_client.create_collection(test_collection_name, basic_schema)
        
        # Verify collection exists in Milvus
        collections = test_client.list_collections()
        assert test_collection_name in collections
        
        # Cleanup
        test_client.drop_collection(test_collection_name)

    def test_drop_collection(self, test_client, test_collection_name, basic_schema):
        """Test collection dropping."""
        # Create collection first
        test_client.create_collection(test_collection_name, basic_schema)
        
        # Drop collection
        test_client.drop_collection(test_collection_name)
        
        # Verify collection doesn't exist
        collections = test_client.list_collections()
        assert test_collection_name not in collections

    def test_insert_data(self, test_client, created_collection, sample_data):
        """Test data insertion."""
        collection_name = created_collection
        
        # Insert data
        result = test_client.insert(collection_name, sample_data[:5])
        assert result is not None
        
        # Verify data count
        time.sleep(1)  # Allow time for synchronization
        count = test_client.count(collection_name)
        assert count['milvus_count'] == 5

    def test_query_data(self, test_client, created_collection, sample_data):
        """Test data querying."""
        collection_name = created_collection
        
        # Insert data
        test_client.insert(collection_name, sample_data[:5])
        time.sleep(1)
        
        # Query data
        results = test_client.query(collection_name, filter_expression="age > 20")
        assert len(results) > 0
        
        # Verify result structure
        for result in results:
            assert 'id' in result
            assert 'name' in result
            assert 'age' in result

    def test_delete_data(self, test_client, created_collection, sample_data):
        """Test data deletion."""
        collection_name = created_collection
        
        # Insert data
        test_client.insert(collection_name, sample_data[:5])
        time.sleep(1)
        
        # Delete some data
        ids_to_delete = [0, 1, 2]
        test_client.delete(collection_name, ids=ids_to_delete)
        time.sleep(1)
        
        # Verify deletion
        count = test_client.count(collection_name)
        assert count['milvus_count'] == 2

    def test_upsert_data(self, test_client, created_collection, sample_data):
        """Test data upserting."""
        collection_name = created_collection
        
        # Insert initial data
        test_client.insert(collection_name, sample_data[:3])
        time.sleep(1)
        
        # Prepare upsert data (update existing + insert new)
        upsert_data = [
            {
                "id": 0,  # Update existing
                "name": "updated_entity_0",
                "age": 999,
                "score": 99.9,
                "is_active": False,
                "json_field": {"updated": True},
                "array_field": [999, 1000, 1001],
                "embedding": [random.random() for _ in range(8)]
            },
            {
                "id": 10,  # Insert new
                "name": "new_entity_10",
                "age": 30,
                "score": 88.0,
                "is_active": True,
                "json_field": {"new": True},
                "array_field": [10, 11, 12],
                "embedding": [random.random() for _ in range(8)]
            }
        ]
        
        # Perform upsert
        test_client.upsert(collection_name, upsert_data)
        time.sleep(1)
        
        # Verify count (3 original + 1 new = 4 total)
        count = test_client.count(collection_name)
        assert count['milvus_count'] == 4
        
        # Verify updated data
        updated_entity = test_client.query(collection_name, filter_expression="id == 0")
        assert len(updated_entity) == 1
        assert updated_entity[0]['name'] == "updated_entity_0"
        assert updated_entity[0]['age'] == 999


@pytest.mark.integration
class TestMilvusPGClientValidation:
    """Test validation and comparison features."""

    def test_export_data(self, test_client, created_collection, sample_data):
        """Test data export from PostgreSQL."""
        collection_name = created_collection
        
        # Insert data
        test_client.insert(collection_name, sample_data[:3])
        time.sleep(1)
        
        # Export data
        exported_data = test_client.export(collection_name)
        assert len(exported_data) == 3
        
        # Verify exported data structure
        for row in exported_data:
            assert 'id' in row
            assert 'name' in row

    def test_query_result_compare(self, test_client, created_collection, sample_data):
        """Test query result comparison between Milvus and PostgreSQL."""
        collection_name = created_collection
        
        # Insert data
        test_client.insert(collection_name, sample_data[:5])
        time.sleep(1)
        
        # Compare query results
        comparison_result = test_client.query_result_compare(
            collection_name, 
            filter_expression="age > 20"
        )
        
        assert 'is_equal' in comparison_result
        assert comparison_result['is_equal'] is True

    def test_entity_compare(self, test_client, created_collection, sample_data):
        """Test full entity comparison between Milvus and PostgreSQL."""
        collection_name = created_collection
        
        # Insert data
        test_client.insert(collection_name, sample_data[:3])
        time.sleep(1)
        
        # Perform entity comparison
        comparison_result = test_client.entity_compare(collection_name, batch_size=10)
        
        assert 'total_batches' in comparison_result
        assert 'successful_batches' in comparison_result
        assert 'failed_batches' in comparison_result

    def test_generate_milvus_filter(self, test_client, created_collection, sample_data):
        """Test automatic filter generation."""
        collection_name = created_collection
        
        # Insert data
        test_client.insert(collection_name, sample_data[:5])
        time.sleep(1)
        
        # Generate filters
        filters = test_client.generate_milvus_filter(collection_name, num_samples=2)
        
        assert isinstance(filters, list)
        assert len(filters) > 0
        
        # Test generated filters
        for filter_expr in filters[:2]:  # Test first 2 filters
            results = test_client.query(collection_name, filter_expression=filter_expr)
            assert isinstance(results, list)

    def test_compare_primary_keys(self, test_client, created_collection, sample_data):
        """Test primary key comparison between Milvus and PostgreSQL."""
        collection_name = created_collection
        
        # Insert data
        test_client.insert(collection_name, sample_data[:5])
        time.sleep(1)
        
        # Compare primary keys
        pk_comparison = test_client.compare_primary_keys(collection_name)
        
        assert 'milvus_only' in pk_comparison
        assert 'pg_only' in pk_comparison
        assert 'common' in pk_comparison
        assert len(pk_comparison['common']) == 5


@pytest.mark.slow
class TestMilvusPGClientPerformance:
    """Test performance-related functionality."""

    def test_batch_operations(self, test_client, created_collection):
        """Test batch operations with larger datasets."""
        collection_name = created_collection
        
        # Generate larger dataset
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                "id": i,
                "name": f"batch_entity_{i}",
                "age": 20 + (i % 50),
                "score": 85.0 + random.random() * 15,
                "is_active": i % 2 == 0,
                "json_field": {"batch": True, "index": i},
                "array_field": [i, i + 1, i + 2],
                "embedding": [random.random() for _ in range(8)]
            })
        
        # Insert in batch
        test_client.insert(collection_name, large_dataset)
        time.sleep(2)
        
        # Verify count
        count = test_client.count(collection_name)
        assert count['milvus_count'] == 100
        
        # Test batch query
        results = test_client.query(collection_name, filter_expression="age > 30")
        assert len(results) > 0


class TestMilvusPGClientEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_insert(self, test_client, created_collection):
        """Test inserting empty data."""
        collection_name = created_collection
        
        # Test empty list
        with pytest.raises(Exception):
            test_client.insert(collection_name, [])

    def test_invalid_filter_expression(self, test_client, created_collection, sample_data):
        """Test query with invalid filter expression."""
        collection_name = created_collection
        
        # Insert some data first
        test_client.insert(collection_name, sample_data[:3])
        time.sleep(1)
        
        # Test invalid filter (this should handle gracefully or raise appropriate error)
        try:
            results = test_client.query(collection_name, filter_expression="invalid_field > 0")
            # If no exception, results should be empty or handle gracefully
            assert isinstance(results, list)
        except Exception as e:
            # If exception is raised, it should be a meaningful error
            assert "invalid_field" in str(e) or "field" in str(e).lower()

    def test_nonexistent_collection_operations(self, test_client):
        """Test operations on non-existent collection."""
        nonexistent_collection = "nonexistent_collection_12345"
        
        with pytest.raises(Exception):
            test_client.query(nonexistent_collection)
            
        with pytest.raises(Exception):
            test_client.count(nonexistent_collection) 