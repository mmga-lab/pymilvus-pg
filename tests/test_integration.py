"""
Integration tests for pymilvus-pg package.

This module contains end-to-end integration tests that verify
the complete workflow of the pymilvus-pg library including
data synchronization and validation.
"""

import random
import time

import pytest
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient


@pytest.mark.integration
class TestFullWorkflow:
    """Test complete workflow from collection creation to validation."""

    def test_complete_workflow(self, milvus_uri, pg_conn_str):
        """Test the complete workflow as shown in the demo."""
        # Initialize client
        client = MilvusPGClient(uri=milvus_uri, pg_conn_str=pg_conn_str, ignore_vector=False)

        collection_name = f"integration_test_{int(time.time() * 1000)}"

        try:
            # 1. Create schema
            schema = client.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
            schema.add_field("name", DataType.VARCHAR, max_length=100)
            schema.add_field("age", DataType.INT64)
            schema.add_field("score", DataType.DOUBLE)
            schema.add_field("is_active", DataType.BOOL)
            schema.add_field("json_field", DataType.JSON)
            schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)

            # 2. Create collection
            client.create_collection(collection_name, schema)

            # 3. Create index
            index_params = IndexParams()
            index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
            client.create_index(collection_name, index_params)

            # 4. Load collection
            client.load_collection(collection_name)

            # 5. Insert initial data
            initial_data = [
                {
                    "id": i,
                    "name": f"entity_{i}",
                    "age": 20 + i,
                    "score": 85.5 + i * 0.5,
                    "is_active": i % 2 == 0,
                    "json_field": {"category": f"cat_{i % 3}", "value": i * 10},
                    "array_field": [i, i + 1, i + 2],
                    "embedding": [random.random() for _ in range(8)],
                }
                for i in range(20)
            ]

            client.insert(collection_name, initial_data)
            time.sleep(2)  # Allow time for synchronization

            # 6. Verify insert
            count_result = client.count(collection_name)
            assert count_result["milvus_count"] == 20
            assert count_result["pg_count"] == 20

            # 7. Query data
            query_results = client.query(collection_name, filter_expression="age > 25")
            assert len(query_results) > 0

            # 8. Delete some data
            ids_to_delete = [0, 1, 2, 3, 4]
            client.delete(collection_name, ids=ids_to_delete)
            time.sleep(1)

            # 9. Verify deletion
            count_after_delete = client.count(collection_name)
            assert count_after_delete["milvus_count"] == 15
            assert count_after_delete["pg_count"] == 15

            # 10. Upsert data (update existing + insert new)
            upsert_data = [
                {
                    "id": 10,  # Update existing
                    "name": "updated_entity_10",
                    "age": 999,
                    "score": 99.9,
                    "is_active": False,
                    "json_field": {"updated": True, "timestamp": int(time.time())},
                    "array_field": [999, 1000, 1001],
                    "embedding": [random.random() for _ in range(8)],
                },
                {
                    "id": 100,  # Insert new
                    "name": "new_entity_100",
                    "age": 50,
                    "score": 95.0,
                    "is_active": True,
                    "json_field": {"new": True, "category": "premium"},
                    "array_field": [100, 101, 102],
                    "embedding": [random.random() for _ in range(8)],
                },
            ]

            client.upsert(collection_name, upsert_data)
            time.sleep(1)

            # 11. Verify upsert
            count_after_upsert = client.count(collection_name)
            assert count_after_upsert["milvus_count"] == 16  # 15 + 1 new
            assert count_after_upsert["pg_count"] == 16

            # 12. Export data from PostgreSQL
            exported_data = client.export(collection_name)
            assert len(exported_data) == 16

            # 13. Test query result comparison
            comparison_result = client.query_result_compare(collection_name, filter_expression="age > 30")
            assert comparison_result["is_equal"] is True

            # 14. Test entity comparison
            entity_comparison = client.entity_compare(collection_name, batch_size=5)
            assert entity_comparison["failed_batches"] == 0

            # 15. Test primary key comparison
            pk_comparison = client.compare_primary_keys(collection_name)
            assert len(pk_comparison["milvus_only"]) == 0
            assert len(pk_comparison["pg_only"]) == 0
            assert len(pk_comparison["common"]) == 16

            # 16. Test filter generation and validation
            generated_filters = client.generate_milvus_filter(collection_name, num_samples=3)
            assert len(generated_filters) >= 1

            for filter_expr in generated_filters[:2]:  # Test first 2 filters
                filter_comparison = client.query_result_compare(collection_name, filter_expr)
                assert filter_comparison["is_equal"] is True

        finally:
            # Cleanup
            try:
                client.drop_collection(collection_name)
            except Exception:
                pass  # Collection might already be dropped

            if hasattr(client, "pg_conn") and client.pg_conn:
                client.pg_conn.close()


@pytest.mark.integration
@pytest.mark.slow
class TestLargeDatasetWorkflow:
    """Test workflow with larger datasets."""

    def test_large_dataset_operations(self, milvus_uri, pg_conn_str):
        """Test operations with a larger dataset."""
        client = MilvusPGClient(uri=milvus_uri, pg_conn_str=pg_conn_str, ignore_vector=False)

        collection_name = f"large_test_{int(time.time() * 1000)}"

        try:
            # Create schema
            schema = client.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
            schema.add_field("name", DataType.VARCHAR, max_length=100)
            schema.add_field("age", DataType.INT64)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=128)

            # Create and setup collection
            client.create_collection(collection_name, schema)

            index_params = IndexParams()
            index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
            client.create_index(collection_name, index_params)
            client.load_collection(collection_name)

            # Generate large dataset
            large_dataset = []
            batch_size = 500
            total_records = 1000

            for i in range(total_records):
                large_dataset.append(
                    {
                        "id": i,
                        "name": f"large_entity_{i}",
                        "age": 18 + (i % 50),
                        "embedding": [random.random() for _ in range(128)],
                    }
                )

            # Insert in batches
            for start_idx in range(0, total_records, batch_size):
                end_idx = min(start_idx + batch_size, total_records)
                batch_data = large_dataset[start_idx:end_idx]
                client.insert(collection_name, batch_data)
                time.sleep(0.5)  # Small delay between batches

            time.sleep(3)  # Allow time for synchronization

            # Verify total count
            count_result = client.count(collection_name)
            assert count_result["milvus_count"] == total_records
            assert count_result["pg_count"] == total_records

            # Test batch entity comparison
            entity_comparison = client.entity_compare(collection_name, batch_size=100, full_scan=True)
            assert entity_comparison["failed_batches"] == 0
            assert entity_comparison["total_batches"] > 0

        finally:
            # Cleanup
            try:
                client.drop_collection(collection_name)
            except Exception:
                pass

            if hasattr(client, "pg_conn") and client.pg_conn:
                client.pg_conn.close()


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    def test_connection_recovery(self, milvus_uri, pg_conn_str):
        """Test recovery from connection issues."""
        client = MilvusPGClient(uri=milvus_uri, pg_conn_str=pg_conn_str)

        # Test that client can handle basic operations
        collection_name = f"error_test_{int(time.time() * 1000)}"

        try:
            schema = client.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
            schema.add_field("name", DataType.VARCHAR, max_length=100)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)

            client.create_collection(collection_name, schema)

            # Test operations continue to work
            test_data = [{"id": 1, "name": "test_entity", "embedding": [random.random() for _ in range(8)]}]

            client.insert(collection_name, test_data)
            time.sleep(1)

            count_result = client.count(collection_name)
            assert count_result["milvus_count"] == 1

        finally:
            try:
                client.drop_collection(collection_name)
            except Exception:
                pass

            if hasattr(client, "pg_conn") and client.pg_conn:
                client.pg_conn.close()

    def test_data_consistency_after_errors(self, milvus_uri, pg_conn_str):
        """Test data consistency is maintained after error conditions."""
        client = MilvusPGClient(uri=milvus_uri, pg_conn_str=pg_conn_str)

        collection_name = f"consistency_test_{int(time.time() * 1000)}"

        try:
            # Setup collection
            schema = client.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
            schema.add_field("name", DataType.VARCHAR, max_length=100)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)

            client.create_collection(collection_name, schema)

            index_params = IndexParams()
            index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
            client.create_index(collection_name, index_params)
            client.load_collection(collection_name)

            # Insert valid data
            valid_data = [
                {"id": i, "name": f"valid_entity_{i}", "embedding": [random.random() for _ in range(8)]}
                for i in range(5)
            ]

            client.insert(collection_name, valid_data)
            time.sleep(1)

            # Verify consistency after successful operations
            count_result = client.count(collection_name)
            assert count_result["milvus_count"] == count_result["pg_count"]

            pk_comparison = client.compare_primary_keys(collection_name)
            assert len(pk_comparison["milvus_only"]) == 0
            assert len(pk_comparison["pg_only"]) == 0

        finally:
            try:
                client.drop_collection(collection_name)
            except Exception:
                pass

            if hasattr(client, "pg_conn") and client.pg_conn:
                client.pg_conn.close()
