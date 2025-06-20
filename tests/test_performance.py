"""
Performance tests for pymilvus-pg package.

This module contains performance benchmarks and stress tests
for the pymilvus-pg library.
"""

import statistics
import time

import pytest
from pymilvus import DataType

from tests.test_utils import SchemaBuilder, TestCollectionManager, TestDataGenerator


@pytest.mark.slow
@pytest.mark.integration
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_insert_performance(self, test_client):
        """Test insert operation performance with different batch sizes."""
        collection_manager = TestCollectionManager(test_client)
        
        try:
            # Create schema
            schema = (SchemaBuilder()
                     .add_primary_field("id")
                     .add_varchar_field("name")
                     .add_numeric_field("age")
                     .add_vector_field("embedding", dim=128)
                     .build(test_client))
            
            collection_name = f"perf_insert_{int(time.time() * 1000)}"
            
            # Test different batch sizes
            batch_sizes = [100, 500, 1000]
            performance_results = {}
            
            for batch_size in batch_sizes:
                # Create fresh collection for each test
                test_collection = collection_manager.create_collection_with_data(
                    f"{collection_name}_{batch_size}", schema, []
                )
                
                # Generate test data
                test_data = TestDataGenerator.generate_complex_data(
                    batch_size, embedding_dim=128
                )
                
                # Measure insert performance
                start_time = time.time()
                test_client.insert(test_collection, test_data)
                end_time = time.time()
                
                insert_time = end_time - start_time
                throughput = batch_size / insert_time
                
                performance_results[batch_size] = {
                    "insert_time": insert_time,
                    "throughput": throughput,
                    "records_per_second": throughput
                }
                
                print(f"Batch size {batch_size}: {insert_time:.2f}s, "
                      f"{throughput:.2f} records/sec")
                
                # Verify data consistency
                time.sleep(2)
                count_result = test_client.count(test_collection)
                assert count_result['milvus_count'] == batch_size
                assert count_result['pg_count'] == batch_size
            
            # Assert performance expectations
            # Larger batches should generally have better throughput
            assert performance_results[1000]["throughput"] >= performance_results[100]["throughput"] * 0.5
            
        finally:
            collection_manager.cleanup_all()

    def test_query_performance(self, test_client):
        """Test query operation performance."""
        collection_manager = TestCollectionManager(test_client)
        
        try:
            # Create schema
            schema = (SchemaBuilder()
                     .add_primary_field("id")
                     .add_varchar_field("name")
                     .add_numeric_field("age")
                     .add_numeric_field("score", DataType.DOUBLE)
                     .add_vector_field("embedding", dim=64)
                     .build(test_client))
            
            collection_name = f"perf_query_{int(time.time() * 1000)}"
            
            # Create collection with data
            test_data = TestDataGenerator.generate_complex_data(
                5000, embedding_dim=64
            )
            
            collection = collection_manager.create_collection_with_data(
                collection_name, schema, test_data
            )
            
            time.sleep(3)  # Allow data to be fully synchronized
            
            # Test different query patterns
            query_tests = [
                ("age > 30", "Simple numeric filter"),
                ("age > 30 and score < 90", "Multiple numeric filters"),
                ("name like 'complex_entity_%'", "String pattern matching"),
                ("age in [25, 30, 35]", "IN clause")
            ]
            
            query_times = []
            
            for filter_expr, description in query_tests:
                times = []
                
                # Run each query multiple times for accurate measurement
                for _ in range(5):
                    start_time = time.time()
                    results = test_client.query(collection, filter_expression=filter_expr)
                    end_time = time.time()
                    
                    query_time = end_time - start_time
                    times.append(query_time)
                
                avg_time = statistics.mean(times)
                query_times.append(avg_time)
                
                print(f"{description}: {avg_time:.3f}s avg ({len(results)} results)")
                
                # Basic assertion that query completed in reasonable time
                assert avg_time < 5.0, f"Query too slow: {avg_time}s for {description}"
            
            # Overall query performance should be reasonable
            overall_avg = statistics.mean(query_times)
            assert overall_avg < 2.0, f"Overall query performance too slow: {overall_avg}s"
            
        finally:
            collection_manager.cleanup_all()

    def test_comparison_performance(self, test_client):
        """Test data comparison operation performance."""
        collection_manager = TestCollectionManager(test_client)
        
        try:
            # Create schema
            schema = (SchemaBuilder()
                     .add_primary_field("id")
                     .add_varchar_field("name")
                     .add_numeric_field("age")
                     .add_vector_field("embedding", dim=32)
                     .build(test_client))
            
            collection_name = f"perf_compare_{int(time.time() * 1000)}"
            
            # Create collection with data
            test_data = TestDataGenerator.generate_complex_data(
                2000, embedding_dim=32
            )
            
            collection = collection_manager.create_collection_with_data(
                collection_name, schema, test_data
            )
            
            time.sleep(2)
            
            # Test entity comparison performance
            start_time = time.time()
            comparison_result = test_client.entity_compare(
                collection, batch_size=500, full_scan=True
            )
            end_time = time.time()
            
            comparison_time = end_time - start_time
            
            print(f"Entity comparison: {comparison_time:.2f}s for 2000 records")
            print(f"Batches processed: {comparison_result['total_batches']}")
            print(f"Failed batches: {comparison_result['failed_batches']}")
            
            # Assert performance and accuracy
            assert comparison_time < 30.0, f"Entity comparison too slow: {comparison_time}s"
            assert comparison_result['failed_batches'] == 0
            
            # Test primary key comparison performance
            start_time = time.time()
            pk_comparison = test_client.compare_primary_keys(collection)
            end_time = time.time()
            
            pk_comparison_time = end_time - start_time
            
            print(f"PK comparison: {pk_comparison_time:.2f}s")
            
            assert pk_comparison_time < 10.0, f"PK comparison too slow: {pk_comparison_time}s"
            assert len(pk_comparison['common']) == 2000
            
        finally:
            collection_manager.cleanup_all()


@pytest.mark.slow
@pytest.mark.integration
class TestStressTests:
    """Stress tests for the system."""

    def test_concurrent_operations_stress(self, test_client):
        """Test system behavior under concurrent operations."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        collection_manager = TestCollectionManager(test_client)
        
        try:
            # Create schema
            schema = (SchemaBuilder()
                     .add_primary_field("id")
                     .add_varchar_field("name")
                     .add_numeric_field("age")
                     .add_vector_field("embedding", dim=16)
                     .build(test_client))
            
            collection_name = f"stress_concurrent_{int(time.time() * 1000)}"
            
            # Create collection
            collection = collection_manager.create_collection_with_data(
                collection_name, schema, []
            )
            
            def insert_batch(batch_id: int, batch_size: int = 100):
                """Insert a batch of data."""
                data = TestDataGenerator.generate_complex_data(
                    batch_size, 
                    start_id=batch_id * batch_size,
                    embedding_dim=16
                )
                test_client.insert(collection, data)
                return batch_id, len(data)
            
            def query_data(query_id: int):
                """Query data."""
                filter_expr = f"age > {20 + (query_id % 30)}"
                results = test_client.query(collection, filter_expression=filter_expr)
                return query_id, len(results)
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit insert tasks
                insert_futures = [
                    executor.submit(insert_batch, i) 
                    for i in range(5)
                ]
                
                # Wait for inserts to complete
                insert_results = []
                for future in as_completed(insert_futures):
                    batch_id, count = future.result()
                    insert_results.append((batch_id, count))
                
                # Allow data to synchronize
                time.sleep(3)
                
                # Submit query tasks
                query_futures = [
                    executor.submit(query_data, i) 
                    for i in range(10)
                ]
                
                # Wait for queries to complete
                query_results = []
                for future in as_completed(query_futures):
                    query_id, count = future.result()
                    query_results.append((query_id, count))
            
            # Verify results
            total_inserted = sum(count for _, count in insert_results)
            assert total_inserted == 500, f"Expected 500 records, got {total_inserted}"
            
            # Verify data consistency
            final_count = test_client.count(collection)
            assert final_count['milvus_count'] == 500
            assert final_count['pg_count'] == 500
            
            # All queries should have completed successfully
            assert len(query_results) == 10
            
        finally:
            collection_manager.cleanup_all()

    def test_large_dataset_stress(self, test_client):
        """Test system with large dataset."""
        collection_manager = TestCollectionManager(test_client)
        
        try:
            # Create schema optimized for large dataset
            schema = (SchemaBuilder()
                     .add_primary_field("id")
                     .add_varchar_field("name", max_length=50)
                     .add_numeric_field("age")
                     .add_vector_field("embedding", dim=64)
                     .build(test_client))
            
            collection_name = f"stress_large_{int(time.time() * 1000)}"
            
            # Create collection
            collection = collection_manager.create_collection_with_data(
                collection_name, schema, []
            )
            
            # Insert data in batches
            total_records = 10000
            batch_size = 1000
            
            for i in range(0, total_records, batch_size):
                batch_data = TestDataGenerator.generate_complex_data(
                    batch_size, 
                    start_id=i,
                    embedding_dim=64
                )
                
                start_time = time.time()
                test_client.insert(collection, batch_data)
                insert_time = time.time() - start_time
                
                print(f"Inserted batch {i//batch_size + 1}: {insert_time:.2f}s")
                
                # Brief pause between batches
                time.sleep(0.5)
            
            # Allow full synchronization
            time.sleep(10)
            
            # Verify final count
            final_count = test_client.count(collection)
            print(f"Final count - Milvus: {final_count['milvus_count']}, "
                  f"PostgreSQL: {final_count['pg_count']}")
            
            assert final_count['milvus_count'] == total_records
            assert final_count['pg_count'] == total_records
            
            # Test query performance on large dataset
            start_time = time.time()
            results = test_client.query(collection, filter_expression="age > 30")
            query_time = time.time() - start_time
            
            print(f"Query on large dataset: {query_time:.2f}s ({len(results)} results)")
            assert query_time < 10.0, f"Query too slow on large dataset: {query_time}s"
            
            # Test comparison on large dataset (sample only)
            start_time = time.time()
            test_client.entity_compare(
                collection, 
                batch_size=1000, 
                full_scan=False  # Sample comparison for performance
            )
            comparison_time = time.time() - start_time
            
            print(f"Sample comparison: {comparison_time:.2f}s")
            assert comparison_time < 60.0, f"Comparison too slow: {comparison_time}s"
            
        finally:
            collection_manager.cleanup_all() 