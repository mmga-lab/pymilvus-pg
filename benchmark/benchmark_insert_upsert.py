#!/usr/bin/env python3
"""
Benchmark script for PyMilvus-PG insert/upsert operations.

This benchmark measures the performance of insert and upsert operations
on both Milvus and PostgreSQL to evaluate synchronization overhead.
"""

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pymilvus import CollectionSchema, DataType, FieldSchema

from pymilvus_pg import MilvusPGClient

load_dotenv()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    collection_name: str = "benchmark_collection"
    record_counts: list[int] = None
    batch_sizes: list[int] = None
    vector_dims: list[int] = None
    concurrent_workers: list[int] = None
    runs_per_config: int = 3
    ignore_vector: bool = False
    sample_vector: bool = False
    vector_sample_size: int = 8
    warmup_runs: int = 1

    def __post_init__(self):
        if self.record_counts is None:
            self.record_counts = [100, 500, 1000, 5000, 10000]
        if self.batch_sizes is None:
            self.batch_sizes = [100, 500, 1000]
        if self.vector_dims is None:
            self.vector_dims = [128, 512]
        if self.concurrent_workers is None:
            self.concurrent_workers = [1, 4, 8]


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    operation: str
    record_count: int
    batch_size: int
    vector_dim: int
    concurrent_workers: int
    run_number: int
    duration_seconds: float
    throughput_records_per_sec: float
    pg_duration_seconds: float
    milvus_duration_seconds: float
    success: bool
    error_message: str | None = None


class InsertUpsertBenchmark:
    """Benchmark suite for insert/upsert operations."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client: MilvusPGClient | None = None
        self.results: list[BenchmarkResult] = []

    @staticmethod
    def _sanitize_collection_name(name: str) -> str:
        """Sanitize collection name for PostgreSQL compatibility."""
        # Replace hyphens and other special chars with underscores
        import re

        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Ensure it starts with a letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f"c_{sanitized}"
        return sanitized

    def setup_client(self) -> None:
        """Initialize the MilvusPGClient."""
        self.client = MilvusPGClient(
            uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            pg_conn_str=os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/default"),
            ignore_vector=self.config.ignore_vector,
            sample_vector=self.config.sample_vector,
            vector_sample_size=self.config.vector_sample_size,
        )

    def create_test_schema(self, vector_dim: int) -> CollectionSchema:
        """Create a test collection schema with various field types."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="text_field", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="int_field", dtype=DataType.INT32),
            FieldSchema(name="float_field", dtype=DataType.FLOAT),
            FieldSchema(name="json_field", dtype=DataType.JSON),
            FieldSchema(name="array_field", dtype=DataType.ARRAY, element_type=DataType.INT32, max_capacity=10),
            FieldSchema(name="vector_field", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        ]
        return CollectionSchema(fields=fields, description="Benchmark test collection")

    def generate_test_data(self, count: int, vector_dim: int) -> list[dict[str, Any]]:
        """Generate test data for benchmark."""
        data = []
        for i in range(count):
            record = {
                "id": i,
                "text_field": f"text_value_{i}_{random.randint(1000, 9999)}",
                "int_field": random.randint(1, 1000),
                "float_field": random.uniform(0.0, 100.0),
                "json_field": {"key": f"value_{i}", "number": random.randint(1, 100)},
                "array_field": [random.randint(1, 10) for _ in range(random.randint(1, 5))],
                "vector_field": np.random.random(vector_dim).tolist(),
            }
            data.append(record)
        return data

    def benchmark_insert(
        self,
        record_count: int,
        batch_size: int,
        vector_dim: int,
        concurrent_workers: int,
        run_number: int,
    ) -> BenchmarkResult:
        """Benchmark insert operation."""
        collection_name = self._sanitize_collection_name(f"{self.config.collection_name}_insert_{abs(run_number)}")

        try:
            # Setup collection
            schema = self.create_test_schema(vector_dim)
            self.client.create_collection(collection_name, schema)

            # Generate data
            data = self.generate_test_data(record_count, vector_dim)

            # Measure insert performance
            start_time = time.time()

            if concurrent_workers == 1:
                # Single-threaded insert
                for i in range(0, len(data), batch_size):
                    batch = data[i : i + batch_size]
                    self.client.insert(collection_name, batch)
            else:
                # Multi-threaded insert
                self._concurrent_insert(collection_name, data, batch_size, concurrent_workers)

            end_time = time.time()
            duration = end_time - start_time
            throughput = record_count / duration

            # Cleanup
            self.client.drop_collection(collection_name)

            return BenchmarkResult(
                operation="insert",
                record_count=record_count,
                batch_size=batch_size,
                vector_dim=vector_dim,
                concurrent_workers=concurrent_workers,
                run_number=run_number,
                duration_seconds=duration,
                throughput_records_per_sec=throughput,
                pg_duration_seconds=0.0,  # Not measured separately in this version
                milvus_duration_seconds=0.0,  # Not measured separately in this version
                success=True,
            )

        except Exception as e:
            return BenchmarkResult(
                operation="insert",
                record_count=record_count,
                batch_size=batch_size,
                vector_dim=vector_dim,
                concurrent_workers=concurrent_workers,
                run_number=run_number,
                duration_seconds=0.0,
                throughput_records_per_sec=0.0,
                pg_duration_seconds=0.0,
                milvus_duration_seconds=0.0,
                success=False,
                error_message=str(e),
            )

    def benchmark_upsert(
        self,
        record_count: int,
        batch_size: int,
        vector_dim: int,
        concurrent_workers: int,
        run_number: int,
    ) -> BenchmarkResult:
        """Benchmark upsert operation."""
        collection_name = self._sanitize_collection_name(f"{self.config.collection_name}_upsert_{abs(run_number)}")

        try:
            # Setup collection with initial data
            schema = self.create_test_schema(vector_dim)
            self.client.create_collection(collection_name, schema)

            # Insert initial data (50% of total)
            initial_count = record_count // 2
            initial_data = self.generate_test_data(initial_count, vector_dim)
            self.client.insert(collection_name, initial_data)

            # Generate upsert data (mix of updates and new records)
            upsert_data = []
            # 50% updates to existing records
            for i in range(initial_count // 2):
                record = {
                    "id": i,  # Existing ID
                    "text_field": f"updated_text_{i}_{random.randint(1000, 9999)}",
                    "int_field": random.randint(1, 1000),
                    "float_field": random.uniform(0.0, 100.0),
                    "json_field": {"key": f"updated_value_{i}", "number": random.randint(1, 100)},
                    "array_field": [random.randint(1, 10) for _ in range(random.randint(1, 5))],
                    "vector_field": np.random.random(vector_dim).tolist(),
                }
                upsert_data.append(record)

            # 50% new records
            new_records_count = record_count - len(upsert_data)
            for i in range(initial_count, initial_count + new_records_count):
                record = {
                    "id": i,  # New ID
                    "text_field": f"new_text_{i}_{random.randint(1000, 9999)}",
                    "int_field": random.randint(1, 1000),
                    "float_field": random.uniform(0.0, 100.0),
                    "json_field": {"key": f"new_value_{i}", "number": random.randint(1, 100)},
                    "array_field": [random.randint(1, 10) for _ in range(random.randint(1, 5))],
                    "vector_field": np.random.random(vector_dim).tolist(),
                }
                upsert_data.append(record)

            # Measure upsert performance
            start_time = time.time()

            if concurrent_workers == 1:
                # Single-threaded upsert
                for i in range(0, len(upsert_data), batch_size):
                    batch = upsert_data[i : i + batch_size]
                    self.client.upsert(collection_name, batch)
            else:
                # Multi-threaded upsert
                self._concurrent_upsert(collection_name, upsert_data, batch_size, concurrent_workers)

            end_time = time.time()
            duration = end_time - start_time
            throughput = len(upsert_data) / duration

            # Cleanup
            self.client.drop_collection(collection_name)

            return BenchmarkResult(
                operation="upsert",
                record_count=len(upsert_data),
                batch_size=batch_size,
                vector_dim=vector_dim,
                concurrent_workers=concurrent_workers,
                run_number=run_number,
                duration_seconds=duration,
                throughput_records_per_sec=throughput,
                pg_duration_seconds=0.0,
                milvus_duration_seconds=0.0,
                success=True,
            )

        except Exception as e:
            return BenchmarkResult(
                operation="upsert",
                record_count=record_count,
                batch_size=batch_size,
                vector_dim=vector_dim,
                concurrent_workers=concurrent_workers,
                run_number=run_number,
                duration_seconds=0.0,
                throughput_records_per_sec=0.0,
                pg_duration_seconds=0.0,
                milvus_duration_seconds=0.0,
                success=False,
                error_message=str(e),
            )

    def _concurrent_insert(
        self, collection_name: str, data: list[dict[str, Any]], batch_size: int, workers: int
    ) -> None:
        """Perform concurrent insert operations."""
        batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.client.insert, collection_name, batch) for batch in batches]

            for future in as_completed(futures):
                future.result()  # Wait for completion and raise any exceptions

    def _concurrent_upsert(
        self, collection_name: str, data: list[dict[str, Any]], batch_size: int, workers: int
    ) -> None:
        """Perform concurrent upsert operations."""
        batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.client.upsert, collection_name, batch) for batch in batches]

            for future in as_completed(futures):
                future.result()  # Wait for completion and raise any exceptions

    def run_benchmark_suite(self) -> None:
        """Run the complete benchmark suite."""
        print("Starting PyMilvus-PG Insert/Upsert Benchmark Suite")
        print("=" * 60)

        self.setup_client()

        total_configurations = (
            len(self.config.record_counts)
            * len(self.config.batch_sizes)
            * len(self.config.vector_dims)
            * len(self.config.concurrent_workers)
            * 2  # insert + upsert
        )

        current_config = 0

        for record_count in self.config.record_counts:
            for batch_size in self.config.batch_sizes:
                for vector_dim in self.config.vector_dims:
                    for workers in self.config.concurrent_workers:
                        current_config += 1

                        print(f"\nConfiguration {current_config}/{total_configurations}:")
                        print(
                            f"  Records: {record_count}, Batch: {batch_size}, Vector Dim: {vector_dim}, Workers: {workers}"
                        )

                        # Warmup runs
                        for warmup in range(self.config.warmup_runs):
                            print(f"  Warmup {warmup + 1}/{self.config.warmup_runs}...")
                            try:
                                warmup_id = 9000 + warmup  # Use high number to avoid conflicts
                                self.benchmark_insert(record_count, batch_size, vector_dim, workers, warmup_id)
                                self.benchmark_upsert(record_count, batch_size, vector_dim, workers, warmup_id)
                            except Exception as e:
                                print(f"    Warmup failed: {e}")

                        # Actual benchmark runs
                        for run in range(self.config.runs_per_config):
                            print(f"  Run {run + 1}/{self.config.runs_per_config}...")

                            # Insert benchmark
                            insert_result = self.benchmark_insert(record_count, batch_size, vector_dim, workers, run)
                            self.results.append(insert_result)
                            if insert_result.success:
                                print(f"    Insert: {insert_result.throughput_records_per_sec:.1f} records/sec")
                            else:
                                print(f"    Insert: FAILED - {insert_result.error_message}")

                            # Upsert benchmark
                            upsert_result = self.benchmark_upsert(record_count, batch_size, vector_dim, workers, run)
                            self.results.append(upsert_result)
                            if upsert_result.success:
                                print(f"    Upsert: {upsert_result.throughput_records_per_sec:.1f} records/sec")
                            else:
                                print(f"    Upsert: FAILED - {upsert_result.error_message}")

        print("\nBenchmark suite completed!")

    def export_results(self, output_file: str = "benchmark_results.json") -> None:
        """Export results to JSON file."""
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "operation": result.operation,
                    "record_count": result.record_count,
                    "batch_size": result.batch_size,
                    "vector_dim": result.vector_dim,
                    "concurrent_workers": result.concurrent_workers,
                    "run_number": result.run_number,
                    "duration_seconds": result.duration_seconds,
                    "throughput_records_per_sec": result.throughput_records_per_sec,
                    "success": result.success,
                    "error_message": result.error_message,
                }
            )

        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Results exported to {output_file}")

    def generate_summary_report(self) -> None:
        """Generate a summary report of benchmark results."""
        if not self.results:
            print("No results to summarize.")
            return

        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("No successful benchmark runs to summarize.")
            return

        print("\nBenchmark Summary Report")
        print("=" * 50)

        # Group results by operation
        insert_results = [r for r in successful_results if r.operation == "insert"]
        upsert_results = [r for r in successful_results if r.operation == "upsert"]

        for operation, results in [("Insert", insert_results), ("Upsert", upsert_results)]:
            if not results:
                continue

            print(f"\n{operation} Performance:")
            print("-" * 20)

            throughputs = [r.throughput_records_per_sec for r in results]
            print(f"  Average Throughput: {statistics.mean(throughputs):.1f} records/sec")
            print(f"  Median Throughput:  {statistics.median(throughputs):.1f} records/sec")
            print(f"  Max Throughput:     {max(throughputs):.1f} records/sec")
            print(f"  Min Throughput:     {min(throughputs):.1f} records/sec")
            if len(throughputs) > 1:
                print(f"  Std Deviation:      {statistics.stdev(throughputs):.1f} records/sec")

        # Best performing configurations
        print("\nBest Performing Configurations:")
        print("-" * 35)

        for operation in ["insert", "upsert"]:
            op_results = [r for r in successful_results if r.operation == operation]
            if not op_results:
                continue

            best = max(op_results, key=lambda x: x.throughput_records_per_sec)
            print(f"\n{operation.capitalize()}:")
            print(f"  Throughput: {best.throughput_records_per_sec:.1f} records/sec")
            print(f"  Records: {best.record_count}, Batch: {best.batch_size}")
            print(f"  Vector Dim: {best.vector_dim}, Workers: {best.concurrent_workers}")

        # Failure analysis
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print(f"\nFailures: {len(failed_results)} out of {len(self.results)} runs failed")
            error_types = {}
            for result in failed_results:
                error_msg = result.error_message or "Unknown error"
                error_types[error_msg] = error_types.get(error_msg, 0) + 1

            print("Most common errors:")
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {error[:50]}{'...' if len(error) > 50 else ''}: {count} times")


def create_config_from_args(args) -> BenchmarkConfig:
    """Create BenchmarkConfig from command line arguments."""
    config = BenchmarkConfig()

    if args.records:
        config.record_counts = args.records
    if args.batches:
        config.batch_sizes = args.batches
    if args.vector_dims:
        config.vector_dims = args.vector_dims
    if args.workers:
        config.concurrent_workers = args.workers
    if args.runs:
        config.runs_per_config = args.runs
    if args.warmup_runs:
        config.warmup_runs = args.warmup_runs

    config.ignore_vector = args.ignore_vector
    config.sample_vector = args.sample_vector
    config.vector_sample_size = args.vector_sample_size

    return config


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="PyMilvus-PG Insert/Upsert Benchmark")

    parser.add_argument(
        "--records",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000],
        help="Record counts to test (default: 100 500 1000 5000)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Batch sizes to test (default: 100 500 1000)",
    )
    parser.add_argument(
        "--vector-dims",
        type=int,
        nargs="+",
        default=[128, 512],
        help="Vector dimensions to test (default: 128 512)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Concurrent worker counts to test (default: 1 4 8)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs per configuration (default: 1)",
    )
    parser.add_argument(
        "--ignore-vector",
        action="store_true",
        help="Ignore vector fields in PostgreSQL operations",
    )
    parser.add_argument(
        "--sample-vector",
        action="store_true",
        help="Sample vector fields for PostgreSQL storage",
    )
    parser.add_argument(
        "--vector-sample-size",
        type=int,
        default=8,
        help="Vector sample size when sampling is enabled (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )

    args = parser.parse_args()

    # Create configuration
    config = create_config_from_args(args)

    # Run benchmark
    benchmark = InsertUpsertBenchmark(config)
    benchmark.run_benchmark_suite()

    # Export results and generate report
    benchmark.export_results(args.output)
    benchmark.generate_summary_report()


if __name__ == "__main__":
    main()
