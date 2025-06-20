"""
Utility functions and classes for testing.

This module provides helper functions and utilities used across
different test modules.
"""

import random
import time
from typing import Any

from pymilvus import DataType


class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_basic_data(count: int, start_id: int = 0) -> list[dict[str, Any]]:
        """Generate basic test data with id, name, age, and embedding fields."""
        return [
            {
                "id": start_id + i,
                "name": f"test_entity_{start_id + i}",
                "age": 20 + i,
                "embedding": [random.random() for _ in range(8)],
            }
            for i in range(count)
        ]

    @staticmethod
    def generate_complex_data(count: int, start_id: int = 0, embedding_dim: int = 8) -> list[dict[str, Any]]:
        """Generate complex test data with all field types."""
        return [
            {
                "id": start_id + i,
                "name": f"complex_entity_{start_id + i}",
                "age": 18 + (i % 50),
                "score": 85.0 + random.random() * 15,
                "is_active": i % 2 == 0,
                "json_field": {
                    "category": f"category_{i % 5}",
                    "metadata": {"index": i, "timestamp": int(time.time())},
                    "tags": [f"tag_{i}", f"tag_{i + 1}"],
                },
                "array_field": [i, i + 1, i + 2, i + 3],
                "embedding": [random.random() for _ in range(embedding_dim)],
            }
            for i in range(count)
        ]

    @staticmethod
    def generate_update_data(
        original_data: list[dict[str, Any]], fields_to_modify: list[str] = None
    ) -> list[dict[str, Any]]:
        """Generate updated version of existing data."""
        if fields_to_modify is None:
            fields_to_modify = ["name", "age", "score"]

        updated_data = []
        for item in original_data:
            updated_item = item.copy()

            if "name" in fields_to_modify:
                updated_item["name"] = f"updated_{item['name']}"
            if "age" in fields_to_modify and "age" in item:
                updated_item["age"] = item["age"] + 1000
            if "score" in fields_to_modify and "score" in item:
                updated_item["score"] = 99.9
            if "json_field" in fields_to_modify and "json_field" in item:
                updated_item["json_field"]["updated"] = True
            if "embedding" in item:
                updated_item["embedding"] = [random.random() for _ in range(len(item["embedding"]))]

            updated_data.append(updated_item)

        return updated_data


class SchemaBuilder:
    """Helper class to build schemas for testing."""

    def __init__(self):
        self.fields = []

    def add_primary_field(self, name: str = "id", dtype: DataType = DataType.INT64):
        """Add primary key field."""
        self.fields.append({"name": name, "dtype": dtype, "is_primary": True, "auto_id": False})
        return self

    def add_varchar_field(self, name: str, max_length: int = 100):
        """Add VARCHAR field."""
        self.fields.append({"name": name, "dtype": DataType.VARCHAR, "max_length": max_length})
        return self

    def add_numeric_field(self, name: str, dtype: DataType = DataType.INT64):
        """Add numeric field (INT64, FLOAT, DOUBLE)."""
        self.fields.append({"name": name, "dtype": dtype})
        return self

    def add_bool_field(self, name: str):
        """Add boolean field."""
        self.fields.append({"name": name, "dtype": DataType.BOOL})
        return self

    def add_json_field(self, name: str):
        """Add JSON field."""
        self.fields.append({"name": name, "dtype": DataType.JSON})
        return self

    def add_array_field(self, name: str, element_type: DataType = DataType.INT64, max_capacity: int = 10):
        """Add ARRAY field."""
        self.fields.append(
            {"name": name, "dtype": DataType.ARRAY, "element_type": element_type, "max_capacity": max_capacity}
        )
        return self

    def add_vector_field(self, name: str, dim: int = 8):
        """Add FLOAT_VECTOR field."""
        self.fields.append({"name": name, "dtype": DataType.FLOAT_VECTOR, "dim": dim})
        return self

    def build(self, client):
        """Build the schema using the client."""
        schema = client.create_schema()

        for field_config in self.fields:
            if field_config["dtype"] == DataType.VARCHAR:
                schema.add_field(
                    field_config["name"],
                    field_config["dtype"],
                    max_length=field_config["max_length"],
                    is_primary=field_config.get("is_primary", False),
                    auto_id=field_config.get("auto_id", False),
                )
            elif field_config["dtype"] == DataType.ARRAY:
                schema.add_field(
                    field_config["name"],
                    field_config["dtype"],
                    element_type=field_config["element_type"],
                    max_capacity=field_config["max_capacity"],
                )
            elif field_config["dtype"] == DataType.FLOAT_VECTOR:
                schema.add_field(field_config["name"], field_config["dtype"], dim=field_config["dim"])
            else:
                schema.add_field(
                    field_config["name"],
                    field_config["dtype"],
                    is_primary=field_config.get("is_primary", False),
                    auto_id=field_config.get("auto_id", False),
                )

        return schema


class TestCollectionManager:
    """Helper class to manage test collections."""

    def __init__(self, client):
        self.client = client
        self.created_collections = []

    def create_collection_with_data(
        self, collection_name: str, schema, data: list[dict[str, Any]], index_field: str = "embedding"
    ):
        """Create collection, add index, load, and insert data."""
        # Create collection
        self.client.create_collection(collection_name, schema)
        self.created_collections.append(collection_name)

        # Create index if vector field exists
        if index_field:
            from pymilvus.milvus_client import IndexParams

            index_params = IndexParams()
            index_params.add_index(index_field, metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
            self.client.create_index(collection_name, index_params)

        # Load collection
        self.client.load_collection(collection_name)

        # Insert data if provided
        if data:
            self.client.insert(collection_name, data)
            time.sleep(1)  # Allow synchronization

        return collection_name

    def cleanup_all(self):
        """Clean up all created collections."""
        for collection_name in self.created_collections:
            try:
                self.client.drop_collection(collection_name)
            except Exception:
                pass  # Collection might already be dropped
        self.created_collections.clear()


def wait_for_synchronization(seconds: float = 1.0):
    """Wait for data synchronization between Milvus and PostgreSQL."""
    time.sleep(seconds)


def assert_collection_consistency(client, collection_name: str):
    """Assert that Milvus and PostgreSQL have consistent data."""
    count_result = client.count(collection_name)
    assert count_result["milvus_count"] == count_result["pg_count"], (
        f"Data count mismatch: Milvus={count_result['milvus_count']}, PG={count_result['pg_count']}"
    )

    pk_comparison = client.compare_primary_keys(collection_name)
    assert len(pk_comparison["milvus_only"]) == 0, f"Found {len(pk_comparison['milvus_only'])} records only in Milvus"
    assert len(pk_comparison["pg_only"]) == 0, f"Found {len(pk_comparison['pg_only'])} records only in PostgreSQL"


def generate_filter_expressions(field_names: list[str], sample_count: int = 5) -> list[str]:
    """Generate various filter expressions for testing."""
    filters = []

    # Numeric comparisons
    for field in field_names:
        if field in ["age", "score", "id"]:
            filters.extend([f"{field} > 10", f"{field} < 100", f"{field} >= 20", f"{field} <= 50", f"{field} == 25"])

    # String comparisons
    for field in field_names:
        if field in ["name"]:
            filters.extend([f'{field} like "test%"', f'{field} like "%entity%"'])

    # Boolean comparisons
    for field in field_names:
        if field in ["is_active"]:
            filters.extend([f"{field} == true", f"{field} == false"])

    # Return a sample of filters
    return filters[:sample_count] if len(filters) > sample_count else filters
