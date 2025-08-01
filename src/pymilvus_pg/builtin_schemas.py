"""Built-in schema definitions for testing various Milvus features.

This module provides predefined schemas with rich data types, nullable fields,
default values, and dynamic field support for comprehensive testing.
"""

from typing import Any

import numpy as np


def get_ecommerce_schema(
    vector_dim: int = 768,
    enable_dynamic: bool = True,
    include_sparse_vector: bool = False,
) -> dict[str, Any]:
    """E-commerce product schema with rich data types.

    Features:
    - Multiple vector fields for multi-modal search
    - Nullable fields for optional attributes
    - Default values for common fields
    - JSON field for flexible attributes
    - Array fields for categories and tags

    Parameters
    ----------
    vector_dim : int
        Dimension for dense vector fields
    enable_dynamic : bool
        Enable dynamic field support
    include_sparse_vector : bool
        Include sparse vector field for keyword search

    Returns
    -------
    dict
        Schema configuration dictionary
    """
    schema = {
        "fields": [
            {"name": "product_id", "type": "INT64", "is_primary": True},
            {"name": "title", "type": "VARCHAR", "max_length": 500, "nullable": True},
            {"name": "price", "type": "DOUBLE", "default_value": 0.0},
            {"name": "in_stock", "type": "BOOL", "default_value": True},
            {"name": "rating", "type": "FLOAT", "nullable": True},
            {"name": "categories", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 10, "max_length": 100},
            {"name": "attributes", "type": "JSON"},
            {"name": "image_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim},
            {"name": "created_at", "type": "INT64"},
            {"name": "vendor_id", "type": "INT64", "nullable": True, "default_value": 0},
        ],
        "enable_dynamic_field": enable_dynamic,
        "description": "E-commerce product catalog with multi-modal embeddings",
    }

    if include_sparse_vector:
        fields = schema["fields"]
        if isinstance(fields, list):
            fields.append({"name": "keyword_sparse_vector", "type": "SPARSE_FLOAT_VECTOR"})

    return schema


def get_document_schema(
    vector_dim: int = 768,
    enable_dynamic: bool = True,
    include_sparse_vector: bool = True,
) -> dict[str, Any]:
    """Document/RAG schema for text processing and retrieval.

    Features:
    - Dense and sparse vectors for hybrid search
    - Metadata storage with JSON
    - Hierarchical document structure support
    - Nullable fields for optional metadata

    Parameters
    ----------
    vector_dim : int
        Dimension for dense vector field
    enable_dynamic : bool
        Enable dynamic field support
    include_sparse_vector : bool
        Include sparse vector for keyword search

    Returns
    -------
    dict
        Schema configuration dictionary
    """
    schema = {
        "fields": [
            {"name": "doc_id", "type": "INT64", "is_primary": True},
            {"name": "content", "type": "VARCHAR", "max_length": 65535},
            {"name": "title", "type": "VARCHAR", "max_length": 500, "nullable": True},
            {"name": "source", "type": "VARCHAR", "max_length": 500, "nullable": True, "default_value": "unknown"},
            {"name": "metadata", "type": "JSON"},
            {"name": "tags", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 15, "max_length": 50},
            {"name": "dense_vector", "type": "FLOAT_VECTOR", "dim": vector_dim},
            {"name": "created_at", "type": "INT64"},
        ],
        "enable_dynamic_field": enable_dynamic,
        "description": "Document storage for RAG applications",
    }

    if include_sparse_vector:
        fields = schema["fields"]
        if isinstance(fields, list):
            fields.append({"name": "sparse_vector", "type": "SPARSE_FLOAT_VECTOR"})

    return schema


def get_multimedia_schema(
    image_vector_dim: int = 512,
    audio_vector_dim: int = 128,
    enable_dynamic: bool = True,
) -> dict[str, Any]:
    """Multimedia schema for image, video, and audio data.

    Features:
    - Multiple vector types for different modalities
    - Rich metadata with JSON
    - Support for various media attributes
    - Nullable fields for optional properties

    Parameters
    ----------
    image_vector_dim : int
        Dimension for image vector field
    audio_vector_dim : int
        Dimension for audio vector field
    enable_dynamic : bool
        Enable dynamic field support

    Returns
    -------
    dict
        Schema configuration dictionary
    """
    schema = {
        "fields": [
            {"name": "media_id", "type": "INT64", "is_primary": True},
            {"name": "media_type", "type": "VARCHAR", "max_length": 20},
            {"name": "file_name", "type": "VARCHAR", "max_length": 500},
            {"name": "file_size", "type": "INT64", "nullable": True},
            {"name": "tags", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 30, "max_length": 50},
            {"name": "metadata", "type": "JSON"},
            {"name": "image_vector", "type": "FLOAT_VECTOR", "dim": image_vector_dim},
            {"name": "view_count", "type": "INT64", "default_value": 0},
        ],
        "enable_dynamic_field": enable_dynamic,
        "description": "Multimedia content storage with multi-modal embeddings",
    }

    return schema


def get_iot_timeseries_schema(
    vector_dim: int = 128,
    enable_dynamic: bool = True,
) -> dict[str, Any]:
    """IoT/Time-series schema for sensor data.

    Features:
    - Time-series data support
    - Sensor readings with nullable fields
    - JSON for complex sensor metadata
    - Embeddings for anomaly detection

    Parameters
    ----------
    vector_dim : int
        Dimension for vector field
    enable_dynamic : bool
        Enable dynamic field support

    Returns
    -------
    dict
        Schema configuration dictionary
    """
    schema = {
        "fields": [
            {"name": "reading_id", "type": "INT64", "is_primary": True},
            {"name": "device_id", "type": "VARCHAR", "max_length": 100},
            {"name": "timestamp", "type": "INT64"},
            {"name": "temperature", "type": "DOUBLE", "nullable": True},
            {"name": "battery_level", "type": "FLOAT", "nullable": True, "default_value": np.float32(100.0)},
            {"name": "sensor_values", "type": "ARRAY", "element_type": "DOUBLE", "max_capacity": 10},
            {"name": "metadata", "type": "JSON"},
            {"name": "feature_vector", "type": "FLOAT_VECTOR", "dim": vector_dim},
        ],
        "enable_dynamic_field": enable_dynamic,
        "description": "IoT sensor data with time-series support",
    }

    return schema


def get_social_media_schema(
    text_vector_dim: int = 768,
    image_vector_dim: int = 512,
    enable_dynamic: bool = True,
) -> dict[str, Any]:
    """Social media schema for user profiles and content.

    Features:
    - User profile data with preferences
    - Multi-modal content support
    - Social graph connections
    - Dynamic fields for user-defined attributes

    Parameters
    ----------
    text_vector_dim : int
        Dimension for text vector field
    image_vector_dim : int
        Dimension for image vector field
    enable_dynamic : bool
        Enable dynamic field support

    Returns
    -------
    dict
        Schema configuration dictionary
    """
    schema = {
        "fields": [
            {"name": "user_id", "type": "INT64", "is_primary": True},
            {"name": "username", "type": "VARCHAR", "max_length": 100},
            {"name": "bio", "type": "VARCHAR", "max_length": 5000, "nullable": True},
            {"name": "verified", "type": "BOOL", "default_value": False},
            {"name": "follower_count", "type": "INT64", "default_value": 0},
            {"name": "interests", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 20, "max_length": 100},
            {"name": "preferences", "type": "JSON"},
            {"name": "bio_embedding", "type": "FLOAT_VECTOR", "dim": text_vector_dim},
            {"name": "reputation_score", "type": "FLOAT", "nullable": True, "default_value": np.float32(0.0)},
        ],
        "enable_dynamic_field": enable_dynamic,
        "description": "Social media user profiles with embeddings",
    }

    return schema


def get_all_datatypes_schema(
    vector_dim: int = 128,
    enable_dynamic: bool = False,
) -> dict[str, Any]:
    """Schema demonstrating all supported Milvus data types.

    This schema is designed for comprehensive testing of data type handling
    and includes examples of all supported types with various configurations.

    Parameters
    ----------
    vector_dim : int
        Dimension for vector fields
    enable_dynamic : bool
        Enable dynamic field support

    Returns
    -------
    dict
        Schema configuration dictionary
    """
    schema = {
        "fields": [
            {"name": "id", "type": "INT64", "is_primary": True},
            {"name": "bool_field", "type": "BOOL"},
            {"name": "bool_nullable_field", "type": "BOOL", "nullable": True},
            {"name": "int_field", "type": "INT64", "nullable": True},
            {"name": "float_field", "type": "FLOAT", "default_value": np.float32(3.0)},
            {"name": "varchar_field", "type": "VARCHAR", "max_length": 1000},
            {"name": "varchar_nullable_default", "type": "VARCHAR", "max_length": 500, "nullable": True, "default_value": "default_text"},
            {"name": "json_field", "type": "JSON"},
            {"name": "array_int", "type": "ARRAY", "element_type": "INT64", "max_capacity": 10},
            {
                "name": "array_varchar",
                "type": "ARRAY",
                "element_type": "VARCHAR",
                "max_capacity": 20,
                "max_length": 100,
            },
            {"name": "float_vector", "type": "FLOAT_VECTOR", "dim": vector_dim},
            {"name": "double_field", "type": "DOUBLE", "nullable": True, "default_value": 2.718},
        ],
        "enable_dynamic_field": enable_dynamic,
        "description": "Schema demonstrating all Milvus data types with nullable and default examples",
    }

    return schema


# Schema preset mapping
SCHEMA_PRESETS = {
    "ecommerce": get_ecommerce_schema,
    "document": get_document_schema,
    "multimedia": get_multimedia_schema,
    "iot": get_iot_timeseries_schema,
    "social": get_social_media_schema,
    "all_datatypes": get_all_datatypes_schema,
}


def get_schema_by_name(name: str, **kwargs: Any) -> dict[str, Any]:
    """Get a schema preset by name with custom parameters.

    Parameters
    ----------
    name : str
        Name of the schema preset
    **kwargs : Any
        Additional parameters passed to the schema function

    Returns
    -------
    dict
        Schema configuration dictionary

    Raises
    ------
    ValueError
        If schema name is not found
    """
    if name not in SCHEMA_PRESETS:
        raise ValueError(f"Unknown schema preset: {name}. Available presets: {', '.join(SCHEMA_PRESETS.keys())}")

    schema_func = SCHEMA_PRESETS[name]
    return schema_func(**kwargs)  # type: ignore[no-any-return,operator]


def list_schema_presets() -> list[str]:
    """Get list of available schema presets.

    Returns
    -------
    list[str]
        List of schema preset names
    """
    return list(SCHEMA_PRESETS.keys())


def describe_schema_preset(name: str, **kwargs: Any) -> dict[str, Any]:
    """Get detailed description of a schema preset including field properties.

    Parameters
    ----------
    name : str
        Name of the schema preset
    **kwargs : Any
        Additional parameters passed to the schema function

    Returns
    -------
    dict
        Dictionary containing:
        - name: Schema preset name
        - description: Schema description
        - field_count: Number of fields
        - fields: List of field details with properties

    Raises
    ------
    ValueError
        If schema name is not found
    """
    if name not in SCHEMA_PRESETS:
        raise ValueError(f"Unknown schema preset: {name}. Available presets: {', '.join(SCHEMA_PRESETS.keys())}")

    schema = get_schema_by_name(name, **kwargs)
    fields_info = []

    for field in schema["fields"]:
        field_info = {
            "name": field["name"],
            "type": field["type"],
        }

        # Add primary key info
        if field.get("is_primary"):
            field_info["primary_key"] = True

        # Add nullable info
        if field.get("nullable"):
            field_info["nullable"] = True

        # Add default value info
        if "default_value" in field:
            field_info["default"] = field["default_value"]

        # Add type-specific properties
        if field["type"] == "VARCHAR":
            field_info["max_length"] = field.get("max_length")
        elif field["type"] in ["FLOAT_VECTOR", "BINARY_VECTOR"]:
            field_info["dim"] = field.get("dim")
        elif field["type"] == "ARRAY":
            field_info["element_type"] = field.get("element_type")
            field_info["max_capacity"] = field.get("max_capacity")
            if field.get("element_type") == "VARCHAR":
                field_info["max_length"] = field.get("max_length")

        fields_info.append(field_info)

    return {
        "name": name,
        "description": schema.get("description", ""),
        "field_count": len(schema["fields"]),
        "enable_dynamic_field": schema.get("enable_dynamic_field", False),
        "fields": fields_info,
    }
