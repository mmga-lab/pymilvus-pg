"""Built-in schema definitions for testing various Milvus features.

This module provides predefined schemas with rich data types, nullable fields,
default values, and dynamic field support for comprehensive testing.
"""

from typing import Any


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
            {"name": "sku", "type": "VARCHAR", "max_length": 100},
            {"name": "title", "type": "VARCHAR", "max_length": 500, "nullable": True},
            {"name": "description", "type": "VARCHAR", "max_length": 5000, "nullable": True},
            {"name": "brand", "type": "VARCHAR", "max_length": 200, "nullable": True, "default_value": "Generic"},
            {"name": "price", "type": "DOUBLE", "nullable": True, "default_value": 0.0},
            {"name": "discount_price", "type": "DOUBLE", "nullable": True},
            {"name": "currency", "type": "VARCHAR", "max_length": 3, "default_value": "USD"},
            {"name": "in_stock", "type": "BOOL", "default_value": True},
            {"name": "stock_quantity", "type": "INT64", "nullable": True, "default_value": 0},
            {"name": "rating", "type": "FLOAT", "nullable": True},
            {"name": "review_count", "type": "INT64", "default_value": 0},
            {"name": "categories", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 10, "max_length": 100},
            {"name": "tags", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 20, "max_length": 50},
            {"name": "attributes", "type": "JSON"},
            {"name": "specifications", "type": "JSON", "nullable": True},
            {"name": "image_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim},
            {"name": "text_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim},
            {"name": "created_at", "type": "INT64"},
            {"name": "updated_at", "type": "INT64", "nullable": True},
            {"name": "vendor_id", "type": "INT64", "nullable": True},
            {"name": "warehouse_location", "type": "VARCHAR", "max_length": 100, "nullable": True},
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
            {"name": "source_type", "type": "VARCHAR", "max_length": 50, "default_value": "document"},
            {"name": "language", "type": "VARCHAR", "max_length": 10, "nullable": True, "default_value": "en"},
            {"name": "chunk_index", "type": "INT64", "default_value": 0},
            {"name": "total_chunks", "type": "INT64", "nullable": True},
            {"name": "parent_doc_id", "type": "INT64", "nullable": True},
            {"name": "metadata", "type": "JSON"},
            {"name": "tags", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 15, "max_length": 50},
            {"name": "dense_vector", "type": "FLOAT_VECTOR", "dim": vector_dim},
            {"name": "created_at", "type": "INT64"},
            {"name": "processed_at", "type": "INT64", "nullable": True},
            {"name": "relevance_score", "type": "FLOAT", "nullable": True},
            {"name": "access_count", "type": "INT64", "default_value": 0},
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
            {"name": "media_type", "type": "VARCHAR", "max_length": 20},  # image, video, audio
            {"name": "file_name", "type": "VARCHAR", "max_length": 500},
            {"name": "file_path", "type": "VARCHAR", "max_length": 1000, "nullable": True},
            {"name": "file_size", "type": "INT64", "nullable": True},
            {"name": "mime_type", "type": "VARCHAR", "max_length": 100, "nullable": True},
            {"name": "duration_ms", "type": "INT64", "nullable": True},  # for video/audio
            {"name": "width", "type": "INT64", "nullable": True},  # for image/video
            {"name": "height", "type": "INT64", "nullable": True},  # for image/video
            {"name": "fps", "type": "FLOAT", "nullable": True},  # for video
            {"name": "bit_rate", "type": "INT64", "nullable": True},  # for audio/video
            {"name": "title", "type": "VARCHAR", "max_length": 500, "nullable": True},
            {"name": "description", "type": "VARCHAR", "max_length": 5000, "nullable": True},
            {"name": "tags", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 30, "max_length": 50},
            {"name": "metadata", "type": "JSON"},
            {"name": "technical_metadata", "type": "JSON", "nullable": True},
            {"name": "image_vector", "type": "FLOAT_VECTOR", "dim": image_vector_dim},
            {"name": "audio_vector", "type": "FLOAT_VECTOR", "dim": audio_vector_dim, "nullable": True},
            {"name": "thumbnail_vector", "type": "FLOAT_VECTOR", "dim": 256, "nullable": True},
            {"name": "created_at", "type": "INT64"},
            {"name": "modified_at", "type": "INT64", "nullable": True},
            {"name": "view_count", "type": "INT64", "default_value": 0},
            {"name": "like_count", "type": "INT64", "default_value": 0},
            {"name": "is_public", "type": "BOOL", "default_value": True},
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
            {"name": "device_type", "type": "VARCHAR", "max_length": 50},
            {"name": "location_id", "type": "VARCHAR", "max_length": 100, "nullable": True},
            {"name": "timestamp", "type": "INT64"},
            {"name": "temperature", "type": "DOUBLE", "nullable": True},
            {"name": "humidity", "type": "DOUBLE", "nullable": True},
            {"name": "pressure", "type": "DOUBLE", "nullable": True},
            {"name": "battery_level", "type": "FLOAT", "nullable": True, "default_value": 100.0},
            {"name": "signal_strength", "type": "INT64", "nullable": True},
            {"name": "is_online", "type": "BOOL", "default_value": True},
            {"name": "error_code", "type": "INT64", "nullable": True, "default_value": 0},
            {"name": "sensor_values", "type": "ARRAY", "element_type": "DOUBLE", "max_capacity": 10},
            {"name": "metadata", "type": "JSON"},
            {"name": "anomaly_score", "type": "FLOAT", "nullable": True},
            {"name": "feature_vector", "type": "FLOAT_VECTOR", "dim": vector_dim},
            {"name": "raw_data", "type": "JSON", "nullable": True},
            {"name": "processed", "type": "BOOL", "default_value": False},
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
            {"name": "display_name", "type": "VARCHAR", "max_length": 200, "nullable": True},
            {"name": "email", "type": "VARCHAR", "max_length": 255, "nullable": True},
            {"name": "bio", "type": "VARCHAR", "max_length": 5000, "nullable": True},
            {"name": "location", "type": "VARCHAR", "max_length": 200, "nullable": True},
            {"name": "verified", "type": "BOOL", "default_value": False},
            {"name": "follower_count", "type": "INT64", "default_value": 0},
            {"name": "following_count", "type": "INT64", "default_value": 0},
            {"name": "post_count", "type": "INT64", "default_value": 0},
            {"name": "account_type", "type": "VARCHAR", "max_length": 50, "default_value": "personal"},
            {"name": "interests", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 20, "max_length": 100},
            {"name": "languages", "type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 5, "max_length": 10},
            {"name": "preferences", "type": "JSON"},
            {"name": "profile_metadata", "type": "JSON", "nullable": True},
            {"name": "bio_embedding", "type": "FLOAT_VECTOR", "dim": text_vector_dim},
            {"name": "profile_image_embedding", "type": "FLOAT_VECTOR", "dim": image_vector_dim, "nullable": True},
            {"name": "created_at", "type": "INT64"},
            {"name": "last_active", "type": "INT64", "nullable": True},
            {"name": "reputation_score", "type": "FLOAT", "nullable": True, "default_value": 0.0},
            {"name": "is_active", "type": "BOOL", "default_value": True},
            {"name": "privacy_settings", "type": "JSON", "default_value": {"public": True}},
        ],
        "enable_dynamic_field": enable_dynamic,
        "description": "Social media user profiles with embeddings",
    }

    return schema


def get_test_all_types_schema(
    vector_dim: int = 128,
    enable_dynamic: bool = False,
) -> dict[str, Any]:
    """Test schema with all supported Milvus data types.

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
            # Primary key
            {"name": "id", "type": "INT64", "is_primary": True},
            # Scalar types - basic
            {"name": "bool_field", "type": "BOOL"},
            {"name": "int8_field", "type": "INT64"},
            {"name": "int16_field", "type": "INT64"},
            {"name": "int32_field", "type": "INT64"},
            {"name": "int64_field", "type": "INT64"},
            {"name": "float_field", "type": "FLOAT"},
            {"name": "double_field", "type": "DOUBLE"},
            {"name": "varchar_field", "type": "VARCHAR", "max_length": 1000},
            # Scalar types - with nullable
            {"name": "bool_nullable", "type": "BOOL", "nullable": True},
            {"name": "int32_nullable", "type": "INT64", "nullable": True},
            {"name": "float_nullable", "type": "FLOAT", "nullable": True},
            {"name": "varchar_nullable", "type": "VARCHAR", "max_length": 500, "nullable": True},
            # Scalar types - with default values
            {"name": "bool_default", "type": "BOOL", "default_value": True},
            {"name": "int32_default", "type": "INT64", "default_value": 42},
            {"name": "float_default", "type": "DOUBLE", "default_value": 3.14},
            {"name": "varchar_default", "type": "VARCHAR", "max_length": 100, "default_value": "default_text"},
            # Scalar types - nullable with defaults
            {"name": "int64_nullable_default", "type": "INT64", "nullable": True, "default_value": 100},
            {"name": "double_nullable_default", "type": "DOUBLE", "nullable": True, "default_value": 2.718},
            # Complex types
            {"name": "json_field", "type": "JSON"},
            {"name": "json_nullable", "type": "JSON", "nullable": True},
            {"name": "array_int", "type": "ARRAY", "element_type": "INT64", "max_capacity": 10},
            {
                "name": "array_varchar",
                "type": "ARRAY",
                "element_type": "VARCHAR",
                "max_capacity": 20,
                "max_length": 100,
            },
            {"name": "array_double", "type": "ARRAY", "element_type": "DOUBLE", "max_capacity": 15},
            # Vector types
            {"name": "float_vector", "type": "FLOAT_VECTOR", "dim": vector_dim},
            {"name": "float_vector_nullable", "type": "FLOAT_VECTOR", "dim": vector_dim},
        ],
        "enable_dynamic_field": enable_dynamic,
        "description": "Comprehensive test schema with all data types",
    }

    return schema


# Schema preset mapping
SCHEMA_PRESETS = {
    "ecommerce": get_ecommerce_schema,
    "document": get_document_schema,
    "multimedia": get_multimedia_schema,
    "iot": get_iot_timeseries_schema,
    "social": get_social_media_schema,
    "test_all": get_test_all_types_schema,
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
