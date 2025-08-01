# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyMilvus-PG is a Python library designed to validate Milvus (vector database) data correctness by synchronizing write operations to a PostgreSQL shadow database. It extends the standard pymilvus client to maintain parallel data copies for validation and comparison purposes.

## Development Commands

### Essential Commands
- `make install-dev` - Install all dependencies including dev tools
- `make lint` - Run ruff linting checks
- `make format` - Auto-format code with ruff
- `make typecheck` - Run mypy type checking
- `make test` - Run tests (automatically starts required Docker containers)
- `make test-cov` - Run tests with coverage report
- `make check` - Run all checks (lint, typecheck, test)

### Docker Environment
- `make docker-up` - Start Milvus and PostgreSQL containers
- `make docker-down` - Stop and clean up containers

### Build and Publish
- `make build` - Build distribution packages
- `make publish` - Publish to PyPI

### Running a Single Test
```bash
pdm run pytest tests/test_demo_operations.py::test_name -v
```

## Architecture

### Core Components

1. **MilvusPGClient** (`src/pymilvus_pg/milvus_pg_client.py`):
   - Main client class extending `pymilvus.MilvusClient`
   - Implements synchronized operations between Milvus and PostgreSQL
   - Features connection pooling (2-20 connections) and transaction management
   - Supports ignoring vector fields for metadata-only validation
   - Key methods include insert, upsert, delete, query, search, export, compare_entities

2. **Type System** (`src/pymilvus_pg/types.py`):
   - Type aliases for primary keys, entities, filters, and operation results
   - Ensures type safety across the codebase

3. **Exception Hierarchy** (`src/pymilvus_pg/exceptions.py`):
   - Custom exceptions: ConnectionError, SchemaError, SyncError, ValidationError
   - Additional specialized exceptions: CollectionNotFoundError, DataTypeMismatchError, FilterConversionError, TransactionError
   - Provides clear error handling for different failure scenarios

4. **Logging** (`src/pymilvus_pg/logger_config.py`):
   - Centralized logging using loguru
   - Configurable log levels with file output
   - Log files stored in `logs/` directory with timestamp naming

5. **LMDB Manager** (`src/pymilvus_pg/lmdb_manager.py`):
   - Third validation source using LMDB key-value store
   - Tracks primary key states (EXISTS/DELETED) and operations
   - Serves as tiebreaker when Milvus and PostgreSQL diverge
   - Default location: `.pymilvus_pg_lmdb/` directory

6. **Comparators** (`src/pymilvus_pg/comparators.py`):
   - High-performance data comparison replacing DeepDiff for large datasets
   - Vector sampling for efficient embedding comparisons
   - Schema-aware comparison logic with precision tolerance configuration

### Key Operations Flow

1. **Data Synchronization**: All write operations (insert, upsert, delete) to Milvus are automatically mirrored to PostgreSQL
2. **Schema Mapping**: Milvus collection schemas are automatically mapped to PostgreSQL tables with appropriate type conversions
3. **Query Validation**: Query results can be compared between Milvus and PostgreSQL to verify correctness
4. **Performance Optimization**: Uses connection pooling, streaming, and concurrent processing with ProcessPoolExecutor
5. **Three-Way Validation**: LMDB integration provides automatic error source identification when inconsistencies are detected

### Testing Approach

- Tests use pytest with fixtures defined in `tests/conftest.py`
- Docker containers (Milvus + PostgreSQL) are automatically started for tests
- Test data includes various field types: vectors, JSON, arrays, scalars
- Automatic cleanup of test collections after each test
- Session-scoped client fixture for performance

### Development Notes

- Python 3.10+ required
- Uses PDM for dependency management
- Strict type checking with mypy configuration
- Ruff for linting and formatting (line length: 120)
- Pre-commit hooks available for code quality
- PostgreSQL connection pooling with 2-20 connections
- Support for concurrent operations and batch processing

### CLI Interface

- Command-line tool available via `pymilvus-pg` command
- Built with Click framework for easy command-line operations
- Supports data generation, testing, and validation workflows
- Schema presets available in `src/pymilvus_pg/builtin_schemas.py`

### Environment Variables

Key environment variables for configuration:
- `MILVUS_URI` - Milvus connection URI (default: http://localhost:19530)
- `PG_CONN` - PostgreSQL connection string (default: postgresql://postgres:admin@localhost:5432/postgres)
- `PYMILVUS_PG_LOG_DIR` - Custom log directory (default: ./logs)