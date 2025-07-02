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

2. **Type System** (`src/pymilvus_pg/types.py`):
   - Type aliases for primary keys, entities, filters, and operation results
   - Ensures type safety across the codebase

3. **Exception Hierarchy** (`src/pymilvus_pg/exceptions.py`):
   - Custom exceptions: ConnectionError, SchemaError, SyncError, ValidationError
   - Provides clear error handling for different failure scenarios

4. **Logging** (`src/pymilvus_pg/logger_config.py`):
   - Centralized logging using loguru
   - Configurable log levels with file output

### Key Operations Flow

1. **Data Synchronization**: All write operations (insert, upsert, delete) to Milvus are automatically mirrored to PostgreSQL
2. **Schema Mapping**: Milvus collection schemas are automatically mapped to PostgreSQL tables
3. **Query Validation**: Query results can be compared between Milvus and PostgreSQL to verify correctness
4. **Performance Optimization**: Uses connection pooling, streaming, and concurrent processing for 30-50% throughput improvement

### Testing Approach

- Tests use pytest with fixtures defined in `tests/conftest.py`
- Docker containers (Milvus + PostgreSQL) are automatically started for tests
- Test data includes various field types: vectors, JSON, arrays, scalars

### Development Notes

- Python 3.10+ required
- Uses PDM for dependency management
- Strict type checking with mypy configuration
- Ruff for linting and formatting (line length: 120)
- Pre-commit hooks available for code quality