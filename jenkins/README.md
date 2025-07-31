# Jenkins Milvus Data Verify Configuration

This directory contains Jenkins pipeline configurations for comprehensive Milvus data verification. The pipelines are designed to run within the pymilvus-pg repository workspace and verify data consistency between Milvus and PostgreSQL shadow databases.

## Overview

The Jenkins test suite provides:
- **Data Validation Testing**: Comprehensive validation of data synchronization between Milvus and PostgreSQL
- **Schema Mapping Validation**: Tests schema consistency across Milvus collections and PostgreSQL tables
- **LMDB Integration Testing**: Three-way validation using LMDB as a tiebreaker for consistency checks
- **Operation Testing**: Insert, upsert, delete, query, and search operation validation
- **Connection Pool Testing**: Multi-connection scenarios (2-20 connections) for concurrent operations
- **Automated Verification**: Multi-level validation system with query correctness and data consistency testing

## Files Structure

```
jenkins/
├── pymilvus_pg_batch_test.groovy        # Batch data verify orchestrator (triggers multiple verification scenarios)
├── pymilvus_pg_stable_test.groovy       # Individual data verify executor (single verification scenario)
├── pods/
│   └── validation-test-client.yaml      # Kubernetes pod configuration
├── values/
│   ├── cluster-storagev1.yaml           # Cluster mode + Storage V1 configuration
│   └── cluster-storagev2.yaml           # Cluster mode + Storage V2 configuration
└── README.md                            # This documentation
```

## Pipeline Parameters

### pymilvus_pg_stable_test.groovy

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `image_repository` | Milvus Docker image repository | `harbor.milvus.io/milvus/milvus` | - |
| `image_tag` | Milvus Docker image tag | `master-latest` | - |
| `querynode_nums` | Number of QueryNodes | `3` | - |
| `datanode_nums` | Number of DataNodes | `3` | - |
| `proxy_nums` | Number of Proxy nodes | `1` | - |
| `keep_env` | Keep environment after test | `false` | `true`, `false` |
| `schema_preset` | Built-in schema preset | `ecommerce` | See [Schema Presets](#schema-presets) |
| `duration` | Test duration in seconds | `1800` | `0` means run indefinitely |
| `threads` | Number of writer threads | `4` | - |
| `compare_interval` | Seconds between validation checks | `60` | - |
| `include_vector` | Include vector fields in PostgreSQL operations | `true` | `true`, `false` |
| `storage_version` | Storage version | `V2` | `V1`, `V2` |

## Schema Presets

The data verify supports these built-in schema presets for comprehensive validation testing:

### Business Domain Schemas (Primary Focus)
- `ecommerce` - **E-commerce product catalog** with multi-modal embeddings (10 fields, dynamic: True)
- `document` - **Document storage** for RAG applications (9 fields, dynamic: True)
- `multimedia` - **Multimedia content storage** with multi-modal embeddings (8 fields, dynamic: True)
- `social` - **Social media user profiles** with embeddings (9 fields, dynamic: True)

### Specialized Schemas (Extended Coverage)
- `iot` - **IoT sensor data** with time-series support (8 fields, dynamic: True)
- `all_datatypes` - **All Milvus data types** with nullable and default examples (10 fields, dynamic: False)

Each schema includes:
- **Vector fields** with appropriate dimensions for semantic search
- **Various data types** including JSON, arrays, nullable fields
- **Dynamic field support** for schema evolution testing
- **Built-in validation** for data consistency across Milvus and PostgreSQL

## Validation Coverage

### Large-Scale Data Validation
- **Configuration**: Up to 1M records with various field types
- **Purpose**: Test data consistency at scale across Milvus and PostgreSQL
- **Vector Types**: Multiple dimensions (128d, 384d, 768d, 1536d)

### Concurrent Operation Testing  
- **Configuration**: 2-20 concurrent connections with batch operations
- **Purpose**: Test data consistency under concurrent load
- **Operations**: Simultaneous insert, upsert, delete, query operations

### Storage Version Validation
- **V1**: Traditional storage with consistency validation
- **V2**: New storage architecture with enhanced validation

### Multi-Partition Validation
- **Partitions**: Cross-partition data consistency checks
- **Operations**: Partition-specific operations with global consistency verification

## Message Queue Configuration

All cluster values files are pre-configured with:
- **Kafka**: Enabled (reliable message queue for validation workloads)  
- **Pulsar/PulsarV3**: Disabled
- **Cluster mode**: Only cluster deployment is supported (no standalone mode)
- **Resource allocation**: Optimized for validation testing with PostgreSQL integration

The validation process uses Kafka as the default reliable message queue for consistent data flow.

## Quick Start

### Prerequisites
- Jenkins environment with Kubernetes plugin
- Access to Kubernetes cluster in `chaos-testing` namespace
- Helm installed for Milvus deployment
- PostgreSQL database available for shadow database testing
- NFS storage available for data persistence

### Running Tests

The simplest way to run tests is using the batch data verify orchestrator which handles multiple validation scenarios automatically.

## Usage Examples

### Run Single Data Verify Test
```bash
# Trigger individual data verify test with specific parameters
jenkins> build 'pymilvus_pg_stable_test' with parameters:
  - schema_preset: 'ecommerce'
  - duration: '1800'
  - threads: '4'
  - compare_interval: '60'
  - include_vector: true
  - storage_version: 'V2'
```

### Run Batch Data Verify Tests
```bash
# Trigger comprehensive data verify test matrix
jenkins> build 'pymilvus_pg_batch_test' with parameters:
  - test_ecommerce: true
  - test_document: true
  - test_multimedia: true
  - test_iot: true
  - duration: '900'
  - include_vector: true
  - test_storage_v1: true
  - test_storage_v2: true
```

## Test Flow

The validation pipeline executes the following stages:

1. **Install Dependencies**: Install PDM and required Python packages
2. **Prepare Values**: Select appropriate Helm values file based on storage version (cluster mode only)
3. **Deploy Milvus**: Deploy Milvus cluster using Helm with configured parameters
4. **Setup PostgreSQL**: Initialize PostgreSQL shadow database for validation
5. **Wait for Stability**: Allow Milvus cluster and PostgreSQL to fully initialize
6. **Install PyMilvus-PG**: Install PyMilvus-PG library from current workspace using PDM
7. **Run PyMilvus-PG Validation**: Execute `pymilvus-pg ingest` with specified schema preset and parameters
8. **Final Validation Check**: Run `pymilvus-pg validate` to ensure data consistency
9. **Cleanup**: Archive logs and cleanup resources (unless keep_env=true)

### Key Commands Used
- `pymilvus-pg list-schemas` - List available schema presets
- `pymilvus-pg show-schema <preset>` - Display schema configuration
- `pymilvus-pg ingest` - Continuous data ingestion with validation
- `pymilvus-pg validate` - Data consistency validation (20% sample for performance)

## Output Artifacts

- `artifacts-{release-name}-validation-data.tar.gz` - Generated validation test data
- `artifacts-{release-name}-server-logs.tar.gz` - Milvus and PostgreSQL server logs
- `artifacts-{release-name}-validation-results.tar.gz` - Validation results and consistency reports
- Test summary with consistency metrics and performance comparisons

## Technical Notes

### Resource Configuration
- **Pod resources**: 64Gi memory limit, 16 CPU limit for validation testing
- **NFS storage**: Mounted at `/root/pymilvus_pg_data` for data persistence
- **Milvus cluster**: Optimized resource allocation for QueryNode, DataNode, IndexNode
- **PostgreSQL**: Dedicated instance with connection pooling (2-20 connections)

### Pipeline Features
- **Workspace execution**: Runs directly in pymilvus-pg repository (no external clone)
- **Timeout protection**: 120-minute timeout for validation operations
- **Three-way validation**: Milvus, PostgreSQL, and LMDB consistency checking
- **Concurrent testing**: Multi-connection validation with connection pooling
- **Automatic cleanup**: Resources cleaned unless `keep_env=true` is specified

### Supported Validation Scenarios
- **Data scales**: Up to 1M records with various field type combinations
- **Vector dimensions**: 128d, 384d, 768d, 1536d with metadata validation
- **Storage versions**: V1 (traditional) and V2 (optimized) architectures
- **Operation types**: Insert, upsert, delete, query, search with consistency validation
- **Concurrent testing**: 2-20 connection pools for multi-user scenarios

### Error Handling
- **Stage timeouts**: Each stage has appropriate timeout limits
- **Retry mechanisms**: Built-in retry for transient failures and connection issues
- **Log collection**: Comprehensive log archival for debugging validation failures
- **Resource cleanup**: Automatic cleanup prevents resource leaks
- **Consistency alerts**: Automated alerts for data inconsistency detection