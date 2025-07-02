# PyMilvus-PG Benchmark Suite

This benchmark suite evaluates the performance of insert and upsert operations in PyMilvus-PG, measuring the synchronization overhead between Milvus and PostgreSQL.

## Overview

The benchmark suite provides comprehensive performance testing for:
- **Insert operations**: Bulk data insertion into both Milvus and PostgreSQL
- **Upsert operations**: Mixed insert/update operations with conflict resolution
- **Concurrency**: Multi-threaded write operations
- **Vector handling**: Different vector dimensions and sampling strategies
- **Batch processing**: Various batch sizes for optimal throughput

## Quick Start

### Prerequisites

1. Ensure Docker services are running:
```bash
make docker-up
```

2. Install dependencies:
```bash
make install-dev
```

### Running Benchmarks

#### Option 1: Using Predefined Configurations

List available configurations:
```bash
python benchmark/run_benchmark.py --list-configs
```

Run a quick test:
```bash
python benchmark/run_benchmark.py --config benchmark/configs/quick_test.json
```

Run comprehensive benchmark:
```bash
python benchmark/run_benchmark.py --config benchmark/configs/comprehensive.json
```

#### Option 2: Using Command Line Arguments

Run with custom parameters:
```bash
python benchmark/benchmark_insert_upsert.py \
  --records 100 500 1000 \
  --batches 100 500 \
  --vector-dims 128 512 \
  --workers 1 4 8 \
  --runs 3
```

Run without vector synchronization (metadata only):
```bash
python benchmark/benchmark_insert_upsert.py \
  --records 1000 5000 \
  --ignore-vector \
  --workers 1 4 8 16
```

Run with vector sampling:
```bash
python benchmark/benchmark_insert_upsert.py \
  --records 1000 5000 \
  --sample-vector \
  --vector-sample-size 16 \
  --vector-dims 512 1024
```

## Benchmark Configurations

### Available Presets

1. **quick_test.json**: Fast benchmark for development
   - 100-500 records, 128D vectors, 1-2 workers
   - Suitable for quick validation

2. **comprehensive.json**: Full performance evaluation
   - 100K-10K records, 128-1024D vectors, 1-16 workers
   - Comprehensive performance profiling

3. **vector_sampling.json**: Vector sampling performance
   - Tests performance impact of vector sampling
   - 512-1024D vectors with 16-sample compression

4. **no_vectors.json**: Metadata-only synchronization
   - Ignores vector fields entirely
   - Tests PostgreSQL-only overhead

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `record_counts` | List of record counts to test | `[100, 500, 1000, 5000]` |
| `batch_sizes` | List of batch sizes for operations | `[100, 500, 1000]` |
| `vector_dims` | List of vector dimensions to test | `[128, 512]` |
| `concurrent_workers` | List of worker thread counts | `[1, 4, 8]` |
| `runs_per_config` | Repetitions per configuration | `3` |
| `ignore_vector` | Skip vector fields in PostgreSQL | `false` |
| `sample_vector` | Use vector sampling for PostgreSQL | `false` |
| `vector_sample_size` | Number of values per sampled vector | `8` |
| `warmup_runs` | Warmup iterations per configuration | `1` |

## Understanding Results

### Metrics

- **Throughput**: Records processed per second
- **Duration**: Total operation time in seconds
- **Success Rate**: Percentage of successful operations
- **Concurrency Impact**: Performance scaling with worker threads

### Sample Output

```
Configuration 1/8:
  Records: 1000, Batch: 500, Vector Dim: 128, Workers: 4
  Run 1/3...
    Insert: 2547.3 records/sec
    Upsert: 1823.6 records/sec

Benchmark Summary Report
=======================

Insert Performance:
------------------
  Average Throughput: 2234.5 records/sec
  Median Throughput:  2198.3 records/sec
  Max Throughput:     2547.3 records/sec

Best Performing Configurations:
------------------------------

Insert:
  Throughput: 2547.3 records/sec
  Records: 1000, Batch: 500
  Vector Dim: 128, Workers: 4
```

### Performance Analysis

#### Expected Performance Characteristics

1. **Insert vs Upsert**: Inserts typically 20-40% faster than upserts
2. **Batch Size Impact**: Larger batches improve throughput up to optimal point
3. **Concurrency Scaling**: Performance increases with workers up to CPU/IO limits
4. **Vector Dimension**: Higher dimensions reduce throughput due to data size
5. **Vector Sampling**: Can improve throughput by 30-50% for large vectors

#### Optimization Recommendations

Based on benchmark results:

- **Optimal Batch Size**: Usually 500-2000 records per batch
- **Concurrency**: 4-8 workers typically optimal for most configurations
- **Vector Handling**: 
  - Use `ignore_vector=true` for metadata-only validation
  - Use `sample_vector=true` for large vectors (>512D)
- **Record Count**: Performance per record improves with larger batches

## Troubleshooting

### Common Issues

1. **Connection Errors**:
   ```bash
   # Ensure services are running
   make docker-up
   
   # Check connection strings in .env
   MILVUS_URI=http://localhost:19530
   PG_CONN=postgresql://postgres:admin@localhost:5432/default
   ```

2. **Memory Issues**:
   - Reduce `record_counts` for large vector dimensions
   - Lower `concurrent_workers` if seeing memory pressure
   - Consider using `sample_vector=true` for large vectors

3. **Performance Inconsistency**:
   - Increase `warmup_runs` to reduce cold-start effects
   - Increase `runs_per_config` for more stable averages
   - Ensure no other heavy processes during benchmarking

### Environment Variables

Create a `.env` file with your configuration:
```bash
MILVUS_URI=http://localhost:19530
PG_CONN=postgresql://postgres:admin@localhost:5432/default
```

## Integration with Development Workflow

### Adding to CI/CD

Add performance regression testing:
```bash
# Quick performance check
python benchmark/run_benchmark.py --config benchmark/configs/quick_test.json

# Store results for comparison
python benchmark/compare_results.py baseline.json current.json
```

### Custom Configurations

Create your own configuration file:
```json
{
  "collection_name": "my_benchmark",
  "record_counts": [1000],
  "batch_sizes": [500],
  "vector_dims": [256],
  "concurrent_workers": [4],
  "runs_per_config": 5,
  "ignore_vector": false,
  "sample_vector": false
}
```

Then run:
```bash
python benchmark/run_benchmark.py --config my_config.json
```

## Performance Baselines

### Expected Throughput (Records/Second)

| Configuration | Insert | Upsert | Notes |
|---------------|--------|--------|-------|
| 1K records, 128D, 4 workers | ~2500 | ~1800 | Baseline |
| 5K records, 128D, 8 workers | ~3500 | ~2500 | Optimal batch |
| 1K records, 512D, 4 workers | ~1800 | ~1200 | Large vectors |
| 1K records, ignore vectors | ~4000 | ~3200 | Metadata only |
| 1K records, sampled vectors | ~3200 | ~2400 | Vector sampling |

*Note: Actual performance varies by hardware, network, and system configuration.*