# Three-Way Validation with LMDB

PyMilvus-PG now includes built-in three-way validation using LMDB (Lightning Memory-Mapped Database) as a third source of truth. This feature automatically helps identify which database has incorrect data when Milvus and PostgreSQL disagree.

**Note**: LMDB is enabled by default as an internal implementation. The existing `entity_compare()` method will automatically use LMDB for error diagnosis when inconsistencies are detected.

## Overview

The three-way validation system uses:
- **Milvus**: Primary vector database
- **PostgreSQL**: Shadow database for SQL-based validation
- **LMDB**: Lightweight key-value store tracking primary key states

When inconsistencies are detected between Milvus and PostgreSQL, LMDB acts as a tiebreaker using a "majority vote" approach to determine the correct state.

## Benefits

1. **Accurate Error Detection**: Identifies which specific database has incorrect data
2. **Lightweight Overhead**: LMDB only stores primary key states, not full data
3. **High Performance**: LMDB provides extremely fast lookups (millions of QPS)
4. **Operation History**: Tracks the last operation performed on each primary key

## Usage

### Basic Setup

```python
from pymilvus_pg import MilvusPGClient

# Default usage - LMDB is enabled automatically
client = MilvusPGClient(
    uri="http://localhost:19530",
    pg_conn_str="postgresql://user:pass@localhost/db"
)

# Customize LMDB settings if needed
client = MilvusPGClient(
    uri="http://localhost:19530",
    pg_conn_str="postgresql://user:pass@localhost/db",
    lmdb_path="/custom/path/to/lmdb",  # Default: .pymilvus_pg_lmdb
    lmdb_map_size=20 * 1024 * 1024 * 1024  # Default: 10GB
)
```

### Automatic Three-Way Validation

When using the standard `entity_compare()` method, LMDB is automatically used when inconsistencies are detected:

```python
# Standard entity comparison - LMDB is used automatically for error diagnosis
passed = client.entity_compare("my_collection", full_scan=True)
# If inconsistencies are found, LMDB helps identify which database is wrong
```

### Manual Three-Way Validation

You can also explicitly perform three-way validation:

```python
# Explicitly perform three-way validation for a collection
result = client.three_way_pk_validation("my_collection")

print(f"Total PKs checked: {result['total_pks']}")
print(f"Consistent PKs: {result['consistent_pks']}")
print(f"Inconsistent PKs: {result['inconsistent_pks']}")

if result['inconsistent_pks'] > 0:
    print(f"Milvus errors: {len(result['milvus_errors'])}")
    print(f"PostgreSQL errors: {len(result['pg_errors'])}")
    print(f"LMDB errors: {len(result['lmdb_errors'])}")
    
    # Show detailed information for first few inconsistencies
    for detail in result['details'][:5]:
        print(f"\nPK {detail['pk']}:")
        print(f"  In Milvus: {detail['in_milvus']}")
        print(f"  In PostgreSQL: {detail['in_pg']}")
        print(f"  In LMDB: {detail['in_lmdb']}")
        print(f"  Correct state: {detail['correct_state']}")
        print(f"  Vote count: {detail['vote_count']}/3")
```

### Validation with Sampling

For large datasets, you can validate a random sample:

```python
# Validate only 1000 random primary keys
result = client.three_way_pk_validation("my_collection", sample_size=1000)
```

### LMDB Statistics

```python
# Get LMDB database statistics
if client.lmdb_manager:
    stats = client.lmdb_manager.get_stats()
    print(f"LMDB entries: {stats['entries']:,}")
    print(f"Used size: {stats['used_size'] / 1024 / 1024:.2f} MB")
    print(f"Database path: {stats['db_path']}")
```

## How It Works

### Write Operations

All write operations (insert, upsert, delete) automatically update LMDB:

1. **Insert**: Records PK with status "exists" and operation "insert"
2. **Upsert**: Records PK with status "exists" and operation "upsert"
3. **Delete**: Records PK with status "deleted" and operation "delete"

### Validation Logic

When validating primary keys:

1. Fetch all PKs from Milvus, PostgreSQL, and LMDB
2. For each PK, check its presence in all three databases
3. If all three agree → Consistent
4. If they disagree → Use majority vote:
   - 2 or 3 databases say "exists" → PK should exist
   - 0 or 1 database says "exists" → PK should be deleted
5. Identify which database(s) disagree with the majority

### Example Scenarios

**Scenario 1**: PK exists in Milvus and LMDB, missing in PostgreSQL
- Majority vote: "exists" (2/3)
- Error: PostgreSQL is incorrect

**Scenario 2**: PK exists in Milvus only
- Majority vote: "deleted" (0/3 for exists)
- Error: Milvus is incorrect

**Scenario 3**: PK exists in all three databases
- All agree: Consistent

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_lmdb` | `True` | Enable/disable LMDB integration |
| `lmdb_path` | `.pymilvus_pg_lmdb` | Directory for LMDB database files |
| `lmdb_map_size` | 10GB | Maximum size of LMDB database |

## Performance Considerations

- **Write overhead**: ~5-10% additional time for LMDB updates
- **Storage**: Only stores PK + metadata (~100 bytes per key)
- **Memory**: Uses memory-mapped files (efficient OS caching)
- **Validation speed**: Can check millions of PKs per second

## Disabling LMDB

If you want to disable LMDB for any reason:

```python
client = MilvusPGClient(
    uri="...",
    pg_conn_str="...",
    enable_lmdb=False  # Disable LMDB
)
```

Note: Without LMDB, you cannot use `three_way_pk_validation()`. The standard `entity_compare()` method will still work for Milvus-PostgreSQL comparison.