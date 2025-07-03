"""automatic_lmdb_demo.py
Demonstration of automatic LMDB integration for error diagnosis.

This example shows how LMDB works transparently in the background
to help identify data inconsistencies.
"""

import random
import time
from pymilvus import DataType
from pymilvus_pg import MilvusPGClient


def main():
    """Demonstrate automatic LMDB error diagnosis."""

    # Create client - LMDB is enabled by default
    print("Creating MilvusPGClient (LMDB enabled by default)...")
    client = MilvusPGClient(
        uri="http://localhost:19530",
        pg_conn_str="postgresql://postgres:postgres@localhost:5432/postgres",
        # Note: We don't need to specify enable_lmdb=True, it's the default
    )

    collection_name = f"auto_lmdb_demo_{int(time.time())}"

    try:
        # Create collection
        print(f"\nCreating collection '{collection_name}'...")
        schema = client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("name", DataType.VARCHAR, max_length=100)
        schema.add_field("score", DataType.FLOAT)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=4)

        client.create_collection(collection_name, schema)
        client.load_collection(collection_name)

        # Insert data
        print("\nInserting 100 records...")
        data = []
        for i in range(100):
            data.append(
                {
                    "id": i,
                    "name": f"item_{i}",
                    "score": random.uniform(0, 100),
                    "embedding": [random.random() for _ in range(4)],
                }
            )
        client.insert(collection_name, data)

        # Normal comparison - should pass
        print("\n" + "=" * 60)
        print("NORMAL ENTITY COMPARISON (all databases consistent)")
        print("=" * 60)

        passed = client.entity_compare(collection_name, full_scan=True)
        if passed:
            print("✓ Entity comparison passed - all databases are consistent")
        else:
            print("✗ Entity comparison failed")

        # Simulate an inconsistency
        print("\n" + "=" * 60)
        print("SIMULATING DATABASE INCONSISTENCY")
        print("=" * 60)

        # Manually delete a record from PostgreSQL only (simulating an error)
        print("\nSimulating error: Deleting ID=50 from PostgreSQL only...")
        import psycopg2

        with psycopg2.connect(client.pg_conn_str) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DELETE FROM {collection_name} WHERE id = 50")
            conn.commit()

        # Now run entity_compare again - LMDB will automatically help diagnose
        print("\n" + "=" * 60)
        print("ENTITY COMPARISON WITH INCONSISTENCY")
        print("=" * 60)
        print("LMDB will automatically identify which database has the error...")

        passed = client.entity_compare(collection_name, full_scan=True)

        if not passed:
            print("\n✗ Entity comparison failed (as expected)")
            print("Check the logs above - LMDB automatically identified that:")
            print("  - PostgreSQL has an error (missing ID=50)")
            print("  - Milvus and LMDB agree that ID=50 should exist")

        # Show LMDB statistics
        print("\n" + "=" * 60)
        print("LMDB STATISTICS")
        print("=" * 60)

        if client.lmdb_manager:
            stats = client.lmdb_manager.get_stats()
            print(f"LMDB is tracking {stats['entries']} primary keys")
            print(f"Database location: {stats['db_path']}")
            print(f"Used size: {stats['used_size'] / 1024:.2f} KB")

        # Demonstrate that we can still use manual three-way validation
        print("\n" + "=" * 60)
        print("MANUAL THREE-WAY VALIDATION (optional)")
        print("=" * 60)

        result = client.three_way_pk_validation(collection_name, sample_size=10)
        print(f"Sampled {result['total_pks']} PKs:")
        print(f"  - Consistent: {result['consistent_pks']}")
        print(f"  - Inconsistent: {result['inconsistent_pks']}")

        if result["pg_errors"]:
            print(f"  - PostgreSQL errors: {result['pg_errors']}")

    finally:
        # Cleanup
        print(f"\nCleaning up collection '{collection_name}'...")
        try:
            client.drop_collection(collection_name)
            print("✓ Collection dropped")
        except Exception as e:
            print(f"Failed to drop collection: {e}")

        client.close()
        print("\nDemo completed!")


if __name__ == "__main__":
    main()
