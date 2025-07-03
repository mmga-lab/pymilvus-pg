"""three_way_validation_demo.py
Demonstration of three-way validation with LMDB for data correctness verification.
"""

import random
import time
from pymilvus import DataType
from pymilvus_pg import MilvusPGClient


def main():
    """Demonstrate three-way validation functionality."""

    # Initialize client with LMDB enabled
    print("Initializing MilvusPGClient with LMDB support...")
    client = MilvusPGClient(
        uri="http://localhost:19530",
        pg_conn_str="postgresql://postgres:postgres@localhost:5432/postgres",
        enable_lmdb=True,  # Enable LMDB for three-way validation
        lmdb_path=".pymilvus_lmdb_demo",  # LMDB storage location
    )

    collection_name = f"demo_three_way_{int(time.time())}"

    try:
        # Create collection
        print(f"\nCreating collection '{collection_name}'...")
        schema = client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("title", DataType.VARCHAR, max_length=200)
        schema.add_field("category", DataType.VARCHAR, max_length=50)
        schema.add_field("price", DataType.FLOAT)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=128)

        client.create_collection(collection_name, schema)
        client.load_collection(collection_name)

        # Insert initial data
        print("\nInserting 1000 products...")
        products = []
        categories = ["Electronics", "Books", "Clothing", "Food", "Toys"]

        for i in range(1000):
            products.append(
                {
                    "id": i,
                    "title": f"Product {i}",
                    "category": random.choice(categories),
                    "price": round(random.uniform(10, 1000), 2),
                    "embedding": [random.random() for _ in range(128)],
                }
            )

        client.insert(collection_name, products)
        print("✓ Insert completed")

        # Perform some updates
        print("\nUpdating 100 products...")
        updates = []
        for i in range(100, 200):
            updates.append(
                {
                    "id": i,
                    "title": f"Updated Product {i}",
                    "category": "Updated",
                    "price": round(random.uniform(50, 500), 2),
                    "embedding": [random.random() for _ in range(128)],
                }
            )

        client.upsert(collection_name, updates)
        print("✓ Updates completed")

        # Delete some records
        print("\nDeleting 50 products...")
        delete_ids = list(range(200, 250))
        client.delete(collection_name, ids=delete_ids)
        print("✓ Deletions completed")

        # Wait for operations to settle
        print("\nWaiting for operations to sync...")
        time.sleep(2)

        # Perform three-way validation
        print("\n" + "=" * 60)
        print("PERFORMING THREE-WAY VALIDATION")
        print("=" * 60)

        validation_result = client.three_way_pk_validation(collection_name)

        print(f"\nValidation Summary:")
        print(f"  Total PKs checked: {validation_result['total_pks']}")
        print(f"  Consistent PKs: {validation_result['consistent_pks']}")
        print(f"  Inconsistent PKs: {validation_result['inconsistent_pks']}")

        if validation_result["inconsistent_pks"] > 0:
            print(f"\nInconsistency Details:")
            print(f"  Milvus errors: {len(validation_result['milvus_errors'])}")
            print(f"  PostgreSQL errors: {len(validation_result['pg_errors'])}")
            print(f"  LMDB errors: {len(validation_result['lmdb_errors'])}")

            # Show first few inconsistencies
            print(f"\nFirst 5 inconsistencies:")
            for detail in validation_result["details"][:5]:
                print(f"  PK {detail['pk']}:")
                print(f"    - In Milvus: {detail['in_milvus']}")
                print(f"    - In PostgreSQL: {detail['in_pg']}")
                print(f"    - In LMDB: {detail['in_lmdb']}")
                print(f"    - Correct state: {detail['correct_state']} (votes: {detail['vote_count']}/3)")
        else:
            print("\n✓ All databases are consistent!")

        # Demonstrate sampling for large datasets
        print("\n" + "=" * 60)
        print("SAMPLING VALIDATION (100 random PKs)")
        print("=" * 60)

        sample_result = client.three_way_pk_validation(collection_name, sample_size=100)
        print(f"\nSample Validation Summary:")
        print(f"  Sampled PKs: {sample_result['total_pks']}")
        print(f"  Consistent: {sample_result['consistent_pks']}")
        print(f"  Inconsistent: {sample_result['inconsistent_pks']}")

        # Show LMDB statistics
        print("\n" + "=" * 60)
        print("LMDB STATISTICS")
        print("=" * 60)

        if client.lmdb_manager:
            stats = client.lmdb_manager.get_stats()
            print(f"\nLMDB Database Info:")
            print(f"  Location: {stats['db_path']}")
            print(f"  Total entries: {stats['entries']:,}")
            print(f"  Used size: {stats['used_size'] / 1024 / 1024:.2f} MB")
            print(f"  Allocated size: {stats['map_size'] / 1024 / 1024 / 1024:.2f} GB")

        # Demonstrate standard entity comparison
        print("\n" + "=" * 60)
        print("STANDARD ENTITY COMPARISON")
        print("=" * 60)

        print("\nRunning full entity comparison between Milvus and PostgreSQL...")
        comparison_passed = client.entity_compare(
            collection_name, batch_size=500, full_scan=True, compare_pks_first=True
        )

        if comparison_passed:
            print("✓ Entity comparison passed - all data matches!")
        else:
            print("✗ Entity comparison failed - data inconsistencies found")

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
