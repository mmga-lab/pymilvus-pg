import argparse
import os
import random
import time

from dotenv import load_dotenv
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient as MilvusClient
from pymilvus_pg import logger

# Load environment variables from .env file
load_dotenv()

# Configuration section
MILVUS_URI = os.getenv("MILVUS_URI", "http://10.104.21.143:19530")  # Milvus server URI
PG_CONN = os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/default")  # PostgreSQL DSN
COLLECTION_NAME_PREFIX = "complex_test_collection"

DIMENSION = 8  # Embedding vector dimension
LARGE_BATCH_SIZE = 10000  # Number of records for initial large insert
SMALL_BATCH_SIZE = 1000  # Number of records per small batch operation
DELETE_BATCH_SIZE = 500  # Number of records to delete per batch
UPSERT_BATCH_SIZE = 300  # Number of records to upsert per batch

# Global ID counter
global_id = 0


def generate_data(start_id, count, for_upsert=False):
    data = []
    for i in range(count):
        current_id = start_id + i
        record = {
            "id": current_id,
            "name": f"name_{current_id}{'_upserted' if for_upsert else ''}",
            "age": random.randint(18, 60) + (100 if for_upsert else 0),  # Differentiate upserted data
            "json_field": {"attr1": current_id, "attr2": f"val_{current_id}"},
            "array_field": [
                current_id,
                current_id + 1,
                current_id + 2,
                random.randint(0, 100),
            ],
            "embedding": [random.random() for _ in range(DIMENSION)],
        }
        data.append(record)
    return data


def perform_large_insert(milvus_client, collection_name):
    """Perform initial large insert operation"""
    global global_id

    start_id = global_id
    batch_data = generate_data(start_id, LARGE_BATCH_SIZE)
    global_id += LARGE_BATCH_SIZE

    milvus_client.insert(collection_name, batch_data)
    logger.info(f"[LARGE INSERT] Inserted {LARGE_BATCH_SIZE} records, start id: {start_id}")


def perform_small_insert(milvus_client, collection_name):
    """Perform small insert operation"""
    global global_id

    start_id = global_id
    batch_data = generate_data(start_id, SMALL_BATCH_SIZE)
    global_id += SMALL_BATCH_SIZE

    milvus_client.insert(collection_name, batch_data)
    logger.info(f"[SMALL INSERT] Inserted {SMALL_BATCH_SIZE} records, start id: {start_id}")


def perform_delete(milvus_client: MilvusClient, collection_name):
    """Perform delete operation"""
    global global_id

    # Delete from existing data range
    start_id = max(0, global_id - LARGE_BATCH_SIZE)
    end_id = start_id + DELETE_BATCH_SIZE
    ids_batch = list(range(start_id, end_id))

    milvus_client.delete(collection_name, ids=ids_batch)
    logger.info(f"[DELETE] Deleted {len(ids_batch)} records, start id: {start_id}")


def perform_upsert(milvus_client, collection_name):
    """Perform upsert operation"""
    global global_id

    # Upsert existing data range
    start_id = max(0, global_id - LARGE_BATCH_SIZE // 2)
    batch_data = generate_data(start_id, UPSERT_BATCH_SIZE, for_upsert=True)

    milvus_client.upsert(collection_name, batch_data)
    logger.info(f"[UPSERT] Upserted {len(batch_data)} records, start id: {start_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential long run checker with large insert followed by repeated small operations."
    )
    parser.add_argument(
        "--repeat_cycles",
        type=int,
        default=10,
        help="Number of repeat cycles for (small insert -> delete -> upsert) (default 10)",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=MILVUS_URI,
        help="Milvus server URI (default http://10.104.21.143:19530)",
    )
    parser.add_argument(
        "--pg_conn",
        type=str,
        default=PG_CONN,
        help="PostgreSQL DSN (default postgresql://postgres:admin@localhost:5432/default)",
    )
    args = parser.parse_args()

    # Create collection
    milvus_client = MilvusClient(uri=args.uri, pg_conn_str=args.pg_conn)
    collection_name = f"{COLLECTION_NAME_PREFIX}_{int(time.time())}"
    logger.info(f"Using collection: {collection_name}")

    if milvus_client.has_collection(collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Dropping it.")
        milvus_client.drop_collection(collection_name)

    schema = milvus_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("name", DataType.VARCHAR, max_length=256)
    schema.add_field("age", DataType.INT64)
    schema.add_field("json_field", DataType.JSON)
    schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=20)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
    milvus_client.create_collection(collection_name, schema)
    logger.info(f"Collection '{collection_name}' created successfully.")

    index_params = IndexParams()
    index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
    milvus_client.create_index(collection_name, index_params)
    logger.info("Index created successfully.")
    milvus_client.load_collection(collection_name)
    logger.info(f"Collection '{collection_name}' loaded.")
    time.sleep(2)

    try:
        # Step 1: Perform large insert operation
        logger.info("=" * 50)
        logger.info("Starting large insert operation...")
        perform_large_insert(milvus_client, collection_name)
        logger.info("Large insert operation completed.")

        # Step 2: Repeat small operations cycle
        logger.info("=" * 50)
        logger.info(f"Starting {args.repeat_cycles} cycles of (small insert -> delete -> upsert)...")

        for cycle in range(args.repeat_cycles):
            logger.info(f"--- Cycle {cycle + 1}/{args.repeat_cycles} ---")

            # Small insert
            perform_small_insert(milvus_client, collection_name)
            time.sleep(1)  # Brief pause between operations

            # Delete
            perform_delete(milvus_client, collection_name)
            time.sleep(1)  # Brief pause between operations

            # Upsert
            perform_upsert(milvus_client, collection_name)
            time.sleep(1)  # Brief pause between operations

            logger.info(f"Cycle {cycle + 1} completed.")

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, exiting early...")
    except Exception as e:
        logger.error(f"Exception occurred: {e}", exc_info=True)

    logger.info("=" * 50)
    logger.info("Sequential operations completed.")

    # Check collection
    logger.info("Performing entity comparison check...")
    milvus_client.entity_compare(collection_name)
