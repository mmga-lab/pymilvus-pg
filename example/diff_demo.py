"""
This script is used to verify the diff function of milvus pg.
"""
import os
import random
import time

import psycopg2
from dotenv import load_dotenv
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

# Local application imports
from pymilvus_pg import MilvusPGClient as MilvusClient
from pymilvus_pg import logger

load_dotenv()

# Connect to Milvus and PostgreSQL
milvus_client = MilvusClient(
    uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
    pg_conn_str=os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/default"),
)

collection_name = f"diff_demo_{int(time.time())}"

# Define the schema for the collection
schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("name", DataType.VARCHAR, max_length=100)
schema.add_field("age", DataType.INT64)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)

milvus_client.create_collection(collection_name, schema)

# Create an index for the embedding field
index_params = IndexParams()
index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
milvus_client.create_index(collection_name, index_params)

milvus_client.load_collection(collection_name)

# Insert initial data
logger.info("--- Insert initial data ---")
initial_data = [
    {
        "id": i,
        "name": f"name_{i}",
        "age": i,
        "embedding": [random.random() for _ in range(8)],
    }
    for i in range(5)
]
milvus_client.insert(collection_name, initial_data)

# Wait for data to be flushed
time.sleep(2)

# Manually modify data in PostgreSQL to create a difference
logger.info("--- Manually modify data in PostgreSQL ---")
pg_conn_str = os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/default")

try:
    with psycopg2.connect(pg_conn_str) as conn:
        with conn.cursor() as cursor:
            # Change the 'age' of the record with id = 3
            update_query = f'UPDATE "{collection_name}" SET age = 99 WHERE id = 3;'
            logger.info(f"Executing SQL: {update_query}")
            cursor.execute(update_query)
            conn.commit()
            logger.info("Successfully modified data in PostgreSQL.")
except Exception as e:
    logger.error(f"Failed to modify data in PostgreSQL: {e}")
    milvus_client.drop_collection(collection_name)
    exit()

# Compare the data between Milvus and PostgreSQL
logger.info("--- Compare data and show the diff ---")
diff = milvus_client.entity_compare(collection_name, full_scan=True)

logger.info("--- Diff result: ---")
logger.info(diff)

# Clean up
logger.info(f"--- Drop collection: {collection_name} ---")
milvus_client.drop_collection(collection_name)

logger.info("--- Diff demo finished ---")

