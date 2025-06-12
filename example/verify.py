import logging

from pymilvus_pg import MilvusPGClient as MilvusClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
MILVUS_URI = "http://localhost:19530"  # URI for Milvus server
PG_CONN = "postgresql://postgres:admin@localhost:5432/default"
COLLECTION_NAME = "complex_test_collection_1748571650"

milvus_client = MilvusClient(uri=MILVUS_URI, pg_conn_str=PG_CONN)
print("compare")
milvus_client.entity_compare(COLLECTION_NAME)
