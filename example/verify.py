import os

from dotenv import load_dotenv

from pymilvus_pg import MilvusPGClient as MilvusClient

load_dotenv()

# --- Configuration ---
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")  # URI for Milvus server
PG_CONN = os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/default")
COLLECTION_NAME = "complex_test_collection_1748571650"

milvus_client = MilvusClient(uri=MILVUS_URI, pg_conn_str=PG_CONN)
milvus_client.entity_compare(COLLECTION_NAME)
