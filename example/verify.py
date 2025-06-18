import os

from dotenv import load_dotenv

from pymilvus_pg import MilvusPGClient as MilvusClient

load_dotenv()

# --- Configuration ---
MILVUS_URI = os.getenv("MILVUS_URI", "http://10.100.36.238:19530")  # URI for Milvus server
PG_CONN = os.getenv("PG_CONN", "postgresql://postgres:admin@10.100.36.239:5432/postgres")
COLLECTION_NAME = "mt_checker_1750236533"

milvus_client = MilvusClient(uri=MILVUS_URI, pg_conn_str=PG_CONN)
milvus_client.entity_compare(COLLECTION_NAME, full_scan=True)
