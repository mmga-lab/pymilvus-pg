# Configuration section
# Define the Milvus client and collection name
import os
import random
import time

from dotenv import load_dotenv
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient as MilvusClient
from pymilvus_pg import logger

load_dotenv()

milvus_client = MilvusClient(
    uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
    pg_conn_str=os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/default"),
)
collection_name = f"demo_{int(time.time())}"

# Define the schema for the collection
schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("name", DataType.VARCHAR, max_length=100)
schema.add_field("age", DataType.INT64)
schema.add_field("json_field", DataType.JSON)
schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)

milvus_client.create_collection(collection_name, schema)
index_params = IndexParams()
index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})

milvus_client.create_index(collection_name, index_params)

milvus_client.load_collection(collection_name)


milvus_client.insert(
    collection_name,
    [
        {
            "id": i,
            "name": f"test_{i}",
            "age": i,
            "json_field": {"a": i, "b": i + 1},
            "array_field": [i, i + 1, i + 2],
            "embedding": [random.random() for _ in range(8)],
        }
        for i in range(10)
    ],
)

milvus_client.delete(collection_name, ids=[1, 2, 3])

milvus_client.upsert(
    collection_name,
    [
        {
            "id": i,
            "name": f"test_{i + 100}",
            "age": i + 100,
            "json_field": {"a": i + 100, "b": i + 101},
            "array_field": [i + 100, i + 101, i + 102],
            "embedding": [random.random() for _ in range(8)],
        }
        for i in range(4, 8)
    ],
)

time.sleep(1)
res = milvus_client.query(collection_name, "age > 0")
logger.info(res)

res = milvus_client.export(collection_name)
logger.info(res)

res = milvus_client.count(collection_name)
logger.info(res)

milvus_client.entity_compare(collection_name)


# Example: Automatically generate a diverse Milvus filter expression
filter_expr = milvus_client.generate_milvus_filter(collection_name, num_samples=2)
logger.info("\n[Auto Milvus Filter Example]\n", filter_expr)
for filter in filter_expr:
    logger.info(filter)
    res = milvus_client.query_result_compare(collection_name, filter)
    logger.info(res)
