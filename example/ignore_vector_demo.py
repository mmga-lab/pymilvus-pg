"""示例：使用 ignore_vector 跳过向量字段的 PG 写入与比对。

运行前请确认已启动 Milvus 与 PostgreSQL，并修改下方连接信息。
"""

from __future__ import annotations

import random
import time

from pymilvus import DataType
from pymilvus_pg import MilvusPGClient, logger
from pymilvus.milvus_client import IndexParams

# --- 配置 ---
MILVUS_URI = "http://localhost:19530"
PG_CONN_STR = "postgresql://postgres:admin@localhost:5432/default"
COLLECTION_NAME = f"ignore_vec_demo_{int(time.time())}"
DIM = 128

# --- 初始化客户端，开启 ignore_vector ---
client = MilvusPGClient(uri=MILVUS_URI, pg_conn_str=PG_CONN_STR, ignore_vector=True)

# --- 创建 Schema & Collection ---
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("name", DataType.VARCHAR, max_length=128)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
client.create_collection(COLLECTION_NAME, schema)
logger.info("Collection created: %s", COLLECTION_NAME)

# --- 创建向量索引 ---
index_params = IndexParams()
index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
client.create_index(COLLECTION_NAME, index_params)
logger.info("Index created on embedding field.")

# --- 加载 Collection，避免未加载导致查询失败 ---
client.load_collection(COLLECTION_NAME)
logger.info("Collection loaded for querying/comparison.")

# --- 插入示例数据 ---
records = [
    {
        "id": i,
        "name": f"user_{i}",
        "embedding": [random.random() for _ in range(DIM)],
    }
    for i in range(10)
]
client.insert(COLLECTION_NAME, records)
logger.info("Inserted 10 records with vector field, PG 写入已自动忽略向量列")

# --- 对比实体 ---
client.entity_compare(COLLECTION_NAME)
logger.info("Entity compare 完成（忽略向量列）")

# --- 清理 ---
# client.drop_collection(COLLECTION_NAME)
# logger.info("Collection dropped")
