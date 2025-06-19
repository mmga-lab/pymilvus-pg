#!/usr/bin/env python3
"""
使用 query_iterator 遍历主键并进行主键差异比较的示例

此示例展示：
1. 使用 query_iterator 获取 Milvus 中的所有主键
2. 比较 Milvus 和 PostgreSQL 中的主键差异
3. 在 entity_compare 中启用主键比较功能
"""

import random
import time

from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient
from pymilvus_pg.logger_config import logger


def create_demo_collection(client: MilvusPGClient, collection_name: str):
    """创建演示用的集合"""
    # 删除已存在的集合
    try:
        client.drop_collection(collection_name)
        logger.info(f"删除了已存在的集合 '{collection_name}'")
    except Exception:
        pass

    # 创建 schema
    schema = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("name", DataType.VARCHAR, max_length=100),
            FieldSchema("age", DataType.INT64),
            FieldSchema("score", DataType.DOUBLE),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=8),
        ],
        description="主键比较演示集合"
    )

    # 创建集合
    client.create_collection(collection_name, schema=schema)
    logger.info(f"创建了集合 '{collection_name}'")

    # 创建索引
    index_params = IndexParams()
    index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
    client.create_index(collection_name, index_params)
    client.load_collection(collection_name)
    logger.info("创建了索引并加载了集合")


def insert_demo_data(client: MilvusPGClient, collection_name: str, num_records: int = 1000):
    """插入演示数据"""
    logger.info(f"插入 {num_records} 条演示数据...")
    
    batch_size = 100
    for i in range(0, num_records, batch_size):
        batch_data = []
        for j in range(min(batch_size, num_records - i)):
            record_id = i + j
            batch_data.append({
                "id": record_id,
                "name": f"user_{record_id}",
                "age": 20 + (record_id % 50),
                "score": random.uniform(60.0, 100.0),
                "embedding": [random.random() for _ in range(8)]
            })
        
        client.insert(collection_name, batch_data)
        logger.info(f"插入了批次 {i//batch_size + 1}/{(num_records + batch_size - 1)//batch_size}")
    
    # 等待数据同步
    time.sleep(2)
    logger.info("演示数据插入完成")


def demonstrate_pk_operations(client: MilvusPGClient, collection_name: str):
    """演示主键相关操作"""
    
    # 1. 使用 query_iterator 获取所有主键
    logger.info("\n=== 1. 使用 query_iterator 获取所有主键 ===")
    milvus_pks = client.get_all_primary_keys_from_milvus(collection_name, batch_size=200)
    logger.info(f"从 Milvus 获取到 {len(milvus_pks)} 个主键")
    logger.info(f"前 10 个主键: {sorted(milvus_pks)[:10]}")
    
    # 2. 比较主键差异
    logger.info("\n=== 2. 比较 Milvus 和 PostgreSQL 的主键差异 ===")
    client.compare_primary_keys(collection_name)
    
    # 3. 模拟一些数据不一致的情况
    logger.info("\n=== 3. 模拟数据不一致情况 ===")
    
    # 删除一些 Milvus 中的数据（但不删除 PostgreSQL 中的）
    ids_to_delete = [100, 101, 102]
    logger.info(f"从 Milvus 中删除 ID: {ids_to_delete}")
    client.delete(collection_name, ids=ids_to_delete)
    
    # 等待删除操作完成
    time.sleep(2)
    
    # 再次比较主键
    logger.info("\n=== 4. 删除后重新比较主键差异 ===")
    pk_comparison_after_delete = client.compare_primary_keys(collection_name)
    
    # 显示差异详情
    if pk_comparison_after_delete["has_differences"]:
        logger.info("发现主键差异:")
        logger.info(f"  仅在 Milvus 中: {pk_comparison_after_delete['only_in_milvus']}")
        logger.info(f"  仅在 PostgreSQL 中: {pk_comparison_after_delete['only_in_pg']}")
    
    # 4. 使用增强的 entity_compare 功能
    logger.info("\n=== 5. 使用增强的 entity_compare 功能 ===")
    
    # 启用主键比较的 entity_compare
    result_with_pk_check = client.entity_compare(
        collection_name, 
        compare_pks_first=True,
        full_scan=False
    )
    logger.info(f"启用主键比较的 entity_compare 结果: {result_with_pk_check}")
    
    # 不启用主键比较的 entity_compare（原有行为）
    result_without_pk_check = client.entity_compare(
        collection_name, 
        compare_pks_first=False,
        full_scan=False
    )
    logger.info(f"不启用主键比较的 entity_compare 结果: {result_without_pk_check}")


def main():
    """主程序"""
    # 配置连接
    milvus_uri = "http://10.100.36.238:19530"
    pg_conn_str = "postgresql://postgres:admin@10.100.36.239:5432/postgres"
    
    logger.info("初始化 MilvusPGClient...")
    client = MilvusPGClient(
        uri=milvus_uri,
        pg_conn_str=pg_conn_str,
        ignore_vector=False  # 包含向量字段进行比较
    )
    
    collection_name = "pk_compare_demo"
    
    try:
        # 创建演示集合
        create_demo_collection(client, collection_name)
        
        # 插入演示数据
        insert_demo_data(client, collection_name, num_records=1000)
        
        # 演示主键操作
        demonstrate_pk_operations(client, collection_name)
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise
    
    finally:
        # 清理资源
        try:
            client.drop_collection(collection_name)
            logger.info(f"清理了演示集合 '{collection_name}'")
        except Exception as e:
            logger.warning(f"清理集合时出错: {e}")


if __name__ == "__main__":
    main() 