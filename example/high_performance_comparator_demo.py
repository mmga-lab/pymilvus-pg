"""高性能比较器演示脚本

展示如何使用自定义的高性能比较器替换DeepDiff进行数据验证
"""

import os
import random
import time
from dotenv import load_dotenv
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

# Local application imports
from pymilvus_pg import MilvusPGClient as MilvusClient
from pymilvus_pg import logger

load_dotenv()

def main():
    """演示高性能比较器的使用"""
    
    # 使用高性能比较器初始化客户端
    logger.info("初始化MilvusPGClient - 启用高性能比较器")
    client = MilvusClient(
        uri="http://10.104.17.43:19530",  # 使用您提供的Milvus host
        pg_conn_str="postgresql://postgres:admin@10.104.20.96/postgres",  # 使用您提供的PG连接信息
        use_high_performance_comparator=True,  # 启用高性能比较器
        vector_precision_decimals=2,  # 向量精度控制
        vector_comparison_significant_digits=2,  # 有效数字控制
    )

    collection_name = f"perf_demo_{int(time.time())}"

    try:
        # 创建集合 - 使用多种数据类型
        logger.info(f"创建集合 '{collection_name}' - 包含多种数据类型")
        schema = client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("title", DataType.VARCHAR, max_length=200)
        schema.add_field("price", DataType.FLOAT)
        schema.add_field("is_active", DataType.BOOL)
        schema.add_field("category", DataType.VARCHAR, max_length=50)
        schema.add_field("tags", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=30)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=128)

        client.create_collection(collection_name, schema)

        # 创建索引
        index_params = IndexParams()
        index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
        client.create_index(collection_name, index_params)
        client.load_collection(collection_name)

        # 生成测试数据
        logger.info("生成测试数据...")
        categories = ["Electronics", "Books", "Clothing", "Food", "Toys"]
        data = []
        
        for i in range(1000):  # 创建1000条记录进行性能测试
            data.append({
                "id": i,
                "title": f"Product {i}",
                "price": round(random.uniform(10.0, 1000.0), 2),
                "is_active": random.choice([True, False]),
                "category": random.choice(categories),
                "tags": [f"tag_{j}" for j in range(random.randint(1, 5))],
                "metadata": {
                    "brand": f"Brand_{i % 50}",
                    "weight": round(random.uniform(0.1, 10.0), 2),
                    "rating": round(random.uniform(1.0, 5.0), 1)
                },
                "embedding": [random.uniform(-1.0, 1.0) for _ in range(128)]
            })

        # 插入数据
        logger.info("插入数据到Milvus和PostgreSQL...")
        start_insert = time.time()
        client.insert(collection_name, data)
        insert_time = time.time() - start_insert
        logger.info(f"数据插入完成，用时: {insert_time:.2f}秒")

        # 等待数据同步
        time.sleep(2)

        # 性能对比测试 - 先用DeepDiff
        logger.info("\n=== 性能对比测试 ===")
        
        # 临时禁用高性能比较器，使用DeepDiff
        logger.info("1. 使用DeepDiff进行数据验证...")
        client.use_high_performance_comparator = False
        
        start_deepdiff = time.time()
        result_deepdiff = client.entity_compare(
            collection_name, 
            batch_size=100,  # 较小的批次以便观察差异
            full_scan=True
        )
        deepdiff_time = time.time() - start_deepdiff
        
        logger.info(f"DeepDiff验证结果: {'通过' if result_deepdiff else '失败'}")
        logger.info(f"DeepDiff用时: {deepdiff_time:.2f}秒")

        # 启用高性能比较器
        logger.info("\n2. 使用高性能比较器进行数据验证...")
        client.use_high_performance_comparator = True
        
        start_hpc = time.time()
        result_hpc = client.entity_compare(
            collection_name, 
            batch_size=100,
            full_scan=True
        )
        hpc_time = time.time() - start_hpc
        
        logger.info(f"高性能比较器验证结果: {'通过' if result_hpc else '失败'}")
        logger.info(f"高性能比较器用时: {hpc_time:.2f}秒")
        
        # 性能对比
        if deepdiff_time > 0:
            speedup = deepdiff_time / hpc_time
            logger.info(f"\n性能提升: {speedup:.2f}x (高性能比较器比DeepDiff快 {speedup:.1f} 倍)")
        
        # 测试容差功能
        logger.info("\n=== 容差功能测试 ===")
        
        # 模拟轻微的数值差异（在容差范围内）
        logger.info("模拟轻微的数值差异...")
        
        # 这里可以手动修改一些数据来测试容差
        test_data = [{
            "id": 1000,
            "title": "Test Product",
            "price": 99.991,  # 与99.99的差异在容差范围内
            "is_active": True,
            "category": "Test",
            "tags": ["test"],
            "metadata": {"test": True},
            "embedding": [0.1001 for _ in range(128)]  # 轻微的向量差异
        }]
        
        client.insert(collection_name, test_data)
        time.sleep(1)
        
        # 验证容差功能
        tolerance_result = client.entity_compare(collection_name, full_scan=True)
        logger.info(f"容差验证结果: {'通过' if tolerance_result else '失败'}")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        # 清理
        try:
            logger.info(f"清理集合 '{collection_name}'...")
            client.drop_collection(collection_name)
            logger.info("清理完成")
        except Exception as e:
            logger.error(f"清理失败: {e}")
        
        client.close()

    logger.info("\n=== 高性能比较器演示完成 ===")


if __name__ == "__main__":
    main()