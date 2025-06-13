"""multi_thread_checker.py
多线程长期运行脚本：在指定集合上持续执行 insert / delete / upsert 操作，
并定期暂停写入，使用 `pymilvus_pg.MilvusPGClient.entity_compare` 验证
Milvus 与 PostgreSQL 影子表中的数据一致性。

运行示例::

    python multi_thread_checker.py \
        --threads 4 \
        --compare_interval 60 \
        --uri http://localhost:19530 \
        --pg_conn postgresql://postgres:admin@localhost:5432/default \
        --duration 3600

注意：脚本并不会终止，按 Ctrl+C 中断。
"""

from __future__ import annotations

import argparse
import os
import random
import threading
import time

from dotenv import load_dotenv
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient as MilvusClient
from pymilvus_pg import logger

load_dotenv()

# ---------------------------- 默认配置 ---------------------------
DIMENSION = 8  # 向量维度
INSERT_BATCH_SIZE = 1000
DELETE_BATCH_SIZE = 500
UPSERT_BATCH_SIZE = 300
COLLECTION_NAME_PREFIX = "mt_checker"

# 全局主键计数，以及线程安全锁
_global_id: int = 0
_id_lock = threading.Lock()

# 事件，用于控制线程暂停/停止
pause_event = threading.Event()
stop_event = threading.Event()


def _next_id_batch(count: int) -> list[int]:
    """返回一个连续 id 列表，并安全地递增全局计数。"""
    global _global_id
    with _id_lock:
        start = _global_id
        _global_id += count
    return list(range(start, start + count))


def _generate_data(id_list: list[int], for_upsert: bool = False):
    """根据 id 列表生成记录。"""
    data = []
    for _id in id_list:
        record = {
            "id": _id,
            "name": f"name_{_id}{'_upserted' if for_upsert else ''}",
            "age": random.randint(18, 60) + (100 if for_upsert else 0),
            "json_field": {"attr1": _id, "attr2": f"val_{_id}"},
            "array_field": [_id, _id + 1, _id + 2, random.randint(0, 100)],
            "embedding": [random.random() for _ in range(DIMENSION)],
        }
        data.append(record)
    return data


def _insert_op(client: MilvusClient, collection: str):
    ids = _next_id_batch(INSERT_BATCH_SIZE)
    client.insert(collection, _generate_data(ids))
    logger.info("[INSERT] %d rows, start id %d", len(ids), ids[0])


def _delete_op(client: MilvusClient, collection: str):
    global _global_id
    # 仅当已有数据时才删除
    if _global_id == 0:
        return
    # 随机选择一段 id
    start = random.randint(0, max(1, _global_id - DELETE_BATCH_SIZE))
    ids = list(range(start, start + DELETE_BATCH_SIZE))
    client.delete(collection, ids=ids)
    logger.info("[DELETE] %d rows, start id %d", len(ids), start)


def _upsert_op(client: MilvusClient, collection: str):
    global _global_id
    if _global_id == 0:
        return
    start = random.randint(0, max(1, _global_id - UPSERT_BATCH_SIZE))
    ids = list(range(start, start + UPSERT_BATCH_SIZE))
    client.upsert(collection, _generate_data(ids, for_upsert=True))
    logger.info("[UPSERT] %d rows, start id %d", len(ids), start)


OPERATIONS = [_insert_op, _delete_op, _upsert_op]


def worker_loop(client: MilvusClient, collection: str):
    """工作线程：循环执行随机写操作。"""
    while not stop_event.is_set():
        if pause_event.is_set():
            time.sleep(0.1)
            continue
        op = random.choice(OPERATIONS)
        try:
            op(client, collection)
        except Exception:  # noqa: BLE001
            logger.exception("Error during %s", op.__name__)
        # 小睡眠降低压力
        time.sleep(random.uniform(0.05, 0.2))


def create_collection(client: MilvusClient, name: str):
    if client.has_collection(name):
        logger.warning("Collection %s already exists, dropping", name)
        client.drop_collection(name)

    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("name", DataType.VARCHAR, max_length=256)
    schema.add_field("age", DataType.INT64)
    schema.add_field("json_field", DataType.JSON)
    schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=20)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
    client.create_collection(name, schema)

    index_params = IndexParams()
    index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
    client.create_index(name, index_params)
    client.load_collection(name)
    logger.info("Collection %s created and loaded", name)


def main():
    parser = argparse.ArgumentParser(description="Multi-thread write / verify checker for MilvusPGClient")
    parser.add_argument("--threads", type=int, default=4, help="Writer thread count (default 4)")
    parser.add_argument(
        "--compare_interval", type=int, default=60, help="Seconds between consistency checks (default 60)"
    )
    parser.add_argument("--duration", type=int, default=0, help="Total run time in seconds (0 means run indefinitely)")
    parser.add_argument(
        "--uri", type=str, default=os.getenv("MILVUS_URI", "http://localhost:19530"), help="Milvus server URI"
    )
    parser.add_argument("--token", type=str, default=os.getenv("MILVUS_TOKEN", ""), help="Milvus auth token")
    parser.add_argument(
        "--pg_conn",
        type=str,
        default=os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/default"),
        help="PostgreSQL DSN",
    )
    args = parser.parse_args()

    start_time = time.time()

    client = MilvusClient(
        uri=args.uri,
        token=args.token,
        pg_conn_str=args.pg_conn,
    )
    collection_name = f"{COLLECTION_NAME_PREFIX}_{int(time.time())}"
    logger.info("Using collection: %s", collection_name)
    create_collection(client, collection_name)

    # 启动写线程
    threads: list[threading.Thread] = []
    for i in range(args.threads):
        t = threading.Thread(target=worker_loop, name=f"Writer-{i}", args=(client, collection_name), daemon=True)
        t.start()
        threads.append(t)

    last_compare = time.time()
    try:
        while True:
            time.sleep(1)
            if time.time() - last_compare >= args.compare_interval:
                logger.info("Pausing writers for entity compare …")
                pause_event.set()
                # 等待在途操作完成
                time.sleep(2)
                try:
                    client.entity_compare(collection_name)
                except Exception:
                    logger.exception("Error during entity_compare")
                last_compare = time.time()
                pause_event.clear()
                logger.info("Writers resumed")
            # 检查 duration
            if args.duration > 0 and time.time() - start_time >= args.duration:
                logger.info("Duration reached (%ds), stopping …", args.duration)
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping …")
        stop_event.set()
        for t in threads:
            t.join(timeout=5)
    finally:
        logger.info("Finished. Final compare …")
        try:
            client.entity_compare(collection_name)
        except Exception:
            logger.exception("Final entity_compare failed")


if __name__ == "__main__":
    main()
