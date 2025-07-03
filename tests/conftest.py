import os
import time

import pytest
from dotenv import load_dotenv

from pymilvus_pg import MilvusPGClient as MilvusClient
from pymilvus_pg import logger

load_dotenv()


@pytest.fixture(scope="session")
def milvus_client_session():
    """Create a session-scoped MilvusPGClient instance for testing"""
    client = MilvusClient(
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        pg_conn_str=os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/postgres"),
    )
    yield client
    # Cleanup code if needed


@pytest.fixture(autouse=True)
def cleanup_test_collections(milvus_client_session):
    """Cleanup test collections after each test"""
    yield

    # Clean up any remaining test collections from both Milvus and PostgreSQL
    try:
        # Get all collections from Milvus that start with 'test_'
        collections = milvus_client_session.milvus_client.list_collections()
        for collection_name in collections:
            if collection_name.startswith("test_"):
                try:
                    milvus_client_session.milvus_client.drop_collection(collection_name)
                    logger.debug(f"Dropped Milvus collection: {collection_name}")
                except Exception as e:
                    logger.debug(f"Could not drop Milvus collection {collection_name}: {e}")

        # Clean up PostgreSQL tables that start with 'test_'
        with milvus_client_session.pg_conn.cursor() as cursor:
            cursor.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' AND tablename LIKE 'test_%'
            """)
            test_tables = cursor.fetchall()

            for (table_name,) in test_tables:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.debug(f"Dropped PostgreSQL table: {table_name}")
                except Exception as e:
                    logger.debug(f"Could not drop PostgreSQL table {table_name}: {e}")

            milvus_client_session.pg_conn.commit()

    except Exception as e:
        logger.debug(f"Error during cleanup: {e}")

    time.sleep(0.1)  # Small delay to ensure proper cleanup
