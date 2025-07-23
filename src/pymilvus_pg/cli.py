"""Command line interface for pymilvus_pg."""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Any

import click
from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_pg import MilvusPGClient as MilvusClient
from pymilvus_pg import __version__, logger

DIMENSION = 8
INSERT_BATCH_SIZE = 10000
DELETE_BATCH_SIZE = 5000
UPSERT_BATCH_SIZE = 3000
COLLECTION_NAME_PREFIX = "data_correctness_checker"

_global_id: int = 0
_id_lock = threading.Lock()

pause_event = threading.Event()
stop_event = threading.Event()

# Track active operations
active_operations = 0
active_operations_lock = threading.Lock()


def _next_id_batch(count: int) -> list[int]:
    """Return a continuous id list and safely increment the global counter."""
    global _global_id
    with _id_lock:
        start = _global_id
        _global_id += count
    return list(range(start, start + count))


def _generate_data(id_list: list[int], for_upsert: bool = False) -> list[dict[str, Any]]:
    """Generate records based on id list."""
    data = []
    for _id in id_list:
        record = {
            "id": _id,
            "category": f"category_{_id % 1000}",
            "name": f"name_{_id}{'_upserted' if for_upsert else ''}",
            "age": random.randint(18, 60) + (100 if for_upsert else 0),
            "json_field": {"attr1": _id, "attr2": f"val_{_id}"},
            "array_field": [_id, _id + 1, _id + 2, random.randint(0, 100)],
            "embedding": [random.random() for _ in range(DIMENSION)],
        }
        data.append(record)
    return data


def _insert_op(client: MilvusClient, collection: str) -> None:
    """Insert operation with exception handling to ensure thread stability."""
    global active_operations
    with active_operations_lock:
        active_operations += 1
    try:
        ids = _next_id_batch(INSERT_BATCH_SIZE)
        client.insert(collection, _generate_data(ids))
        logger.info(f"[INSERT] {len(ids)} rows, start id {ids[0]}")
    except Exception as e:
        logger.error(f"[INSERT] Exception occurred: {e}")
    finally:
        with active_operations_lock:
            active_operations -= 1


def _delete_op(client: MilvusClient, collection: str) -> None:
    """Delete operation with exception handling to ensure thread stability."""
    global _global_id, active_operations
    with active_operations_lock:
        active_operations += 1
    try:
        if _global_id == 0:
            return
        
        # Query actual existing IDs from PostgreSQL before deletion
        if hasattr(client, '_get_pg_connection'):
            with client._get_pg_connection() as conn:
                with conn.cursor() as cursor:
                    # Get random sample of existing IDs from PostgreSQL
                    cursor.execute(
                        f"SELECT {client.primary_field} FROM {collection} "
                        f"ORDER BY RANDOM() LIMIT {DELETE_BATCH_SIZE}"
                    )
                    result = cursor.fetchall()
                    ids = [row[0] for row in result] if result else []
        else:
            # Fallback to old logic if PostgreSQL connection is not available
            start = random.randint(0, max(1, _global_id - DELETE_BATCH_SIZE))
            ids = list(range(start, min(start + DELETE_BATCH_SIZE, _global_id)))
        
        if ids:
            client.delete(collection, ids=ids)
            logger.info(f"[DELETE] Attempted {len(ids)} rows from PostgreSQL query")
    except Exception as e:
        logger.error(f"[DELETE] Exception occurred: {e}")
    finally:
        with active_operations_lock:
            active_operations -= 1


def _upsert_op(client: MilvusClient, collection: str) -> None:
    """Upsert operation with exception handling to ensure thread stability."""
    global _global_id, active_operations
    with active_operations_lock:
        active_operations += 1
    try:
        if _global_id == 0:
            return
        # Select a random range of IDs that have been inserted
        start = random.randint(0, max(1, _global_id - UPSERT_BATCH_SIZE))
        ids = list(range(start, min(start + UPSERT_BATCH_SIZE, _global_id)))
        if ids:
            client.upsert(collection, _generate_data(ids, for_upsert=True))
            logger.info(f"[UPSERT] {len(ids)} rows, start id {start}")
    except Exception as e:
        logger.error(f"[UPSERT] Exception occurred: {e}")
    finally:
        with active_operations_lock:
            active_operations -= 1


OPERATIONS = [_insert_op, _delete_op, _upsert_op]


def wait_for_operations_to_complete(timeout: float = 30.0) -> bool:
    """Wait for all active operations to complete.

    Returns True if all operations completed within timeout, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        with active_operations_lock:
            if active_operations == 0:
                return True
        logger.debug(f"Waiting for {active_operations} operations to complete...")
        time.sleep(0.1)

    logger.warning(f"Timeout waiting for operations to complete. {active_operations} still active.")
    return False


def worker_loop(client: MilvusClient, collection: str) -> None:
    """Worker thread: loop to execute random write operations."""
    while not stop_event.is_set():
        if pause_event.is_set():
            time.sleep(0.1)
            continue
        op = random.choice(OPERATIONS)
        try:
            op(client, collection)
        except Exception:  # noqa: BLE001
            logger.exception(f"Error during {op.__name__}")
        time.sleep(random.uniform(0.05, 0.2))


def create_collection(client: MilvusClient, name: str, drop_if_exists: bool = True) -> None:
    if client.has_collection(name):
        if drop_if_exists:
            logger.warning(f"Collection {name} already exists, dropping")
            client.drop_collection(name)
        else:
            logger.info(f"Collection {name} already exists, will continue using it")
            return

    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("category", DataType.VARCHAR, max_length=256, is_partition_key=True)
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
    logger.info(f"Collection {name} created and loaded")


@click.group()
@click.version_option(version=__version__, prog_name="pymilvus-pg")
def cli() -> None:
    """PyMilvus-PG CLI for data consistency validation."""
    pass


@cli.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish", "powershell"]),
    help="Shell type (auto-detected if not specified)",
)
@click.option("--install", is_flag=True, help="Install completion to shell config file")
def completion(shell: str | None, install: bool) -> None:
    """Generate shell completion script or install completion."""
    import subprocess
    import sys

    if install:
        # Install completion
        if shell:
            subprocess.run([sys.executable, "-m", "pymilvus_pg.cli", "completion", "--shell", shell], check=True)
        else:
            click.echo("Installing completion...")
            # Auto-detect shell and install
            try:
                shell_name = os.path.basename(os.environ.get("SHELL", "bash"))
                if shell_name in ["bash", "zsh", "fish"]:
                    subprocess.run(
                        [sys.executable, "-m", "pymilvus_pg.cli", "completion", "--shell", shell_name], check=True
                    )
                else:
                    click.echo(f"Unsupported shell: {shell_name}")
                    return
            except Exception as e:
                click.echo(f"Error installing completion: {e}")
                return
    else:
        # Generate completion script
        if not shell:
            shell = os.path.basename(os.environ.get("SHELL", "bash"))

        prog_name = "pymilvus-pg"

        if shell == "bash":
            click.echo(f'eval "$(_PYMILVUS_PG_COMPLETE=bash_source {prog_name})"')
            click.echo("\n# Add this to your ~/.bashrc:")
            click.echo(f'# eval "$(_PYMILVUS_PG_COMPLETE=bash_source {prog_name})"')
        elif shell == "zsh":
            click.echo(f'eval "$(_PYMILVUS_PG_COMPLETE=zsh_source {prog_name})"')
            click.echo("\n# Add this to your ~/.zshrc:")
            click.echo(f'# eval "$(_PYMILVUS_PG_COMPLETE=zsh_source {prog_name})"')
        elif shell == "fish":
            click.echo(f"eval (_PYMILVUS_PG_COMPLETE=fish_source {prog_name})")
            click.echo("\n# Add this to your ~/.config/fish/config.fish:")
            click.echo(f"# eval (_PYMILVUS_PG_COMPLETE=fish_source {prog_name})")
        elif shell == "powershell":
            click.echo(f'$env:_PYMILVUS_PG_COMPLETE="powershell_source"; {prog_name} | Out-String | Invoke-Expression')
            click.echo("\n# Add this to your PowerShell profile")
        else:
            click.echo(f"Unsupported shell: {shell}")
            return

        click.echo("\n# Or run directly:")
        click.echo(f"# {prog_name} completion --install")


@cli.command()
@click.option("--threads", type=int, default=10, help="Writer thread count (default 10)")
@click.option("--compare-interval", type=int, default=60, help="Seconds between validation checks (default 60)")
@click.option("--duration", type=int, default=0, help="Total run time in seconds (0 means run indefinitely)")
@click.option("--uri", type=str, default=None, help="Milvus server URI")
@click.option("--token", type=str, default="", help="Milvus auth token")
@click.option("--pg-conn", type=str, default=None, help="PostgreSQL DSN")
@click.option("--collection", type=str, default=None, help="Collection name (auto-generated if not specified)")
@click.option("--drop-existing", is_flag=True, help="Drop existing collection before starting")
@click.option(
    "--include-vector",
    is_flag=True,
    help="Include vector fields in PostgreSQL operations (default: False)",
)
def ingest(
    threads: int,
    compare_interval: int,
    duration: int,
    uri: str | None,
    token: str,
    pg_conn: str | None,
    collection: str | None,
    drop_existing: bool,
    include_vector: bool,
) -> None:
    """Continuously ingest data with periodic validation checks.

    Performs high-throughput data ingestion using insert/delete/upsert operations
    while periodically validating data consistency between Milvus and PostgreSQL.
    """
    global _global_id

    uri = uri or os.getenv("MILVUS_URI", "http://localhost:19530")
    pg_conn = pg_conn or os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/postgres")

    start_time = time.time()

    client = MilvusClient(
        uri=uri,
        token=token,
        pg_conn_str=pg_conn,
        ignore_vector=not include_vector,
    )

    # Use provided collection name or default fixed name
    collection_name = collection or COLLECTION_NAME_PREFIX
    logger.info(f"Using collection: {collection_name}")

    # Create or reuse collection
    create_collection(client, collection_name, drop_if_exists=drop_existing)

    # Always use timestamp-based starting ID to avoid conflicts across runs
    _global_id = int(time.time() * 1000)
    logger.info(f"Starting with timestamp-based ID: {_global_id}")

    thread_list: list[threading.Thread] = []
    for i in range(threads):
        t = threading.Thread(target=worker_loop, name=f"Writer-{i}", args=(client, collection_name), daemon=True)
        t.start()
        thread_list.append(t)

    last_compare = time.time()
    try:
        while True:
            time.sleep(1)
            if time.time() - last_compare >= compare_interval:
                logger.info("Pausing writers for entity compare …")
                pause_event.set()

                # Wait for all active operations to complete
                if wait_for_operations_to_complete(timeout=30.0):
                    logger.info("All operations completed, starting entity compare")
                else:
                    logger.warning("Some operations still active, proceeding with entity compare")

                # Additional safety wait to ensure data is flushed
                time.sleep(2)

                try:
                    client.entity_compare(collection_name)
                except Exception:
                    logger.exception("Error during entity_compare")

                last_compare = time.time()
                pause_event.clear()
                logger.info("Writers resumed")
            if duration > 0 and time.time() - start_time >= duration:
                logger.info(f"Duration reached ({duration}s), stopping …")
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping …")
        stop_event.set()
        for t in thread_list:
            t.join(timeout=5)
    finally:
        logger.info("Stopping all writers...")
        stop_event.set()

        # Wait for threads to finish
        for t in thread_list:
            t.join(timeout=5)

        # Wait for any remaining operations
        logger.info("Waiting for final operations to complete...")
        wait_for_operations_to_complete(timeout=30.0)

        # Final safety wait
        time.sleep(2)

        logger.info("Final entity compare...")
        try:
            client.entity_compare(collection_name)
        except Exception:
            logger.exception("Final entity_compare failed")


@cli.command()
@click.option("--uri", type=str, default=None, help="Milvus server URI")
@click.option("--pg-conn", type=str, default=None, help="PostgreSQL DSN")
@click.option(
    "--collection",
    type=str,
    default="data_correctness_checker",
    help="Collection name to validate (default: data_correctness_checker)",
)
@click.option("--full-scan/--no-full-scan", default=True, help="Perform full scan validation (default: enabled)")
@click.option(
    "--include-vector",
    is_flag=True,
    help="Include vector fields in PostgreSQL operations (default: False)",
)
def validate(uri: str | None, pg_conn: str | None, collection: str, full_scan: bool, include_vector: bool) -> None:
    """Validate data consistency between Milvus and PostgreSQL for a collection."""
    uri = uri or os.getenv("MILVUS_URI", "http://localhost:19530")
    pg_conn = pg_conn or os.getenv("PG_CONN", "postgresql://postgres:admin@localhost:5432/postgres")

    client = MilvusClient(uri=uri, pg_conn_str=pg_conn, ignore_vector=not include_vector)
    logger.info(f"Verifying collection: {collection}")
    client.entity_compare(collection, full_scan=full_scan)


if __name__ == "__main__":
    cli()
