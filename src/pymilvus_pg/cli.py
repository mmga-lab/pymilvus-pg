"""Command line interface for pymilvus_pg."""

from __future__ import annotations

import json
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
from pymilvus_pg.builtin_schemas import SCHEMA_PRESETS, get_schema_by_name, list_schema_presets

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


# Global schema config for operations
_current_schema_config: dict[str, Any] = {}


def _generate_data(id_list: list[int], for_upsert: bool = False) -> list[dict[str, Any]]:
    """Generate records based on id list using current schema config."""
    if _current_schema_config:
        return _generate_data_for_schema(_current_schema_config, id_list, for_upsert)

    # DEBUG: Log why fallback is being used
    print(f"WARNING: Using fallback data generation. _current_schema_config is: {_current_schema_config}")

    # Fallback to original data generation for backward compatibility
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
        generated_data = _generate_data(ids)
        logger.debug(f"[INSERT] Generated data sample: {generated_data[0] if generated_data else 'empty'}")
        if generated_data:
            # Check for potential $meta conflicts
            sample_keys = set(generated_data[0].keys())
            if '$meta' in sample_keys:
                logger.warning(f"[INSERT] Found '$meta' key in generated data! Keys: {sample_keys}")
            logger.debug(f"[INSERT] Data keys: {sample_keys}")
        client.insert(collection, generated_data)
        logger.info(f"[INSERT] {len(ids)} rows, start id {ids[0]}")
    except Exception as e:
        logger.error(f"[INSERT] Exception occurred: {e}")
        logger.error(f"[INSERT] Exception type: {type(e).__name__}")
        logger.error(f"[INSERT] Exception details: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            logger.error(f"[INSERT] Full traceback: {traceback.format_exc()}")
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
        if hasattr(client, "_get_pg_connection"):
            # Ensure schema is loaded to get primary field
            client._get_schema(collection)
            if client.primary_field:
                with client._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        # Get random sample of existing IDs from PostgreSQL
                        cursor.execute(
                            f"SELECT {client.primary_field} FROM {collection} ORDER BY RANDOM() LIMIT {DELETE_BATCH_SIZE}"
                        )
                        result = cursor.fetchall()
                        ids = [row[0] for row in result] if result else []
            else:
                ids = []
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
            generated_data = _generate_data(ids, for_upsert=True)
            logger.debug(f"[UPSERT] Generated data sample: {generated_data[0] if generated_data else 'empty'}")
            if generated_data:
                # Check for potential $meta conflicts
                sample_keys = set(generated_data[0].keys())
                if '$meta' in sample_keys:
                    logger.warning(f"[UPSERT] Found '$meta' key in generated data! Keys: {sample_keys}")
                logger.debug(f"[UPSERT] Data keys: {sample_keys}")
            client.upsert(collection, generated_data)
            logger.info(f"[UPSERT] {len(ids)} rows, start id {start}")
    except Exception as e:
        logger.error(f"[UPSERT] Exception occurred: {e}")
        logger.error(f"[UPSERT] Exception type: {type(e).__name__}")
        logger.error(f"[UPSERT] Exception details: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            logger.error(f"[UPSERT] Full traceback: {traceback.format_exc()}")
    finally:
        with active_operations_lock:
            active_operations -= 1


OPERATIONS = [_insert_op, _delete_op, _upsert_op]


def create_collection_from_config(
    client: MilvusClient, collection_name: str, schema_config: dict[str, Any], drop_if_exists: bool = True
) -> None:
    """Create collection from schema configuration dict."""
    if client.has_collection(collection_name):
        if drop_if_exists:
            logger.warning(f"Collection {collection_name} already exists, dropping")
            client.drop_collection(collection_name)
        else:
            logger.info(f"Collection {collection_name} already exists, will continue using it")
            return

    # Create schema from config
    schema = client.create_schema(enable_dynamic_field=schema_config.get("enable_dynamic_field", False))

    # Add fields from config
    vector_fields = []
    for field_config in schema_config["fields"]:
        field_name = field_config["name"]
        field_type = getattr(DataType, field_config["type"])

        # Base field parameters
        field_params = {
            "field_name": field_name,
            "datatype": field_type,
        }

        # Add optional parameters
        if field_config.get("is_primary", False):
            field_params["is_primary"] = True
            field_params["auto_id"] = field_config.get("auto_id", False)

        if field_config.get("nullable", False):
            field_params["nullable"] = True

        if "default_value" in field_config:
            field_params["default_value"] = field_config["default_value"]

        if "max_length" in field_config:
            field_params["max_length"] = field_config["max_length"]

        if "dim" in field_config:
            field_params["dim"] = field_config["dim"]

        if field_config.get("type") == "ARRAY":
            field_params["element_type"] = getattr(DataType, field_config["element_type"])
            field_params["max_capacity"] = field_config.get("max_capacity", 100)

        # Track vector fields for indexing
        if field_type in [
            DataType.FLOAT_VECTOR,
            DataType.BINARY_VECTOR,
            DataType.SPARSE_FLOAT_VECTOR,
        ]:
            vector_fields.append(field_name)

        schema.add_field(**field_params)

    # Create collection
    client.create_collection(collection_name, schema)

    # Create indexes for vector fields
    if vector_fields:
        index_params = IndexParams()
        for vector_field in vector_fields:
            # Get field info to determine index type
            field_info = next((f for f in schema_config["fields"] if f["name"] == vector_field), None)
            if field_info:
                field_type = field_info["type"]

                if field_type == "SPARSE_FLOAT_VECTOR":
                    index_params.add_index(vector_field, index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
                elif field_type == "BINARY_VECTOR":
                    index_params.add_index(
                        vector_field, index_type="BIN_IVF_FLAT", metric_type="HAMMING", params={"nlist": 128}
                    )
                else:
                    # FLOAT_VECTOR, FLOAT16_VECTOR, BFLOAT16_VECTOR
                    index_params.add_index(vector_field, index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})

        client.create_index(collection_name, index_params)

    # Load collection
    client.load_collection(collection_name)
    logger.info(
        f"Collection {collection_name} created and loaded with schema: {schema_config.get('description', 'custom')}"
    )


def get_default_test_schema() -> dict[str, Any]:
    """Get the original default test schema for backward compatibility."""
    return {
        "fields": [
            {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False},
            {"name": "category", "type": "VARCHAR", "max_length": 256, "is_partition_key": True},
            {"name": "name", "type": "VARCHAR", "max_length": 256},
            {"name": "age", "type": "INT64"},
            {"name": "json_field", "type": "JSON"},
            {"name": "array_field", "type": "ARRAY", "element_type": "INT64", "max_capacity": 20},
            {"name": "embedding", "type": "FLOAT_VECTOR", "dim": DIMENSION},
        ],
        "enable_dynamic_field": False,
        "description": "Default test schema (backward compatible)",
    }


def _generate_data_for_schema(
    schema_config: dict[str, Any], id_list: list[int], for_upsert: bool = False
) -> list[dict[str, Any]]:
    """Generate test data based on schema configuration."""
    data = []

    for _id in id_list:
        record: dict[str, Any] = {}

        # Generate data for each defined field
        for field_config in schema_config["fields"]:
            field_name = field_config["name"]
            field_type = field_config["type"]

            # Always generate primary key value - auto_id is not supported
            if field_config.get("is_primary", False):
                record[field_name] = _id
                continue

            # Randomly skip nullable fields (30% chance)
            if field_config.get("nullable", False) and random.random() < 0.3:
                if random.random() < 0.5:
                    # Explicitly set to None
                    record[field_name] = None
                # Otherwise, don't set the field (will use default if available)
                continue

            # Generate data based on type
            if field_type == "BOOL":
                record[field_name] = random.choice([True, False])
            elif field_type in ["INT8", "INT16", "INT32", "INT64"]:
                base_value = random.randint(1, 1000)
                record[field_name] = base_value + (1000 if for_upsert else 0)
            elif field_type in ["FLOAT", "DOUBLE"]:
                base_value_float = random.uniform(0.0, 1000.0)
                record[field_name] = base_value_float + (1000.0 if for_upsert else 0.0)
            elif field_type == "VARCHAR":
                max_length = field_config.get("max_length", 100)
                suffix = "_upserted" if for_upsert else ""
                base_value = f"{field_name}_{_id}{suffix}"

                # Respect max_length constraint
                if len(base_value) > max_length:
                    # For short fields, use appropriate values
                    if max_length <= 5:
                        # Very short fields - use simple values
                        short_values = (
                            ["USD", "EUR", "GBP", "JPY", "CNY"]
                            if field_name == "currency"
                            else [f"v{i}" for i in range(10)]
                        )
                        record[field_name] = random.choice(short_values)[:max_length]
                    else:
                        # Truncate to fit max_length
                        record[field_name] = base_value[:max_length]
                else:
                    record[field_name] = base_value
            elif field_type == "JSON":
                record[field_name] = {
                    "id": _id,
                    "type": field_name,
                    "upserted": for_upsert,
                    "nested": {"value": random.randint(1, 100)},
                }
            elif field_type == "ARRAY":
                element_type = field_config.get("element_type", "VARCHAR")
                max_capacity = field_config.get("max_capacity", 10)
                array_size = random.randint(1, min(5, max_capacity))

                if element_type == "VARCHAR":
                    record[field_name] = [f"item_{i}_{_id}" for i in range(array_size)]
                elif element_type in ["INT64", "INT32"]:
                    record[field_name] = [_id + i for i in range(array_size)]
                elif element_type in ["FLOAT", "DOUBLE"]:
                    record[field_name] = [float(_id + i) for i in range(array_size)]
                else:
                    record[field_name] = [f"val_{i}" for i in range(array_size)]
            elif field_type in ["FLOAT_VECTOR"]:
                dim = field_config.get("dim", 128)
                record[field_name] = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            elif field_type == "BINARY_VECTOR":
                dim = field_config.get("dim", 128)
                # Generate binary vector as list of integers (0 or 1)
                record[field_name] = [random.randint(0, 1) for _ in range(dim)]
            elif field_type == "SPARSE_FLOAT_VECTOR":
                # Generate sparse vector as dict {index: value}
                num_non_zero = random.randint(5, 20)
                indices = random.sample(range(1000), num_non_zero)
                record[field_name] = {str(idx): random.uniform(0.1, 1.0) for idx in indices}

        # Add dynamic fields if enabled
        if schema_config.get("enable_dynamic_field", False) and random.random() > 0.5:
            dynamic_field_count = random.randint(1, 3)
            for i in range(dynamic_field_count):
                dynamic_key = f"dynamic_field_{i}"
                if random.random() > 0.5:
                    record[dynamic_key] = f"dynamic_value_{_id}_{i}"
                else:
                    record[dynamic_key] = random.randint(1, 1000)

        data.append(record)

    # Debug: Validate generated data
    if data and all(value is None for record in data[:3] for value in record.values()):
        print("ERROR in _generate_data_for_schema: Generated data contains all None values!")
        print(f"Schema config: {schema_config}")
        print(f"ID list: {id_list}")
        print(f"First record: {data[0]}")

    return data


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
    """Create collection using default schema (backward compatibility)."""
    global _current_schema_config

    # Use default schema for backward compatibility
    schema_config = get_default_test_schema()
    _current_schema_config = schema_config

    create_collection_from_config(client, name, schema_config, drop_if_exists)


@click.group()
@click.version_option(version=__version__, prog_name="pymilvus-pg")
def cli() -> None:
    """PyMilvus-PG CLI for data consistency validation."""
    pass


@cli.command()
def list_schemas() -> None:
    """List available built-in schema presets."""
    presets = list_schema_presets()
    click.echo("Available built-in schema presets:")
    for preset in presets:
        schema_func = SCHEMA_PRESETS[preset]
        schema_config = schema_func()  # type: ignore[operator]
        description = schema_config.get("description", "No description")
        field_count = len(schema_config["fields"])
        dynamic = schema_config.get("enable_dynamic_field", False)
        click.echo(f"  {preset:12} - {description} ({field_count} fields, dynamic: {dynamic})")

    click.echo("\nUse with: pymilvus-pg ingest --schema <name>")


@cli.command()
@click.argument("preset_name")
@click.option("--format", type=click.Choice(["json", "yaml"]), default="json", help="Output format")
def show_schema(preset_name: str, format: str) -> None:
    """Show detailed schema configuration for a preset."""
    try:
        schema_config = get_schema_by_name(preset_name)

        if format == "yaml":
            try:
                import yaml  # type: ignore[import-untyped]

                output = yaml.dump(schema_config, default_flow_style=False, sort_keys=False)
            except ImportError:
                click.echo("PyYAML not installed. Install with: pip install PyYAML", err=True)
                return
        else:
            output = json.dumps(schema_config, indent=2)

        click.echo(output)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)


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
@click.option("--schema", type=click.Choice(list(SCHEMA_PRESETS.keys())), help="Use built-in schema preset")
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
    schema: str | None,
) -> None:
    """Continuously ingest data with periodic validation checks.

    Performs high-throughput data ingestion using insert/delete/upsert operations
    while periodically validating data consistency between Milvus and PostgreSQL.
    """
    global _global_id, _current_schema_config

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

    # Get schema configuration
    try:
        if schema:
            schema_config = get_schema_by_name(schema)
            logger.info(f"Using schema preset: {schema}")
        else:
            schema_config = get_default_test_schema()
            logger.info("Using default test schema")

        # Set global schema config for data generation
        _current_schema_config = schema_config

        # Create collection with schema
        create_collection_from_config(client, collection_name, schema_config, drop_if_exists=drop_existing)

    except (ValueError, KeyError) as e:
        logger.error(f"Schema configuration error: {e}")
        raise click.ClickException(f"Schema error: {e}") from e

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
