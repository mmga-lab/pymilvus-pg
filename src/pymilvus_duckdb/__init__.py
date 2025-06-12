from .logger_config import logger, set_logger_level
from .milvus_pg_client import MilvusPGClient

# Backward compatibility alias – will be removed in future major release
MilvusDuckDBClient = MilvusPGClient

__all__ = [
    "MilvusPGClient",
    "MilvusDuckDBClient",
    "logger",
    "set_logger_level",
]
