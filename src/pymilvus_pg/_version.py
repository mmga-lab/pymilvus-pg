"""Version information for pymilvus-pg."""

import importlib.metadata

try:
    # This will work when the package is installed
    __version__ = importlib.metadata.version("pymilvus-pg")
except importlib.metadata.PackageNotFoundError:
    # During development, when running from source
    __version__ = "0.0.0.dev0"
