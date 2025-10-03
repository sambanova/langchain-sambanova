"""Main entrypoint into package."""

from importlib import metadata

try:
    __version__ = metadata.version("langchain-sambanova")
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = "0.0.0"
