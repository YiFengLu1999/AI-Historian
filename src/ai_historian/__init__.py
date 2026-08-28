"""Public Python package for AI Historian."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-historian")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
