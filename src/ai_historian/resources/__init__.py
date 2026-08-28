"""Versioned reference data shipped with AI Historian."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resource_path(filename: str) -> Path:
    path = ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"AI Historian resource not found: {filename}")
    return path


BOOK_CATALOG = resource_path("book_catalog.json")
CHINESE_ERAS = resource_path("chinese_eras.csv")
TIME_STRING_ISO_MAP = resource_path("time_string_iso_map.json")

__all__ = ["BOOK_CATALOG", "CHINESE_ERAS", "TIME_STRING_ISO_MAP", "resource_path"]
