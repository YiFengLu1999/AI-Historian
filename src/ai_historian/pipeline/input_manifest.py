"""Explicit metadata contract for raw text collections."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Iterable

INPUT_MANIFEST_NAME = "manifest.json"
INPUT_MANIFEST_SCHEMA = "ai_historian_input_v1"


def stable_collection_uuid(
    text_files: Iterable[Path],
    documents: dict[str, dict[str, str]],
) -> str:
    """Return a location-independent UUID for an input collection."""
    identity = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "documents": [
            {
                "file": path.name,
                "person": documents[path.name]["person"],
                "title": documents[path.name]["title"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(text_files, key=lambda item: item.name)
        ],
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ai-historian:{canonical}"))


def load_input_manifest(input_dir: Path, text_files: Iterable[Path]) -> dict[str, dict[str, str]]:
    """Load and validate metadata for every text file in an input directory."""
    manifest_path = input_dir / INPUT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {INPUT_MANIFEST_NAME} in input directory: {input_dir}"
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise ValueError(
            f"{manifest_path} must declare schema={INPUT_MANIFEST_SCHEMA!r}"
        )
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{manifest_path} must contain a non-empty documents list")

    metadata: dict[str, dict[str, str]] = {}
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            raise ValueError(f"documents[{index}] must be an object")
        filename = str(document.get("file", "")).strip()
        person = str(document.get("person", "")).strip()
        title = str(document.get("title", "")).strip()
        if not filename or Path(filename).name != filename or not filename.endswith(".txt"):
            raise ValueError(f"documents[{index}].file must be a top-level .txt filename")
        if not person or not title:
            raise ValueError(f"documents[{index}] requires non-empty person and title")
        if filename in metadata:
            raise ValueError(f"Duplicate document metadata for {filename}")
        metadata[filename] = {"person": person, "title": title}

    actual = {path.name for path in text_files}
    declared = set(metadata)
    if actual != declared:
        missing = sorted(actual - declared)
        unknown = sorted(declared - actual)
        details = []
        if missing:
            details.append(f"missing metadata for {missing}")
        if unknown:
            details.append(f"metadata references missing files {unknown}")
        raise ValueError(f"{manifest_path}: {'; '.join(details)}")
    return metadata


def source_metadata(txt_path: Path, documents: dict[str, dict[str, str]]) -> dict[str, str]:
    """Resolve source fields without encoding metadata into the filename."""
    document = documents[txt_path.name]
    return {
        "file_stem": txt_path.stem.strip(),
        "source_person": document["person"],
        "source_title": document["title"],
    }


__all__ = [
    "INPUT_MANIFEST_NAME",
    "INPUT_MANIFEST_SCHEMA",
    "load_input_manifest",
    "source_metadata",
    "stable_collection_uuid",
]
