#!/usr/bin/env python3
"""Offline validation for an AI Historian model configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from ai_historian.model_config import resolve_chat_config, resolve_embedding_config


def load_env_file(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def resolved_config() -> dict[str, str]:
    configuration_error = ""
    try:
        chat = resolve_chat_config(
            require_api_key=False,
            require_model=False,
            require_base_url=False,
        )
    except RuntimeError as exc:
        configuration_error = str(exc)
        chat = None
    embedding = resolve_embedding_config(required=False)
    return {
        "configuration_error": configuration_error,
        "provider": chat.provider if chat else os.getenv("AIH_CHAT_PROVIDER", "").strip(),
        "api_key": chat.api_key if chat else "",
        "key_source": chat.api_key_env if chat else "",
        "base_url": chat.base_url if chat else "",
        "base_url_source": chat.base_url_env if chat else "",
        "model": chat.model if chat else os.getenv("AIH_CHAT_MODEL", "").strip(),
        "selector": os.getenv("AIH_CROSSDOC_SCOPE_SELECTOR", "lexical").strip().lower(),
        "embed_key": embedding.api_key,
        "embed_base_url": embedding.base_url,
        "embed_model": embedding.model,
    }


def validate(config: dict[str, str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    if config["configuration_error"]:
        errors.append(config["configuration_error"])
    if not config["model"]:
        errors.append("AIH_CHAT_MODEL is required; set the exact model ID for this run.")
    elif any(marker in config["model"].lower() for marker in ("your-model", "replace-with")):
        errors.append("AIH_CHAT_MODEL still contains a template placeholder; set the served model ID.")
    parsed = urlparse(config["base_url"])
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        source = config["base_url_source"] or "The selected provider base URL"
        errors.append(f"{source} must be an absolute HTTP(S) URL.")
    elif (parsed.hostname or "").endswith(".example"):
        errors.append(
            f"{config['base_url_source']} still contains a template placeholder; set the endpoint URL."
        )
    if not config["api_key"] and config["key_source"]:
        errors.append(
            f"{config['key_source']} is required when AIH_CHAT_PROVIDER={config['provider']}."
        )
    if config["selector"] in {"embedding", "hybrid"}:
        if not config["embed_model"] or not config["embed_base_url"]:
            errors.append("Embedding/hybrid retrieval requires AIH_EMBED_MODEL and AIH_EMBED_BASE_URL.")
        if not config["embed_key"]:
            errors.append("Embedding/hybrid retrieval requires AIH_EMBED_API_KEY.")
    else:
        notes.append("Lexical retrieval is ready; embedding configuration is optional.")
    notes.append("Offline validation complete; run smoke_test_model.py for endpoint and JSON-output validation.")
    return errors, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Load configuration from this .env file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    config = resolved_config()
    errors, notes = validate(config)
    public = {
        k: v
        for k, v in config.items()
        if k not in {"api_key", "embed_key", "configuration_error"}
    }
    public["credentials"] = "configured" if config["api_key"] else "missing"
    payload = {"ok": not errors, "config": public, "errors": errors, "notes": notes}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"AIH model configuration: {'READY' if not errors else 'CHECK REQUIRED'}")
        print(json.dumps(public, ensure_ascii=False, indent=2))
        for item in errors:
            print(f"ERROR: {item}")
        for item in notes:
            print(f"NOTE: {item}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
