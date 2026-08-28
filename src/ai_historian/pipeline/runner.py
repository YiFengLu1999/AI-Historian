"""Subprocess stage runner with a reproducible run manifest."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ai_historian import __version__
from ai_historian.pipeline.input_manifest import INPUT_MANIFEST_NAME

SAFE_CONFIG_KEYS = (
    # Provider and endpoint selection. Credentials are intentionally excluded.
    "AIH_CHAT_PROVIDER",
    "AIH_CHAT_MODEL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "GEMINI_BASE_URL",
    "DASHSCOPE_BASE_URL",
    "AIH_COMPATIBLE_BASE_URL",
    "AIH_EMBED_MODEL",
    "AIH_EMBED_BASE_URL",
    "AIH_PIPELINE_CONCURRENCY",
    "AIH_PIPELINE_BATCH_SIZE",
    "AIH_AGENT_CONCURRENCY",
    "AIH_AGENT_BATCH_SIZE",
    "AIH_AGENT_MAX_WORKERS",
    "AIH_REQUEST_TIMEOUT",
    # Runtime retrieval and cross-document evidence controls.
    "AIH_CROSSDOC_SCOPE_MODE",
    "AIH_CROSSDOC_SCOPE_SELECTOR",
    "AIH_CROSSDOC_SCOPE_MAX_CASES",
    "AIH_CROSSDOC_SCOPE_TOP_K_PER_PAIR",
    "AIH_CROSSDOC_SCOPE_WINDOW_SIZE",
    "AIH_CROSSDOC_SCOPE_OVERLAP",
    "AIH_CROSSDOC_SCOPE_CONTEXT_PAD",
    "AIH_CROSSDOC_SCOPE_MIN_SCORE",
    "AIH_CROSSDOC_SCOPE_ANCHOR_SEARCH",
    "AIH_CROSSDOC_SCOPE_PRE_ANCHOR_BACKFILL",
    "AIH_CROSSDOC_SCOPE_FALLBACK_LEXICAL",
    "AIH_CROSSDOC_SCOPE_RERUN_STEP10_FROM_STEP9",
    "AIH_CROSSDOC_SCOPE_EMBEDDING_BATCH_SIZE",
    "AIH_CROSSDOC_SCOPE_EMBEDDING_TEXT_CHARS",
    "AIH_CROSSDOC_PREALIGN_TOP_K",
    "AIH_CROSSDOC_PREALIGN_MIN_SCORE",
    "AIH_CROSSDOC_PREALIGN_MIN_EPISODE_CONF",
    "AIH_CROSSDOC_PREALIGN_MAX_VERIFY",
    "AIH_CROSSDOC_PREALIGN_MAX_TEXT_CHARS",
    "AIH_CROSSDOC_SCHEMA_MAX_TEXT_CHARS",
    "AIH_CROSSDOC_QUOTE_MIN_SCORE",
    "AIH_CROSSDOC_CONTEXT_MIN_CONF",
    "AIH_CROSSDOC_WEAK_CONTEXT_MIN_CONF",
    "AIH_CROSSDOC_TIME_EVIDENCE_MIN_CONF",
    "AIH_CROSSDOC_INTERVAL_MAX_ANCHOR_SPAN",
    "AIH_CROSSDOC_QUALITY_GATE_MODE",
    "AIH_CROSSDOC_CONTEXT_MODE",
    "AIH_CROSSDOC_RECALL_ACCEPT_WEAK_CONTEXT",
    "AIH_APPLY_CROSSDOC_INTERVAL_RANGE",
    "AIH_ENABLE_SOFT_CONTEXT_BOUNDARIES",
    # Stage-specific deterministic and output-affecting controls.
    "AIH_ISO_INPUT_STEP",
    "AIH_ISO_OUTPUT_STEP",
    "AIH_ISO_STEP_LABEL",
    "STEP12_MODE",
    "STEP14_BATCH_SIZE",
    "STEP14_MAX_CHARS_PER_CHUNK",
    "STEP14_MAX_RETRIES",
    "STEP14_RETRY_BASE_SECONDS",
    "STEP14_REQUEST_TIMEOUT_SECONDS",
    "STEP14_SKIP_EXISTING_SUMMARY",
)


def repository_root() -> Path:
    configured = os.getenv("AIH_WORKSPACE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "experiments").is_dir():
            return candidate
    return Path.cwd().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_inventory(input_dir: Path) -> list[dict[str, object]]:
    paths = list(input_dir.glob("*.txt"))
    manifest_path = input_dir / INPUT_MANIFEST_NAME
    if manifest_path.is_file():
        paths.append(manifest_path)
    return [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


@dataclass(frozen=True)
class Stage:
    label: str
    module: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


class RunRecorder:
    def __init__(
        self,
        *,
        profile: str,
        input_dir: Path,
        output_dir: Path,
        argv: Sequence[str],
        stages: Iterable[Stage],
        root: Path | None = None,
    ) -> None:
        self.root = root or repository_root()
        self.output_dir = output_dir
        self.path = output_dir / "run_manifest.json"
        self.payload: dict[str, object] = {
            "schema": "ai_historian_run_v1",
            "status": "running",
            "profile": profile,
            "ai_historian_version": __version__,
            "source_commit": git_commit(self.root),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "command": list(argv),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "inputs": input_inventory(input_dir),
            "configuration": {
                key: os.environ[key]
                for key in SAFE_CONFIG_KEYS
                if os.environ.get(key, "").strip()
            },
            "stage_configuration": {
                stage.label: {
                    key: value
                    for key, value in stage.env.items()
                    if key in SAFE_CONFIG_KEYS and str(value).strip()
                }
                for stage in stages
                if any(
                    key in SAFE_CONFIG_KEYS and str(value).strip()
                    for key, value in stage.env.items()
                )
            },
            "stages": [stage.label for stage in stages],
            "completed_stages": [],
        }
        self.write()

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)

    def stage_completed(self, label: str) -> None:
        completed = self.payload["completed_stages"]
        assert isinstance(completed, list)
        completed.append(label)
        self.write()

    def finish(self, status: str, error: str = "") -> None:
        self.payload["status"] = status
        self.payload["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        )
        if error:
            self.payload["error"] = error
        self.write()


def run_stage(stage: Stage, *, run_root: Path, root: Path | None = None) -> None:
    workspace = root or repository_root()
    env = os.environ.copy()
    env["AIH_WORKSPACE_ROOT"] = str(workspace)
    env.update(stage.env)
    command = [sys.executable, "-m", stage.module, str(run_root), *stage.args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=workspace, env=env, check=True)
