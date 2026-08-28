from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("AIH_WORKSPACE_ROOT", Path.cwd())).expanduser().resolve()
RESULTS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_TEXT_ROOT = PROJECT_ROOT / "examples" / "input"


def derive_result_dir_name_from_input(input_dir: Path) -> str:
    folder_name = input_dir.name.strip() or "step1output"
    folder_name = re.sub(r"_txt_文件$", "", folder_name)
    folder_name = re.sub(r"_txt文件$", "", folder_name)
    folder_name = re.sub(r"_txt$", "", folder_name)
    folder_name = folder_name.strip("_- ") or "step1output"
    if folder_name.startswith("result_"):
        return folder_name
    return f"result_{folder_name}"


def looks_like_step_output_dir(path: Path) -> bool:
    return (
        path.name.startswith("step")
        and path.name.endswith("output")
        and path.parent.name in {"sentence", "timeblock", "sequence"}
    )


def looks_like_run_root(path: Path) -> bool:
    return (
        path.name.startswith("result_")
        or (path / "sentence").is_dir()
        or (path / "timeblock").is_dir()
        or (path / "sequence").is_dir()
    )


def looks_like_text_input_dir(path: Path) -> bool:
    name = path.name.strip()
    return bool(name) and (
        name.endswith("_txt_文件")
        or name.endswith("_txt文件")
        or name.endswith("_txt")
        or any(path.glob("*.txt"))
    )


def sentence_step_dir(run_root: Path, step_num: int) -> Path:
    return Path(run_root) / "sentence" / f"step{step_num}output"


def timeblock_step_dir(run_root: Path, step_num: int) -> Path:
    return Path(run_root) / "timeblock" / f"step{step_num}output"


def sequence_step_dir(run_root: Path, step_num: int) -> Path:
    return Path(run_root) / "sequence" / f"step{step_num}output"


def _candidate_paths(raw_arg: Optional[str]) -> list[Path]:
    if not raw_arg:
        return []

    raw = Path(raw_arg)
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(PROJECT_ROOT / raw)
        if len(raw.parts) == 1:
            candidates.append(RESULTS_ROOT / raw.name)
            candidates.append(DEFAULT_TEXT_ROOT / raw.name)

    seen = set()
    deduped = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def resolve_latest_run_root() -> Optional[Path]:
    if not RESULTS_ROOT.exists():
        return None

    result_dirs = [
        path
        for path in RESULTS_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("result_")
    ]
    if not result_dirs:
        return None

    return max(result_dirs, key=lambda path: path.stat().st_mtime)


def normalize_run_root(path: Path) -> Path:
    path = Path(path)
    if looks_like_step_output_dir(path):
        return path.parent.parent
    return path


def resolve_run_root_from_input_dir(input_dir: Path, output_arg: Optional[str] = None) -> Path:
    if output_arg:
        raw = Path(output_arg)
        if raw.is_absolute():
            return normalize_run_root(raw)

        if len(raw.parts) == 1 and raw.name.startswith("result_"):
            return RESULTS_ROOT / raw.name

        return normalize_run_root(PROJECT_ROOT / raw)

    return RESULTS_ROOT / derive_result_dir_name_from_input(input_dir)


def resolve_run_root(run_root_arg: Optional[str] = None, fallback_to_project_root: bool = True) -> Path:
    for candidate in _candidate_paths(run_root_arg):
        if not candidate.exists():
            continue

        if looks_like_step_output_dir(candidate):
            return candidate.parent.parent

        if looks_like_run_root(candidate):
            return candidate

        if looks_like_text_input_dir(candidate):
            return RESULTS_ROOT / derive_result_dir_name_from_input(candidate)

    if run_root_arg:
        raw = Path(run_root_arg)
        if raw.is_absolute():
            return normalize_run_root(raw)

        if len(raw.parts) == 1 and raw.name.startswith("result_"):
            return RESULTS_ROOT / raw.name

        if looks_like_text_input_dir(raw):
            return RESULTS_ROOT / derive_result_dir_name_from_input(raw)

        return normalize_run_root(PROJECT_ROOT / raw)

    cwd = Path.cwd().resolve()
    if looks_like_step_output_dir(cwd):
        return cwd.parent.parent
    if looks_like_run_root(cwd):
        return cwd

    latest_run_root = resolve_latest_run_root()
    if latest_run_root is not None:
        return latest_run_root

    if fallback_to_project_root:
        return PROJECT_ROOT

    raise FileNotFoundError("未能解析流水线运行目录。")
