#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "src"
STEP11_MODULE = "ai_historian.profiles.evaluation.stages.step_11_iso_normalization"

MONTH_ONLY_RE = re.compile(r"[正一二三四五六七八九十冬腊]+月")
SEASON_RE = re.compile(r"(春天|春季|春|夏天|夏季|夏|秋天|秋季|秋|冬天|冬季|冬)")
YEAR_RE = re.compile(r"(?:(?:汉高祖|汉王|汉|秦二世|秦始皇|秦王子婴)[元一二三四五六七八九十]+年|(?:公元)?前?\d+年)")


def month_precision_iso(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in {"-infinity", "+infinity"}:
        return text
    match = re.match(r"^([+-]?\d{4,})-(\d{2})(?:-\d{2})?$", text)
    return f"{match.group(1)}-{match.group(2)}" if match else ""


def compose_iso_range(start: str, end: str) -> str:
    start = str(start or "").strip()
    end = str(end or "").strip()
    return f"{start}to{end}" if start and end else ""


def infer_granularity(time_text: Any) -> str:
    text = re.sub(r"\s+", "", str(time_text or "").strip())
    if not text:
        return "0"
    if MONTH_ONLY_RE.search(text):
        return "2"
    if SEASON_RE.search(text):
        return "1"
    if YEAR_RE.search(text):
        return "1"
    return "1"


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("case_id", "")).strip(),
        str(row.get("part_id", "")).strip(),
        str(row.get("item_no", "")).strip(),
        str(row.get("sentence_id", "")).strip(),
    )


def numeric_part_id(part_id: str) -> str:
    text = str(part_id or "").strip()
    match = re.search(r"\d+", text)
    if not match:
        raise ValueError(f"part_id does not contain a numeric chapter id: {part_id}")
    return match.group(0)


def make_boundary_id(case_id: str, part_id: str, item_no: str, boundary_type: str) -> tuple[str, str]:
    chapter_id = numeric_part_id(part_id)
    synthetic_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"direct-llm:{case_id}:{part_id}"))
    suffix = "1" if boundary_type == "start" else "2"
    point_id = f"{synthetic_uuid}.{chapter_id}.{int(item_no):04d}.{suffix}"
    return chapter_id, point_id


def build_step11_workspace(rows: list[dict[str, str]], run_dir: Path) -> dict[str, dict[str, Any]]:
    workspace = run_dir / "agent_postprocess_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)

    timeblock_dir = workspace / "timeblock" / "step10output"
    sequence_dir = workspace / "sequence" / "step8output"
    timeblock_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    boundary_lookup: dict[str, dict[str, Any]] = {}

    for row in rows:
        if any(str(row.get(flag, "")).strip() for flag in ("ai_unknown", "ai_sink", "ai_interlude")):
            continue
        for boundary_type, field_name in (("start", "ai_timeblock_start_tm"), ("end", "ai_timeblock_end_tm")):
            boundary_text = str(row.get(field_name, "") or "").strip()
            if not boundary_text:
                continue
            case_id = str(row.get("case_id", "")).strip()
            part_id = str(row.get("part_id", "")).strip()
            item_no = str(row.get("item_no", "")).strip()
            chapter_id, point_id = make_boundary_id(case_id, part_id, item_no, boundary_type)
            group_key = (chapter_id, part_id)
            boundary_obj = {
                "ID": point_id,
                "Range": f"{point_id}-{point_id}",
                "TM": boundary_text,
                "Granularity": infer_granularity(boundary_text),
                "Interlude": False,
                "Sink": False,
                "DirectLLMBoundary": {
                    "case_id": case_id,
                    "part_id": part_id,
                    "item_no": item_no,
                    "sentence_id": str(row.get("sentence_id", "")).strip(),
                    "boundary_type": boundary_type,
                    "boundary_text": boundary_text,
                },
            }
            grouped.setdefault(group_key, []).append(boundary_obj)
            boundary_lookup[point_id] = boundary_obj["DirectLLMBoundary"]

    for (chapter_id, part_id), objects in grouped.items():
        synthetic_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"direct-llm:{part_id}"))
        base_name = f"{chapter_id}_{synthetic_uuid}"
        objects.sort(key=lambda item: item["ID"])
        (timeblock_dir / f"{base_name}_timeblock.json").write_text(
            json.dumps({"TMB": objects}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (sequence_dir / f"{base_name}_sequence.json").write_text(
            json.dumps([item["ID"] for item in objects], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "workspace": workspace,
        "boundary_lookup": boundary_lookup,
    }


def run_step11(workspace: Path) -> tuple[float, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(SOURCE_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    started_at = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", STEP11_MODULE, str(workspace)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Agent Step11 post-processing failed.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    return time.time() - started_at, proc.stdout + proc.stderr


def read_step11_results(workspace: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    output_dir = workspace / "timeblock" / "step11output"
    if not output_dir.exists():
        raise FileNotFoundError(f"Step11 output not found: {output_dir}")

    result: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for path in sorted(output_dir.glob("*_timeblock.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for obj in payload.get("TMB", []):
            meta = obj.get("DirectLLMBoundary")
            if not isinstance(meta, dict):
                continue
            key = (
                str(meta.get("case_id", "")).strip(),
                str(meta.get("part_id", "")).strip(),
                str(meta.get("item_no", "")).strip(),
                str(meta.get("sentence_id", "")).strip(),
                str(meta.get("boundary_type", "")).strip(),
            )
            result[key] = {
                "raw_tm": str(meta.get("boundary_text", "")).strip(),
                "canonical_tm": str(obj.get("TM", "") or "").strip(),
                "granularity": str(obj.get("Granularity", "") or "").strip(),
                "iso_day": str(obj.get("iso", "") or "").strip(),
                "iso_month": month_precision_iso(obj.get("iso", "")),
                "synthetic_timeblock_id": str(obj.get("ID", "") or "").strip(),
            }
    return result


def postprocess_rows_with_agent_step11(rows: list[dict[str, str]], run_dir: Path) -> dict[str, Any]:
    manifest = build_step11_workspace(rows, run_dir)
    workspace = manifest["workspace"]
    boundary_results: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    step11_seconds = 0.0
    step11_log = ""

    if manifest["boundary_lookup"]:
        step11_seconds, step11_log = run_step11(workspace)
        boundary_results = read_step11_results(workspace)

    postprocessed_rows: list[dict[str, str]] = []
    unresolved_rows: list[dict[str, str]] = []

    for row in rows:
        updated = dict(row)
        is_special = any(str(updated.get(flag, "")).strip() for flag in ("ai_unknown", "ai_sink", "ai_interlude"))
        note_parts = [str(updated.get("ai_agent_note", "") or "").strip()]

        start_key = row_key(updated) + ("start",)
        end_key = row_key(updated) + ("end",)
        start_result = boundary_results.get(start_key, {})
        end_result = boundary_results.get(end_key, {})

        if is_special:
            updated["ai_start_ym"] = ""
            updated["ai_end_ym"] = ""
            updated["ai_iso_range"] = ""
        else:
            start_iso = start_result.get("iso_month", "")
            end_iso = end_result.get("iso_month", "")
            updated["ai_start_ym"] = start_iso
            updated["ai_end_ym"] = end_iso
            updated["ai_iso_range"] = compose_iso_range(start_iso, end_iso)
            updated["ai_timeblock_id"] = "|".join(
                part for part in [start_result.get("synthetic_timeblock_id", ""), end_result.get("synthetic_timeblock_id", "")] if part
            )

            for boundary_type, source_field, result in (
                ("start", "ai_timeblock_start_tm", start_result),
                ("end", "ai_timeblock_end_tm", end_result),
            ):
                original_text = str(updated.get(source_field, "") or "").strip()
                canonical_tm = result.get("canonical_tm", "")
                iso_month = result.get("iso_month", "")
                if original_text and not iso_month:
                    unresolved_rows.append(
                        {
                            "case_id": str(updated.get("case_id", "")).strip(),
                            "part_id": str(updated.get("part_id", "")).strip(),
                            "item_no": str(updated.get("item_no", "")).strip(),
                            "sentence_id": str(updated.get("sentence_id", "")).strip(),
                            "boundary_type": boundary_type,
                            "raw_time_text": original_text,
                            "canonical_time_text": canonical_tm,
                            "synthetic_timeblock_id": result.get("synthetic_timeblock_id", ""),
                        }
                    )

            note_parts.append(
                "step11_postprocess="
                f"start[{start_result.get('canonical_tm', '')}->{start_result.get('iso_month', '')}] "
                f"end[{end_result.get('canonical_tm', '')}->{end_result.get('iso_month', '')}]"
            )

        updated["ai_agent_note"] = " | ".join(part for part in note_parts if part)
        postprocessed_rows.append(updated)

    return {
        "rows": postprocessed_rows,
        "unresolved_rows": unresolved_rows,
        "step11_seconds": round(step11_seconds, 3),
        "step11_log": step11_log,
        "workspace_dir": str(workspace),
    }
