from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_historian.pipeline.logging import StepReporter, setup_step_logging
from ai_historian.pipeline.paths import resolve_run_root, sequence_step_dir, timeblock_step_dir
from ai_historian.pipeline.time_canonicalizer import normalize_experiment1_tm
from ai_historian.resources import TIME_STRING_ISO_MAP

RUN_ROOT: Path
TIMEBLOCK_DIR: Path
SEQUENCE_DIR: Path


ISO_RE = re.compile(r"^(?P<year>[+-]?\d{4,})-(?P<month>\d{2})(?:-\d{2})?$")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_iso_lookup() -> Dict[str, str]:
    data = load_json(TIME_STRING_ISO_MAP)
    raw_map = data.get("map", data) if isinstance(data, dict) else {}
    return {
        normalize_experiment1_tm(str(key or "").strip()): str(value or "").strip()
        for key, value in raw_map.items()
        if str(key or "").strip() and str(value or "").strip()
    }


def parse_iso_month(value: str) -> Optional[Tuple[int, int]]:
    value = str(value or "").strip()
    if value in {"-infinity", "+infinity"}:
        return None
    match = ISO_RE.match(value)
    if not match:
        return None
    return int(match.group("year")), int(match.group("month"))


def iso_less(a: str, b: str) -> bool:
    if not a or not b or a in {"-infinity", "+infinity"} or b in {"-infinity", "+infinity"}:
        return False
    a_key = parse_iso_month(a)
    b_key = parse_iso_month(b)
    return bool(a_key and b_key and a_key < b_key)


def iso_greater(a: str, b: str) -> bool:
    if not a or not b or a in {"-infinity", "+infinity"} or b in {"-infinity", "+infinity"}:
        return False
    a_key = parse_iso_month(a)
    b_key = parse_iso_month(b)
    return bool(a_key and b_key and a_key > b_key)


def tm_to_iso(tm: str, lookup: Dict[str, str]) -> str:
    tm = normalize_experiment1_tm(str(tm or "").strip())
    if not tm:
        return ""
    return lookup.get(tm, "")


def granularity(block: Dict[str, Any]) -> str:
    return str(block.get("Granularity", "")).strip()


def is_anchor(block: Dict[str, Any]) -> bool:
    return granularity(block) != "0" and bool(str(block.get("TM", "") or "").strip())


def ordered_blocks(doc_id: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(block.get("ID", "") or ""): block for block in blocks if str(block.get("ID", "") or "")}
    seq_path = SEQUENCE_DIR / f"{doc_id}_sequence.json"
    if not seq_path.exists():
        return blocks
    try:
        sequence = [str(item).strip() for item in load_json(seq_path) if str(item).strip()]
    except Exception:
        return blocks
    ordered = [by_id[item_id] for item_id in sequence if item_id in by_id]
    seen = {str(block.get("ID", "") or "") for block in ordered}
    ordered.extend(block for block in blocks if str(block.get("ID", "") or "") not in seen)
    return ordered


def nearest_anchor_context(
    ordered: List[Dict[str, Any]],
    index: int,
    lookup: Dict[str, str],
) -> Dict[str, Any]:
    prev_anchor: Optional[Dict[str, Any]] = None
    next_anchor: Optional[Dict[str, Any]] = None
    for left in range(index - 1, -1, -1):
        if is_anchor(ordered[left]):
            prev_anchor = ordered[left]
            break
    for right in range(index + 1, len(ordered)):
        if is_anchor(ordered[right]):
            next_anchor = ordered[right]
            break
    return {
        "previous_anchor_id": str(prev_anchor.get("ID", "") or "") if prev_anchor else "",
        "previous_anchor_tm": str(prev_anchor.get("TM", "") or "") if prev_anchor else "",
        "previous_anchor_iso": tm_to_iso(str(prev_anchor.get("TM", "") or ""), lookup) if prev_anchor else "",
        "next_anchor_id": str(next_anchor.get("ID", "") or "") if next_anchor else "",
        "next_anchor_tm": str(next_anchor.get("TM", "") or "") if next_anchor else "",
        "next_anchor_iso": tm_to_iso(str(next_anchor.get("TM", "") or ""), lookup) if next_anchor else "",
    }


def validate_interval_evidence(
    evidence: Dict[str, Any],
    context: Dict[str, Any],
    lookup: Dict[str, str],
) -> Tuple[bool, str, Dict[str, Any]]:
    if evidence.get("relation") != "contained_in_source_interval":
        return False, "relation_not_interval_transferable", {}

    start_tm = normalize_experiment1_tm(str(evidence.get("start_tm", "") or ""))
    end_tm = normalize_experiment1_tm(str(evidence.get("end_tm", "") or ""))
    start_iso = tm_to_iso(start_tm, lookup) if start_tm else "-infinity"
    end_iso = tm_to_iso(end_tm, lookup) if end_tm else "+infinity"

    normalized = {
        "start_tm": start_tm,
        "end_tm": end_tm,
        "start_iso": start_iso,
        "end_iso": end_iso,
        **context,
    }
    if start_tm and not start_iso:
        return False, "start_tm_not_in_iso_lookup", normalized
    if end_tm and not end_iso:
        return False, "end_tm_not_in_iso_lookup", normalized
    if iso_greater(start_iso, end_iso):
        return False, "interval_start_after_end", normalized

    prev_iso = str(context.get("previous_anchor_iso", "") or "")
    next_iso = str(context.get("next_anchor_iso", "") or "")
    if prev_iso and iso_less(end_iso, prev_iso):
        return False, "interval_ends_before_previous_local_anchor", normalized
    if next_iso and iso_greater(start_iso, next_iso):
        return False, "interval_starts_after_next_local_anchor", normalized
    return True, "accepted", normalized


def process_file(path: Path, lookup: Dict[str, str]) -> Dict[str, Any]:
    data = load_json(path)
    blocks = data.get("TMB", []) if isinstance(data, dict) else []
    if not isinstance(blocks, list):
        return {"file": path.name, "total": 0, "accepted": 0, "rejected": 0}

    doc_id = path.stem.replace("_timeblock", "").replace("_timeblocks_updated", "")
    ordered = ordered_blocks(doc_id, blocks)
    stats = {
        "file": path.name,
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "rejected_by_reason": {},
        "samples": [],
    }

    for index, block in enumerate(ordered):
        evidence = block.get("crossdoc_interval_evidence")
        if not isinstance(evidence, dict):
            block.pop("crossdoc_temporal_constraints", None)
            continue
        stats["total"] += 1
        context = nearest_anchor_context(ordered, index, lookup)
        accepted, reason, normalized = validate_interval_evidence(evidence, context, lookup)
        constraint = {
            "method": "generic_crossdoc_temporal_constraint_graph",
            "status": "accepted" if accepted else "rejected",
            "reason": reason,
            **normalized,
        }
        block["crossdoc_temporal_constraints"] = constraint
        if accepted:
            evidence.pop("disabled", None)
            evidence.pop("disabled_reason", None)
            evidence["temporal_graph_status"] = "accepted"
            evidence["temporal_graph"] = constraint
            stats["accepted"] += 1
        else:
            evidence["disabled"] = True
            evidence["disabled_reason"] = reason
            evidence["temporal_graph_status"] = "rejected"
            evidence["temporal_graph"] = constraint
            block.pop("iso", None)
            block.pop("iso_range", None)
            stats["rejected"] += 1
            stats["rejected_by_reason"][reason] = stats["rejected_by_reason"].get(reason, 0) + 1
        if len(stats["samples"]) < 20:
            stats["samples"].append({
                "id": block.get("ID", ""),
                "tm": block.get("TM", ""),
                "status": constraint["status"],
                "reason": reason,
                "start_tm": constraint.get("start_tm", ""),
                "end_tm": constraint.get("end_tm", ""),
                "previous_anchor_tm": context.get("previous_anchor_tm", ""),
                "next_anchor_tm": context.get("next_anchor_tm", ""),
            })

    save_json(path, data)
    return stats


def main() -> None:
    global RUN_ROOT, TIMEBLOCK_DIR, SEQUENCE_DIR

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    TIMEBLOCK_DIR = timeblock_step_dir(RUN_ROOT, 10)
    SEQUENCE_DIR = sequence_step_dir(RUN_ROOT, 8)
    setup_step_logging(RUN_ROOT, "step_10d_crossdoc_temporal_graph")

    reporter = StepReporter("Step10d")
    files = sorted(TIMEBLOCK_DIR.glob("*.json"))
    reporter.start(input_dir=TIMEBLOCK_DIR, output_dir=TIMEBLOCK_DIR)
    reporter.info(f"待处理 timeblock 文件数={len(files)}")
    lookup = load_iso_lookup()
    report = {
        "schema": "AIH_experiment1_crossdoc_temporal_graph.v1",
        "run_root": str(RUN_ROOT),
        "iso_lookup_size": len(lookup),
        "files": [],
        "total_interval_evidence": 0,
        "accepted": 0,
        "rejected": 0,
        "rejected_by_reason": {},
    }
    for path in files:
        stats = process_file(path, lookup)
        report["files"].append(stats)
        report["total_interval_evidence"] += stats.get("total", 0)
        report["accepted"] += stats.get("accepted", 0)
        report["rejected"] += stats.get("rejected", 0)
        for reason, count in stats.get("rejected_by_reason", {}).items():
            report["rejected_by_reason"][reason] = report["rejected_by_reason"].get(reason, 0) + count
    save_json(RUN_ROOT / "timeblock" / "step10d_temporal_graph_report.json", report)
    reporter.item_ok(
        "temporal_graph",
        detail=f"intervals={report['total_interval_evidence']} accepted={report['accepted']} rejected={report['rejected']}",
    )
    reporter.finish()


if __name__ == "__main__":
    main()
