#!/usr/bin/env python3
"""Repair duplicate Step10 timeblock IDs before Step11.

Step11 computes ISO ranges by the `ID` field in each TMB item. Some historical
outputs can contain duplicate IDs after upstream segmentation. This utility
keeps the first block for each ID and merges useful fields from later duplicates,
then writes a backup of the original step10output directory.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

MERGE_EVENT_KEYS = ("crossdoc_context_evidence", "crossdoc_interval_evidence")
MERGE_IF_EMPTY_KEYS = (
    "TM",
    "iso",
    "iso_range",
    "timeblock_range",
    "summary",
    "sentence",
    "content",
    "text",
)


def evidence_rank(ev: dict[str, Any]) -> tuple[int, float, int]:
    return (
        0 if ev.get("recall_relation_relaxed") else 1,
        float(ev.get("confidence") or 0),
        -len(ev.get("quality_warnings") or []),
    )


def merge_duplicate_block(kept: dict[str, Any], duplicate: dict[str, Any]) -> None:
    for key in MERGE_EVENT_KEYS:
        new_ev = duplicate.get(key)
        old_ev = kept.get(key)
        if isinstance(new_ev, dict) and not isinstance(old_ev, dict):
            kept[key] = new_ev
        elif isinstance(new_ev, dict) and isinstance(old_ev, dict):
            if evidence_rank(new_ev) > evidence_rank(old_ev):
                kept[key] = new_ev

    for key in MERGE_IF_EMPTY_KEYS:
        if not kept.get(key) and duplicate.get(key):
            kept[key] = duplicate[key]


def duplicate_ids(tmb: list[Any]) -> list[str]:
    ids = [
        str(block.get("ID", "")).strip()
        for block in tmb
        if isinstance(block, dict) and str(block.get("ID", "")).strip()
    ]
    return [key for key, count in collections.Counter(ids).items() if count > 1]


def repair_file(path: Path, *, dry_run: bool) -> tuple[bool, list[str], int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tmb = data.get("TMB", [])
    if not isinstance(tmb, list):
        return False, [], 0, 0

    dups = duplicate_ids(tmb)
    if not dups:
        return False, [], len(tmb), len(tmb)

    seen: dict[str, dict[str, Any]] = {}
    repaired: list[Any] = []

    for block in tmb:
        if not isinstance(block, dict):
            repaired.append(block)
            continue

        block_id = str(block.get("ID", "")).strip()
        if not block_id or block_id not in seen:
            if block_id:
                seen[block_id] = block
            repaired.append(block)
            continue

        merge_duplicate_block(seen[block_id], block)

    if not dry_run:
        data["TMB"] = repaired
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return True, dups, len(tmb), len(repaired)


def iter_step10_dirs(target: Path) -> list[Path]:
    if (target / "timeblock" / "step10output").is_dir():
        return [target / "timeblock" / "step10output"]
    if target.name == "step10output" and target.is_dir():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.glob("result_*/timeblock/step10output") if p.is_dir())
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help="A run root such as result/result_三国志, a step10output dir, or result/",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report duplicates without writing files.")
    args = parser.parse_args()

    target = Path(args.target)
    step10_dirs = iter_step10_dirs(target)
    if not step10_dirs:
        raise FileNotFoundError(f"No step10output directory found under: {target}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    total_changed = 0

    for step10_dir in step10_dirs:
        changed: list[tuple[str, list[str], int, int]] = []

        for path in sorted(step10_dir.glob("*.json")):
            did_change, dups, before, after = repair_file(path, dry_run=True)
            if did_change:
                changed.append((path.name, dups, before, after))

        if not changed:
            print(f"OK no duplicate IDs: {step10_dir}")
            continue

        if not args.dry_run:
            backup = step10_dir.with_name(f"step10output.before_ID_dedupe_{stamp}")
            if not backup.exists():
                shutil.copytree(step10_dir, backup)
            print(f"backup = {backup}")

            for path_name, _, _, _ in changed:
                repair_file(step10_dir / path_name, dry_run=False)

        total_changed += len(changed)
        print(f"DEDUPED {step10_dir}")
        for path_name, dups, before, after in changed:
            print(f"  {path_name}: {before} -> {after}; duplicate IDs={dups[:10]}")

    print(f"changed_files = {total_changed}")


if __name__ == "__main__":
    main()
