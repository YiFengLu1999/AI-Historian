# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_historian.pipeline.logging import StepReporter, setup_step_logging
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir, timeblock_step_dir


# =========================
# 路径配置
# =========================
# =========================
# 工具函数
# =========================
def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """读取单个 step5 JSON 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"文件顶层必须是 list: {path}")

    return data


def save_json(data: Dict[str, Any], path: Path) -> None:
    """保存 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_number(number: str) -> Dict[str, Any]:
    """
    解析新的 number 格式:
    书籍uuid.篇章id.段落id.段落内句子id

    例如:
    94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.8.1.1
    """
    if not isinstance(number, str):
        raise TypeError(f"number 必须是字符串，实际为: {type(number)}")

    parts = number.rsplit(".", 3)
    if len(parts) != 4 or any(part == "" for part in parts):
        raise ValueError(
            f"number 格式错误，期望 4 段: 书籍uuid.篇章id.段落id.句子id，实际得到: {number}"
        )

    book_uuid, chapter_id, paragraph_id, sentence_id = parts

    return {
        "book_uuid": book_uuid,
        "chapter_id": chapter_id,
        "paragraph_id": paragraph_id,
        "sentence_id": sentence_id,
    }


def extract_file_identity(records: List[Dict[str, Any]], file_path: Optional[Path] = None) -> Tuple[str, str]:
    """
    从 records 中提取 (book_uuid, chapter_id)
    优先从第一条记录的 number 解析。
    如果 records 为空，则尝试从文件名解析。
    """
    if records:
        first_number = records[0].get("number", "")
        parsed = parse_number(first_number)
        return parsed["book_uuid"], parsed["chapter_id"]

    # records 为空时，尝试从文件名兜底
    # 文件名可能长这样:
    # 8_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_sentence.json
    if file_path is not None:
        stem = file_path.stem  # 8_..._sentence
        parts = stem.split("_")
        if len(parts) >= 3:
            chapter_id = parts[0]
            book_uuid = parts[1]
            return book_uuid, chapter_id

    raise ValueError("无法提取 book_uuid 和 chapter_id。")


def validate_records(records: List[Dict[str, Any]], file_path: Path) -> None:
    """
    做基本校验：
    - number 可解析
    - 同一个文件中的 book_uuid / chapter_id 应一致
    """
    if not records:
        return

    first = parse_number(records[0]["number"])
    expected_book_uuid = first["book_uuid"]
    expected_chapter_id = first["chapter_id"]

    for idx, rec in enumerate(records):
        if "number" not in rec:
            raise KeyError(f"{file_path.name} 第 {idx} 条记录缺少 number 字段")

        parsed = parse_number(rec["number"])

        if parsed["book_uuid"] != expected_book_uuid:
            raise ValueError(
                f"{file_path.name} 第 {idx} 条记录的 book_uuid 不一致: "
                f"{parsed['book_uuid']} != {expected_book_uuid}"
            )

        if parsed["chapter_id"] != expected_chapter_id:
            raise ValueError(
                f"{file_path.name} 第 {idx} 条记录的 chapter_id 不一致: "
                f"{parsed['chapter_id']} != {expected_chapter_id}"
            )


def get_bool(record: Dict[str, Any], path: List[str], default: bool = False) -> bool:
    """安全取布尔值。"""
    cur = record
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return bool(cur)


def get_str(record: Dict[str, Any], path: List[str], default: str = "") -> str:
    """安全取字符串。"""
    cur = record
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if isinstance(cur, str) else default


# =========================
# 核心逻辑
# =========================
def extract_timeblocks(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    根据 step5 输出生成 timeblocks。

    当前规则：
    1. 仅处理 sink.Is_it_sinking == False 的句子
    2. Interlude == True 的可用句子，按“连续索引片段”生成 interlude timeblock
    3. 非 Interlude 的可用句子中：
       - 有 Original_time_information.exist == True 的句子作为锚点
       - 如果第一条非 interlude 可用句子不是锚点，则它自己作为一个起始块
       - 一个块从当前起点延续到“下一个锚点之前的最后一条非 interlude 可用句子”
    4. Conversion information.reasoning 始终为空字符串
    5. 其余新增字段先按空值占位
    """
    n = len(records)

    def is_usable(i: int) -> bool:
        # 只保留 sink.Is_it_sinking == False 的句子
        return records[i].get("sink", {}).get("Is_it_sinking") is False

    def is_interlude(i: int) -> bool:
        return bool(records[i].get("Interlude", False))

    def oti_exist(i: int) -> bool:
        return records[i].get("Original_time_information", {}).get("exist") is True

    def oti_text(i: int) -> str:
        return records[i].get("Original_time_information", {}).get("OTI", "")

    usable = [i for i in range(n) if is_usable(i)]
    non_interlude = [i for i in usable if not is_interlude(i)]
    interlude_only = [i for i in usable if is_interlude(i)]

    # -------------------------
    # 1) 处理 Interlude 连续片段
    # -------------------------
    interlude_runs: List[Tuple[int, int]] = []
    j = 0
    while j < len(interlude_only):
        start = interlude_only[j]
        end = start

        while j + 1 < len(interlude_only) and interlude_only[j + 1] == end + 1:
            j += 1
            end = interlude_only[j]

        interlude_runs.append((start, end))
        j += 1

    interlude_blocks = []
    for start_idx, end_idx in interlude_runs:
        start_id = records[start_idx]["number"]
        end_id = records[end_idx]["number"]

        interlude_blocks.append({
            "ID": start_id,
            "timeblock_range": f"{start_id}-{end_id}",
            "Interlude": True,
            "Conversion information": {
                "time_information_original": "",
                "is_conversion_required": False,
                "basis_of_conversion": "",
                "reasoning": ""
            },
            "Granularity": "0",
            "TM": "",
            "iso": "",
            "iso_range": "",
            "TB_Update": "",
            "summary": "",
            "_order": start_idx
        })

    # -------------------------
    # 2) 处理普通 timeblocks
    # -------------------------
    anchors = [i for i in non_interlude if oti_exist(i)]
    blocks = []

    start_candidates = []
    if non_interlude:
        # 第一条非 interlude 可用句若不是 anchor，也要从它开始起一个块
        if non_interlude[0] not in anchors:
            start_candidates.append(non_interlude[0])
        start_candidates.extend(anchors)

    pos_in_non_interlude = {idx: pos for pos, idx in enumerate(non_interlude)}

    for s in start_candidates:
        # 找下一个 anchor
        next_anchors = [a for a in anchors if a > s]

        if next_anchors:
            nxt = min(next_anchors)
            pos = pos_in_non_interlude[nxt]
            # 当前块终点 = 下一个 anchor 的前一条 non_interlude 可用句
            end_idx = non_interlude[pos - 1] if pos - 1 >= pos_in_non_interlude[s] else s
        else:
            # 没有下一个 anchor，则延续到最后一条 non_interlude 可用句
            end_idx = non_interlude[-1]

        if end_idx < s:
            end_idx = s

        start_id = records[s]["number"]
        end_id = records[end_idx]["number"]
        time_info = oti_text(s) if oti_exist(s) else ""

        basis = ""
        if not oti_exist(s):
            basis = "首段无时间锚点，按规则从第一条可用记录起算"

        blocks.append({
            "ID": start_id,
            "timeblock_range": f"{start_id}-{end_id}",
            "Interlude": False,
            "Conversion information": {
                "time_information_original": time_info,
                "is_conversion_required": False,
                "basis_of_conversion": basis,
                "reasoning": ""
            },
            "Granularity": "0",
            "TM": "",
            "iso": "",
            "iso_range": "",
            "TB_Update": "",
            "summary": "",
            "_order": s
        })

    # -------------------------
    # 3) 合并排序
    # -------------------------
    all_blocks = blocks + interlude_blocks
    all_blocks.sort(key=lambda x: x["_order"])

    for block in all_blocks:
        block.pop("_order", None)

    return {"TMB": all_blocks}


# =========================
# 单文件处理
# =========================
def process_one_file(input_path: Path, output_dir: Path) -> Path:
    """
    处理单个 step5 文件，输出为:
    篇章id_书籍uuid_timeblock.json
    """
    records = load_json_records(input_path)

    if not records:
        # 空文件也给一个空输出，文件名尽量从原文件名推断
        book_uuid, chapter_id = extract_file_identity(records, input_path)
        output_name = f"{chapter_id}_{book_uuid}_timeblock.json"
        output_path = output_dir / output_name
        save_json({"TMB": []}, output_path)
        return output_path

    validate_records(records, input_path)
    book_uuid, chapter_id = extract_file_identity(records, input_path)

    output = extract_timeblocks(records)

    output_name = f"{chapter_id}_{book_uuid}_timeblock.json"
    output_path = output_dir / output_name
    save_json(output, output_path)

    return output_path


# =========================
# 批量处理
# =========================
def batch_process_step5_to_step6(
    step5_dir: Path | None = None,
    step6_dir: Path | None = None,
) -> None:
    """
    批量处理 sentence/step5output 下的所有 *_sentence.json，
    输出到 timeblock/step6output/
    """
    if step5_dir is None or step6_dir is None:
        run_root = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
        step5_dir = step5_dir or sentence_step_dir(run_root, 5)
        step6_dir = step6_dir or timeblock_step_dir(run_root, 6)

    if not step5_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {step5_dir.resolve()}")

    step6_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(step5_dir.glob("*.json"))
    if not json_files:
        reporter = StepReporter("Step6", total=0)
        reporter.start(input_dir=step5_dir, output_dir=step6_dir)
        reporter.info(f"没有可处理文件: {step5_dir.resolve()}")
        reporter.finish(output_dir=step6_dir)
        return

    total = len(json_files)
    success = 0
    failed = 0
    reporter = StepReporter("Step6", total=total)
    reporter.start(input_dir=step5_dir, output_dir=step6_dir)

    for idx, input_path in enumerate(json_files, 1):
        output_name = input_path.name.replace("_sentence.json", "_timeblock.json")
        output_path = step6_dir / output_name
        if output_path.exists():
            reporter.item_skip(input_path.name, detail=output_path.name)
            continue
        try:
            output_path = process_one_file(input_path, step6_dir)
            success += 1
            reporter.item_ok(input_path.name, detail=output_path.name)
        except Exception as e:
            failed += 1
            reporter.item_fail(input_path.name, e)

    reporter.finish(output_dir=step6_dir, extra=f"成功={success} 失败={failed}")


def main() -> None:
    run_root = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    setup_step_logging(run_root, "step_06_timeblock_generation")
    batch_process_step5_to_step6(
        sentence_step_dir(run_root, 5),
        timeblock_step_dir(run_root, 6),
    )


if __name__ == "__main__":
    main()
