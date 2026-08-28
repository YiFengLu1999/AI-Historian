import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_historian.pipeline.logging import StepReporter, setup_step_logging
from ai_historian.pipeline.paths import resolve_run_root, timeblock_step_dir
from ai_historian.pipeline.time_canonicalizer import normalize_experiment1_tm

# =========================
# 路径配置
# =========================
RUN_ROOT: Path
TIMEBLOCK_INPUT_DIR: Path
TIMEBLOCK_OUTPUT_DIR: Path
SENTENCE_ROOT_DIR: Path

# 匹配 step1output / step5output 这种目录名
STEP_DIR_PATTERN = re.compile(r"step(\d+)output$", re.IGNORECASE)

# 兼容新的 range：
# 94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1-94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
# 注意这里不能直接 split("-")，因为 uuid 本身就带 "-"
RANGE_PATTERN = re.compile(
    r"^(?P<start>.+?\.\d+\.\d+\.\d+)-(?P<end>.+?\.\d+\.\d+\.\d+)$"
)


# =========================
# 基础读写
# =========================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pick_tm(conv_info: Dict[str, Any]) -> str:
    """
    规则：
    1) 若存在且非空 time_information_converted -> TM = 它
    2) 否则 TM = time_information_original（可能为空）
    """
    if not isinstance(conv_info, dict):
        return ""

    converted = conv_info.get("time_information_converted", None)
    if isinstance(converted, str) and converted.strip() != "":
        return normalize_experiment1_tm(converted)

    original = conv_info.get("time_information_original", "")
    if isinstance(original, str):
        return normalize_experiment1_tm(original)

    return ""


# =========================
# 新版 number / range 解析
# =========================
def parse_number(number: str) -> Optional[Dict[str, Any]]:
    """
    新版 number:
    书籍uuid.篇章id.段落id.段落内句子id

    例如:
    94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1

    不能按 '-' 解析，只能从右边按 '.' 拆 3 次。
    """
    if not isinstance(number, str):
        return None

    s = number.strip()
    if not s:
        return None

    parts = s.rsplit(".", 3)
    if len(parts) != 4:
        return None

    book_uuid, chapter_id, paragraph_id, sentence_id = parts

    try:
        return {
            "book_uuid": book_uuid,
            "chapter_id": int(chapter_id),
            "paragraph_id": int(paragraph_id),
            "sentence_id": int(sentence_id),
        }
    except ValueError:
        return None


def parse_range(range_text: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    解析新版 range:
    start-end

    例如:
    uuid.7.50.1-uuid.7.50.3

    注意 uuid 自己包含 '-'，所以不能 range_text.split('-')
    必须把左右两端都识别成完整的 number。
    """
    if not isinstance(range_text, str):
        return None

    s = range_text.strip()
    if not s:
        return None

    m = RANGE_PATTERN.match(s)
    if not m:
        return None

    start_str = m.group("start")
    end_str = m.group("end")

    start = parse_number(start_str)
    end = parse_number(end_str)

    if start is None or end is None:
        return None

    return start, end


def order_key(parsed_number: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    按“篇章id -> 段落id -> 句子id”排序
    这是你要求不能改的核心原则。
    """
    return (
        parsed_number["chapter_id"],
        parsed_number["paragraph_id"],
        parsed_number["sentence_id"],
    )


# =========================
# sentence 文件索引
# =========================
def extract_step_num(path: Path) -> int:
    """
    从父目录名里提取 step 编号。
    比如 sentence/step5output/xxx_sentence.json -> 5
    提取不到就返回 -1
    """
    m = STEP_DIR_PATTERN.match(path.name)
    if m:
        return int(m.group(1))
    return -1


def build_sentence_file_index(sentence_root: Path) -> Dict[str, Path]:
    """
    递归扫描 sentence 目录下所有 *_sentence.json，
    若同名文件出现在多个 stepXoutput 中，优先选择 step 编号更大的那个。
    """
    best: Dict[str, Tuple[int, Path]] = {}

    if not sentence_root.exists():
        return {}

    for fp in sentence_root.rglob("*_sentence.json"):
        if not fp.is_file():
            continue

        step_num = extract_step_num(fp.parent)
        name = fp.name

        if name not in best or step_num > best[name][0]:
            best[name] = (step_num, fp)

    return {name: path for name, (_, path) in best.items()}


# =========================
# 递归扫描 sentence JSON，构建句子级 TM 索引
# =========================
def walk_all_dicts(obj: Any):
    """
    递归遍历任意 JSON 结构，产出所有 dict。
    这样不用假设 sentence JSON 的内部结构必须固定。
    """
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_all_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_all_dicts(item)


def build_sentence_tm_entries(sentence_data: Any) -> List[Dict[str, Any]]:
    """
    从 sentence JSON 中，递归找到所有带 number 的记录，
    建立句子级索引，供 timeblock 根据 isorange / timeblockrange 回查。

    每条 entry 结构：
    {
        "number": "...",
        "parsed": {...},
        "TM": "..."
    }
    """
    entries: List[Dict[str, Any]] = []
    seen = set()

    for d in walk_all_dicts(sentence_data):
        number = d.get("number")
        if not isinstance(number, str):
            continue

        parsed = parse_number(number)
        if parsed is None:
            continue

        conv_info = d.get("Conversion information", {})
        tm = pick_tm(conv_info)

        # sentence 里如果本身已经有 TM，也拿来兜底
        if not tm:
            raw_tm = d.get("TM", "")
            if isinstance(raw_tm, str):
                tm = raw_tm.strip()

        # 同一个 number 只保留一次；若后面遇到更丰富 TM，可覆盖空值
        if number not in seen:
            entries.append({
                "number": number,
                "parsed": parsed,
                "TM": tm,
            })
            seen.add(number)
        else:
            # 已存在但 TM 为空，后面如果有非空 TM，则补上
            for item in entries:
                if item["number"] == number and not item["TM"] and tm:
                    item["TM"] = tm
                    break

    entries.sort(key=lambda x: (
        x["parsed"]["book_uuid"],
        x["parsed"]["chapter_id"],
        x["parsed"]["paragraph_id"],
        x["parsed"]["sentence_id"],
    ))
    return entries


# =========================
# 基于 range，从 sentence 中回填 TM
# =========================
def infer_tm_from_range(
    range_text: str,
    sentence_entries: List[Dict[str, Any]]
) -> Tuple[str, bool]:
    """
    根据 range 去 sentence entries 中找覆盖到的句子，并汇总 TM。
    返回:
        (tm_value, parse_ok)

    parse_ok=False 表示 range 连解析都失败了
    """
    parsed_range = parse_range(range_text)
    if parsed_range is None:
        return "", False

    start, end = parsed_range

    # 同一个 timeblock 原则上应落在同一本书里
    if start["book_uuid"] != end["book_uuid"]:
        return "", True

    start_key = order_key(start)
    end_key = order_key(end)
    if start_key > end_key:
        start, end = end, start
        start_key, end_key = end_key, start_key

    collected: List[str] = []
    seen_tm = set()

    for entry in sentence_entries:
        p = entry["parsed"]

        if p["book_uuid"] != start["book_uuid"]:
            continue

        key = order_key(p)
        if start_key <= key <= end_key:
            tm = entry["TM"].strip()
            if tm and tm not in seen_tm:
                collected.append(tm)
                seen_tm.add(tm)

    # 多个不同 TM 用全角分号拼起来，避免信息丢失
    return "；".join(collected), True


def infer_tm_from_tmb_ranges(
    tmb_obj: Dict[str, Any],
    sentence_entries: List[Dict[str, Any]]
) -> Tuple[str, int]:
    """
    依次尝试：
    1) isorange
    2) timeblockrange

    返回:
        (tm_value, parse_fail_count)
    """
    parse_fail_count = 0

    for key in ("isorange", "timeblockrange"):
        value = tmb_obj.get(key)
        if not (isinstance(value, str) and value.strip()):
            continue

        tm, parse_ok = infer_tm_from_range(value, sentence_entries)
        if not parse_ok:
            parse_fail_count += 1
            continue

        if tm:
            return tm, parse_fail_count

    return "", parse_fail_count


# =========================
# 单文件处理
# =========================
def process_one_file(
    timeblock_path: Path,
    sentence_path: Optional[Path],
    sentence_cache: Dict[Path, List[Dict[str, Any]]]
) -> Tuple[Path, Dict[str, int]]:
    """
    处理单个 timeblock 文件：
    - 优先按 TMB 自身 Conversion information 取 TM
    - 若取不到，则尝试根据 isorange / timeblockrange 去对应 sentence 文件补 TM
    - 若还取不到，则保留原 TM（若原本有的话）
    - 最终输出到 timeblock/step10output/原文件名
    """
    data = load_json(timeblock_path)

    if not isinstance(data, dict) or "TMB" not in data or not isinstance(data["TMB"], list):
        raise ValueError(
            f"文件结构不符合预期：{timeblock_path}（需要 dict 顶层且包含 list 类型的 'TMB'）"
        )

    sentence_entries: List[Dict[str, Any]] = []
    if sentence_path is not None:
        if sentence_path not in sentence_cache:
            sentence_data = load_json(sentence_path)
            sentence_cache[sentence_path] = build_sentence_tm_entries(sentence_data)
        sentence_entries = sentence_cache[sentence_path]

    total = 0
    changed = 0
    tm_empty_after = 0
    from_self = 0
    from_sentence = 0
    kept_existing = 0
    parse_fail_range = 0

    for obj in data["TMB"]:
        if not isinstance(obj, dict):
            continue

        total += 1
        before = obj.get("TM", None)

        # 1) 先按你原来的逻辑，从 TMB 自己的 Conversion information 里取
        conv_info = obj.get("Conversion information", {})
        tm_value = pick_tm(conv_info)

        if tm_value:
            from_self += 1
        else:
            # 2) 若自身拿不到，再尝试从对应 sentence 的 range 回填
            if sentence_entries:
                inferred_tm, fail_cnt = infer_tm_from_tmb_ranges(obj, sentence_entries)
                parse_fail_range += fail_cnt
                if inferred_tm:
                    tm_value = inferred_tm
                    from_sentence += 1

            # 3) 如果还拿不到，保留原 TM（若原本有）
            if not tm_value:
                if isinstance(before, str) and before.strip():
                    tm_value = before.strip()
                    kept_existing += 1
                else:
                    tm_value = ""

        obj["TM"] = tm_value

        if before != tm_value:
            changed += 1
        if not tm_value.strip():
            tm_empty_after += 1

    out_path = TIMEBLOCK_OUTPUT_DIR / timeblock_path.name
    save_json(data, out_path)

    stats = {
        "total": total,
        "changed": changed,
        "tm_empty_after": tm_empty_after,
        "from_self": from_self,
        "from_sentence": from_sentence,
        "kept_existing": kept_existing,
        "parse_fail_range": parse_fail_range,
        "sentence_found": 1 if sentence_path is not None else 0,
    }
    return out_path, stats


# =========================
# 主程序
# =========================
def main():
    global RUN_ROOT, TIMEBLOCK_INPUT_DIR, TIMEBLOCK_OUTPUT_DIR, SENTENCE_ROOT_DIR

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    TIMEBLOCK_INPUT_DIR = timeblock_step_dir(RUN_ROOT, 9)
    TIMEBLOCK_OUTPUT_DIR = timeblock_step_dir(RUN_ROOT, 10)
    SENTENCE_ROOT_DIR = RUN_ROOT / "sentence"
    TIMEBLOCK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_step_logging(RUN_ROOT, "step_10_tm_generation")

    if not TIMEBLOCK_INPUT_DIR.exists():
        raise FileNotFoundError(f"找不到输入目录：{TIMEBLOCK_INPUT_DIR.resolve()}")

    json_files = sorted(TIMEBLOCK_INPUT_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"在 {TIMEBLOCK_INPUT_DIR.resolve()} 下没找到 .json 文件")

    sentence_index = build_sentence_file_index(SENTENCE_ROOT_DIR)
    sentence_cache: Dict[Path, List[Dict[str, Any]]] = {}

    total_files = 0
    total_records = 0
    total_changed = 0
    total_empty = 0
    total_from_self = 0
    total_from_sentence = 0
    total_kept_existing = 0
    total_parse_fail_range = 0
    total_missing_sentence = 0
    failed_files = 0

    reporter = StepReporter("Step10", total=len(json_files))
    reporter.start(input_dir=TIMEBLOCK_INPUT_DIR, output_dir=TIMEBLOCK_OUTPUT_DIR)

    for tb_fp in json_files:
        try:
            # 对应规则：
            # 7_xxx_timeblock.json  ->  7_xxx_sentence.json
            sentence_name = tb_fp.name.replace("_timeblock.json", "_sentence.json")
            sentence_path = sentence_index.get(sentence_name)

            if sentence_path is None:
                total_missing_sentence += 1

            out_path, stats = process_one_file(
                timeblock_path=tb_fp,
                sentence_path=sentence_path,
                sentence_cache=sentence_cache,
            )

            total_files += 1
            total_records += stats["total"]
            total_changed += stats["changed"]
            total_empty += stats["tm_empty_after"]
            total_from_self += stats["from_self"]
            total_from_sentence += stats["from_sentence"]
            total_kept_existing += stats["kept_existing"]
            total_parse_fail_range += stats["parse_fail_range"]

            sentence_note = "matched_sentence=YES" if sentence_path else "matched_sentence=NO"
            reporter.item_ok(
                tb_fp.name,
                detail=(
                    f"TMB={stats['total']} changed={stats['changed']} "
                    f"empty={stats['tm_empty_after']} {sentence_note}"
                ),
            )

        except Exception as e:
            failed_files += 1
            reporter.item_fail(tb_fp.name, e)

    reporter.finish(
        output_dir=TIMEBLOCK_OUTPUT_DIR,
        extra=(
            f"TMB={total_records} changed={total_changed} empty={total_empty} "
            f"缺sentence={total_missing_sentence}"
        ),
    )


if __name__ == "__main__":
    main()
