import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from ai_historian.pipeline.logging import StepReporter, setup_step_logging
from ai_historian.pipeline.paths import (
    resolve_run_root,
    sentence_step_dir,
    sequence_step_dir,
    timeblock_step_dir,
)

# =========================
# 可配置变量
# =========================
# 如需只处理某一个文件，可写成：
# TARGET_DOC_KEYS = {"7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b"}
# 默认 None 表示处理 step11output 里的全部 timeblock 文件
TARGET_DOC_KEYS = None

RUN_ROOT: Path
TIMEBLOCK_DIR: Path
SENTENCE_DIR: Path
SEQUENCE_DIR: Path
OUTPUT_DIR: Path

BASE = 10 ** 6


# =========================
# 文件名解析
# 规则：篇章id_uuid_文件属性.json
# 例子：7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_timeblock.json
# =========================
FILENAME_RE = re.compile(
    r"^(?P<chapter_id>\d+)_(?P<book_uuid>.+)_(?P<kind>timeblock|sentence|sequence)\.json$"
)


def parse_artifact_filename(filename):
    """
    解析文件名，返回:
    {
        "chapter_id": "7",
        "book_uuid": "94d18bb5-29cc-51b5-b0c3-70afe2b6f85b",
        "kind": "timeblock",
        "doc_key": "7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b"
    }
    """
    m = FILENAME_RE.match(filename)
    if not m:
        return None

    chapter_id = m.group("chapter_id")
    book_uuid = m.group("book_uuid")
    kind = m.group("kind")
    doc_key = f"{chapter_id}_{book_uuid}"

    return {
        "chapter_id": chapter_id,
        "book_uuid": book_uuid,
        "kind": kind,
        "doc_key": doc_key,
    }


def index_json_dir(directory, expected_kind=None):
    """
    扫描目录下所有 json，按 doc_key 建索引。
    """
    directory = Path(directory)
    mapping = {}

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".json":
            continue

        info = parse_artifact_filename(path.name)
        if info is None:
            print(f"[跳过] 文件名不符合规则: {path.name}")
            continue

        if expected_kind is not None and info["kind"] != expected_kind:
            continue

        mapping[info["doc_key"]] = {
            "path": path,
            "filename": path.name,
            **info
        }

    return mapping


# =========================
# 基础 IO
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# ID / Range 解析
# 新格式：
#   uuid.chapter_id.paragraph_id.sentence_id
#   例：94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
#
# 旧格式仍兼容：
#   chapter_id.paragraph_id.sentence_id
#   例：53.1.5
# =========================
ID_PATTERN = r"(?:.+?\.)?\d+\.\d+\.\d+"


def parse_id(id_str):
    """
    新格式:
      '94d18...f85b.7.50.1' -> ('94d18...f85b', 7, 50, 1)

    旧格式:
      '53.1.5' -> ('', 53, 1, 5)
    """
    s = str(id_str).strip()
    parts = s.rsplit(".", 3)
    if len(parts) == 4:
        book_uuid, chapter_id, paragraph_id, sentence_id = parts
    elif len(parts) == 3:
        book_uuid = ""
        chapter_id, paragraph_id, sentence_id = parts
    else:
        raise ValueError(f"无法解析 ID: {id_str}")
    return book_uuid, int(chapter_id), int(paragraph_id), int(sentence_id)


def id_order_key(id_str):
    """
    返回一个可比较的顺序键：
      (chapter_id, paragraph_id, sentence_id)

    注意：
    区间比较逻辑仍然遵循旧原则，只比较篇章 / 段落 / 句子。
    uuid 只用于文档一致性校验，不参与排序。
    """
    _, chapter_id, paragraph_id, sentence_id = parse_id(id_str)
    return chapter_id, paragraph_id, sentence_id


def id_to_abs(id_str):
    """
    用于比较范围大小。只取 chapter/paragraph/sentence 数值部分。
    uuid 不参与“大小差值”计算。
    """
    _, chapter_id, paragraph_id, sentence_id = parse_id(id_str)
    return chapter_id * (BASE ** 2) + paragraph_id * BASE + sentence_id


def make_doc_key(book_uuid, chapter_id):
    """
    从 ID 里的 uuid + chapter_id 反推出文件索引键。
    文件名规则是: 篇章id_uuid_*.json
    所以 doc_key = '{chapter_id}_{book_uuid}'
    """
    if book_uuid:
        return f"{chapter_id}_{book_uuid}"
    return str(chapter_id)


def extract_doc_key_from_number(number):
    book_uuid, chapter_id, _, _ = parse_id(number)
    return make_doc_key(book_uuid, chapter_id)


def split_any_range(range_str):
    """
    同时支持两种 range:
      1) start-end
      2) starttoend

    返回:
      (left, right, sep)
    其中 sep 为 '-' 或 'to'
    """
    s = str(range_str).strip()
    if not s:
        raise ValueError("空 range 无法解析")

    # 先尝试 'to'
    m = re.match(
        rf"^\s*(?P<left>{ID_PATTERN})\s*to\s*(?P<right>{ID_PATTERN})\s*$",
        s
    )
    if m:
        return m.group("left").strip(), m.group("right").strip(), "to"

    # 再尝试 '-'
    m = re.match(
        rf"^\s*(?P<left>{ID_PATTERN})\s*-\s*(?P<right>{ID_PATTERN})\s*$",
        s
    )
    if m:
        return m.group("left").strip(), m.group("right").strip(), "-"

    raise ValueError(f"无法解析 range: {range_str}")


def split_range(range_str):
    left, right, _ = split_any_range(range_str)
    return left, right


def in_timeblock_range(number, timeblock_range):
    start, end = split_range(timeblock_range)
    uuids = {book_uuid for book_uuid, _, _, _ in (parse_id(start), parse_id(end), parse_id(number)) if book_uuid}
    if len(uuids) > 1:
        return False
    return id_order_key(start) <= id_order_key(number) <= id_order_key(end)


def range_size(timeblock_range):
    start, end = split_range(timeblock_range)
    return id_to_abs(end) - id_to_abs(start)


def split_iso_range(old_iso_range, pivot_iso):
    """
    把 old_iso_range 按 pivot_iso 切成左右两部分。
    会保留原来的分隔符风格（'-' 或 'to'）。

    例如:
      old_iso_range = 'a-b', pivot_iso='p'
      -> ('a-p', 'p-b')

      old_iso_range = 'ato b' 这种不支持
      old_iso_range = 'ato b'? 也不支持
      old_iso_range = 'a to b'
      -> ('a to p', 'p to b') 这里实际输出无空格: 'atop' / 'ptob'
    """
    try:
        start, end, sep = split_any_range(old_iso_range)
    except Exception:
        return old_iso_range, old_iso_range

    left_iso_range = f"{start}{sep}{pivot_iso}"
    right_iso_range = f"{pivot_iso}{sep}{end}"
    return left_iso_range, right_iso_range


# =========================
# 数据归一化
# =========================
def normalize_timeblock_root(data):
    """
    timeblock 预期是:
      {"TMB": [...]}
    """
    if isinstance(data, dict):
        if "TMB" in data and isinstance(data["TMB"], list):
            return data
    raise ValueError("timeblock JSON 格式不符合预期，缺少 TMB 列表")


def normalize_sentence_items(data):
    """
    sentence 预期是 list。
    如未来格式变成 dict，也尝试兼容常见字段。
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["sentence", "sentences", "Sentence", "data", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("sentence JSON 格式不符合预期，无法提取句子列表")


def normalize_sequence_list(data):
    """
    sequence 预期是 list。
    如未来格式变成 dict，也尝试兼容常见字段。
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["sequence", "Sequence", "SEQ", "data", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("sequence JSON 格式不符合预期，无法提取 sequence 列表")


# =========================
# 业务工具
# =========================
def find_timeblock_for_number(number, tmb_list):
    """
    找出 number 所属的 timeblock。
    若同时落在多个 timeblock 中，按“range 更小”优先。
    """
    candidates = []
    for idx, block in enumerate(tmb_list):
        tb_range = block.get("timeblock_range", "")
        if not tb_range:
            continue

        try:
            if in_timeblock_range(number, tb_range):
                start, _ = split_range(tb_range)
                candidates.append(
                    (
                        range_size(tb_range),  # 先比 range 大小
                        id_to_abs(start),      # 再比起点
                        idx
                    )
                )
        except Exception as e:
            print(f"[警告] timeblock_range 解析失败，已跳过: {tb_range} | 错误: {e}")

    if not candidates:
        return None, None

    candidates.sort()
    chosen_idx = candidates[0][2]
    return chosen_idx, tmb_list[chosen_idx]


def build_sentence_context(sentence_items):
    """
    为每个 doc 建立：
    - sentence_numbers: 原句子顺序
    - sink_map: number -> bool
    - sentence_index_map: number -> 顺序索引
    """
    sentence_numbers = []
    sink_map = {}

    for item in sentence_items:
        if not isinstance(item, dict):
            continue

        number = item.get("number")
        if number is None:
            continue

        sentence_numbers.append(number)

        sink_val = False
        sink_field = item.get("sink")
        if isinstance(sink_field, dict):
            sink_val = sink_field.get("Is_it_sinking", sink_field.get("is_it_sinking", False))
        elif isinstance(sink_field, bool):
            sink_val = sink_field

        sink_map[number] = bool(sink_val)

    sentence_index_map = {num: i for i, num in enumerate(sentence_numbers)}
    return sentence_numbers, sink_map, sentence_index_map


def build_sequence_index_map(sequence_list):
    """
    sequence 可能是:
    - ["uuid.7.1.1", "uuid.7.1.2", ...]
    - [{"number": "..."} , ...]
    """
    seq_numbers = []

    for item in sequence_list:
        if isinstance(item, str):
            seq_numbers.append(item)
        elif isinstance(item, dict):
            number = item.get("number") or item.get("id")
            if number:
                seq_numbers.append(number)

    return {num: i for i, num in enumerate(seq_numbers)}


def find_prev_non_sink_number_within_block(target_number, block_range, sentence_numbers, sink_map):
    """
    在当前 block 内，找 target_number 之前、离它最近且不是 sink 的句子。
    用于确定分割后“左半块”的结束位置。
    """
    start, end = split_range(block_range)
    start_key = id_order_key(start)
    target_key = id_order_key(target_number)
    end_key = id_order_key(end)

    candidates = [
        num for num in sentence_numbers
        if start_key <= id_order_key(num) < target_key <= end_key
    ]

    for num in reversed(candidates):
        if not sink_map.get(num, False):
            return num

    return None


def get_timeblock_order_key(block, seq_index_map, sentence_index_map, list_index):
    """
    给 timeblock 一个“顺序键”，优先按 sequence 排序。
    若该块没有任何句子出现在 sequence 中，则退回到句子原始顺序。
    """
    start, end = split_range(block["timeblock_range"])
    start_key = id_order_key(start)
    end_key = id_order_key(end)

    seq_hits = []
    for num, seq_idx in seq_index_map.items():
        num_key = id_order_key(num)
        if start_key <= num_key <= end_key:
            seq_hits.append(seq_idx)

    first_seq_idx = min(seq_hits) if seq_hits else 10 ** 15
    start_sentence_idx = sentence_index_map.get(start, 10 ** 15)

    return (
        first_seq_idx,
        start_sentence_idx,
        id_order_key(start),
        range_size(block["timeblock_range"]),
        list_index
    )


def copy_source_fields_for_interlude(block, source_block, source_id):
    """
    Interlude=True 时，直接覆盖 source block 的
    Granularity / TM / iso / iso_range
    """
    for field in ["Granularity", "TM", "iso", "iso_range"]:
        block[field] = source_block.get(field, "")
    block["TB_Update"] = source_id


def copy_source_fields_for_split_new_block(block, source_block, new_iso_range, source_id):
    """
    Interlude=False 且发生分割时：
    新块只继承 source block 的 Granularity / TM / iso
    iso_range 按分割规则写入右半块
    """
    for field in ["Granularity", "TM", "iso"]:
        block[field] = source_block.get(field, "")
    block["iso_range"] = new_iso_range
    block["TB_Update"] = source_id


# =========================
# 核心处理
# =========================
def process_one_doc(
    doc_key,
    source_timeblock_data,      # 原始 step11 快照，只读
    sentence_data_map,
    sequence_data_map,
    timeblock_file_map
):
    if doc_key not in source_timeblock_data:
        print(f"[跳过] doc_key={doc_key} 缺少 timeblock 文件")
        return {
            "status": "skipped",
            "doc_key": doc_key,
            "reason": "缺少 timeblock 文件",
        }

    if doc_key not in sentence_data_map:
        print(f"[跳过] doc_key={doc_key} 缺少对应的 sentence 文件")
        return {
            "status": "skipped",
            "doc_key": doc_key,
            "reason": "缺少 sentence 文件",
        }

    # target 使用可变副本；source 一律从原始快照读取
    tmb_root = deepcopy(source_timeblock_data[doc_key])
    tmb_root = normalize_timeblock_root(tmb_root)
    tmb_list = tmb_root.get("TMB", [])

    sentence_items = normalize_sentence_items(sentence_data_map[doc_key])
    sequence_list = normalize_sequence_list(sequence_data_map.get(doc_key, []))

    sentence_numbers, sink_map, sentence_index_map = build_sentence_context(sentence_items)
    seq_index_map = build_sequence_index_map(sequence_list)

    stats = {
        "skip_sink": 0,
        "skip_not_same": 0,
        "skip_no_same_id": 0,
        "skip_no_target_tb": 0,
        "skip_no_source_tb": 0,
        "skip_source_iso_empty": 0,
        "interlude_direct_cover": 0,
        "split": 0,
        "replace_without_left_block": 0,
    }

    for sent_item in sentence_items:
        if not isinstance(sent_item, dict):
            continue

        number = sent_item.get("number")
        if not number:
            continue

        # 规则1：当前句子若 sink=True，直接跳过
        if sink_map.get(number, False):
            stats["skip_sink"] += 1
            continue

        cross = sent_item.get("crossDocTransfer", {})
        if not isinstance(cross, dict):
            cross = {}

        is_same = cross.get("isSame", cross.get("IsSame", False))
        if not is_same:
            stats["skip_not_same"] += 1
            continue

        same_ids = cross.get("same_timeblock_id", [])
        if not same_ids:
            stats["skip_no_same_id"] += 1
            continue

        source_id = same_ids[0]
        source_doc_key = extract_doc_key_from_number(source_id)

        # 当前句子落在哪个 target timeblock 里（在当前 doc 的可变 timeblock 列表中找）
        target_idx, target_block = find_timeblock_for_number(number, tmb_list)
        if target_block is None:
            stats["skip_no_target_tb"] += 1
            print(f"[警告] doc_key={doc_key}, number={number} 找不到所属 target timeblock，已跳过")
            continue

        # source_id 落在哪个 source timeblock（从原始 step11 快照中找）
        source_root = source_timeblock_data.get(source_doc_key)
        if not source_root:
            stats["skip_no_source_tb"] += 1
            print(
                f"[警告] number={number}, source_id={source_id} 所属 doc_key={source_doc_key} "
                f"的 source timeblock 文件不存在，已跳过"
            )
            continue

        source_root = normalize_timeblock_root(source_root)
        source_tmb_list = source_root.get("TMB", [])
        source_idx, source_block = find_timeblock_for_number(source_id, source_tmb_list)
        if source_block is None:
            stats["skip_no_source_tb"] += 1
            print(f"[警告] number={number}, source_id={source_id} 找不到 source timeblock，已跳过")
            continue

        # ========== 情况 A：当前 target_block 的 Interlude=True ==========
        if bool(target_block.get("Interlude", False)) is True:
            copy_source_fields_for_interlude(target_block, source_block, source_id)
            stats["interlude_direct_cover"] += 1
            continue

        # ========== 情况 B：当前 target_block 的 Interlude=False，需要分割 ==========
        source_iso = source_block.get("iso", "")
        if not source_iso:
            stats["skip_source_iso_empty"] += 1
            print(
                f"[警告] doc_key={doc_key}, number={number}, source_id={source_id} "
                f"的 source iso 为空，无法分割 iso_range，已跳过"
            )
            continue

        old_block = target_block
        old_block_range = old_block.get("timeblock_range", "")
        old_block_iso_range = old_block.get("iso_range", "")

        if not old_block_range:
            stats["skip_no_target_tb"] += 1
            print(f"[警告] doc_key={doc_key}, number={number} 的 target_block 缺少 timeblock_range，已跳过")
            continue

        try:
            old_order_key = get_timeblock_order_key(
                old_block, seq_index_map, sentence_index_map, target_idx
            )
            old_start, old_end = split_range(old_block_range)
        except Exception as e:
            stats["skip_no_target_tb"] += 1
            print(f"[警告] doc_key={doc_key}, number={number} 的 target_block 解析失败，已跳过 | 错误: {e}")
            continue

        # 找“左半块”的结束点：number 前最近的非 sink 句子
        left_end = find_prev_non_sink_number_within_block(
            number, old_block_range, sentence_numbers, sink_map
        )

        left_iso_range, right_iso_range = split_iso_range(old_block_iso_range, source_iso)

        # 新建右半块
        new_block = deepcopy(old_block)
        new_block["ID"] = number
        new_block["timeblock_range"] = f"{number}-{old_end}"
        new_block["Interlude"] = False
        copy_source_fields_for_split_new_block(new_block, source_block, right_iso_range, source_id)

        excluded_indices = set()

        if left_end is not None:
            # 原块改成左半块
            old_block["timeblock_range"] = f"{old_start}-{left_end}"
            old_block["iso_range"] = left_iso_range

            # 在原块后面插入新块
            tmb_list.insert(target_idx + 1, new_block)
            excluded_indices = {target_idx, target_idx + 1}
            stats["split"] += 1
        else:
            # 左侧没有非 sink 句子时，直接替换原块
            tmb_list[target_idx] = new_block
            excluded_indices = {target_idx}
            stats["replace_without_left_block"] += 1
            print(
                f"[提示] doc_key={doc_key}, number={number} 分割点左侧没有非 sink 句子，"
                f"原块已直接替换为新块：{new_block['timeblock_range']}"
            )

        # 广播：寻找所有“原来 iso_range == old_block_iso_range”的其他块
        for idx, block in enumerate(tmb_list):
            if idx in excluded_indices:
                continue

            if block.get("iso_range", "") != old_block_iso_range:
                continue

            try:
                current_order_key = get_timeblock_order_key(
                    block, seq_index_map, sentence_index_map, idx
                )
            except Exception:
                continue

            if current_order_key < old_order_key:
                # 原块之前 -> 全部改为左半段
                block["iso_range"] = left_iso_range
            elif current_order_key > old_order_key:
                # 原块之后 -> 全部改为右半段，同时 iso 改成 source iso
                block["iso_range"] = right_iso_range
                block["iso"] = source_iso
            else:
                # 极少数顺序键完全相同，保守跳过
                pass

    # 保存到 timeblock/step13output，文件名保持不变
    out_name = timeblock_file_map[doc_key]["filename"]
    out_path = OUTPUT_DIR / out_name
    save_json(tmb_root, out_path)

    print(f"[完成] {doc_key} -> {out_path}")
    print(f"        统计信息: {stats}")
    return {
        "status": "done",
        "doc_key": doc_key,
        "out_path": out_path,
        "stats": stats,
    }


# =========================
# 主流程
# =========================
def main():
    global RUN_ROOT, TIMEBLOCK_DIR, SENTENCE_DIR, SEQUENCE_DIR, OUTPUT_DIR

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    TIMEBLOCK_DIR = timeblock_step_dir(RUN_ROOT, 11)
    SENTENCE_DIR = sentence_step_dir(RUN_ROOT, 12)
    SEQUENCE_DIR = sequence_step_dir(RUN_ROOT, 8)
    OUTPUT_DIR = timeblock_step_dir(RUN_ROOT, 13)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_step_logging(RUN_ROOT, "step_13_timeblock_iso_update")

    # 1) 建立文件索引
    timeblock_file_map = index_json_dir(TIMEBLOCK_DIR, expected_kind="timeblock")
    sentence_file_map = index_json_dir(SENTENCE_DIR, expected_kind="sentence")
    sequence_file_map = index_json_dir(SEQUENCE_DIR, expected_kind="sequence")

    if not timeblock_file_map:
        raise FileNotFoundError(f"{TIMEBLOCK_DIR} 下没有找到符合规则的 timeblock json 文件")

    # 2) 目标文件集合：以 timeblock 为准
    all_doc_keys = sorted(
        timeblock_file_map.keys(),
        key=lambda x: (
            int(x.split("_", 1)[0]),   # chapter_id
            x.split("_", 1)[1]         # uuid
        )
    )

    if TARGET_DOC_KEYS is None:
        target_doc_keys = all_doc_keys
    else:
        target_doc_keys = [doc_key for doc_key in all_doc_keys if doc_key in TARGET_DOC_KEYS]
    reporter = StepReporter("Step13", total=len(target_doc_keys))
    reporter.start(
        input_dir=TIMEBLOCK_DIR,
        output_dir=OUTPUT_DIR,
        extra=f"sentence={SENTENCE_DIR.name} sequence={SEQUENCE_DIR.name}",
    )

    pending_doc_keys = []
    for doc_key in target_doc_keys:
        out_path = OUTPUT_DIR / timeblock_file_map[doc_key]["filename"]
        if out_path.exists():
            reporter.item_skip(doc_key, detail="已存在输出")
            continue
        pending_doc_keys.append(doc_key)

    if not pending_doc_keys:
        reporter.finish(output_dir=OUTPUT_DIR, extra="无需处理")
        return

    # 3) 读取全部 timeblock 原始快照（作为 source 使用）
    source_timeblock_data = {}
    for doc_key, info in timeblock_file_map.items():
        try:
            source_timeblock_data[doc_key] = normalize_timeblock_root(load_json(info["path"]))
        except Exception as e:
            print(f"[跳过] timeblock 读取失败: {info['filename']} | 错误: {e}")

    # 4) 读取全部 sentence
    sentence_data_map = {}
    for doc_key, info in sentence_file_map.items():
        try:
            sentence_data_map[doc_key] = normalize_sentence_items(load_json(info["path"]))
        except Exception as e:
            print(f"[跳过] sentence 读取失败: {info['filename']} | 错误: {e}")

    # 5) 读取全部 sequence（允许缺失）
    sequence_data_map = {}
    for doc_key in all_doc_keys:
        if doc_key in sequence_file_map:
            try:
                sequence_data_map[doc_key] = normalize_sequence_list(
                    load_json(sequence_file_map[doc_key]["path"])
                )
            except Exception as e:
                print(f"[提示] sequence 读取失败，将回退为空序列: {sequence_file_map[doc_key]['filename']} | 错误: {e}")
                sequence_data_map[doc_key] = []
        else:
            print(f"[提示] doc_key={doc_key} 缺少对应 sequence 文件，将只用句子原始顺序作为后备排序")
            sequence_data_map[doc_key] = []

    # 6) 处理
    for doc_key in pending_doc_keys:
        try:
            result = process_one_doc(
                doc_key=doc_key,
                source_timeblock_data=source_timeblock_data,
                sentence_data_map=sentence_data_map,
                sequence_data_map=sequence_data_map,
                timeblock_file_map=timeblock_file_map
            )
            if result["status"] == "skipped":
                reporter.item_skip(doc_key, detail=result["reason"])
            else:
                stats = result["stats"]
                reporter.item_ok(
                    doc_key,
                    detail=f"split={stats['split']} cover={stats['interlude_direct_cover']}",
                )
        except Exception as e:
            reporter.item_fail(doc_key, e)

    reporter.finish(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
