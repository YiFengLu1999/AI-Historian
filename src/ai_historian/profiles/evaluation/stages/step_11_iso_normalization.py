import csv
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, ValidationError
from rich import print as rprint

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion,
    make_sync_chat_client,
)
from ai_historian.pipeline.logging import StepReporter, setup_step_logging, step_tqdm
from ai_historian.pipeline.paths import resolve_run_root, sequence_step_dir, timeblock_step_dir
from ai_historian.pipeline.time_canonicalizer import (
    is_context_dependent_tm,
    is_non_anchor_tm,
    normalize_experiment1_tm,
)
from ai_historian.resources import CHINESE_ERAS, TIME_STRING_ISO_MAP


# =========================
# 0) Pydantic output schema
# =========================
class JUICE(BaseModel):
    TSTR: str = Field(default="")  # ISO string or ""

# =========================
# 1) Paths
# =========================
RUN_ROOT: Path
TIMEBLOCK_INPUT_DIR: Path
SEQUENCE_INPUT_DIR: Path
OUTPUT_DIR: Path

# =========================
# 2) Reference table + rules
# =========================
REFERENCE_TABLE_CANDIDATES = [
    CHINESE_ERAS,
]

DEFAULT_STR2ISO_TABLE_TEXT = """\
皇帝名称,年号/纪元,开始的第一天是ISO 8601的那一天
"""


def load_reference_table_text() -> Tuple[str, str]:
    for path in REFERENCE_TABLE_CANDIDATES:
        if not path.exists():
            continue

        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = path.read_text(encoding=encoding).strip()
                break
            except UnicodeDecodeError:
                text = ""
        else:
            text = ""

        if not text:
            continue

        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames and all(
            key in reader.fieldnames
            for key in ("皇帝名称", "年号/纪元", "开始的第一天是ISO 8601的那一天")
        ):
            return text, str(path.resolve())

    return DEFAULT_STR2ISO_TABLE_TEXT.strip(), "built-in fallback"


STR2ISO_TABLE_TEXT = DEFAULT_STR2ISO_TABLE_TEXT.strip()
STR2ISO_TABLE_SOURCE = "built-in fallback"

RULES_TEXT = """\
任务：将输入 timeblock 的 TM 转换为 iso（ISO 8601，天文纪年），如果无法准确判断则输出空字符串 ""。

你必须遵守：
- ISO 8601 表示公元前年份时采用天文纪年：允许存在年 0，因此 0000 对应传统的 1 BCE，-0001 对应 2 BCE。一般规律：BCE 的 N 年 → ISO 年 = 1 − N。
- 纪年推理：若给出 “某帝某年”的开始日，可推得下一年：例如已知“某帝元年”的起始日，则“某帝二年”的起始日通常是在此基础上顺推一年。
- 规则1：若仅提及某一年，未指明具体月份和日期，则默认该日期为当年 10 月 1 日。
- 规则2：若 TM 中包含季节信息，则推定日期：
    春季 → 3 月 1 日
    夏季 → 6 月 1 日
    秋季 → 9 月 1 日
    冬季 → 12 月 1 日
- 规则3：如果只提及在那个朝代，没有提及君王及月份信息，我们认为这个是该王朝第一个皇帝的第一年的第一天。
- 规则4：优先参考给定的年号表；如果表中没有足够信息支撑精确换算，就输出空字符串，不要猜测。
- 输出格式严格：只输出 JSON 对象 {"TSTR": "..."}；若不能确定则 {"TSTR": ""}。
"""

def find_experiment_iso_map_path() -> Path | None:
    return TIME_STRING_ISO_MAP


def month_iso_to_step11_date(value: str) -> str:
    text = str(value or "").strip()
    if text in {"-infinity", "+infinity", ""}:
        return ""
    match = re.match(r"^([+-]?\d{4,})-(\d{2})(?:-\d{2})?$", text)
    return f"{match.group(1)}-{match.group(2)}-01" if match else ""


def load_experiment_iso_lookup() -> Tuple[Dict[str, str], str]:
    path = find_experiment_iso_map_path()
    if path is None:
        return {}, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, str(path)
    raw_map = payload.get("map", {})
    if not isinstance(raw_map, dict):
        return {}, str(path)
    lookup = {
        normalize_experiment1_tm(key): month_iso_to_step11_date(value)
        for key, value in raw_map.items()
        if normalize_experiment1_tm(key) and month_iso_to_step11_date(value)
    }
    return lookup, str(path)


EXPERIMENT_ISO_LOOKUP: Dict[str, str] = {}
EXPERIMENT_ISO_LOOKUP_SOURCE = ""

# =========================
# 3) 文件名 / 新 number / 新 range 解析
# =========================
# 文件名规则：篇章id_uuid_文件属性.json
# 例子：
#   7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_timeblock.json
#   7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_sequence.json
FILE_NAME_RE = re.compile(
    r"^(?P<chapter_id>\d+)_(?P<book_uuid>[^_]+)_(?P<kind>timeblock|sequence)\.json$"
)

# 新 number 规则：
#   书籍uuid.篇章id.段落id.段落内句子id
# 例子：
#   94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
#
# 注意：不能按 "-" 切，因为 uuid 自己就有 "-"
# 正确做法：从右边按 "." 只切 3 次
def parse_point_number(number_str: str) -> Dict[str, Any]:
    s = str(number_str).strip()
    parts = s.rsplit(".", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid point number: {number_str}")

    book_uuid, chapter_id, paragraph_id, sentence_id = parts

    try:
        chapter_id_i = int(chapter_id)
        paragraph_id_i = int(paragraph_id)
        sentence_id_i = int(sentence_id)
    except ValueError:
        raise ValueError(f"Invalid numeric suffix in point number: {number_str}")

    return {
        "book_uuid": book_uuid,
        "chapter_id": chapter_id_i,
        "paragraph_id": paragraph_id_i,
        "sentence_id": sentence_id_i,
    }

def point_order_key(number_str: str) -> Tuple[int, int, int]:
    """
    保持原原则：比较顺序时仍然按
    篇章id、段落id、段落内句子id
    来比较。
    """
    parsed = parse_point_number(number_str)
    return (
        parsed["chapter_id"],
        parsed["paragraph_id"],
        parsed["sentence_id"],
    )


def obj_point_order_key(obj: Dict[str, Any]) -> Tuple[int, int, int]:
    try:
        return point_order_key(str(obj.get("ID", "") or ""))
    except Exception:
        return 10**9, 10**9, 10**9

# 新 range 规则：
#   start_point-end_point
# 例子：
#   94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1-94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
#
# 注意：不能简单 split("-")
RANGE_RE = re.compile(
    r"^(?P<start>.+?\.\d+\.\d+\.\d+)-(?P<end>.+?\.\d+\.\d+\.\d+)$"
)

def split_range_with_uuid(range_str: str) -> Tuple[str, str]:
    s = str(range_str).strip()
    m = RANGE_RE.match(s)
    if not m:
        raise ValueError(f"Invalid range string: {range_str}")

    start = m.group("start")
    end = m.group("end")

    # 再做一次 number 级别校验，确保不会被 uuid 的 - 搞崩
    parse_point_number(start)
    parse_point_number(end)

    return start, end

def parse_structured_filename(filename: str) -> Dict[str, str]:
    name = Path(filename).name
    m = FILE_NAME_RE.match(name)
    if not m:
        raise ValueError(
            f"Invalid filename format: {filename}\n"
            f"Expected: <chapter_id>_<book_uuid>_<timeblock|sequence>.json"
        )
    return {
        "chapter_id": m.group("chapter_id"),
        "book_uuid": m.group("book_uuid"),
        "kind": m.group("kind"),
    }

def build_pair_key_from_filename(filename: str) -> str:
    info = parse_structured_filename(filename)
    return f"{info['chapter_id']}_{info['book_uuid']}"

# =========================
# 4) DeepSeek helper (sync)
# =========================
DEFAULT_MODEL = CHAT_MODEL
SDK_KIND = "deepseek"
OPENAI_CLIENT = None


def get_client():
    if OPENAI_CLIENT is None:
        raise RuntimeError("LLM client is not initialized; call main() first")
    return OPENAI_CLIENT

def call_llm_json(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 200,
    max_retries: int = 6,
) -> Dict[str, Any]:
    """
    Returns parsed JSON dict. Retries on transient errors.
    """
    import json as _json

    backoff = 1.0
    last_err = None

    for _ in range(max_retries):
        try:
            resp = create_chat_completion(
                get_client(),
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            txt = resp.choices[0].message.content or ""

            return _json.loads(txt)

        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)

    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {repr(last_err)}")

# =========================
# 5) Validation / Constants
# =========================
ISO_RE = re.compile(r"^-?\d{4}-\d{2}-\d{2}$")  # e.g. -0209-10-01

NEG_INF = "-infinity"
POS_INF = "+infinity"
APPLY_CROSSDOC_INTERVAL_RANGE = os.getenv("AIH_APPLY_CROSSDOC_INTERVAL_RANGE", "0").strip().lower() in {"1", "true", "yes", "on"}
MIN_CROSSDOC_CONTEXT_CONFIDENCE = float(os.getenv("AIH_CROSSDOC_CONTEXT_MIN_CONF", "0.72"))
CROSSDOC_CONTEXT_MODE = os.getenv("AIH_CROSSDOC_CONTEXT_MODE", "start_only").strip().lower()
ENABLE_SOFT_CONTEXT_BOUNDARIES = os.getenv("AIH_ENABLE_SOFT_CONTEXT_BOUNDARIES", "1").strip().lower() in {"1", "true", "yes", "on"}

def validate_iso(s: str) -> str:
    s = (s or "").strip()
    if s == "":
        return ""
    return s if ISO_RE.match(s) else ""

def granularity_str(obj: Dict[str, Any]) -> str:
    return str(obj.get("Granularity", "")).strip()

def is_nonzero_granularity(obj: Dict[str, Any]) -> bool:
    return granularity_str(obj) != "0"

def finalize_boundary(start: str, end: str) -> Tuple[str, str]:
    """
    在算完原始 start / end 以后：
    - start 为空 => -infinity
    - end 为空   => +infinity
    """
    start = (start or "").strip()
    end = (end or "").strip()

    if start == "":
        start = NEG_INF
    if end == "":
        end = POS_INF
    return start, end

def compose_iso_range(start: str, end: str) -> str:
    start, end = finalize_boundary(start, end)
    return f"{start}to{end}"


def iso_day_index(value: str) -> int | None:
    if value in {NEG_INF, POS_INF}:
        return None
    match = ISO_RE.match(str(value or "").strip())
    if not match:
        return None
    year, month, day = str(value).split("-")[-3:]
    signed_year = int(str(value)[:-6])
    return signed_year * 372 + int(month) * 31 + int(day)


def later_iso(left: str, right: str) -> str:
    left = validate_iso(left)
    right = validate_iso(right)
    if not left:
        return right
    if not right:
        return left
    left_idx = iso_day_index(left)
    right_idx = iso_day_index(right)
    if left_idx is None:
        return right
    if right_idx is None:
        return left
    return right if right_idx > left_idx else left


def earlier_iso(left: str, right: str) -> str:
    left = validate_iso(left)
    right = validate_iso(right)
    if not left:
        return right
    if not right:
        return left
    left_idx = iso_day_index(left)
    right_idx = iso_day_index(right)
    if left_idx is None:
        return right
    if right_idx is None:
        return left
    return right if right_idx < left_idx else left


def repair_inverted_range(obj: Dict[str, Any], start: str, end: str) -> Tuple[str, str]:
    start_idx = iso_day_index(start)
    end_idx = iso_day_index(end)
    if start_idx is None or end_idx is None or start_idx <= end_idx:
        obj.pop("iso_range_conflict", None)
        return start, end
    obj["iso_range_conflict"] = {
        "start_before_repair": start,
        "end_before_repair": end,
        "repair": "finite_start_after_end_collapsed_to_start",
    }
    return start, start


def next_distinct_anchor_iso(ordered_objs: List[Dict[str, Any]], start_pos: int, current_iso: str) -> str:
    """
    If adjacent/successive anchor blocks repeat the same ISO, treat them as the
    same temporal heading and use the next distinct anchor as the range end.
    """
    current_iso = validate_iso(current_iso)
    for pos in range(start_pos + 1, len(ordered_objs)):
        candidate = ordered_objs[pos]
        if not is_nonzero_granularity(candidate):
            continue
        candidate_iso = validate_iso(str(candidate.get("iso", "")).strip())
        if not candidate_iso:
            continue
        if current_iso and candidate_iso == current_iso:
            continue
        return candidate_iso
    return ""


def boundary_tm_to_iso(tm: str, granularity: str = "1") -> str:
    iso_val, _status, _err_type, _err_msg = worker_call(tm, granularity)
    return validate_iso(iso_val)


MONTH_ONLY_RE = re.compile(r"^([正一二三四五六七八九十冬腊]+月)$")
MONTH_WITH_DAYPART_RE = re.compile(
    r"^([正一二三四五六七八九十冬腊]+月)(?:[,，、和至]*(?:早晨|中午|午间|傍晚|晚上|夜间|夜))*$"
)
SEASON_ONLY_RE = re.compile(r"^(春|春天|春季|夏|夏天|夏季|秋|秋天|秋季|冬|冬天|冬季)$")
REGNAL_PREFIX_RE = re.compile(r"((?:汉高祖|汉|秦二世|秦始皇|秦王子婴)[元一二三四五六七八九十]+年)")
EXPLICIT_YEAR_ONLY_RE = re.compile(
    r"^(?:(?:汉高祖|汉王|汉|秦二世|秦始皇|秦王子婴)[元一二三四五六七八九十]+年|(?:公元)?前?\d+年)$"
)


def latest_regnal_prefix_from_tm(tm: str) -> str:
    matches = REGNAL_PREFIX_RE.findall(str(tm or ""))
    return matches[-1] if matches else ""


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def is_explicit_year_only_text(value: str) -> bool:
    text = compact_text(value)
    if not text:
        return False
    if re.search(r"[正一二三四五六七八九十冬腊]+月", text):
        return False
    if re.search(r"(春天|春季|春|夏天|夏季|夏|秋天|秋季|秋|冬天|冬季|冬)", text):
        return False
    return bool(EXPLICIT_YEAR_ONLY_RE.fullmatch(text))


def conversion_original_text(obj: Dict[str, Any]) -> str:
    info = obj.get("Conversion information")
    if not isinstance(info, dict):
        info = obj.get("conversion_information")
    if not isinstance(info, dict):
        return ""
    return compact_text(str(info.get("time_information_original", "") or ""))


def conversion_required(obj: Dict[str, Any]) -> bool:
    info = obj.get("Conversion information")
    if not isinstance(info, dict):
        info = obj.get("conversion_information")
    return isinstance(info, dict) and bool(info.get("is_conversion_required"))


def looks_like_time_expression(value: str) -> bool:
    text = compact_text(value)
    if not text:
        return False
    if MONTH_WITH_DAYPART_RE.fullmatch(text) or SEASON_ONLY_RE.fullmatch(text):
        return True
    if re.search(r"(?:公元)?前?\d+年", text):
        return True
    if re.search(r"(?:汉高祖|汉王|汉|秦二世|秦始皇|秦王子婴)[元一二三四五六七八九十]+年", text):
        return True
    if re.search(r"[正一二三四五六七八九十冬腊]+月", text):
        return True
    if re.search(r"(春天|春季|春|夏天|夏季|夏|秋天|秋季|秋|冬天|冬季|冬)", text):
        return True
    return False


def is_dynasty_only_tm(value: str) -> bool:
    return compact_text(value) in {"秦朝", "汉朝", "楚朝"}


def is_event_phase_like_text(value: str) -> bool:
    """Return True for event/status phases that should not become ISO anchors.

    This is intentionally semantic rather than case-specific. Examples include
    "任县令期间", "攻占城邑后", "奉命巡边时", and "被封侯以后": they place an
    event in a narrative phase, but they do not themselves provide a month-level
    calendar anchor.
    """
    text = compact_text(value)
    if not text:
        return False
    if looks_like_time_expression(text):
        return False

    role_or_status = re.search(
        r"(?:任|为|作|做|担任|充任|拜为|封为|受封为|被封为).{1,20}(?:时|之时|期间|以后|之后|前后)?$",
        text,
    )
    administrative_phase = re.search(
        r"(?:任职|居官|在任|当政|执政|督察|巡察|巡行|奉命|受命|留守|镇守|治理|工作)",
        text,
    )
    military_or_political_event = re.search(
        r"(?:起兵|举兵|举事|反叛|叛乱|征讨|出征|进军|退军|东进|西进|南下|北上|"
        r"攻|围|破|败|战|入关|迁徙|就国|称王|即位|废立|诛杀|被杀|推荐|任命)",
        text,
    )
    phase_suffix = re.search(r"(?:时|之时|期间|阶段|以后|之后|以前|之前|前后)$", text)

    return bool(role_or_status or administrative_phase or military_or_political_event or phase_suffix)


def set_time_anchor(obj: Dict[str, Any], tm: str, granularity: str, reason: str) -> None:
    anchor = obj.get("time_anchor") if isinstance(obj.get("time_anchor"), dict) else {}
    granularity_value = int(granularity) if str(granularity).isdigit() else granularity
    anchor.update({
        "time_type": "absolute_anchor",
        "is_anchor": True,
        "canonical_time_text": tm,
        "granularity": granularity_value,
        "reason": reason,
    })
    obj["Granularity"] = str(granularity_value)
    obj["time_anchor"] = anchor


def demote_time_anchor(obj: Dict[str, Any], reason: str) -> None:
    obj["Granularity"] = "0"
    obj.pop("iso", None)
    obj.pop("iso_range", None)
    anchor = obj.get("time_anchor") if isinstance(obj.get("time_anchor"), dict) else {}
    anchor.update({
        "time_type": "context_dependent_or_non_time",
        "is_anchor": False,
        "granularity": 0,
        "reason": reason,
    })
    obj["time_anchor"] = anchor


def inferred_granularity_for_tm(tm: str, fallback: str) -> str:
    text = compact_text(tm)
    if re.search(r"[正一二三四五六七八九十冬腊]+月", text):
        return "2"
    if re.search(r"(春天|春季|春|夏天|夏季|夏|秋天|秋季|秋|冬天|冬季|冬)", text):
        return "1"
    if is_explicit_year_only_text(text):
        return "1"
    return str(fallback or "1")


def resolve_relative_month_anchors(ordered_objs: List[Dict[str, Any]]) -> int:
    """
    Resolve bare month expressions using the nearest previous explicit regnal-year anchor.
    Example: previous anchor "汉高祖元年正月" + current "四月" => "汉高祖元年四月".
    """
    resolved = 0
    previous_anchor_tm = ""
    for obj in ordered_objs:
        if not isinstance(obj, dict):
            continue
        tm = normalize_experiment1_tm(str(obj.get("TM", "") or "").strip())
        gran = granularity_str(obj)
        own_iso = validate_iso(str(obj.get("iso", "") or "").strip())
        month_match = MONTH_ONLY_RE.fullmatch(tm)
        if gran == "0" and month_match and previous_anchor_tm:
            prefix = latest_regnal_prefix_from_tm(previous_anchor_tm)
            if prefix:
                candidate_tm = normalize_experiment1_tm(f"{prefix}{month_match.group(1)}")
                candidate_iso = boundary_tm_to_iso(candidate_tm, "2")
                if candidate_iso:
                    obj["TM"] = candidate_tm
                    obj["Granularity"] = "2"
                    obj["iso"] = candidate_iso
                    obj["relative_month_anchor"] = {
                        "source": tm,
                        "previous_anchor_tm": previous_anchor_tm,
                        "resolved_tm": candidate_tm,
                        "resolved_iso": candidate_iso,
                    }
                    obj.pop("iso_range", None)
                    resolved += 1
                    previous_anchor_tm = candidate_tm
                    continue

        if gran != "0" and (own_iso or validate_iso(str(obj.get("iso", "") or "").strip())):
            current_tm = normalize_experiment1_tm(str(obj.get("TM", "") or "").strip())
            if latest_regnal_prefix_from_tm(current_tm):
                previous_anchor_tm = current_tm
    return resolved


def should_use_document_order_bounds(obj: Dict[str, Any]) -> bool:
    if granularity_str(obj) != "0":
        return False
    if isinstance(obj.get("crossdoc_context_evidence"), dict) or isinstance(obj.get("crossdoc_interval_evidence"), dict):
        return False
    tm = compact_text(str(obj.get("TM", "") or ""))
    anchor = obj.get("time_anchor") if isinstance(obj.get("time_anchor"), dict) else {}
    time_type = str(anchor.get("time_type", "") or "")
    return bool(
        obj.get("Interlude")
        or (not tm and time_type in {"non_time", "comparison_reference", "context_dependent_or_non_time"})
    )


def document_order_bounds(obj: Dict[str, Any], ordered_objs: List[Dict[str, Any]]) -> Tuple[str, str]:
    current_key = obj_point_order_key(obj)
    prev_iso = ""
    next_iso = ""
    prev_key = (-1, -1, -1)
    next_key = (10**9, 10**9, 10**9)

    for candidate in ordered_objs:
        if candidate is obj or not is_nonzero_granularity(candidate):
            continue
        candidate_iso = validate_iso(str(candidate.get("iso", "")).strip())
        if not candidate_iso:
            continue
        key = obj_point_order_key(candidate)
        if key < current_key and key > prev_key:
            prev_key = key
            prev_iso = candidate_iso
        elif key > current_key and key < next_key:
            next_key = key
            next_iso = candidate_iso

    return prev_iso or NEG_INF, next_iso or POS_INF


def normalize_ordered_contextual_tms(fname: str, data: Dict[str, Any]) -> Dict[str, int]:
    """
    Before ISO lookup/LLM conversion, normalize context-dependent expressions in
    sequence order. Bare months and seasons inherit the nearest previous regnal
    year anchor; if no such anchor exists, they are demoted to non-anchors.
    """
    stats = {
        "changed": 0,
        "demoted": 0,
        "context_resolved": 0,
    }
    if "TMB" not in data or not isinstance(data["TMB"], list):
        return stats

    id_to_obj: Dict[str, Dict[str, Any]] = {}
    original_objs: List[Dict[str, Any]] = []
    for obj in data["TMB"]:
        if not isinstance(obj, dict):
            continue
        original_objs.append(obj)
        obj_id = str(obj.get("ID", "")).strip()
        if obj_id and obj_id not in id_to_obj:
            id_to_obj[obj_id] = obj

    try:
        seq_ids = load_sequence_ids_for_timeblock(fname)
        ordered_objs = [id_to_obj[item] for item in seq_ids if item in id_to_obj]
        seen = {id(obj) for obj in ordered_objs}
        ordered_objs.extend(obj for obj in original_objs if id(obj) not in seen)
    except Exception:
        ordered_objs = original_objs

    previous_anchor_tm = ""
    for obj in ordered_objs:
        original_tm = str(obj.get("TM", "") or "").strip()
        original_conversion_tm = conversion_original_text(obj)
        original_gran = granularity_str(obj)
        source_for_context = original_conversion_tm or original_tm
        canonical_from_source = normalize_experiment1_tm(source_for_context, previous_anchor_tm)
        canonical_tm = canonical_from_source
        if original_conversion_tm and is_explicit_year_only_text(original_conversion_tm):
            canonical_tm = canonical_from_source
            if canonical_tm and canonical_tm != original_tm:
                obj["contextual_tm_source"] = {
                    "source": "conversion_information.time_information_original",
                    "original": original_conversion_tm,
                    "resolved_tm": canonical_tm,
                    "reason": "Explicit year-only source is preserved without inheriting a month.",
                }
        elif (
            original_conversion_tm
            and previous_anchor_tm
            and canonical_from_source
            and canonical_from_source != source_for_context
        ):
            obj["contextual_tm_source"] = {
                "source": "conversion_information.time_information_original",
                "original": original_conversion_tm,
                "previous_anchor_tm": previous_anchor_tm,
                "resolved_tm": canonical_from_source,
            }
        elif original_conversion_tm and not (
            MONTH_WITH_DAYPART_RE.fullmatch(original_conversion_tm)
            or SEASON_ONLY_RE.fullmatch(original_conversion_tm)
        ):
            canonical_tm = normalize_experiment1_tm(original_tm, previous_anchor_tm)

        if canonical_tm != original_tm:
            obj["TM"] = canonical_tm
            obj.pop("iso", None)
            obj.pop("iso_range", None)
            stats["changed"] += 1
            set_time_anchor(
                obj,
                canonical_tm,
                inferred_granularity_for_tm(canonical_tm, original_gran),
                "Context-dependent time expression resolved from nearest previous regnal-year anchor.",
            )
            if previous_anchor_tm and (
                MONTH_WITH_DAYPART_RE.fullmatch(compact_text(source_for_context))
                or SEASON_ONLY_RE.fullmatch(compact_text(source_for_context))
            ):
                stats["context_resolved"] += 1

        if (
            conversion_required(obj)
            and original_conversion_tm
            and not looks_like_time_expression(original_conversion_tm)
        ):
            demote_time_anchor(
                obj,
                "Conversion original text is an event/status description, not an explicit time expression.",
            )
            stats["demoted"] += 1
            continue

        if (
            is_dynasty_only_tm(canonical_tm)
            and original_conversion_tm
            and is_event_phase_like_text(original_conversion_tm)
        ):
            demote_time_anchor(
                obj,
                "Dynasty-only TM came from an event-phase original expression, so it is not a hard ISO anchor.",
            )
            stats["demoted"] += 1
            continue

        if original_tm and (
            not canonical_tm
            or is_non_anchor_tm(canonical_tm)
            or is_context_dependent_tm(canonical_tm)
        ):
            if original_gran != "0":
                demote_time_anchor(
                    obj,
                    "Context-dependent time expression has no previous regnal-year anchor.",
                )
                stats["demoted"] += 1
            continue

        if canonical_tm and granularity_str(obj) != "0" and latest_regnal_prefix_from_tm(canonical_tm):
            previous_anchor_tm = canonical_tm

    return stats


def apply_crossdoc_time_evidence(obj: Dict[str, Any]) -> bool:
    evidence = obj.get("crossdoc_time_evidence")
    if not isinstance(evidence, dict):
        return False
    if str(obj.get("iso", "")).strip() and granularity_str(obj) != "0":
        evidence["applied"] = False
        evidence["skip_reason"] = "target_already_has_iso_anchor"
        return False
    relation = str(evidence.get("relation", "") or "").strip()
    if relation not in {"same_atomic_event", "same_event_different_granularity", "same_event"}:
        evidence["applied"] = False
        evidence["skip_reason"] = "relation_not_anchor_transferable"
        return False
    anchor_tm = normalize_experiment1_tm(str(evidence.get("anchor_tm", "") or "").strip())
    if not anchor_tm:
        evidence["applied"] = False
        evidence["skip_reason"] = "empty_anchor_tm"
        return False
    anchor_granularity = str(evidence.get("anchor_granularity", "") or "1").strip()
    if not anchor_granularity or anchor_granularity == "0":
        anchor_granularity = "1"
    anchor_iso = boundary_tm_to_iso(anchor_tm, anchor_granularity)
    if not anchor_iso:
        evidence["applied"] = False
        evidence["skip_reason"] = "anchor_tm_iso_empty"
        return False
    obj["TM"] = anchor_tm
    obj["Granularity"] = anchor_granularity
    obj["iso"] = anchor_iso
    obj.pop("iso_range", None)
    evidence["applied"] = True
    evidence["canonical_anchor_tm"] = anchor_tm
    evidence["canonical_anchor_granularity"] = anchor_granularity
    evidence["canonical_anchor_iso"] = anchor_iso
    return True


def crossdoc_interval_iso_range(obj: Dict[str, Any]) -> str:
    evidence = obj.get("crossdoc_interval_evidence")
    if not isinstance(evidence, dict):
        return ""
    if not APPLY_CROSSDOC_INTERVAL_RANGE:
        evidence["applied"] = False
        evidence["skip_reason"] = "interval_range_application_disabled"
        return ""
    if evidence.get("disabled"):
        evidence["applied"] = False
        evidence["skip_reason"] = evidence.get("disabled_reason") or "disabled_by_temporal_graph"
        return ""
    if evidence.get("relation") != "contained_in_source_interval":
        evidence["applied"] = False
        evidence["skip_reason"] = "relation_not_interval_transferable"
        return ""
    start_tm = normalize_experiment1_tm(str(evidence.get("start_tm", "") or "").strip())
    end_tm = normalize_experiment1_tm(str(evidence.get("end_tm", "") or "").strip())
    if not start_tm and not end_tm:
        evidence["applied"] = False
        evidence["skip_reason"] = "empty_interval_boundaries"
        return ""
    start_iso = boundary_tm_to_iso(start_tm, "1") if start_tm else NEG_INF
    end_iso = boundary_tm_to_iso(end_tm, "1") if end_tm else POS_INF
    if not start_iso and start_tm:
        evidence["applied"] = False
        evidence["skip_reason"] = "start_tm_iso_empty"
        return ""
    if not end_iso and end_tm:
        evidence["applied"] = False
        evidence["skip_reason"] = "end_tm_iso_empty"
        return ""
    evidence["applied"] = True
    evidence["start_iso"] = start_iso
    evidence["end_iso"] = end_iso
    return compose_iso_range(start_iso, end_iso)


def crossdoc_context_boundaries(obj: Dict[str, Any], mode_override: str = "") -> Tuple[str, str]:
    evidence = obj.get("crossdoc_context_evidence")
    if not isinstance(evidence, dict):
        return "", ""
    context_mode = (mode_override or CROSSDOC_CONTEXT_MODE).strip().lower()
    if context_mode in {"0", "false", "no", "off", "disabled"}:
        evidence["applied"] = False
        evidence["skip_reason"] = "context_application_disabled"
        return "", ""
    if not evidence.get("quote_verified"):
        evidence["applied"] = False
        evidence["skip_reason"] = "quote_not_verified"
        return "", ""
    try:
        confidence = float(evidence.get("confidence", 0) or 0)
    except Exception:
        confidence = 0.0
    if confidence < MIN_CROSSDOC_CONTEXT_CONFIDENCE:
        evidence["applied"] = False
        evidence["skip_reason"] = "confidence_below_threshold"
        return "", ""
    relation = str(evidence.get("relation", "") or "")
    if relation not in {"episode_context", "same_sequence_phase"}:
        evidence["applied"] = False
        evidence["skip_reason"] = "relation_not_context_usable"
        return "", ""
    start_tm = normalize_experiment1_tm(str(evidence.get("source_anchor_before_tm", "") or "").strip())
    end_tm = normalize_experiment1_tm(str(evidence.get("source_anchor_after_tm", "") or "").strip())
    start_gran = str(evidence.get("source_anchor_before_granularity", "") or "1").strip() or "1"
    end_gran = str(evidence.get("source_anchor_after_granularity", "") or "1").strip() or "1"
    start_iso = boundary_tm_to_iso(start_tm, start_gran) if start_tm else ""
    end_iso = boundary_tm_to_iso(end_tm, end_gran) if end_tm else ""
    if not start_iso and not end_iso:
        evidence["applied"] = False
        evidence["skip_reason"] = "context_anchor_iso_empty"
        return "", ""
    evidence["applied"] = True
    evidence["source_anchor_before_iso"] = start_iso
    evidence["source_anchor_after_iso"] = end_iso
    evidence["context_mode"] = context_mode
    if context_mode in {"start", "start_only", "start-only"}:
        return start_iso, ""
    if context_mode in {"end", "end_only", "end-only"}:
        return "", end_iso
    return start_iso, end_iso

# =========================
# 6) Prompt builder
# =========================
def build_messages(tm: str, granularity: str) -> List[Dict[str, str]]:
    user_text = f"""\
输入：
- Granularity: {granularity}
- TM: {tm}

参考 str2iso 表（起始日）：
{STR2ISO_TABLE_TEXT}

规则：
{RULES_TEXT}

请输出严格 JSON：{{"TSTR": "<ISO或空字符串>"}}。
"""
    return [
        {"role": "system", "content": "你是一个严格的时间转换器，只做时间到ISO 8601（天文纪年）转换，不要输出额外解释。"},
        {"role": "user", "content": user_text},
    ]

# =========================
# 7) Batch execution
# =========================
CONCURRENCY = int(os.getenv("AIH_PIPELINE_CONCURRENCY", "40"))
TABLE_REFRESH_EVERY = 20

ENABLE_CACHE = True

_cache_lock = threading.Lock()
_cache: Dict[Tuple[str, str], str] = {}

def worker_call(tm: str, gran: str, model: str = DEFAULT_MODEL) -> Tuple[str, str, str, str]:
    """
    returns: (iso, status, err_type, err_msg)
    status in {"ok","empty","cached","invalid","fail","no_tm"}
    """
    tm = normalize_experiment1_tm((tm or "").strip())
    gran = (gran or "").strip()

    if tm == "":
        return ("", "no_tm", "", "")

    key = (tm, gran)
    if ENABLE_CACHE:
        with _cache_lock:
            if key in _cache:
                cached_iso = _cache[key]
                return (cached_iso, "cached" if cached_iso else "empty", "", "")

    lookup_iso = EXPERIMENT_ISO_LOOKUP.get(tm)
    if lookup_iso:
        if ENABLE_CACHE:
            with _cache_lock:
                _cache[key] = lookup_iso
        return (lookup_iso, "lookup", "", "")

    messages = build_messages(tm=tm, granularity=gran)
    try:
        raw = call_llm_json(messages, model=model)
        juice = JUICE.model_validate(raw) if hasattr(JUICE, "model_validate") else JUICE.parse_obj(raw)
        iso_val = validate_iso(juice.TSTR)

        if ENABLE_CACHE:
            with _cache_lock:
                _cache[key] = iso_val

        if iso_val:
            return (iso_val, "ok", "", "")
        else:
            return ("", "empty", "", "")
    except (ValidationError, ValueError) as e:
        if ENABLE_CACHE:
            with _cache_lock:
                _cache[key] = ""
        return ("", "invalid", type(e).__name__, str(e)[:200])
    except Exception as e:
        return ("", "fail", type(e).__name__, str(e)[:200])

# =========================
# 8) Sequence helpers
# =========================
def find_sequence_file_for_timeblock(timeblock_filename: str) -> Path:
    """
    按文件名精确匹配：
    7_uuid_timeblock.json -> 7_uuid_sequence.json
    """
    info = parse_structured_filename(timeblock_filename)
    if info["kind"] != "timeblock":
        raise ValueError(f"Not a timeblock filename: {timeblock_filename}")

    expected = SEQUENCE_INPUT_DIR / f"{info['chapter_id']}_{info['book_uuid']}_sequence.json"
    if expected.exists():
        return expected

    # 兜底扫描：同 chapter_id + uuid 的 sequence
    prefix = f"{info['chapter_id']}_{info['book_uuid']}_"
    candidates = [
        p for p in sorted(SEQUENCE_INPUT_DIR.glob(f"{prefix}*.json"))
        if p.name.endswith("_sequence.json")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find sequence file for timeblock file: {timeblock_filename}\n"
            f"Expected: {expected.resolve()}"
        )

    return candidates[0]

def load_sequence_ids_for_timeblock(timeblock_filename: str) -> List[str]:
    seq_path = find_sequence_file_for_timeblock(timeblock_filename)
    data = json.loads(seq_path.read_text(encoding="utf-8"))

    raw_ids = None

    if isinstance(data, list):
        raw_ids = data
    elif isinstance(data, dict):
        for key in ["sequence", "Sequence", "SEQ", "ordered_ids", "ids", "ID"]:
            if key in data and isinstance(data[key], list):
                raw_ids = data[key]
                break

    if raw_ids is None:
        raise ValueError(
            f"{seq_path.name}: sequence json must be either a list "
            f"or a dict containing one of keys "
            f"['sequence', 'Sequence', 'SEQ', 'ordered_ids', 'ids', 'ID']."
        )

    seq_ids = [str(x).strip() for x in raw_ids if str(x).strip()]
    if not seq_ids:
        raise ValueError(f"{seq_path.name}: sequence list is empty.")

    return seq_ids

# =========================
# 9) iso_range computation
# =========================
def compute_iso_ranges_for_one_file(fname: str, data: Dict[str, Any]) -> Dict[str, int]:
    """
    按对应 sequence 文件中的 ID 顺序，为单个 timeblock 文件计算所有 timeblock 的 iso_range。
    """
    if "TMB" not in data or not isinstance(data["TMB"], list):
        raise ValueError(f"{fname}: JSON must contain key 'TMB' as a list.")

    tmb: List[Dict[str, Any]] = data["TMB"]
    seq_ids = load_sequence_ids_for_timeblock(fname)

    # 建立 ID -> obj index
    id_to_index: Dict[str, int] = {}
    duplicate_ids = []
    original_ids_in_file: List[str] = []

    for i, obj in enumerate(tmb):
        if not isinstance(obj, dict):
            continue
        _id = str(obj.get("ID", "")).strip()
        if not _id:
            continue

        # 这里不强制要求新 number 格式必须合法，但一旦你后续要解析，下面的函数可直接用：
        # parse_point_number(_id)

        original_ids_in_file.append(_id)
        if _id in id_to_index:
            duplicate_ids.append(_id)
        else:
            id_to_index[_id] = i

    if duplicate_ids:
        raise ValueError(f"{fname}: duplicated timeblock IDs found: {duplicate_ids[:10]}")

    if not id_to_index:
        raise ValueError(f"{fname}: no valid timeblock IDs found in TMB.")

    # 按 sequence 顺序取交集
    seq_set = set(seq_ids)
    ordered_ids = [tb_id for tb_id in seq_ids if tb_id in id_to_index]

    # 如果文件里有 ID 不在 sequence 里，则追加到末尾（按原文件顺序）
    missing_in_sequence = [tb_id for tb_id in original_ids_in_file if tb_id not in seq_set]
    if missing_in_sequence:
        rprint(
            f"[yellow]Warning:[/yellow] {fname} has IDs missing in sequence file. "
            f"Appending them at the end in original TMB order. "
            f"Missing count={len(missing_in_sequence)}"
        )
        ordered_ids.extend(missing_in_sequence)

    if not ordered_ids:
        raise ValueError(f"{fname}: after matching with sequence, no ordered IDs remain.")

    ordered_indices = [id_to_index[tb_id] for tb_id in ordered_ids]
    ordered_objs = [tmb[idx] for idx in ordered_indices]
    n = len(ordered_objs)

    relative_month_anchor_resolved = resolve_relative_month_anchors(ordered_objs)

    # prev_anchor_pos[i]: i 往上找，第一个 Granularity != 0 的位置（不含自己）
    prev_anchor_pos = [-1] * n
    last_anchor = -1
    for i in range(n):
        prev_anchor_pos[i] = last_anchor
        if is_nonzero_granularity(ordered_objs[i]):
            last_anchor = i

    # next_anchor_pos[i]: i 往下找，第一个 Granularity != 0 的位置（不含自己）
    next_anchor_pos = [-1] * n
    next_anchor = -1
    for i in range(n - 1, -1, -1):
        next_anchor_pos[i] = next_anchor
        if is_nonzero_granularity(ordered_objs[i]):
            next_anchor = i

    soft_context_starts = [""] * n
    soft_context_ends = [""] * n
    soft_context_candidates = 0
    if ENABLE_SOFT_CONTEXT_BOUNDARIES:
        for i, obj in enumerate(ordered_objs):
            if granularity_str(obj) != "0":
                continue
            context_start, context_end = crossdoc_context_boundaries(obj, mode_override="both")
            if context_start or context_end:
                soft_context_starts[i] = context_start
                soft_context_ends[i] = context_end
                soft_context_candidates += 1

    prev_soft_boundary = [""] * n
    latest_soft = ""
    for i in range(n):
        prev_soft_boundary[i] = latest_soft
        latest_soft = later_iso(latest_soft, soft_context_starts[i])
        latest_soft = later_iso(latest_soft, soft_context_ends[i])

    next_soft_boundary = [""] * n
    nearest_soft = ""
    for i in range(n - 1, -1, -1):
        next_soft_boundary[i] = nearest_soft
        nearest_soft = earlier_iso(nearest_soft, soft_context_starts[i])
        nearest_soft = earlier_iso(nearest_soft, soft_context_ends[i])

    stats = {
        "total_timeblocks": n,
        "iso_range_filled": 0,
        "relative_month_anchor_resolved": relative_month_anchor_resolved,
        "crossdoc_interval_range_filled": 0,
        "crossdoc_context_boundary_used": 0,
        "soft_context_boundary_candidates": soft_context_candidates,
        "soft_context_boundary_used": 0,
        "duplicate_anchor_end_skipped": 0,
    }

    for i, obj in enumerate(ordered_objs):
        interval_range = crossdoc_interval_iso_range(obj)
        if interval_range:
            obj["iso_range"] = interval_range
            stats["iso_range_filled"] += 1
            stats["crossdoc_interval_range_filled"] += 1
            continue

        gran = granularity_str(obj)
        own_iso = validate_iso(str(obj.get("iso", "")).strip())
        context_start, context_end = ("", "")
        pa = prev_anchor_pos[i]
        na = next_anchor_pos[i]
        doc_order_start, doc_order_end = ("", "")
        if should_use_document_order_bounds(obj):
            doc_order_start, doc_order_end = document_order_bounds(obj, ordered_objs)
        if gran == "0" and pa == -1:
            context_start, context_end = crossdoc_context_boundaries(obj)
            if context_start or context_end:
                stats["crossdoc_context_boundary_used"] += 1
        if gran == "0" and ENABLE_SOFT_CONTEXT_BOUNDARIES:
            context_start = later_iso(context_start, soft_context_starts[i])
            context_start = later_iso(context_start, prev_soft_boundary[i])
            context_end = earlier_iso(context_end, soft_context_ends[i])
            context_end = earlier_iso(context_end, next_soft_boundary[i])
            if context_start or context_end:
                obj["soft_context_boundary"] = {
                    "start_iso": context_start,
                    "end_iso": context_end,
                    "self_start_iso": soft_context_starts[i],
                    "self_end_iso": soft_context_ends[i],
                    "prev_soft_iso": prev_soft_boundary[i],
                    "next_soft_iso": next_soft_boundary[i],
                }

        # ---------- 先按原逻辑算 raw start ----------
        if gran != "0":
            raw_start = own_iso
        elif context_start:
            raw_start = context_start
        elif doc_order_start:
            raw_start = doc_order_start
        else:
            if pa == -1:
                raw_start = NEG_INF
            else:
                raw_start = validate_iso(str(ordered_objs[pa].get("iso", "")).strip())

        # ---------- 再按原逻辑算 raw end ----------
        if gran == "0" and context_end:
            raw_end = context_end
            stats["soft_context_boundary_used"] += 1
        elif gran == "0" and doc_order_end:
            raw_end = doc_order_end
        else:
            if na == -1:
                raw_end = POS_INF
            else:
                raw_end = validate_iso(str(ordered_objs[na].get("iso", "")).strip())
                if gran != "0" and own_iso and raw_end == own_iso:
                    distinct_end = next_distinct_anchor_iso(ordered_objs, i, own_iso)
                    if distinct_end:
                        raw_end = distinct_end
                        stats["duplicate_anchor_end_skipped"] += 1
                    else:
                        raw_end = POS_INF

        # ---------- 兜底 ----------
        start, end = finalize_boundary(raw_start, raw_end)
        start, end = repair_inverted_range(obj, start, end)

        obj["iso_range"] = compose_iso_range(start, end)
        stats["iso_range_filled"] += 1

    return stats

def compute_all_iso_ranges(file_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    对所有文件计算 iso_range
    """
    all_stats: Dict[str, Dict[str, int]] = {}

    pbar = step_tqdm(total=len(file_data), desc="Compute iso_range", unit="file")
    for fname, data in file_data.items():
        stats = compute_iso_ranges_for_one_file(fname, data)
        all_stats[fname] = stats
        pbar.update(1)
    pbar.close()

    return all_stats

# =========================
# 10) Main pipeline
# =========================
def run_batch(concurrency: int = CONCURRENCY):
    timeblock_files = sorted(TIMEBLOCK_INPUT_DIR.glob("*.json"))
    if not timeblock_files:
        raise FileNotFoundError(f"No json files found in {TIMEBLOCK_INPUT_DIR.resolve()}")
    reporter = StepReporter("Step11", total=len(timeblock_files))
    reporter.start(
        input_dir=TIMEBLOCK_INPUT_DIR,
        output_dir=OUTPUT_DIR,
        extra=f"sequence={SEQUENCE_INPUT_DIR.name} 并发={concurrency}",
    )

    # 先校验文件名格式 & sequence 是否存在
    pair_preview = []
    for p in timeblock_files:
        info = parse_structured_filename(p.name)
        if info["kind"] != "timeblock":
            raise ValueError(f"Unexpected non-timeblock file in input dir: {p.name}")
        seq_path = find_sequence_file_for_timeblock(p.name)
        pair_preview.append((p.name, seq_path.name))

    rprint("[bold cyan]Matched file pairs:[/bold cyan]")
    for tb_name, seq_name in pair_preview:
        rprint(f" - {tb_name}  <->  {seq_name}")

    # 1) load all input files into memory
    file_data: Dict[str, Dict[str, Any]] = {}
    for p in timeblock_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        if "TMB" not in data or not isinstance(data["TMB"], list):
            raise ValueError(f"{p.name}: JSON must contain key 'TMB' as a list.")
        file_data[p.name] = data

    contextual_tm_stats = {
        "changed": 0,
        "demoted": 0,
        "context_resolved": 0,
    }
    for fname, data in file_data.items():
        current_stats = normalize_ordered_contextual_tms(fname, data)
        for key, value in current_stats.items():
            contextual_tm_stats[key] = contextual_tm_stats.get(key, 0) + int(value or 0)
    if any(contextual_tm_stats.values()):
        reporter.info(
            "上下文时间规范化="
            f"changed={contextual_tm_stats.get('changed', 0)} "
            f"demoted={contextual_tm_stats.get('demoted', 0)} "
            f"context_resolved={contextual_tm_stats.get('context_resolved', 0)}"
        )

    # 2) create LLM work items for iso
    # work item = (filename, obj_index, ID, TM, Granularity)
    work: List[Tuple[str, int, str, str, str]] = []
    crossdoc_time_evidence_applied = 0
    for fname, data in file_data.items():
        for i, obj in enumerate(data["TMB"]):
            if not isinstance(obj, dict):
                continue
            evidence_applied = apply_crossdoc_time_evidence(obj)
            if evidence_applied:
                crossdoc_time_evidence_applied += 1
            anchor = obj.get("time_anchor") if isinstance(obj.get("time_anchor"), dict) else None
            if anchor is not None and not anchor.get("is_anchor") and not evidence_applied:
                obj["Granularity"] = "0"
                obj.pop("iso", None)
                obj.pop("iso_range", None)
                continue

            gran = str(anchor.get("granularity") if anchor else obj.get("Granularity", "")).strip()
            if gran == "0":
                continue
            if str(obj.get("iso", "")).strip():  # already filled
                continue
            tm_source = anchor.get("canonical_time_text") if anchor else obj.get("TM", "")
            tm = normalize_experiment1_tm(str(tm_source or "").strip())
            obj["TM"] = tm
            obj["Granularity"] = gran
            _id = str(obj.get("ID", "")).strip()
            work.append((fname, i, _id, tm, gran))

    reporter.info(f"待判定 ISO 对象={len(work)}")
    if crossdoc_time_evidence_applied:
        reporter.info(f"跨文本事件时间证据应用={crossdoc_time_evidence_applied}")

    # 3) submit jobs with concurrency
    pbar = step_tqdm(total=len(work), desc="LLM objects", unit="obj")

    stats = {
        "ok": 0,
        "lookup": 0,
        "empty": 0,
        "cached": 0,
        "invalid": 0,
        "fail": 0,
        "no_tm": 0,
        "crossdoc_time_evidence_applied": crossdoc_time_evidence_applied,
        "contextual_tm_changed": contextual_tm_stats.get("changed", 0),
        "contextual_tm_demoted": contextual_tm_stats.get("demoted", 0),
        "contextual_tm_resolved": contextual_tm_stats.get("context_resolved", 0),
    }

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {}
        for idx, (fname, obj_i, _id, tm, gran) in enumerate(work, start=1):
            fut = ex.submit(worker_call, tm, gran, DEFAULT_MODEL)
            futures[fut] = (idx, fname, obj_i, _id, tm, gran)

        done_count = 0
        for fut in as_completed(futures):
            idx, fname, obj_i, _id, tm, gran = futures[fut]
            iso_val, status, err_type, err_msg = fut.result()

            # write back iso
            file_data[fname]["TMB"][obj_i]["iso"] = iso_val

            stats[status] = stats.get(status, 0) + 1

            done_count += 1
            pbar.update(1)

            if done_count % TABLE_REFRESH_EVERY == 0:
                reporter.info(f"ISO 进度={done_count}/{len(work)}")

    pbar.close()

    # 4) compute iso_range for all files（按对应 sequence 顺序，不按 TMB 原始顺序）
    rprint("[bold cyan]Start computing iso_range...[/bold cyan]")
    range_stats = compute_all_iso_ranges(file_data)
    stats["crossdoc_interval_range_filled"] = sum(
        item.get("crossdoc_interval_range_filled", 0)
        for item in range_stats.values()
    )
    stats["crossdoc_context_boundary_used"] = sum(
        item.get("crossdoc_context_boundary_used", 0)
        for item in range_stats.values()
    )

    # 5) write all outputs（保持 timeblock 原始文件名）
    for fname, data in file_data.items():
        (OUTPUT_DIR / fname).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        reporter.item_ok(fname)

    reporter.finish(
        output_dir=OUTPUT_DIR,
        extra=" ".join(f"{k}={v}" for k, v in stats.items()),
    )

def main() -> None:
    global RUN_ROOT, TIMEBLOCK_INPUT_DIR, SEQUENCE_INPUT_DIR, OUTPUT_DIR
    global STR2ISO_TABLE_TEXT, STR2ISO_TABLE_SOURCE
    global EXPERIMENT_ISO_LOOKUP, EXPERIMENT_ISO_LOOKUP_SOURCE, OPENAI_CLIENT

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    TIMEBLOCK_INPUT_DIR = timeblock_step_dir(RUN_ROOT, 10)
    SEQUENCE_INPUT_DIR = sequence_step_dir(RUN_ROOT, 8)
    OUTPUT_DIR = timeblock_step_dir(RUN_ROOT, 11)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_step_logging(RUN_ROOT, "step_11_iso_normalization")

    if not TIMEBLOCK_INPUT_DIR.exists():
        raise FileNotFoundError(f"Cannot find timeblock input dir: {TIMEBLOCK_INPUT_DIR.resolve()}")
    if not SEQUENCE_INPUT_DIR.exists():
        raise FileNotFoundError(f"Cannot find sequence input dir: {SEQUENCE_INPUT_DIR.resolve()}")

    STR2ISO_TABLE_TEXT, STR2ISO_TABLE_SOURCE = load_reference_table_text()
    EXPERIMENT_ISO_LOOKUP, EXPERIMENT_ISO_LOOKUP_SOURCE = load_experiment_iso_lookup()
    OPENAI_CLIENT = make_sync_chat_client()

    rprint("[bold cyan]Input dirs:[/bold cyan]")
    rprint(" - timeblock:", TIMEBLOCK_INPUT_DIR.resolve())
    rprint(" - sequence :", SEQUENCE_INPUT_DIR.resolve())
    rprint("[bold cyan]Output dir:[/bold cyan]", OUTPUT_DIR.resolve())
    rprint("[bold cyan]str2iso reference:[/bold cyan]", STR2ISO_TABLE_SOURCE)
    if EXPERIMENT_ISO_LOOKUP_SOURCE:
        rprint(
            "[bold cyan]experiment iso lookup:[/bold cyan]",
            f"{EXPERIMENT_ISO_LOOKUP_SOURCE} entries={len(EXPERIMENT_ISO_LOOKUP)}",
        )
    rprint(f"[bold cyan]LLM client:[/bold cyan] {SDK_KIND} | model={DEFAULT_MODEL}")
    run_batch(CONCURRENCY)


if __name__ == "__main__":
    main()
