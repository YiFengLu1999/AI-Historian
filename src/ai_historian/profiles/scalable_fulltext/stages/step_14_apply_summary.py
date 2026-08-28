# =========================
# timeblock step14 summary pipeline
# 输入：
#   sentence/step12output/*.json
#   timeblock/step13output/*.json
#
# 输出：
#   timeblock/step14output/*.json
# =========================

import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion,
    load_json_object,
    make_sync_chat_client,
    validate_json_text,
)
from ai_historian.pipeline.logging import StepReporter, setup_step_logging, step_tqdm
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir, timeblock_step_dir

# -------------------------
# 可调参数
# -------------------------
MODEL = CHAT_MODEL
BATCH_SIZE = max(1, int(os.getenv("STEP14_BATCH_SIZE", "40")))
MAX_CHARS_PER_CHUNK = max(500, int(os.getenv("STEP14_MAX_CHARS_PER_CHUNK", "6000")))
MAX_RETRIES = max(1, int(os.getenv("STEP14_MAX_RETRIES", "5")))
RETRY_BASE_SECONDS = max(1, int(os.getenv("STEP14_RETRY_BASE_SECONDS", "2")))
REQUEST_TIMEOUT_SECONDS = max(10, int(os.getenv("STEP14_REQUEST_TIMEOUT_SECONDS", "120")))
BATCH_PROGRESS_EVERY = max(1, int(os.getenv("STEP14_BATCH_PROGRESS_EVERY", "10")))
SKIP_EXISTING_SUMMARY = os.getenv("STEP14_SKIP_EXISTING_SUMMARY", "1").strip() != "0"


# -------------------------
# Pydantic 输出模型
# -------------------------
class SummaryOutput(BaseModel):
    summary: str = Field(
        ...,
        description="对该时间块内容的中文总结，必须是100字以内，不要编号，不要引号。",
    )


# -------------------------
# Qwen client（每个线程一个）
# -------------------------
_THREAD_LOCAL = threading.local()

def get_client():
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = make_sync_chat_client()
        _THREAD_LOCAL.client = client
    return client

def assert_sdk_ready():
    get_client()


# -------------------------
# 路径定位
# -------------------------
def find_project_root(start: Path) -> Path:
    """
    向上查找同时包含以下目录的项目根目录：
      - sentence/step12output
      - timeblock/step13output
    """
    for p in [start, *start.parents]:
        if (p / "sentence" / "step12output").is_dir() and (p / "timeblock" / "step13output").is_dir():
            return p
    raise FileNotFoundError(
        f"从 {start} 往上查找，未找到同时包含 sentence/step12output 和 timeblock/step13output 的项目根目录。"
    )


# -------------------------
# 文件工具
# -------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def list_json_files(directory: Path) -> List[Path]:
    return sorted(
        [path for path in Path(directory).glob("*.json") if path.is_file()],
        key=lambda path: path.name,
    )

def extract_doc_key_from_filename(path: Path, suffix_kind: str) -> str:
    """
    从文件名中抽取 doc_key。
    例如：
      7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_timeblock.json
        -> doc_key = "7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b"

      7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_sentence.json
        -> doc_key = "7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b"
    """
    suffix = f"_{suffix_kind}.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"文件名不符合预期，必须以 {suffix} 结尾：{path.name}")
    return path.name[:-len(suffix)]

def sort_key_for_doc_key(doc_key: str):
    """
    按 doc_key 前缀排序。
    假设文件名形如：篇章id_uuid_文件属性.json
    那么前缀一般是篇章 id，例如：
      7_xxx -> 7
      53_xxx -> 53
    """
    first_part, _, rest = doc_key.partition("_")
    if first_part.isdigit():
        return (0, int(first_part), rest)
    return (1, first_part, rest)


def derive_app_export_dir_name(run_root: Path) -> str:
    run_name = Path(run_root).name.strip() or "result"
    if run_name.startswith("result_") and len(run_name) > len("result_"):
        return f"app_base_input_{run_name[len('result_'):]}"
    return f"app_base_input_{run_name}"


APP_EXPORT_SKIP_NAME_SUBSTRINGS = ("step12",)


def export_result_to_app(run_root: Path) -> Tuple[Path, int, int]:
    sentence_source_dir = sentence_step_dir(run_root, 12)
    timeblock_source_dir = timeblock_step_dir(run_root, 14)

    if not sentence_source_dir.is_dir():
        raise FileNotFoundError(f"缺少 sentence 导出源目录: {sentence_source_dir}")
    if not timeblock_source_dir.is_dir():
        raise FileNotFoundError(f"缺少 timeblock 导出源目录: {timeblock_source_dir}")

    sentence_files = [
        path
        for path in list_json_files(sentence_source_dir)
        if not any(token in path.name for token in APP_EXPORT_SKIP_NAME_SUBSTRINGS)
    ]
    timeblock_files = list_json_files(timeblock_source_dir)

    if not sentence_files:
        raise FileNotFoundError(f"在 {sentence_source_dir} 下未找到可导出的 JSON 文件。")
    if not timeblock_files:
        raise FileNotFoundError(f"在 {timeblock_source_dir} 下未找到可导出的 JSON 文件。")

    export_root = run_root / "export"
    target_root = export_root / derive_app_export_dir_name(run_root)
    sentence_target_dir = target_root / "sentence"
    timeblock_target_dir = target_root / "timeblock"

    if target_root.exists():
        shutil.rmtree(target_root)

    sentence_target_dir.mkdir(parents=True, exist_ok=True)
    timeblock_target_dir.mkdir(parents=True, exist_ok=True)

    for source_file in sentence_files:
        shutil.copy2(source_file, sentence_target_dir / source_file.name)

    for source_file in timeblock_files:
        shutil.copy2(source_file, timeblock_target_dir / source_file.name)

    return target_root, len(sentence_files), len(timeblock_files)


# -------------------------
# 新 number / range 解析
# 现在的格式：
#   书籍uuid.篇章id.段落id.段落内句子id
# 例如：
#   94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
# -------------------------
UUID_NUMBER_PATTERN = re.compile(
    r"^[0-9a-fA-F-]+\.\d+\.\d+\.\d+$"
)

UUID_RANGE_PATTERN = re.compile(
    r"^\s*(?P<start>[0-9a-fA-F-]+\.\d+\.\d+\.\d+)\s*-\s*(?P<end>[0-9a-fA-F-]+\.\d+\.\d+\.\d+)\s*$"
)

def parse_scoped_number(number_str: str) -> Dict[str, Any]:
    """
    解析：
      '94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1'
    返回：
      {
        "book_uuid": "...",
        "chapter_id": 7,
        "paragraph_id": 50,
        "sentence_id": 1
      }
    """
    s = str(number_str).strip()
    if not UUID_NUMBER_PATTERN.fullmatch(s):
        raise ValueError(f"无法解析 number: {number_str}")

    book_uuid, chapter_id, paragraph_id, sentence_id = s.rsplit(".", 3)
    return {
        "book_uuid": book_uuid,
        "chapter_id": int(chapter_id),
        "paragraph_id": int(paragraph_id),
        "sentence_id": int(sentence_id),
    }

def number_to_order_key(parsed: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    排序 / 区间判断仍然遵循旧原则：
    按 篇章id -> 段落id -> 段落内句子id 比较
    """
    return (
        parsed["chapter_id"],
        parsed["paragraph_id"],
        parsed["sentence_id"],
    )

def parse_range_string(range_str: str) -> Tuple[Dict[str, Any], Dict[str, Any], Tuple[int, int, int], Tuple[int, int, int]]:
    """
    支持两种情况：
      1. 单点：
         94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
      2. 区间：
         94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1-94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.8
    """
    s = str(range_str).strip()
    if not s:
        raise ValueError(f"非法 range: {range_str}")

    if UUID_NUMBER_PATTERN.fullmatch(s):
        start_str = s
        end_str = s
    else:
        m = UUID_RANGE_PATTERN.fullmatch(s)
        if not m:
            raise ValueError(f"range 格式不合法: {range_str}")
        start_str = m.group("start")
        end_str = m.group("end")

    start_meta = parse_scoped_number(start_str)
    end_meta = parse_scoped_number(end_str)

    # 同一个区间必须属于同一个 book_uuid
    if start_meta["book_uuid"] != end_meta["book_uuid"]:
        raise ValueError(f"range 起止 book_uuid 不一致: {range_str}")

    start_key = number_to_order_key(start_meta)
    end_key = number_to_order_key(end_meta)

    if start_key > end_key:
        start_meta, end_meta = end_meta, start_meta
        start_key, end_key = end_key, start_key

    return start_meta, end_meta, start_key, end_key

def in_range(number_str: str, start_meta: Dict[str, Any], end_meta: Dict[str, Any], start_key: Tuple[int, int, int], end_key: Tuple[int, int, int]) -> bool:
    parsed = parse_scoped_number(number_str)

    # 先保证 book_uuid 一致
    if parsed["book_uuid"] != start_meta["book_uuid"]:
        return False

    # 再按 章-段-句 判断区间
    key = number_to_order_key(parsed)
    return start_key <= key <= end_key


# -------------------------
# 从 timeblock 对象里取 range
# 兼容不同字段名，避免你后面结构一改就崩
# -------------------------
RANGE_FIELD_CANDIDATES = [
    "timeblock_range",
    "timeblockRange",
    "timeblockrange",
    "iso_range",
    "isoRange",
    "isorange",
]

def get_range_str_from_obj(obj: dict) -> str:
    for key in RANGE_FIELD_CANDIDATES:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# -------------------------
# 句子提取
# -------------------------
def collect_sentences_for_range(sentence_items: List[dict], range_str: str) -> List[str]:
    """
    从 sentence 文件中提取落在 range_str 内的句子。
    保留原文件顺序，不带 number。
    """
    start_meta, end_meta, start_key, end_key = parse_range_string(range_str)

    selected = []
    for item in sentence_items:
        num = item.get("number")
        sent = item.get("sentence", "")

        if not num or not sent:
            continue

        try:
            if in_range(num, start_meta, end_meta, start_key, end_key):
                sent = str(sent).strip()
                if sent:
                    selected.append(sent)
        except Exception:
            # 某条 sentence 的 number 如果格式有问题，直接跳过，不让全局炸掉
            continue

    return selected


# -------------------------
# 文本分块
# -------------------------
def pack_units_by_chars(units: List[str], max_chars: int) -> List[str]:
    """
    把文本单元打包成多个 chunk，每个 chunk 不超过 max_chars。
    使用 \n 连接，便于 LLM 阅读。
    """
    chunks = []
    current = []
    current_len = 0

    for unit in units:
        unit = str(unit).strip()
        if not unit:
            continue

        if len(unit) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

            for i in range(0, len(unit), max_chars):
                piece = unit[i:i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        add_len = len(unit) + (1 if current else 0)

        if current and current_len + add_len > max_chars:
            chunks.append("\n".join(current))
            current = [unit]
            current_len = len(unit)
        else:
            current.append(unit)
            current_len += add_len

    if current:
        chunks.append("\n".join(current))

    return chunks


# -------------------------
# LLM 总结
# -------------------------
SUMMARY_INSTRUCTIONS = """
你是一个严谨的中国古代传记年谱整理助手。
你的任务是根据给定文本，写出一个高度凝练的中文总结。

要求：
1. 只输出与文本内容对应的总结。
2. 总结必须是100字以内。
3. 不要编号，不要项目符号，不要引号，不要解释。
4. 不要说“这段文字”“该文本”“本段”等元话语。
5. 尽量保留关键人物、关键事件、时间推进和结果。
6. 输出必须严格符合给定的 JSON 结构。
""".strip()

def normalize_summary(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", "", text)
    text = text.strip("\"'“”‘’")
    return text


def _walk_string_values(value: Any) -> List[str]:
    results: List[str] = []
    if isinstance(value, str):
        cleaned = normalize_summary(value)
        if cleaned:
            results.append(cleaned)
        return results

    if isinstance(value, dict):
        for item in value.values():
            results.extend(_walk_string_values(item))
        return results

    if isinstance(value, list):
        for item in value:
            results.extend(_walk_string_values(item))
        return results

    return results


def try_extract_summary_from_content(content: str) -> str:
    payload = load_json_object(content)

    preferred_keys = (
        "summary", "Summary", "摘要", "summary_text", "result", "Result",
        "output", "Output", "content", "Content", "text", "Text", "answer", "Answer",
    )
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, str):
            cleaned = normalize_summary(value)
            if cleaned:
                return cleaned[:100]

    candidates = _walk_string_values(payload)
    if candidates:
        return max(candidates, key=len)[:100]

    raise ValueError(f"无法从模型输出中提取 summary: {content[:200]!r}")


def summarize_plaintext_once(text: str) -> str:
    client = get_client()
    response = create_chat_completion(
        client,
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    "请直接输出100字以内的中文总结，不要 JSON，不要解释：\n\n"
                    f"{text}"
                ),
            },
        ],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    content = normalize_summary(response.choices[0].message.content or "")
    if not content:
        raise ValueError("plain summary 为空")
    return content[:100]

def summarize_once(text: str) -> str:
    """
    单次调用 LLM，总结为 100 字以内。
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_client()
            response = create_chat_completion(
                client,
                model=MODEL,
                messages=[
                    {"role": "system", "content": SUMMARY_INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": f"请总结下面这个时间块文本，严格控制在100字以内：\n\n{text}",
                    },
                ],
                response_format={"type": "json_object"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            content = response.choices[0].message.content or ""
            try:
                parsed = validate_json_text(SummaryOutput, content)
                summary = normalize_summary(parsed.summary)
            except Exception:
                summary = try_extract_summary_from_content(content)

            if not summary:
                raise ValueError("summary 为空")

            if len(summary) > 100:
                summary = summary[:100]

            return summary

        except Exception as e:
            last_error = e
            try:
                summary = summarize_plaintext_once(text)
                if summary:
                    return summary
            except Exception as fallback_error:
                last_error = fallback_error
            if attempt < MAX_RETRIES:
                sleep_s = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(sleep_s)
            else:
                raise RuntimeError(f"LLM 总结失败（已重试 {MAX_RETRIES} 次）: {last_error}") from last_error

    raise RuntimeError(f"LLM 总结失败: {last_error}")

def summarize_with_chunking(sentences: List[str]) -> str:
    """
    对句子列表做分块总结：
    - 如果整体不长：直接总结
    - 如果太长：分块总结
    - 若分块摘要仍很多：继续压缩，直到得到最终摘要
    """
    sentences = [s.strip() for s in sentences if str(s).strip()]
    if not sentences:
        return ""

    current_units = sentences

    while True:
        chunks = pack_units_by_chars(current_units, MAX_CHARS_PER_CHUNK)

        if len(chunks) == 1:
            return summarize_once(chunks[0])

        current_units = [summarize_once(chunk) for chunk in chunks]


# -------------------------
# 任务处理
# -------------------------
def batched(seq: List[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def process_one_timeblock_task(task: dict, sentence_store: Dict[str, List[dict]]) -> dict:
    """
    task:
    {
        "doc_key": "7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b",
        "timeblock_file": Path(...),
        "obj_index": 0,
        "range_str": "94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1-94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.8"
    }
    """
    doc_key = task["doc_key"]
    range_str = task["range_str"]

    sentence_items = sentence_store[doc_key]
    sentences = collect_sentences_for_range(sentence_items, range_str)

    if not sentences:
        return {
            "ok": False,
            "doc_key": doc_key,
            "timeblock_file": task["timeblock_file"],
            "obj_index": task["obj_index"],
            "range_str": range_str,
            "summary": "",
            "error": f"未在对应 sentence 文件中找到区间 {range_str} 对应的句子"
        }

    summary = summarize_with_chunking(sentences)

    return {
        "ok": True,
        "doc_key": doc_key,
        "timeblock_file": task["timeblock_file"],
        "obj_index": task["obj_index"],
        "range_str": range_str,
        "summary": summary,
        "error": None
    }


# -------------------------
# 主流程
# -------------------------
def main():
    root = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    if __name__ == "__main__":
        setup_step_logging(root, "step_14_apply_summary")

    assert_sdk_ready()

    sentence_dir = sentence_step_dir(root, 12)
    timeblock_dir = timeblock_step_dir(root, 13)
    output_dir = timeblock_step_dir(root, 14)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_timeblock_files = sorted(
        timeblock_dir.glob("*_timeblock.json"),
        key=lambda p: sort_key_for_doc_key(extract_doc_key_from_filename(p, "timeblock"))
    )

    if not all_timeblock_files:
        raise FileNotFoundError(f"在 {timeblock_dir} 下未找到 *_timeblock.json 文件。")
    reporter = StepReporter("Step14", total=len(all_timeblock_files))
    reporter.start(
        input_dir=timeblock_dir,
        output_dir=output_dir,
        extra=(
            f"sentence={sentence_dir.name} batch={BATCH_SIZE} "
            f"timeout={REQUEST_TIMEOUT_SECONDS}s retries={MAX_RETRIES}"
        ),
    )

    timeblock_files = []
    for tb_file in all_timeblock_files:
        out_path = output_dir / tb_file.name
        if out_path.exists():
            timeblock_files.append(tb_file)
            reporter.info(f"检测到已有 step14 输出，将尝试断点续跑: {tb_file.name}")
            continue
        timeblock_files.append(tb_file)

    if not timeblock_files:
        reporter.finish(output_dir=output_dir, extra="无需处理")
        return

    # 先加载所有对应的 sentence 文件
    doc_keys = [extract_doc_key_from_filename(p, "timeblock") for p in timeblock_files]
    sentence_store: Dict[str, List[dict]] = {}

    for doc_key in doc_keys:
        sentence_file = sentence_dir / f"{doc_key}_sentence.json"
        if not sentence_file.exists():
            raise FileNotFoundError(f"缺少对应 sentence 文件: {sentence_file}")

        data = load_json(sentence_file)
        if not isinstance(data, list):
            raise TypeError(f"{sentence_file} 不是 list 结构。")
        sentence_store[doc_key] = data

    # 加载 timeblock 文件并构建任务
    payloads_by_file: Dict[Path, dict] = {}
    tasks: List[dict] = []
    skipped_existing = 0

    for tb_file in timeblock_files:
        doc_key = extract_doc_key_from_filename(tb_file, "timeblock")
        out_path = output_dir / tb_file.name
        payload_source = out_path if out_path.exists() else tb_file
        payload = load_json(payload_source)

        if not isinstance(payload, dict) or "TMB" not in payload or not isinstance(payload["TMB"], list):
            raise TypeError(f"{payload_source} 结构不符合预期，应为 {{'TMB': [...]}}")

        payloads_by_file[tb_file] = payload

        for idx, obj in enumerate(payload["TMB"]):
            range_str = get_range_str_from_obj(obj)
            existing_summary = str(obj.get("summary", "")).strip()

            if not range_str:
                tasks.append({
                    "doc_key": doc_key,
                    "timeblock_file": tb_file,
                    "obj_index": idx,
                    "range_str": "",
                })
                continue

            if SKIP_EXISTING_SUMMARY and existing_summary:
                skipped_existing += 1
                continue

            tasks.append({
                "doc_key": doc_key,
                "timeblock_file": tb_file,
                "obj_index": idx,
                "range_str": range_str,
            })

    reporter.info(f"待处理 summary={len(tasks)} 跳过现有={skipped_existing}")

    if not tasks:
        for tb_file, payload in payloads_by_file.items():
            out_path = output_dir / tb_file.name
            save_json(payload, out_path)
            reporter.item_ok(tb_file.name, detail="无需重算")
        exported_dir, sentence_count, timeblock_count = export_result_to_app(root)
        reporter.info(
            f"已导出到 {exported_dir} | sentence={sentence_count} | timeblock={timeblock_count}"
        )
        reporter.finish(output_dir=output_dir, extra="无需处理")
        return

    success_count = 0
    fail_results = []

    pbar = step_tqdm(total=len(tasks), desc="生成 timeblock summary")

    for batch_tasks in batched(tasks, BATCH_SIZE):
        max_workers = min(BATCH_SIZE, len(batch_tasks))
        batch_done = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_one_timeblock_task, task, sentence_store): task
                for task in batch_tasks
            }

            for future in as_completed(future_map):
                result = None
                try:
                    result = future.result()
                except Exception as e:
                    task = future_map[future]
                    result = {
                        "ok": False,
                        "doc_key": task["doc_key"],
                        "timeblock_file": task["timeblock_file"],
                        "obj_index": task["obj_index"],
                        "range_str": task["range_str"],
                        "summary": "",
                        "error": str(e),
                    }

                tb_file = result["timeblock_file"]
                obj_index = result["obj_index"]

                if result["ok"]:
                    payloads_by_file[tb_file]["TMB"][obj_index]["summary"] = result["summary"]
                    success_count += 1
                else:
                    payloads_by_file[tb_file]["TMB"][obj_index]["summary"] = ""
                    fail_results.append(result)

                pbar.update(1)
                batch_done += 1
                finished_total = success_count + len(fail_results)
                if batch_done % BATCH_PROGRESS_EVERY == 0 or batch_done == len(batch_tasks):
                    reporter.info(
                        f"summary 批内进度={finished_total}/{len(tasks)} "
                        f"当前批次={batch_done}/{len(batch_tasks)}"
                    )

        for tb_file, payload in payloads_by_file.items():
            out_path = output_dir / tb_file.name
            save_json(payload, out_path)
        reporter.info(f"summary 进度={success_count + len(fail_results)}/{len(tasks)}")

    pbar.close()

    # 保存到 step14output，文件名保持原样
    for tb_file, payload in payloads_by_file.items():
        out_path = output_dir / tb_file.name
        save_json(payload, out_path)
        reporter.item_ok(tb_file.name)

    exported_dir, sentence_count, timeblock_count = export_result_to_app(root)
    reporter.info(
        f"已导出到 {exported_dir} | sentence={sentence_count} | timeblock={timeblock_count}"
    )

    if fail_results:
        for item in fail_results[:20]:
            reporter.info(
                f"失败 | 文件={item['timeblock_file'].name} index={item['obj_index']} "
                f"range={item['range_str']} 错误={item['error']}"
            )
    reporter.finish(output_dir=output_dir, extra=f"summary成功={success_count} 失败={len(fail_results)}")


if __name__ == "__main__":
    main()
