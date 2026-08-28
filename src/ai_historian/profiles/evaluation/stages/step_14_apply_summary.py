# =========================
# timeblock step14 summary pipeline
# 输入：
#   sentence/step5output/*.json
#   timeblock/step11output/*.json
#
# 输出：
#   timeblock/step14output/*.json
# =========================

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field
from rich import print
from tqdm.auto import tqdm

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion,
    make_sync_chat_client,
)
from ai_historian.pipeline.logging import setup_step_logging
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir, timeblock_step_dir

# -------------------------
# 可调参数
# -------------------------
MODEL = os.getenv("AIH_CHAT_MODEL", CHAT_MODEL)
BATCH_SIZE = int(os.getenv("AIH_AGENT_BATCH_SIZE", os.getenv("AIH_AGENT_CONCURRENCY", "40")))
MAX_CHARS_PER_CHUNK = 6000
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 2
SKIP_EXISTING_SUMMARY = False  # True: 跳过已有 summary；False: 全部重算


# -------------------------
# Pydantic 输出模型
# -------------------------
class SummaryOutput(BaseModel):
    summary: str = Field(
        ...,
        description="对该时间块内容的中文总结，必须是100字以内，不要编号，不要引号。",
        max_length=100,
    )


# -------------------------
# LLM client（每个线程一个）
# -------------------------
_THREAD_LOCAL = threading.local()

def get_client():
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = make_sync_chat_client()
        _THREAD_LOCAL.client = client
    return client

def extract_first_json(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        return text[left:right + 1]
    return text


# -------------------------
# 路径定位
# -------------------------
def find_project_root(start: Path) -> Path:
    """
    向上查找同时包含以下目录的项目根目录：
      - sentence/step5output
      - timeblock/step11output
    """
    for p in [start, *start.parents]:
        if (p / "sentence" / "step5output").is_dir() and (p / "timeblock" / "step11output").is_dir():
            return p
    raise FileNotFoundError(
        f"从 {start} 往上查找，未找到同时包含 sentence/step5output 和 timeblock/step11output 的项目根目录。"
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
6. 输出必须严格是 JSON 对象，格式为 {"summary":"100字以内中文总结"}。
""".strip()

def normalize_summary(text: str) -> str:
    text = str(text).strip()
    text = text.strip("`")
    text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    text = text.strip("\"'“”‘’")
    return text

def parse_summary_text(raw_text: str) -> str:
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return ""
    try:
        parsed = SummaryOutput.model_validate_json(extract_first_json(raw_text))
        return normalize_summary(parsed.summary)
    except Exception:
        return normalize_summary(raw_text)

def summarize_once_plain(client: Any, text: str) -> str:
    response = create_chat_completion(
        client,
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严谨的中国古代传记年谱整理助手。"
                    "请直接输出100字以内中文总结，不要JSON，不要解释。"
                ),
            },
            {"role": "user", "content": f"请总结下面这个时间块文本，严格控制在100字以内：\n\n{text}"},
        ],
        temperature=0,
    )
    return parse_summary_text(response.choices[0].message.content or "")

def summarize_once(text: str) -> str:
    """
    单次调用 LLM，总结为 100 字以内。
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_client()
            kwargs = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SUMMARY_INSTRUCTIONS},
                    {"role": "user", "content": f"请总结下面这个时间块文本，严格控制在100字以内：\n\n{text}"},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            response = create_chat_completion(client, **kwargs)
            raw_text = response.choices[0].message.content or ""
            summary = parse_summary_text(raw_text)
            if not summary:
                summary = summarize_once_plain(client, text)

            if not summary:
                raise ValueError("summary 为空")

            if len(summary) > 100:
                summary = summary[:100]

            return summary

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                sleep_s = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(sleep_s)
            else:
                raise RuntimeError(f"LLM 总结失败（已重试 {MAX_RETRIES} 次）: {e}") from e

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
    setup_step_logging(root, "step_14_apply_summary")

    sentence_dir = sentence_step_dir(root, 5)
    timeblock_dir = timeblock_step_dir(root, 11)
    output_dir = timeblock_step_dir(root, 14)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bold cyan]项目根目录:[/bold cyan] {root}")
    print(f"[bold cyan]sentence 输入目录:[/bold cyan] {sentence_dir}")
    print(f"[bold cyan]timeblock 输入目录:[/bold cyan] {timeblock_dir}")
    print(f"[bold cyan]输出目录:[/bold cyan] {output_dir}")

    timeblock_files = sorted(
        timeblock_dir.glob("*_timeblock.json"),
        key=lambda p: sort_key_for_doc_key(extract_doc_key_from_filename(p, "timeblock"))
    )

    if not timeblock_files:
        raise FileNotFoundError(f"在 {timeblock_dir} 下未找到 *_timeblock.json 文件。")

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
        payload = load_json(tb_file)

        if not isinstance(payload, dict) or "TMB" not in payload or not isinstance(payload["TMB"], list):
            raise TypeError(f"{tb_file} 结构不符合预期，应为 {{'TMB': [...]}}")

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

    print(f"[bold green]待处理 timeblock 数量:[/bold green] {len(tasks)}")
    if SKIP_EXISTING_SUMMARY:
        print(f"[bold yellow]跳过已有 summary 数量:[/bold yellow] {skipped_existing}")

    if not tasks:
        print("[bold yellow]没有需要处理的对象。[/bold yellow]")
        for tb_file, payload in payloads_by_file.items():
            out_path = output_dir / tb_file.name
            save_json(payload, out_path)
        print("[bold green]已完成。[/bold green]")
        return

    success_count = 0
    fail_results = []

    pbar = tqdm(total=len(tasks), desc="生成 timeblock summary")

    for batch_tasks in batched(tasks, BATCH_SIZE):
        max_workers = min(BATCH_SIZE, len(batch_tasks))

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

    pbar.close()

    # 保存到 step14output，文件名保持原样
    for tb_file, payload in payloads_by_file.items():
        out_path = output_dir / tb_file.name
        save_json(payload, out_path)

    print(f"[bold green]成功写入 summary 数量:[/bold green] {success_count}")
    print(f"[bold cyan]输出文件已保存到:[/bold cyan] {output_dir}")

    if fail_results:
        print(f"[bold red]失败数量:[/bold red] {len(fail_results)}")
        print("[bold red]失败明细：[/bold red]")
        for item in fail_results[:20]:
            print(
                f"  - 文件: {item['timeblock_file'].name} | "
                f"index: {item['obj_index']} | "
                f"range: {item['range_str']} | "
                f"错误: {item['error']}"
            )
        if len(fail_results) > 20:
            print(f"  ... 其余 {len(fail_results) - 20} 条失败未展开。")
    else:
        print("[bold green]全部处理完成，没有失败项。[/bold green]")


if __name__ == "__main__":
    main()
