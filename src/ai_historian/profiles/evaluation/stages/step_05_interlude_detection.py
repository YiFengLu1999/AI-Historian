# === Step5 Interlude 批处理流水线 ===
# 功能：
# 1. 处理 sentence/step4output 里的每一个 json 文件
# 2. 输出到 sentence/step5output
# 3. 输出文件名保持原始名称不变
# 4. 文件内部顺序判定（因为有前后依赖）
# 5. 文件之间并发处理，最大并发数 = 40
# 6. 使用统一模型配置和 client factory

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel
from rich import print
from tqdm.auto import tqdm

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion,
    make_sync_chat_client,
)
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir

# =========================================================
# 可配置区域
# =========================================================
CTX_PREV = 80
CTX_NEXT = 80

MODEL_NAME = CHAT_MODEL

# 并发数：最多同时处理 40 个文件
MAX_WORKERS = int(os.getenv("AIH_AGENT_MAX_WORKERS", os.getenv("AIH_AGENT_CONCURRENCY", "40")))

# 单次请求的最大 token
MAX_TOKENS = 64

# 重试次数
MAX_RETRIES = 6


# =========================================================
# 输出约束
# =========================================================
class EA(BaseModel):
    Interlude: bool


# =========================================================
# 插叙定义
# =========================================================
INTERLUDE_DEF = (
    "“插叙”指在连续叙述过程中，暂时中断当前时间线的推进，插入一个或一组与当前叙述时间不一致的内容。"
    "插叙可能回溯过去（倒叙）、提前描述未来（预叙），也可能是补充背景或人物经历；其共同特征是使叙事时间结构短暂偏离或中断。"
    "例如：“多年以后，人们才知道他早年曾在边郡避难。”是插叙。"
    "特别注意：我们不分析某人说出的内容，也就是某人说“……”中的引语不作为插叙判定依据；"
    "出现“某人说：‘……’（或“……”）”这一类直接引语，直接判定其不是插叙。"
)


# =========================================================
# Chat client：每个工作线程独立创建
# =========================================================
_thread_local = threading.local()

def _get_thread_client():
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = make_sync_chat_client()
        _thread_local.client = client
    return client


# =========================================================
# LLM 调用：强制 JSON 输出 + 退避重试
# =========================================================
def _chat_json(
    messages: List[Dict[str, str]],
    model: str = MODEL_NAME,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    retries: int = MAX_RETRIES,
) -> str:
    delay = 1.0

    for attempt in range(retries):
        try:
            resp = create_chat_completion(
                _get_thread_client(),
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content

        except Exception as e:
            msg = str(e).lower()
            retriable = any(
                k in msg for k in [
                    "rate limit", "timeout", "overloaded", "temporarily",
                    "503", "500", "connection", "server error"
                ]
            )

            if (not retriable) or (attempt == retries - 1):
                raise

            sleep_s = delay * (1.5 + 0.5 * (time.time() % 1))
            print(f"[yellow]LLM 调用重试 {attempt + 1}/{retries}：{e}，sleep={sleep_s:.2f}s[/yellow]")
            time.sleep(sleep_s)
            delay *= 1.8


# =========================================================
# 解析 JSON 输出
# =========================================================
def _parse_ea(json_text: str) -> bool:
    try:
        data = json.loads(json_text)
        return EA(**data).Interlude
    except Exception:
        m = re.search(r"\{.*\}", json_text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                return EA(**data).Interlude
            except Exception:
                pass
        raise ValueError(f"无法解析为 EA: {json_text[:200]}")


# =========================================================
# Prompt 构造
# =========================================================
def _messages_ctx(
    prev_text: str,
    next_text: str,
    cur_sent: str,
    prev_window: int,
    next_window: int,
) -> List[Dict[str, str]]:
    sys = "你是一个严格的中文叙事分析器，只输出JSON，不要多余文字。"
    usr = (
        "任务：判断“当前句子”是否构成插叙（Interlude）。\n"
        f"插叙定义：{INTERLUDE_DEF}\n\n"
        f"前文（最多{prev_window}句，供参考）：\n{prev_text}\n\n"
        f"后文（最多{next_window}句，供参考）：\n{next_text}\n\n"
        f"当前句子：{cur_sent}\n\n"
        '只输出一个 JSON 对象，格式精确为：{"Interlude": true} 或 {"Interlude": false}'
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]


def _messages_continuity(block_text: str, next_text: str, cur_sent: str) -> List[Dict[str, str]]:
    sys = "你是一个严格的中文叙事分析器，只输出JSON，不要多余文字。"
    usr = (
        "任务：判断“当前句子”与“前文插叙片段”是否属于同一个事件的连续叙述。\n"
        "若连续，视为仍处于插叙（Interlude=true）；若不连续，则 Interlude=false。\n"
        f"插叙定义：{INTERLUDE_DEF}\n\n"
        f"前文插叙片段（已连贯的 Interlude 片段）：\n{block_text}\n\n"
        f"后文（供参考）：\n{next_text}\n\n"
        f"当前句子：{cur_sent}\n\n"
        '只输出一个 JSON 对象，格式精确为：{"Interlude": true} 或 {"Interlude": false}'
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]


# =========================================================
# 上下文工具
# =========================================================
def _concat_prev_sentences(records: List[Dict[str, Any]], idx: int, limit: int) -> str:
    start = max(0, idx - limit)
    prev = [r.get("sentence", "") for r in records[start:idx]]
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(prev))


def _concat_next_sentences(records: List[Dict[str, Any]], idx: int, limit: int) -> str:
    end = min(len(records), idx + 1 + limit)
    nxt = [records[k].get("sentence", "") for k in range(idx + 1, end)]
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(nxt))


def _contiguous_true_block(records: List[Dict[str, Any]], flags: List[bool], idx: int) -> str:
    """
    给定当前 idx（其前一个已是 True），向前回溯找最近的 False+1 到 idx-1 之间的 True 连续片段，
    拼接 sentence 为“前文插叙片段”
    """
    j = idx - 1
    if j < 0 or not flags[j]:
        return ""

    start = j
    while start - 1 >= 0 and flags[start - 1]:
        start -= 1

    seg = [records[k].get("sentence", "") for k in range(start, j + 1)]
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(seg))


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{m:02d}m{s:04.1f}s"


# =========================================================
# 单个文件处理
# 说明：
# - 单文件内部必须顺序跑
# - 因为 inter_flags[i] 依赖 inter_flags[i-1]
# =========================================================
def detect_interludes_single_file(
    input_json: Path,
    output_json: Path,
    model: str = MODEL_NAME,
    prev_window: int = CTX_PREV,
    next_window: int = CTX_NEXT,
) -> Path:
    t0 = time.time()

    data = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"输入 JSON 顶层必须是数组(List[Object])：{input_json}")

    n = len(data)
    inter_flags: List[bool] = [False] * n

    for i in range(n):
        obj = data[i] if isinstance(data[i], dict) else {}
        sent = obj.get("sentence", "")
        sink = obj.get("sink", {}) or {}
        oti = obj.get("Original_time_information", {}) or {}

        is_sink = bool(sink.get("Is_it_sinking", False))
        exist_time = bool(oti.get("exist", False))
        prev_inter = inter_flags[i - 1] if i > 0 else False

        # 规则 1：描述性沉底 => False
        if is_sink:
            inter_flags[i] = False
            data[i]["Interlude"] = False
            continue

        # 规则 2：无时间信息
        if not exist_time:
            if not prev_inter:
                inter_flags[i] = False
                data[i]["Interlude"] = False
            else:
                block_text = _contiguous_true_block(data, inter_flags, i)
                next_text = _concat_next_sentences(data, i, limit=next_window)
                try:
                    messages = _messages_continuity(block_text, next_text, sent)
                    resp = _chat_json(
                        messages,
                        model=model,
                        max_tokens=MAX_TOKENS,
                        temperature=0.0,
                    )
                    inter_flags[i] = _parse_ea(resp)
                except Exception as e:
                    print(
                        f"[red]LLM/解析失败，默认 False[/red] "
                        f"@ file={input_json.name}, idx={i}, num={obj.get('number', '?')}：{e}"
                    )
                    inter_flags[i] = False

                data[i]["Interlude"] = bool(inter_flags[i])
            continue

        # 规则 3：有时间信息
        prev_text = _concat_prev_sentences(data, i, limit=prev_window)
        next_text = _concat_next_sentences(data, i, limit=next_window)

        try:
            messages = _messages_ctx(prev_text, next_text, sent, prev_window, next_window)
            resp = _chat_json(
                messages,
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            inter_flags[i] = _parse_ea(resp)
        except Exception as e:
            print(
                f"[red]LLM/解析失败，默认 False[/red] "
                f"@ file={input_json.name}, idx={i}, num={obj.get('number', '?')}：{e}"
            )
            inter_flags[i] = False

        data[i]["Interlude"] = bool(inter_flags[i])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.time() - t0
    print(f"[green]完成[/green] {input_json.name} -> {output_json}  |  {n}条  |  {_fmt_dur(elapsed)}")
    return output_json


# =========================================================
# 批量处理目录
# 说明：
# - 处理 sentence/step4output/*.json
# - 输出到 sentence/step5output/*.json
# - 文件名保持不变
# - 并发上限 40
# =========================================================
def batch_detect_interludes(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    model: str = MODEL_NAME,
    prev_window: int = CTX_PREV,
    next_window: int = CTX_NEXT,
    max_workers: int = MAX_WORKERS,
) -> List[Path]:
    t0 = time.time()

    if input_dir is None or output_dir is None:
        run_root = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
        input_dir = input_dir or sentence_step_dir(run_root, 4)
        output_dir = output_dir or sentence_step_dir(run_root, 5)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"输入目录下没有 json 文件：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[blue]输入目录：[/blue]{input_dir}")
    print(f"[blue]输出目录：[/blue]{output_dir}")
    print(f"[blue]文件数量：[/blue]{len(json_files)}")
    print(f"[blue]上下文窗口：[/blue]prev={prev_window}, next={next_window}")
    print(f"[blue]最大并发文件数：[/blue]{max_workers}")
    print("[yellow]说明：单个文件内部必须顺序处理；并发仅针对多个文件之间。[/yellow]")

    results: List[Path] = []
    errors: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}

        for input_json in json_files:
            output_json = output_dir / input_json.name
            future = executor.submit(
                detect_interludes_single_file,
                input_json=input_json,
                output_json=output_json,
                model=model,
                prev_window=prev_window,
                next_window=next_window,
            )
            future_to_file[future] = input_json.name

        for future in tqdm(
            as_completed(future_to_file),
            total=len(future_to_file),
            desc="处理 step4output 全部文件",
        ):
            fname = future_to_file[future]
            try:
                out_path = future.result()
                results.append(out_path)
            except Exception as e:
                errors.append((fname, str(e)))
                print(f"[red]文件处理失败[/red] {fname}: {e}")

    elapsed = time.time() - t0

    print("\n[bold cyan]批处理完成[/bold cyan]")
    print(f"[cyan]成功：[/cyan]{len(results)}")
    print(f"[cyan]失败：[/cyan]{len(errors)}")
    print(f"[cyan]总耗时：[/cyan]{_fmt_dur(elapsed)}")

    if errors:
        print("\n[bold red]失败文件列表[/bold red]")
        for fname, err in errors:
            print(f"[red]- {fname}[/red] -> {err}")

    return sorted(results)


def main() -> None:
    run_root = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    result_files = batch_detect_interludes(
        input_dir=sentence_step_dir(run_root, 4),
        output_dir=sentence_step_dir(run_root, 5),
        model=MODEL_NAME,
        prev_window=CTX_PREV,
        next_window=CTX_NEXT,
        max_workers=MAX_WORKERS,
    )

    print("\n[bold green]输出文件：[/bold green]")
    for p in result_files:
        print(p)


if __name__ == "__main__":
    main()
