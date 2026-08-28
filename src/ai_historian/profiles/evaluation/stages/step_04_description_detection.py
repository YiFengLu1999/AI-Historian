# ===========================================
# step3output -> step4output
# 批并发版句子下沉类型判定（描述性语句识别）
# - 处理 sentence/step3output 下的每一个 json 文件
# - 输出到 sentence/step4output
# - 输出文件名保持不变
# - 每一批并发 40 个 API 请求
# - 带重试 / 退避
# - 记录总运行时间
# - 独立 CLI 运行
# ===========================================

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel
from rich import print
from tqdm.auto import tqdm

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion_async,
    make_async_chat_client,
)
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir

# =========================================================
# 0. Runtime state
# =========================================================
RUN_ROOT: Path
INPUT_DIR: Path
OUTPUT_DIR: Path
async_client = None

# =========================================================
# 1. 运行参数配置
# =========================================================
MODEL_NAME = CHAT_MODEL

BATCH_SIZE   = int(os.getenv("AIH_AGENT_BATCH_SIZE", os.getenv("AIH_AGENT_CONCURRENCY", "40")))
MAX_RETRY    = 5       # 单条请求最大重试次数
BASE_BACKOFF = 1.5     # 指数退避基数

# =========================================================
# 4. 输出结构模型
# =========================================================
class EA(BaseModel):
    Is_it_sinking: bool
    reason: str

# =========================================================
# 5. 构造 messages
# =========================================================
def build_messages(sentence_text: str) -> List[Dict[str, str]]:
    system_msg = {
        "role": "system",
        "content": (
            "你是一个研究历史文献的历史学家，擅长从文献中分析句子的时间属性。"
            "用户会提供一句历史文献原文、翻译或整理后的句子，你需要根据这句话的内容判断它的时间属性类型是否是描述性语句。\n\n"
            "你必须严格按照以下规则进行判断，并给出判断理由：\n\n"
            "描述性语句：句子主要用于状态描述或背景介绍，虽可能包含动词，但不指明具体时间点或持续行为，"
            "比如“此人为人清廉”。\n\n"
            "有一种情况是，虽然你已经判断出这是一个描述性语句，但是这里面有时间信息，比如：这时、某年春天、某帝元年，"
            "我们就认为他不是描述性信息。\n\n"
            "你必须输出一个格式严格符合 JSON 的对象：\n"
            "{\n"
            "  \"Is_it_sinking\": true 或 false,\n"
            "  \"reason\": \"你为什么作出这样的判断？\"\n"
            "}\n"
            "只返回这个 JSON 格式，不要额外解释或输出。"
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            "请判断下面这个句子是否是描述性语句，并给出理由。\n"
            "输出格式必须为 JSON，例如：\n"
            "{\n"
            "  \"Is_it_sinking\": true,\n"
            "  \"reason\": \"因为句中没有任何动词，所以这句话仅仅起一个描述性作用，所以判定为 true。\"\n"
            "}\n"
            "不要有其他内容，确保 JSON 格式闭合，也不要加 ```json 或其他 markdown 代码块标记。\n\n"
            f"句子是：{sentence_text}"
        )
    }

    return [system_msg, user_msg]

# =========================================================
# 6. 从对象中尽量稳健地提取句子
#    默认优先找 obj["sentence"]
# =========================================================
def extract_sentence(obj: Dict[str, Any]) -> str:
    """
    尽量从常见字段中取句子。
    如果你的 step3 数据结构就是 {"sentence": "..."}，这里会直接命中。
    """
    candidate_keys = [
        "sentence",
        "text",
        "content",
        "句子",
        "原文",
        "翻译",
    ]
    for key in candidate_keys:
        if key in obj and isinstance(obj[key], str):
            return obj[key].strip()

    raise KeyError(
        f"当前对象中找不到句子字段。可用字段有: {list(obj.keys())}"
    )

# =========================================================
# 7. 清理模型输出
# =========================================================
def clean_json_text(content: str) -> str:
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    # 有些模型会吐点前后废料，粗暴截取第一个 { 到最后一个 }
    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last != -1 and first < last:
        content = content[first:last + 1]

    # 偶发缺右花括号，补一次
    if content.startswith("{") and not content.endswith("}"):
        content += "}"

    return content

# =========================================================
# 8. 单句请求（带重试）
# =========================================================
async def call_llm_for_sentence(sentence: str) -> EA:
    if async_client is None:
        raise RuntimeError("Step 4 client has not been initialized")
    msgs = build_messages(sentence)

    for attempt in range(1, MAX_RETRY + 1):
        try:
            rsp = await create_chat_completion_async(
                async_client,
                model=MODEL_NAME,
                messages=msgs,
                stream=False,
                temperature=0
            )

            content = rsp.choices[0].message.content or ""
            content = clean_json_text(content)

            parsed = EA.model_validate_json(content)
            return parsed

        except Exception as e:
            if attempt == MAX_RETRY:
                raise RuntimeError(f"多次重试仍失败: {repr(e)}")

            backoff = (BASE_BACKOFF ** attempt) + random.uniform(0, 0.6)
            print(
                f"[red]单句请求失败，第 {attempt} 次重试，"
                f"等待 {backoff:.2f} 秒。错误: {repr(e)}"
            )
            await asyncio.sleep(backoff)

# =========================================================
# 9. 处理一个 batch（并发 40 个）
# =========================================================
async def process_batch(batch_objs: List[Dict[str, Any]]) -> List[EA]:
    tasks = []
    for obj in batch_objs:
        sentence_text = extract_sentence(obj)
        tasks.append(asyncio.create_task(call_llm_for_sentence(sentence_text)))

    results = await asyncio.gather(*tasks)
    return results

# =========================================================
# 10. 处理单个文件
# =========================================================
async def process_one_file(input_file: Path, output_file: Path):
    file_start_time = time.time()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{input_file.name} 的顶层不是 list，而是 {type(data)}")

    updated_data = []

    pbar = tqdm(
        total=len(data),
        desc=f"处理 {input_file.name}",
        unit="条"
    )

    for start in range(0, len(data), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(data))
        batch = data[start:end]

        try:
            batch_results = await process_batch(batch)
        except Exception as e:
            print(f"[red]❌ 文件 {input_file.name} 中 batch {start}:{end} 处理失败: {e}")
            for obj in batch:
                obj["sink"] = {
                    "Is_it_sinking": False,
                    "reason": f"批处理失败，未能解析：{str(e)}"
                }
                updated_data.append(obj)
                pbar.update(1)
            continue

        for obj, parsed in zip(batch, batch_results):
            obj["sink"] = {
                "Is_it_sinking": parsed.Is_it_sinking,
                "reason": parsed.reason
            }
            updated_data.append(obj)
            pbar.update(1)

    pbar.close()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)

    elapsed = time.time() - file_start_time
    print(f"[green]✅ 完成文件: {input_file.name} -> {output_file.name}，耗时 {elapsed:.1f} 秒")

async def main() -> None:
    global async_client, RUN_ROOT, INPUT_DIR, OUTPUT_DIR

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    INPUT_DIR = sentence_step_dir(RUN_ROOT, 3)
    OUTPUT_DIR = sentence_step_dir(RUN_ROOT, 4)
    async_client = make_async_chat_client()

    print(f"[cyan]当前模型: {MODEL_NAME}")
    print(f"[cyan]输入目录: {INPUT_DIR}")
    print(f"[cyan]输出目录: {OUTPUT_DIR}")
    print(f"[magenta]并发批大小: {BATCH_SIZE}")

    total_start_time = time.time()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    json_files = sorted(INPUT_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"在 {INPUT_DIR} 下没有找到任何 .json 文件")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[blue]共发现 {len(json_files)} 个 json 文件，开始处理...")

    success_count = 0
    failed_count = 0
    failed_files = []

    for json_file in json_files:
        out_file = OUTPUT_DIR / json_file.name
        try:
            await process_one_file(json_file, out_file)
            success_count += 1
        except Exception as e:
            failed_count += 1
            failed_files.append((json_file.name, str(e)))
            print(f"[red]❌ 文件处理失败: {json_file.name}")
            print(f"[red]错误信息: {e}")

    total_elapsed = time.time() - total_start_time
    h, rem = divmod(total_elapsed, 3600)
    m, s = divmod(rem, 60)

    print("\n[bold green]🎉 全部处理结束")
    print(f"[cyan]成功文件数: {success_count}")
    print(f"[red]失败文件数: {failed_count}")
    print(f"[magenta]输出目录: {OUTPUT_DIR}")
    print(f"[magenta]并发设置: 每批 {BATCH_SIZE} 个 API 请求")
    print(f"[cyan]总运行时间: {int(h)} 小时 {int(m)} 分 {s:.1f} 秒")

    if failed_files:
        print("\n[bold red]失败文件列表：")
        for fname, err in failed_files:
            print(f"[red]- {fname}: {err}")


if __name__ == "__main__":
    asyncio.run(main())
