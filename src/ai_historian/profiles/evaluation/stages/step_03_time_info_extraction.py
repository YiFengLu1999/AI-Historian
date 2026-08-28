# ===========================================
# step2output -> step3output 批量并发信息抽取
# 每批并发10条 API
# ===========================================

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from pydantic import BaseModel
from rich import print

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion_async,
    make_async_chat_client,
    validate_json_text,
)
from ai_historian.pipeline.logging import StepReporter, setup_step_logging, step_tqdm
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir

# -------- 0. Runtime state --------
client = None

MODEL_NAME = CHAT_MODEL
BATCH_SIZE = int(os.getenv("AIH_PIPELINE_BATCH_SIZE", "40"))
MAX_RETRY  = 5
BASE_BACKOFF = 1.5

# -------- 1. Runtime paths --------
RUN_ROOT: Path
INPUT_DIR: Path
OUTPUT_DIR: Path

# -------- 3. Pydantic schema --------
class EA(BaseModel):
    exist: bool
    OTI: str

# -------- 4. prompt 构造 --------
def build_prompt(sentence: str) -> str:

    return f"""
你是一个用于信息抽取的助手。
请阅读以下句子，并根据规则判断是否包含 **时间信息原文**：

规则：
- 时间信息原文是能够明确指示事件发生或动作施行具体时间点的名词性或介词性短语。
  例如：“汉十二年的秋天”、“做亭长时”、“秦二世二年”、“正月”。
- 如果句子中只有“最初”、“后来”、“曾经”、“当时”等模糊的时间表达，则不算时间信息原文。
- 如果有明确时间信息原文，则返回 exist=True，并在 OTI 字段写出时间信息原文。
- 如果没有明确时间信息原文，则返回 exist=False，OTI 为空字符串。

请严格按照以下 JSON Schema 返回结果：
{EA.model_json_schema()}

句子是：
"{sentence}"
""".strip()

# -------- 5. 单句调用 --------
async def call_llm_for_sentence(sentence: str) -> EA:
    if client is None:
        raise RuntimeError("Step 3 client has not been initialized")

    msgs = [
        {"role": "system", "content": "你是一个帮助用户做信息抽取的助手。"},
        {"role": "user", "content": build_prompt(sentence)},
    ]

    for attempt in range(1, MAX_RETRY + 1):

        try:
            rsp = await create_chat_completion_async(
                client,
                model=MODEL_NAME,
                messages=msgs,
                response_format={"type": "json_object"},
            )

            content = rsp.choices[0].message.content
            return validate_json_text(EA, content)

        except Exception as e:

            if attempt == MAX_RETRY:
                raise RuntimeError(f"多次重试仍失败: {repr(e)}")

            backoff = (BASE_BACKOFF ** attempt) + random.uniform(0, 0.5)

            print(f"[red]请求失败，第{attempt}次重试，等待 {backoff:.2f} 秒")
            await asyncio.sleep(backoff)

# -------- 6. 批处理 --------
async def process_batch(batch_items):

    tasks = [
        asyncio.create_task(call_llm_for_sentence(item["sentence"]))
        for item in batch_items
    ]

    results = await asyncio.gather(*tasks)

    return results

# -------- 7. 单文件 pipeline --------
async def process_file(input_path: Path):

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    pbar = step_tqdm(total=total, desc=f"Processing {input_path.name}")

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        sub_data = data[start:end]

        batch_results = await process_batch(sub_data)

        for item, ea_result in zip(sub_data, batch_results):

            if "Original_time_information" not in item or item["Original_time_information"] is None:
                item["Original_time_information"] = {}

            item["Original_time_information"]["exist"] = ea_result.exist
            item["Original_time_information"]["OTI"]   = ea_result.OTI

            pbar.update(1)

    pbar.close()

    # 输出文件路径（保持原名）
    output_path = OUTPUT_DIR / input_path.name

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[green]完成: {input_path.name}")

# -------- 8. 总流程 --------
async def main_pipeline():
    files = sorted(INPUT_DIR.glob("*.json"))
    reporter = StepReporter("Step3", total=len(files))
    reporter.start(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        extra=f"并发批次={BATCH_SIZE}",
    )

    for file in files:
        try:
            await process_file(file)
            reporter.item_ok(file.name)
        except Exception as e:
            reporter.item_fail(file.name, e)
            raise SystemExit(1)

    reporter.finish(output_dir=OUTPUT_DIR)

def main() -> None:
    global client, RUN_ROOT, INPUT_DIR, OUTPUT_DIR

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    INPUT_DIR = sentence_step_dir(RUN_ROOT, 2)
    OUTPUT_DIR = sentence_step_dir(RUN_ROOT, 3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_step_logging(RUN_ROOT, "step_03_time_info_extraction")
    client = make_async_chat_client()
    asyncio.run(main_pipeline())


if __name__ == "__main__":
    main()
