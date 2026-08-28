from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion_async,
    make_async_chat_client,
)
from ai_historian.pipeline.logging import StepReporter, setup_step_logging, step_tqdm
from ai_historian.pipeline.paths import resolve_run_root, timeblock_step_dir

# =======================
# 配置
# =======================
RUN_ROOT: Path
INPUT_DIR: Path
OUTPUT_DIR: Path

MODEL = CHAT_MODEL
MAX_CONCURRENCY = int(os.getenv("AIH_PIPELINE_CONCURRENCY", "20"))
MAX_RETRIES = 6
BASE_BACKOFF = 0.8

# 是否做范围字段的格式检查（只警告，不阻断）
CHECK_RANGE_FORMAT = True

# =======================
# DeepSeek SDK 检查
# =======================
aclient = None


def get_async_client():
    if aclient is None:
        raise RuntimeError("LLM client is not initialized; call main() first")
    return aclient

# =======================
# Pydantic v1 / v2 兼容
# =======================
def pydantic_validate(cls, obj: Any):
    if hasattr(cls, "model_validate"):   # pydantic v2
        return cls.model_validate(obj)
    return cls.parse_obj(obj)            # pydantic v1

# =======================
# 只用于验证 LLM 输出
# =======================
class GranularityOut(BaseModel):
    Granularity: str

def validate_granularity_payload(data: dict) -> str:
    out = pydantic_validate(GranularityOut, data)
    if out.Granularity not in {"0", "1", "2", "3"}:
        raise ValueError("Granularity 必须是 {'0','1','2','3'} 之一")
    return out.Granularity

# =======================
# 文件名 / 编号 / range 解析
# 兼容新格式：
# 1) 文件名：篇章id_uuid_文件属性.json
#    例：7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_timeblock.json
#
# 2) number / ISO 点位：
#    uuid.篇章id.段落id.句子id
#    例：94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
#
# 3) range：
#    uuid.篇章id.段落id.句子id-uuid.篇章id.段落id.句子id
#    例：94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1-94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
# =======================
FILE_RE = re.compile(
    r"^(?P<chapter_id>\d+)_(?P<book_uuid>[0-9a-fA-F-]+)_(?P<kind>sentence|timeblock|sequence)\.json$"
)

POINT_ID_RE = re.compile(
    r"^(?P<book_uuid>[0-9a-fA-F-]+)\.(?P<chapter_id>\d+)\.(?P<paragraph_id>\d+)\.(?P<sentence_id>\d+)$"
)

RANGE_RE = re.compile(
    r"^(?P<start>[0-9a-fA-F-]+\.\d+\.\d+\.\d+)-(?P<end>[0-9a-fA-F-]+\.\d+\.\d+\.\d+)$"
)

def parse_filename_meta(filename: str) -> Dict[str, str]:
    """
    解析文件名：篇章id_uuid_属性.json
    """
    m = FILE_RE.match(filename)
    if not m:
        raise ValueError(f"文件名不符合新规则：{filename}")
    return m.groupdict()

def corresponding_sentence_filename(timeblock_filename: str) -> str:
    """
    根据 timeblock 文件名推导对应的 sentence 文件名
    例：
    7_xxx_timeblock.json -> 7_xxx_sentence.json
    """
    meta = parse_filename_meta(timeblock_filename)
    return f"{meta['chapter_id']}_{meta['book_uuid']}_sentence.json"

def parse_point_id(text: str) -> Optional[Dict[str, Any]]:
    """
    解析单个点位：
    uuid.chapter.paragraph.sentence
    """
    text = (text or "").strip()
    m = POINT_ID_RE.match(text)
    if not m:
        return None
    d = m.groupdict()
    return {
        "book_uuid": d["book_uuid"],
        "chapter_id": int(d["chapter_id"]),
        "paragraph_id": int(d["paragraph_id"]),
        "sentence_id": int(d["sentence_id"]),
    }

def parse_point_range(text: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    解析范围：
    uuid.chapter.paragraph.sentence-uuid.chapter.paragraph.sentence

    注意：
    不能用 split("-")，因为 uuid 里本身带 '-'
    """
    text = (text or "").strip()
    m = RANGE_RE.match(text)
    if not m:
        return None

    start = parse_point_id(m.group("start"))
    end = parse_point_id(m.group("end"))
    if start is None or end is None:
        return None

    return {"start": start, "end": end}

# =======================
# JSON 工具函数
# =======================
def load_one_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} 顶层不是 dict")
    if "TMB" not in data:
        raise ValueError(f"{path.name} 缺少顶层字段 'TMB'")
    if not isinstance(data["TMB"], list):
        raise ValueError(f"{path.name} 的 'TMB' 不是 list")

    return data

def load_folder(folder: Path) -> Dict[str, Dict[str, Any]]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"找不到目录：{folder.resolve()}")

    files = sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"目录里没有 .json 文件：{folder.resolve()}")

    out: Dict[str, Dict[str, Any]] = {}
    for p in files:
        # 文件名规则检查
        meta = parse_filename_meta(p.name)
        if meta["kind"] != "timeblock":
            raise ValueError(f"{p.name} 不是 timeblock 文件，但它出现在 {folder}")

        out[p.name] = load_one_json(p)

    return out

def get_conversion_info(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容几种可能的写法，但优先使用你当前流水线里的：
    'Conversion information'
    """
    if "Conversion information" in item and isinstance(item["Conversion information"], dict):
        return item["Conversion information"]
    if "Conversion_information" in item and isinstance(item["Conversion_information"], dict):
        return item["Conversion_information"]
    return {}

def pick_time_text(item: Dict[str, Any]) -> str:
    """
    优先取 converted，若为空则退回 original
    """
    ci = get_conversion_info(item)
    conv = str(ci.get("time_information_converted") or "").strip()
    orig = str(ci.get("time_information_original") or "").strip()
    return conv if conv else orig

def collect_range_warnings(filename: str, data: Dict[str, Any]) -> List[str]:
    """
    只做温和检查，不阻断主流程。
    """
    warnings = []
    meta = parse_filename_meta(filename)
    tmb_items = data.get("TMB", [])

    for idx, item in enumerate(tmb_items, start=1):
        for field_name in ["timeblock_range", "timeblockrange", "ISO_range", "isorange"]:
            value = item.get(field_name)
            if not value:
                continue

            parsed = parse_point_range(str(value).strip())
            if parsed is None:
                warnings.append(
                    f"{filename} | TMB[{idx}] | 字段 {field_name} 无法按新规则解析：{value}"
                )
                continue

            # 检查 range 起点是否与文件名里的 uuid / chapter 对齐
            start = parsed["start"]
            if (
                start["book_uuid"] != meta["book_uuid"]
                or start["chapter_id"] != int(meta["chapter_id"])
            ):
                warnings.append(
                    f"{filename} | TMB[{idx}] | 字段 {field_name} 与文件名不一致：{value}"
                )

    return warnings

# =======================
# LLM 分类
# =======================
SYSTEM_PROMPT = """你是一个时间表达精度分类器。你只根据给定的“时间表达字符串”本身判断，不引入外部上下文推断。

只输出一个 JSON 对象，严格为：
{"Granularity":"0"} 或 {"Granularity":"1"} 或 {"Granularity":"2"} 或 {"Granularity":"3"}

分类规则：
- 3：具备明确无歧义的“年-月-日”信息（如：2020年3月5日 / 公元前206年十月初一 等明确到日）。
- 2：仅到“年-月”层级（如：2020年3月；或“某帝某年三月”这种明确到月）。
- 1：仅到“年”层级（包含：年号/帝王纪年/某某元年/某年；即使带“春夏秋冬/上半年”等但没有明确月份，也仍按年份级）。
- 0：缺乏明确时间锚点或仅相对/模糊描述（如：这时、刚才、两天后、夜里、年少时、年终时、某日、春天、正月(无年份)）。

补充：
- 只有“月/日/季节/昼夜”等但缺少年份锚点 -> 0
- “年号/帝王年/元年/某年” -> 1
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)

def extract_json_from_text(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError(f"无法从模型输出中提取 JSON：{text[:200]}")
        return json.loads(m.group(0))

_cache: Dict[str, str] = {}

async def classify_expr(expr: str, sem: asyncio.Semaphore) -> str:
    expr = (expr or "").strip()
    if not expr:
        return "0"

    if expr in _cache:
        return _cache[expr]

    async with sem:
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await create_chat_completion_async(
                    get_async_client(),
                    model=MODEL,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"时间表达：{expr}\n只输出 JSON。"},
                    ],
                )
                text = resp.choices[0].message.content
                data = extract_json_from_text(text)
                granularity = validate_granularity_payload(data)
                _cache[expr] = granularity
                return granularity

            except Exception as e:
                last_err = e
                backoff = BASE_BACKOFF * (2 ** (attempt - 1)) + random.random() * 0.2
                await asyncio.sleep(backoff)

        raise RuntimeError(f"分类失败（重试耗尽）：{expr!r}；最后错误={repr(last_err)}")

async def classify_many(exprs: List[str]) -> Dict[str, str]:
    uniq = sorted({(e or "").strip() for e in exprs if (e or "").strip()})
    if not uniq:
        return {}

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _one(e: str):
        g = await classify_expr(e, sem)
        return e, g

    tasks = [asyncio.create_task(_one(e)) for e in uniq]
    mapping: Dict[str, str] = {}

    try:
        for fut in step_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Granularity"):
            e, g = await fut
            mapping[e] = g
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return mapping

# =======================
# 主流程
# =======================
async def main():
    global RUN_ROOT, INPUT_DIR, OUTPUT_DIR, aclient

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    INPUT_DIR = timeblock_step_dir(RUN_ROOT, 7)
    OUTPUT_DIR = timeblock_step_dir(RUN_ROOT, 9)
    setup_step_logging(RUN_ROOT, "step_09_granularity_classification")
    aclient = make_async_chat_client()

    loaded = load_folder(INPUT_DIR)

    reporter = StepReporter("Step9", total=len(loaded))
    reporter.start(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

    # 可选：检查 range / isorange 是否符合新规则
    all_warnings: List[str] = []
    if CHECK_RANGE_FORMAT:
        for fname, data in loaded.items():
            all_warnings.extend(collect_range_warnings(fname, data))

    if all_warnings:
        print(f"\n⚠️ 范围字段警告 {len(all_warnings)} 条（不影响本轮 Granularity 生成）")
        for w in all_warnings[:20]:
            print(" -", w)
        if len(all_warnings) > 20:
            print(f" - 其余 {len(all_warnings) - 20} 条已省略")

    # 1) 收集时间表达
    exprs_all: List[str] = []
    total_items = 0

    for fname, data in loaded.items():
        for item in data["TMB"]:
            total_items += 1
            expr = pick_time_text(item)
            if expr:
                exprs_all.append(expr)
            else:
                item["Granularity"] = "0"

    uniq_expr_count = len({e.strip() for e in exprs_all if e.strip()})
    reporter.info(f"TMB={total_items} 非空时间表达={len(exprs_all)} 去重表达={uniq_expr_count}")

    # 2) LLM 批量分类
    mapping = await classify_many(exprs_all)

    # 3) 写回 Granularity
    granularity_counter = Counter()

    for fname, data in loaded.items():
        for item in data["TMB"]:
            expr = pick_time_text(item)
            g = "0" if not expr.strip() else mapping[expr]
            item["Granularity"] = g
            granularity_counter[g] += 1

    # 4) 写出到 timeblock/step9output，保持原始文件名
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname, data in loaded.items():
        out_path = OUTPUT_DIR / fname
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        reporter.item_ok(fname)

    reporter.finish(
        output_dir=OUTPUT_DIR,
        extra=" ".join(f"{k}={granularity_counter.get(k, 0)}" for k in ["0", "1", "2", "3"]),
    )

    return {
        "file_count": len(loaded),
        "item_count": total_items,
        "unique_expr_count": uniq_expr_count,
        "granularity_counter": dict(granularity_counter),
        "warnings": all_warnings,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "corresponding_sentence_examples": {
            fname: corresponding_sentence_filename(fname)
            for fname in list(loaded.keys())[:5]
        },
    }

if __name__ == "__main__":
    asyncio.run(main())
