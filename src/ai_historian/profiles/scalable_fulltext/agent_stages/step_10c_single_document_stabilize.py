import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion,
    make_sync_chat_client,
)
from ai_historian.pipeline.logging import StepReporter, setup_step_logging, step_tqdm
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir, timeblock_step_dir
from ai_historian.pipeline.time_canonicalizer import normalize_experiment1_tm

TIMEBLOCK_STEP = int(os.getenv("AIH_ANCHOR_STABILIZE_TIMEBLOCK_STEP", "10"))
RUN_ROOT: Path
TIMEBLOCK_DIR: Path
SENTENCE_DIR: Path
CONCURRENCY = int(os.getenv("AIH_PIPELINE_CONCURRENCY", "8"))
DEFAULT_MODEL = os.getenv("AIH_CHAT_MODEL", CHAT_MODEL)
STEP_LABEL = os.getenv("AIH_ANCHOR_STABILIZE_STEP_LABEL", "Step10c")
CLIENT = None


class AnchorDecision(BaseModel):
    time_type: str = Field(
        ...,
        pattern=r"^(absolute_anchor|relative_anchor|event_phase|comparison_reference|day_part|non_time)$",
    )
    is_anchor: bool
    canonical_time_text: str = ""
    granularity: int = Field(0, ge=0, le=3)
    reason: str = ""


RANGE_RE = re.compile(r"^(?P<start>.+?\.\d+\.\d+\.\d+)-(?P<end>.+?\.\d+\.\d+\.\d+)$")
ANCHOR_TYPES = {"absolute_anchor", "relative_anchor"}
DYNASTY_RE = re.compile(r"^[\u4e00-\u9fff]{1,4}朝(?:时期|期|时代)?$")
PURE_DYNASTY_RE = re.compile(r"^[\u4e00-\u9fff]{1,4}朝(?:时期|期|时代)?$")


def has_explicit_temporal_anchor(text: str) -> bool:
    value = normalize_experiment1_tm(str(text or "").strip())
    if not value:
        return False
    return bool(
        re.search(r"(?:公元前|前)?\d{1,4}年", value)
        or re.search(r"(?:元|[一二三四五六七八九十百廿卅]+)年", value)
        or re.search(r"[正一二三四五六七八九十冬腊]+月", value)
        or re.search(r"(春季?|夏季?|秋季?|冬季?)", value)
    )


def fallback_anchor_decision(tm: str, reason: str) -> AnchorDecision:
    canonical = normalize_experiment1_tm(tm)
    granularity = 2 if re.search(r"[正一二三四五六七八九十冬腊]+月", canonical) else 1
    return AnchorDecision(
        time_type="absolute_anchor",
        is_anchor=True,
        canonical_time_text=canonical,
        granularity=granularity,
        reason=reason,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_range(value: str) -> Tuple[str, str]:
    match = RANGE_RE.match(str(value or "").strip())
    if not match:
        raise ValueError(f"bad range: {value}")
    return match.group("start"), match.group("end")


def number_key(value: str) -> Tuple[int, int, int]:
    parts = str(value).rsplit(".", 3)
    if len(parts) != 4:
        return (0, 0, 0)
    return int(parts[1]), int(parts[2]), int(parts[3])


def in_range(number: str, range_text: str) -> bool:
    start, end = split_range(range_text)
    return number_key(start) <= number_key(number) <= number_key(end)


def sentence_file_for_timeblock(path: Path) -> Path | None:
    name = path.name.replace("_timeblock.json", "_sentence.json")
    matches = sorted(SENTENCE_DIR.glob(name))
    return matches[-1] if matches else None


def sentence_text_by_number(sentence_path: Path | None) -> Dict[str, str]:
    if sentence_path is None or not sentence_path.exists():
        return {}
    data = load_json(sentence_path)
    rows = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
    return {
        str(row.get("number")): str(row.get("sentence", ""))
        for row in rows
        if isinstance(row, dict) and row.get("number")
    }


def block_text(block: Dict[str, Any], sentence_map: Dict[str, str]) -> str:
    range_text = str(block.get("timeblock_range", "") or "")
    chunks = []
    for number, text in sentence_map.items():
        try:
            if in_range(number, range_text):
                chunks.append(text)
        except Exception:
            continue
    return "".join(chunks)


def demote_non_anchor(block: Dict[str, Any], decision: AnchorDecision) -> None:
    block["Granularity"] = "0"
    block.pop("iso", None)
    block.pop("iso_range", None)
    block["time_anchor"] = decision.model_dump()


def apply_anchor(block: Dict[str, Any], decision: AnchorDecision) -> None:
    canonical = normalize_experiment1_tm(decision.canonical_time_text or str(block.get("TM", "") or ""))
    block["TM"] = canonical
    block["Granularity"] = str(decision.granularity)
    block.pop("iso", None)
    block.pop("iso_range", None)
    payload = decision.model_dump()
    payload["canonical_time_text"] = canonical
    block["time_anchor"] = payload


def call_llm_json(messages: List[Dict[str, str]], max_retries: int = 5) -> Dict[str, Any]:
    backoff = 1.0
    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            response = create_chat_completion(
                CLIENT,
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            return json.loads(text)
        except Exception as exc:
            last_error = exc
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
    raise RuntimeError(f"time anchor classification failed: {last_error!r}")


def is_nominal_dynasty_modifier(tm: str, text: str) -> bool:
    compact_tm = normalize_experiment1_tm(tm)
    compact_text = re.sub(r"\s+", "", text or "")
    if not compact_tm or not DYNASTY_RE.match(compact_tm):
        return False
    dynasty_base = re.sub(r"(?:时期|期|时代)$", "", compact_tm)
    if re.search(re.escape(dynasty_base) + r"(时|年间|末年|初年|末|初|中|以来|以前|以后)", compact_text):
        return False
    if compact_tm != dynasty_base:
        return True
    return bool(re.search(re.escape(dynasty_base) + r"的[\u4e00-\u9fff]{1,12}", compact_text))


def is_pure_dynasty_background(tm: str) -> bool:
    compact_tm = normalize_experiment1_tm(tm)
    return bool(PURE_DYNASTY_RE.match(compact_tm))


def build_messages(tm: str, granularity: str, text: str, prev_anchor: str, next_anchor: str) -> List[Dict[str, str]]:
    user = f"""\
输入 TimeBlock：
- TM: {tm}
- Granularity: {granularity}
- block_text: {text}
- previous_anchor_time: {prev_anchor}
- next_anchor_time: {next_anchor}

任务：判断 TM 是否是当前 TimeBlock 的真实时间锚点，并规范化为可交给 ISO 表推理的时间表达。

分类定义：
- absolute_anchor：TM 自身明确包含王年、年月、季节、朝代纪年等，可直接作为当前事件锚点。
- relative_anchor：TM 是相对时间或省略年号的时间，但能根据上下文明确解析为当前事件锚点。
- event_phase：人物身份、人生阶段、官职阶段、事件阶段，只说明阶段，不足以推出 ISO。
- comparison_reference：比较、类比、回忆、引用过去事件的时间，不是当前事件锚点。
- day_part：只有早晨、中午、夜间等日内时间，不足以推出月级 ISO。
- non_time：地点、人物、普通事件状态，或不是时间表达。

锚点规则：
- 只有 absolute_anchor 和可解析的 relative_anchor 可以 is_anchor=true。
- event_phase、comparison_reference、day_part、non_time 必须 is_anchor=false。
- 如果表达同时包含年月/季节和日内时间，canonical_time_text 只保留年月/季节，丢弃日内时间。
- 如果只有日内时间，is_anchor=false。
- 如果表达是比较/类比/回忆，不产生当前锚点，即使里面出现过去时间。
- 朝代词只有在作为时间状语/时代背景时才是 absolute_anchor，例如“某朝时”“某朝末年”“某朝年间”。
- 不要把纯朝代泛称或改写形式（如“某朝”“某朝时期”“某朝期”“某朝时代”）单独作为当前事件锚点。
- 如果朝代词只是官职、机构、文书、身份、对象的名词性修饰语，例如“某朝的官员”“某朝的御史”“某朝丞相掌管的文献”，它不是当前事件时间锚点，判为 non_time。
- granularity 使用 0=非锚点，1=年/季节，2=月，3=日。当前实验只需要月级或更粗，不要因为日内时间输出 3。

只输出 JSON：
{{
  "time_type": "absolute_anchor|relative_anchor|event_phase|comparison_reference|day_part|non_time",
  "is_anchor": true,
  "canonical_time_text": "...",
  "granularity": 0,
  "reason": "..."
}}
"""
    return [
        {"role": "system", "content": "你是时间表达锚点分类器。只判断表达是否能作为当前事件时间锚点，不做 ISO 换算。只输出 JSON。"},
        {"role": "user", "content": user},
    ]


def classify_anchor(tm: str, granularity: str, text: str, prev_anchor: str, next_anchor: str) -> AnchorDecision:
    tm = normalize_experiment1_tm(tm)
    if not tm:
        return AnchorDecision(time_type="non_time", is_anchor=False, canonical_time_text="", granularity=0, reason="empty TM")
    if is_pure_dynasty_background(tm) or is_nominal_dynasty_modifier(tm, text):
        return AnchorDecision(
            time_type="non_time",
            is_anchor=False,
            canonical_time_text="",
            granularity=0,
            reason="dynasty term is generic or nominal rather than a current-event temporal anchor",
        )
    raw = call_llm_json(build_messages(tm, granularity, text, prev_anchor, next_anchor))
    decision = AnchorDecision.model_validate(raw)
    if decision.time_type not in ANCHOR_TYPES:
        decision.is_anchor = False
        decision.granularity = 0
        decision.canonical_time_text = ""
    if decision.is_anchor and decision.granularity == 0:
        decision.is_anchor = False
    if decision.is_anchor:
        decision.canonical_time_text = normalize_experiment1_tm(decision.canonical_time_text or tm)
    return decision


def nearest_anchor_hints(tmb: List[Dict[str, Any]], index: int) -> Tuple[str, str]:
    prev_anchor = ""
    for obj in reversed(tmb[:index]):
        if not isinstance(obj, dict):
            continue
        anchor = obj.get("time_anchor") if isinstance(obj.get("time_anchor"), dict) else {}
        if anchor.get("is_anchor") and anchor.get("canonical_time_text"):
            prev_anchor = str(anchor["canonical_time_text"])
            break
        if str(obj.get("Granularity", "")).strip() != "0" and obj.get("TM"):
            prev_anchor = str(obj.get("TM"))
            break

    next_anchor = ""
    for obj in tmb[index + 1 :]:
        if not isinstance(obj, dict):
            continue
        if str(obj.get("Granularity", "")).strip() != "0" and obj.get("TM"):
            next_anchor = str(obj.get("TM"))
            break
    return prev_anchor, next_anchor


def process_file(path: Path) -> Dict[str, int]:
    payload = load_json(path)
    tmb = payload.get("TMB") if isinstance(payload, dict) else payload
    if not isinstance(tmb, list):
        return {"anchors": 0, "non_anchors": 0, "failed": 0}

    sentence_map = sentence_text_by_number(sentence_file_for_timeblock(path))
    work: List[Tuple[int, Dict[str, Any], str, str, str, str, str]] = []
    for index, block in enumerate(tmb):
        if not isinstance(block, dict):
            continue
        tm = normalize_experiment1_tm(str(block.get("TM", "") or ""))
        gran = str(block.get("Granularity", "") or "").strip()
        text = block_text(block, sentence_map)
        prev_anchor, next_anchor = nearest_anchor_hints(tmb, index)
        work.append((index, block, tm, gran, text, prev_anchor, next_anchor))

    stats = {"anchors": 0, "non_anchors": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(classify_anchor, tm, gran, text, prev_anchor, next_anchor): (index, block, tm)
            for index, block, tm, gran, text, prev_anchor, next_anchor in work
        }
        for future in as_completed(futures):
            _index, block, tm = futures[future]
            try:
                decision = future.result()
            except Exception as exc:
                if has_explicit_temporal_anchor(tm) and not is_nominal_dynasty_modifier(tm, str(block_text(block, sentence_map))):
                    decision = fallback_anchor_decision(
                        tm,
                        f"classification failed; preserved explicit temporal anchor: {type(exc).__name__}: {str(exc)[:120]}",
                    )
                else:
                    decision = AnchorDecision(
                        time_type="non_time",
                        is_anchor=False,
                        canonical_time_text="",
                        granularity=0,
                        reason=f"classification failed conservatively: {type(exc).__name__}: {str(exc)[:160]}",
                    )
                stats["failed"] += 1

            if decision.is_anchor:
                apply_anchor(block, decision)
                stats["anchors"] += 1
            else:
                if tm:
                    block["TM"] = tm
                demote_non_anchor(block, decision)
                stats["non_anchors"] += 1

    save_json(path, payload)
    return stats


def main() -> None:
    global RUN_ROOT, TIMEBLOCK_DIR, SENTENCE_DIR, CLIENT

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    TIMEBLOCK_DIR = timeblock_step_dir(RUN_ROOT, TIMEBLOCK_STEP)
    SENTENCE_DIR = sentence_step_dir(RUN_ROOT, 5)
    setup_step_logging(RUN_ROOT, "step_10c_time_anchor_classification")
    CLIENT = make_sync_chat_client()

    files = sorted(TIMEBLOCK_DIR.glob("*_timeblock.json"))
    reporter = StepReporter(STEP_LABEL, total=len(files))
    reporter.start(input_dir=TIMEBLOCK_DIR, output_dir=TIMEBLOCK_DIR)
    totals = {"anchors": 0, "non_anchors": 0, "failed": 0}
    pbar = step_tqdm(total=len(files), desc="Time anchor classification", unit="file")
    for index, path in enumerate(files, 1):
        stats = process_file(path)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
        print(
            f"{STEP_LABEL} | {index}/{len(files)} | OK | {path.name} | "
            + " ".join(f"{key}={value}" for key, value in stats.items())
        )
        pbar.update(1)
    pbar.close()
    print(
        f"{STEP_LABEL} | summary | "
        + " ".join(f"{key}={value}" for key, value in totals.items())
    )
    reporter.finish()


if __name__ == "__main__":
    main()
