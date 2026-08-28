import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field
from rich import print as rprint

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion,
    make_sync_chat_client,
)
from ai_historian.pipeline.logging import StepReporter, setup_step_logging, step_tqdm
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir, timeblock_step_dir
from ai_historian.pipeline.time_canonicalizer import normalize_experiment1_tm

client = None

# -----------------------------
# 基本路径
# -----------------------------
RUN_ROOT: Path
SENTENCE_DIR: Path
TIMEBLOCK_DIR: Path
OUTPUT_DIR: Path

MODEL_NAME = CHAT_MODEL
MAX_WORKERS = int(os.getenv("AIH_PIPELINE_CONCURRENCY", "40"))
MAX_RETRIES = 3               # API 重试次数
RETRY_SLEEP = 2               # 初始重试等待秒数


def get_client():
    if client is None:
        raise RuntimeError("LLM client is not initialized; call main() first")
    return client

# -----------------------------
# Pydantic 数据模型
# -----------------------------
class JUDGE(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # LLM 必须返回 {"continue": true/false}
    cont: bool = Field(..., alias="continue")

class TURN(BaseModel):
    TM: str
    reasoning: str


# -----------------------------
# 基础工具函数
# -----------------------------
def load_json(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"找不到文件：{p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_sentence_id(num_str: str) -> Tuple[str, int, int, int]:
    """
    解析新的句子编号格式：
    书籍uuid.篇章id.段落id.段落内句子id

    例如：
    94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
    -> (uuid, 7, 50, 1)

    注意：uuid 内部可能包含 '-'，因此不能靠 '-' 分割。
    """
    s = num_str.strip()
    m = re.match(r"^(?P<uuid>.+?)\.(?P<chapter>\d+)\.(?P<para>\d+)\.(?P<sent>\d+)$", s)
    if not m:
        raise ValueError(f"非法 sentence number 格式：{num_str}")
    return (
        m.group("uuid"),
        int(m.group("chapter")),
        int(m.group("para")),
        int(m.group("sent")),
    )

def sentence_id_to_order_tuple(num_str: str) -> Tuple[int, int, int]:
    """
    只取“篇章、段落、句子”三元组用于排序与闭区间判断。
    原则不变：范围判断仍然只按 篇章id/段落id/句子id 进行。
    """
    _, chapter, para, sent = parse_sentence_id(num_str)
    return (chapter, para, sent)

def tuple_le(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    return a <= b

def parse_id_range(range_str: str) -> Tuple[str, str]:
    """
    解析类似：
    uuid.7.50.1-uuid.7.50.1

    不能直接 split('-')，因为 uuid 自己有 '-'
    所以用正则从“两个完整 sentence id”层面解析。
    """
    s = range_str.strip()
    pattern = (
        r"^(?P<start>.+?\.\d+\.\d+\.\d+)"
        r"-"
        r"(?P<end>.+?\.\d+\.\d+\.\d+)$"
    )
    m = re.match(pattern, s)
    if not m:
        raise ValueError(f"非法 range 格式：{range_str}")
    return m.group("start"), m.group("end")

def expand_range_using_sentence_ids(range_str: str, all_ids_sorted: List[str]) -> List[str]:
    """
    根据 sentence json 中所有 number 的顺序来扩展范围。
    范围是闭区间，例如：
    'uuid.7.50.1-uuid.7.50.3'
    """
    start_str, end_str = parse_id_range(range_str)

    start_t = sentence_id_to_order_tuple(start_str)
    end_t = sentence_id_to_order_tuple(end_str)

    out = []
    for nid in all_ids_sorted:
        t = sentence_id_to_order_tuple(nid)
        if tuple_le(start_t, t) and tuple_le(t, end_t):
            out.append(nid)
    return out

def join_sentences(nums: List[str], num2sent: Dict[str, str]) -> str:
    sents = [num2sent[n] for n in nums if n in num2sent]
    # 原句自带标点，这里直接拼接即可
    return " ".join(sents).strip()

def parse_json_with_model(model_cls, data: dict):
    """
    兼容不同 pydantic 版本
    """
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


# -----------------------------
# 文件名配对相关
# -----------------------------
def get_sentence_key(path: Path) -> Optional[str]:
    """
    例如：
    7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_sentence.json
    -> 7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b
    """
    name = path.name
    suffix = "_sentence.json"
    if not name.endswith(suffix):
        return None
    return name[:-len(suffix)]

def get_timeblock_key(path: Path) -> Optional[str]:
    """
    例如：
    7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_timeblock.json
    -> 7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b
    """
    name = path.name
    suffix = "_timeblock.json"
    if not name.endswith(suffix):
        return None
    return name[:-len(suffix)]

def find_matched_file_pairs(sentence_dir: Path, timeblock_dir: Path) -> List[Tuple[Path, Path]]:
    sentence_files = sorted(sentence_dir.glob("*.json"))
    timeblock_files = sorted(timeblock_dir.glob("*.json"))

    sentence_map = {}
    for p in sentence_files:
        k = get_sentence_key(p)
        if k is not None:
            sentence_map[k] = p

    timeblock_map = {}
    for p in timeblock_files:
        k = get_timeblock_key(p)
        if k is not None:
            timeblock_map[k] = p

    common_keys = sorted(set(sentence_map.keys()) & set(timeblock_map.keys()))
    pairs = [(sentence_map[k], timeblock_map[k]) for k in common_keys]

    missing_sentence = sorted(set(timeblock_map.keys()) - set(sentence_map.keys()))
    missing_timeblock = sorted(set(sentence_map.keys()) - set(timeblock_map.keys()))

    if missing_sentence:
        rprint("[yellow]这些 timeblock 文件没有匹配到 sentence 文件：[/yellow]")
        for k in missing_sentence:
            rprint(f"  - {k}")

    if missing_timeblock:
        rprint("[yellow]这些 sentence 文件没有匹配到 timeblock 文件：[/yellow]")
        for k in missing_timeblock:
            rprint(f"  - {k}")

    return pairs


# -----------------------------
# OpenAI 调用（带重试）
# -----------------------------
def _chat_json(messages: List[Dict[str, str]], model: str = MODEL_NAME) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = create_chat_completion(
                get_client(),
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                sleep_s = RETRY_SLEEP * attempt
                time.sleep(sleep_s)
            else:
                raise last_err


# -----------------------------
# LLM 调用（JUDGE）
# -----------------------------
def call_judge(
    time_info_original: str,
    background_sentence: str,
    candidate_context_text: str,
    model: str = MODEL_NAME
) -> JUDGE:
    """
    判断“候选上文（判断辅助文本）”是否足以把 time_information_original
    补全为更具体的“时间标志物”。

    返回：
    - {"continue": false} => 证据足够，可以转化
    - {"continue": true}  => 证据不足，需要继续向上找
    """
    system = (
        "你是严格的历史时间对齐评审器。"
        "你的输出必须是 JSON，键为 'continue'（布尔）。"
        "判断标准：如果仅凭给定的“候选上文（判断辅助文本）”，就能把原始时间信息补全成具体且有据可依的时间标志物，"
        "则返回 false；若仍证据不足，返回 true。不得猜测。"
    )
    user = (
        f"【当前 timeblock 的背景句】\n{background_sentence}\n\n"
        f"【原始时间信息 time_information_original】\n{time_info_original}\n\n"
        f"【候选上文（判断辅助文本）】\n{candidate_context_text}\n\n"
        "请只返回 JSON：{\"continue\": true/false}。\n"
        "注意：false 表示证据足够（可以完成转化），true 表示证据不足（需要继续往上找）。"
    )

    data = _chat_json(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
    )
    return parse_json_with_model(JUDGE, data)


# -----------------------------
# LLM 调用（TURN）
# -----------------------------
def call_turn(
    time_info_original: str,
    basis_context_text: str,
    model: str = MODEL_NAME
) -> TURN:
    """
    输入：原始时间信息 + 依据文本（即“判断辅助文本”）
    产出：TM（转化后的时间标志物）与 reasoning（简明理由）
    """
    system = (
        "你是历史时间对齐与规范化助手。"
        "目标：把原始时间信息补全为明确、具体的“时间标志物”（TM）。"
        "做法：只依据提供的依据文本进行补全，例如补齐朝代、年号、干支、月份或主体。"
        "实验1规范：不得输出“公元前206年”或“前206年”作为 TM；遇到这些表达，必须改写成汉元年/汉高祖元年的纪年表达。"
        "谨慎：不得凭空发挥；若依据文本只能确定到某一层级，就到该层级为止。"
        "输出必须是 JSON，包含键：TM（字符串）、reasoning（中文解释依据与步骤）。"
    )
    user = (
        f"【原始时间信息】\n{time_info_original}\n\n"
        f"【依据文本（判断辅助文本）】\n{basis_context_text}\n\n"
        "请只返回 JSON：{\"TM\": \"...\", \"reasoning\": \"...\"}。\n"
        "TM 禁止包含“公元前206年”或“前206年”；例如公元前206年十二月应写作汉元年十二月，前206年四月应写作汉高祖元年四月。\n"
        "注意：不得编造未出现的纪年或朝代；若文本证据明确，尽量补全至最具体层级。"
    )

    data = _chat_json(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
    )
    return parse_json_with_model(TURN, data)


# -----------------------------
# 单个 timeblock 的处理逻辑
# -----------------------------
def process_one_timeblock(
    idx: int,
    tmb_list: List[dict],
    num2sent: Dict[str, str],
    all_ids_sorted: List[str],
) -> Dict[str, Any]:
    """
    返回一个结果 dict，由主线程统一回写。
    这样线程之间不会直接共享写入原对象，少一点妖气。
    """
    cur_tb = tmb_list[idx]
    cur_id = cur_tb.get("ID", "")

    result = {
        "idx": idx,
        "cur_id": cur_id,
        "judged_total": 0,
        "converted": False,
        "message": "",
        "update": None,
    }

    if not cur_id:
        result["message"] = f"警告：第 {idx} 个 timeblock 缺少 ID，跳过。"
        return result

    background_sentence = num2sent.get(cur_id, "")

    conv_info = cur_tb.get("Conversion information", {})
    time_info_original = (conv_info.get("time_information_original", "") or "").strip()

    # 没有原始时间信息则跳过
    if not time_info_original:
        result["message"] = f"跳过 {cur_id}（无 time_information_original）"
        return result

    decided = False
    chosen_basis_prev_id: Optional[str] = None
    chosen_basis_text: Optional[str] = None

    prev_idx = idx - 1
    while prev_idx >= 0:
        tb_prev = tmb_list[prev_idx]
        basis_id = tb_prev.get("ID", "")

        range_str = (tb_prev.get("timeblock_range", "") or "").strip()
        if not range_str:
            prev_idx -= 1
            continue

        try:
            nums_in_range = expand_range_using_sentence_ids(range_str, all_ids_sorted)
        except Exception:
            prev_idx -= 1
            continue

        basis_text = join_sentences(nums_in_range, num2sent)

        if not basis_text:
            prev_idx -= 1
            continue

        try:
            judge = call_judge(
                time_info_original=time_info_original,
                background_sentence=background_sentence,
                candidate_context_text=basis_text,
            )
            result["judged_total"] += 1
        except Exception:
            prev_idx -= 1
            continue

        if judge.cont is False:
            decided = True
            chosen_basis_prev_id = basis_id or f"INDEX:{prev_idx}"
            chosen_basis_text = basis_text
            break
        else:
            prev_idx -= 1

    if decided and chosen_basis_prev_id and chosen_basis_text:
        try:
            turn = call_turn(
                time_info_original=time_info_original,
                basis_context_text=chosen_basis_text
            )
        except Exception as e:
            result["message"] = f"TURN 出错于当前 {cur_id}（依据 {chosen_basis_prev_id}）：{e}"
            return result

        result["converted"] = True
        normalized_tm = normalize_experiment1_tm(turn.TM)
        result["message"] = f"已转化 {cur_id} ← 依据 {chosen_basis_prev_id} → TM = {normalized_tm}"
        result["update"] = {
            "basis_of_conversion": chosen_basis_prev_id,
            "time_information_converted": normalized_tm,
            "reasoning": turn.reasoning,
            "is_conversion_required": True,
        }
        return result

    result["message"] = f"未转化 {cur_id}（未找到足够依据）"
    return result


# -----------------------------
# 处理一对文件
# -----------------------------
def process_file_pair(sentence_file: Path, timeblock_file: Path) -> Dict[str, Any]:
    sentence_obj = load_json(sentence_file)   # list[dict]
    tb_obj = load_json(timeblock_file)        # {'TMB': [...]}

    if not isinstance(sentence_obj, list):
        raise ValueError(f"sentence 文件格式不对，应为 list：{sentence_file}")

    if not isinstance(tb_obj, dict) or "TMB" not in tb_obj:
        raise ValueError(f"timeblock 文件格式不对，应包含 TMB：{timeblock_file}")

    tmb_list: List[dict] = tb_obj["TMB"]

    # 建立句子索引
    num2sent: Dict[str, str] = {}
    for d in sentence_obj:
        num = d.get("number", "")
        sent = d.get("sentence", "")
        if num:
            num2sent[num] = sent

    all_ids_sorted = sorted(num2sent.keys(), key=sentence_id_to_order_tuple)

    judged_total = 0
    converted_count = 0

    # 第 0 个无法向上回溯，所以从第 1 个开始
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx in range(1, len(tmb_list)):
            futures.append(
                executor.submit(
                    process_one_timeblock,
                    idx,
                    tmb_list,
                    num2sent,
                    all_ids_sorted
                )
            )

        for fut in step_tqdm(as_completed(futures), total=len(futures), desc=f"Processing {timeblock_file.name}"):
            res = fut.result()
            judged_total += res["judged_total"]

            msg = res["message"]
            if res["converted"]:
                converted_count += 1
                rprint(f"[green]{msg}[/green]")
            else:
                # 只打印关键信息，避免刷屏过猛
                rprint(f"[cyan]{msg}[/cyan]")

            if res["update"] is not None:
                idx = res["idx"]
                cur_tb = tmb_list[idx]
                if "Conversion information" not in cur_tb or not isinstance(cur_tb["Conversion information"], dict):
                    cur_tb["Conversion information"] = {}

                cur_tb["Conversion information"]["is_conversion_required"] = res["update"]["is_conversion_required"]
                cur_tb["Conversion information"]["basis_of_conversion"] = res["update"]["basis_of_conversion"]
                cur_tb["Conversion information"]["time_information_converted"] = res["update"]["time_information_converted"]
                cur_tb["Conversion information"]["reasoning"] = res["update"]["reasoning"]

    # 保存
    output_path = OUTPUT_DIR / timeblock_file.name
    tb_obj["TMB"] = tmb_list
    save_json(tb_obj, output_path)

    return {
        "sentence_file": str(sentence_file),
        "timeblock_file": str(timeblock_file),
        "output_file": str(output_path),
        "judged_total": judged_total,
        "converted_count": converted_count,
        "timeblock_count": len(tmb_list),
    }


# -----------------------------
# 主流程
# -----------------------------
def main():
    global RUN_ROOT, SENTENCE_DIR, TIMEBLOCK_DIR, OUTPUT_DIR, client

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    SENTENCE_DIR = sentence_step_dir(RUN_ROOT, 5)
    TIMEBLOCK_DIR = timeblock_step_dir(RUN_ROOT, 6)
    OUTPUT_DIR = timeblock_step_dir(RUN_ROOT, 7)
    setup_step_logging(RUN_ROOT, "step_07_timeblock_conversion")
    client = make_sync_chat_client()

    if not SENTENCE_DIR.exists():
        raise FileNotFoundError(f"找不到目录：{SENTENCE_DIR}")
    if not TIMEBLOCK_DIR.exists():
        raise FileNotFoundError(f"找不到目录：{TIMEBLOCK_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_matched_file_pairs(SENTENCE_DIR, TIMEBLOCK_DIR)
    if not pairs:
        raise RuntimeError("没有找到可匹配的 sentence/timeblock 文件对。")

    reporter = StepReporter("Step7", total=len(pairs))
    reporter.start(input_dir=SENTENCE_DIR, output_dir=OUTPUT_DIR)

    grand_judged = 0
    grand_converted = 0

    for sentence_file, timeblock_file in pairs:
        try:
            stats = process_file_pair(sentence_file, timeblock_file)
        except Exception as e:
            reporter.item_fail(timeblock_file.name, e)
            raise SystemExit(1)

        grand_judged += stats["judged_total"]
        grand_converted += stats["converted_count"]
        reporter.item_ok(
            timeblock_file.name,
            detail=f"JUDGE={stats['judged_total']} 转化={stats['converted_count']}",
        )

    reporter.finish(output_dir=OUTPUT_DIR, extra=f"JUDGE={grand_judged} 转化={grand_converted}")


# -----------------------------
# 执行
# -----------------------------
if __name__ == "__main__":
    main()
