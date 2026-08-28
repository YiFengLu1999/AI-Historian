import json
import os
import re
import sys
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
from ai_historian.pipeline.paths import (
    resolve_run_root,
    sentence_step_dir,
    sequence_step_dir,
    timeblock_step_dir,
)

# =========================================================
# Paths
# =========================================================
RUN_ROOT: Path
SENTENCE_DIR: Path
TIMEBLOCK_DIR: Path
OUT_DIR: Path
client = None

# =========================================================
# Pydantic schemas
# =========================================================
class EA(BaseModel):
    location: str = Field(..., pattern=r"^(front|back|irrelevant)$")
    reason: str
    Credibility: float = Field(..., ge=1, le=10)

class EF(BaseModel):
    ID: str
    location: str = Field(..., pattern=r"^(front|back|irrelevant)$")

# =========================================================
# Regex
# =========================================================
# 文件名格式：篇章id_uuid_sentence.json / 篇章id_uuid_timeblock.json
FILE_RE = re.compile(
    r"^(?P<chapter>[^_]+)_(?P<uuid>[0-9a-fA-F-]+)_(?P<kind>sentence|timeblock)\.json$"
)

# 新 sentence id 格式：uuid.篇章id.段落id.段落内句子id
# 例如：94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
ID_RE = re.compile(
    r"^(?P<uuid>[0-9a-fA-F-]+)\.(?P<chapter>\d+)\.(?P<para>\d+)\.(?P<sent>\d+)$"
)

# 新 range 格式：
# uuid.篇章id.段落id.句子id-uuid.篇章id.段落id.句子id
RANGE_RE = re.compile(
    r"^(?P<start>[0-9a-fA-F-]+\.\d+\.\d+\.\d+)-(?P<end>[0-9a-fA-F-]+\.\d+\.\d+\.\d+)$"
)

# =========================================================
# Utility: JSON I/O
# =========================================================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =========================================================
# Utility: filename parsing
# =========================================================
def parse_filename(path: Path) -> Tuple[str, str, str]:
    """
    返回: (chapter_id, uuid, kind)
    """
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"文件名不符合预期格式: {path.name}")
    return m.group("chapter"), m.group("uuid"), m.group("kind")

# =========================================================
# Utility: sentence ID / range parsing
# =========================================================
def parse_sentence_id(id_str: str) -> Tuple[str, int, int, int]:
    """
    'uuid.chapter.para.sent' -> (uuid, chapter, para, sent)
    """
    m = ID_RE.match(id_str.strip())
    if not m:
        raise ValueError(f"非法 sentence ID: {id_str}")
    return (
        m.group("uuid"),
        int(m.group("chapter")),
        int(m.group("para")),
        int(m.group("sent")),
    )

def id_key(id_str: str) -> Tuple[int, int, int]:
    """
    只按 (篇章id, 段落id, 段落内句子id) 排序/比较
    不把 uuid 作为时序依据，这和你原来的逻辑原则一致。
    """
    _, chapter, para, sent = parse_sentence_id(id_str)
    return (chapter, para, sent)

def parse_range(rng: str) -> Tuple[str, str]:
    """
    兼容带 uuid 且 uuid 里含 '-' 的 range。
    不能再用 rng.split('-')，因为 uuid 本身有 '-'
    """
    rng = rng.strip()
    m = RANGE_RE.match(rng)
    if not m:
        raise ValueError(f"非法 range: {rng}")
    return m.group("start"), m.group("end")

def ids_in_range(all_ids_sorted: List[str], start_id: str, end_id: str) -> List[str]:
    """
    从现有 sentence ID 中，取出位于 [start_id, end_id] 范围内的所有句子 ID
    比较规则仍然是：篇章id -> 段落id -> 句子id
    """
    start_uuid, start_ch, start_para, start_sent = parse_sentence_id(start_id)
    end_uuid, end_ch, end_para, end_sent = parse_sentence_id(end_id)

    if start_uuid != end_uuid:
        raise ValueError(f"range 起点和终点 UUID 不一致: {start_id} / {end_id}")

    ks = (start_ch, start_para, start_sent)
    ke = (end_ch, end_para, end_sent)

    out = []
    for sid in all_ids_sorted:
        sid_uuid, ch, para, sent = parse_sentence_id(sid)
        if sid_uuid != start_uuid:
            continue
        k = (ch, para, sent)
        if ks <= k <= ke:
            out.append(sid)
    return out

def build_text_for_range(sentence_map: Dict[str, str], all_ids_sorted: List[str], rng: str) -> str:
    start_id, end_id = parse_range(rng)
    ids = ids_in_range(all_ids_sorted, start_id, end_id)
    return "".join(sentence_map[i] for i in ids if i in sentence_map)

# =========================================================
# Utility: load sentence list / timeblock list robustly
# =========================================================
def extract_sentence_list(obj: Any) -> List[Dict[str, Any]]:
    """
    兼容几种常见结构：
    1) 直接是 list
    2) {"data": [...]}
    3) {"sentences": [...]}
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("data"), list):
            return obj["data"]
        if isinstance(obj.get("sentences"), list):
            return obj["sentences"]
    raise ValueError("sentence json 结构无法识别，预期为 list / {'data': [...]} / {'sentences': [...]}")

def extract_timeblock_list(obj: Any) -> List[Dict[str, Any]]:
    """
    兼容几种常见结构：
    1) {"TMB": [...]}
    2) 直接是 list
    3) {"timeblocks": [...]}
    """
    if isinstance(obj, dict) and isinstance(obj.get("TMB"), list):
        return obj["TMB"]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("timeblocks"), list):
        return obj["timeblocks"]
    raise ValueError("timeblock json 结构无法识别，预期为 {'TMB': [...]} / list / {'timeblocks': [...]}")

# =========================================================
# Utility: LLM call -> strict JSON -> Pydantic validate
# =========================================================
def get_client():
    if client is None:
        raise RuntimeError("LLM client is not initialized; call main() first")
    return client


def _extract_first_json(text: str) -> str:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    left = s.find("{")
    right = s.rfind("}")
    if left != -1 and right != -1 and right > left:
        return s[left:right + 1]
    return s

def llm_json(model: str, instruction: str, payload: str, schema_model, max_retries: int = 4):
    last_err = None
    last_txt = ""
    for _ in range(max_retries + 1):
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": payload},
            ],
            "response_format": {"type": "json_object"},
        }
        resp = create_chat_completion(get_client(), **kwargs)
        last_txt = (resp.choices[0].message.content or "").strip()
        try:
            js = json.loads(_extract_first_json(last_txt))
            return schema_model.model_validate(js)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"LLM output parse/validate failed after retries.\n"
        f"Last error: {last_err}\n"
        f"Raw output (truncated):\n{last_txt[:2000]}"
    )

def fallback_ef_from_pairs(pairs: List[Dict[str, Any]]) -> EF:
    candidates = [
        p for p in pairs
        if p.get("location") in {"front", "back"} and p.get("judge_ID")
    ]
    if not candidates:
        return EF(ID="irrelevant", location="irrelevant")
    best = max(candidates, key=lambda p: float(p.get("Credibility") or 0))
    return EF(ID=str(best["judge_ID"]), location=str(best["location"]))

# =========================================================
# Prompts
# =========================================================
EA_SYSTEM = (
    "你是一个严谨的历史文本事件时间顺序判断器。\n"
    "给你两段文本：A 是“插叙内容”，B 是“判断文本（一个时间块对应的事件描述）”。\n"
    "你的任务：判断 A 所描述的事件整体在 B 之前(front)、之后(back)、或无关(irrelevant)。\n"
    "规则：\n"
    "1) 若 A 与 B 完全无关，输出 irrelevant。\n"
    "2) 若两者存在交叉/重叠/难以严格分割：以各自文本中的“第一个事件”作为比较基准。\n"
    "3) 必须只输出一个 JSON 对象，不要输出任何额外文字。\n"
    "4) Credibility 为 1-10 分（可以是小数），表示你对结论的可信度。\n"
    "JSON 格式：\n"
    '{"location":"front|back|irrelevant","reason":"...","Credibility":7}\n'
)

def make_EA_user(interlude_text: str, judge_id: str, judge_text: str) -> str:
    return (
        f"【插叙内容 A】\n{interlude_text}\n\n"
        f"【判断文本 B】(ID={judge_id})\n{judge_text}\n\n"
        "请按要求输出 JSON。"
    )

EF_SYSTEM = (
    "你是一个谨慎的裁决器。\n"
    "输入是一张表：每一行都是同一个插叙内容 A 与某个判断文本 B(ID=...) 的比较结果，"
    "包含 location/front/back/irrelevant、reason、Credibility。\n"
    "你的任务：综合全表，选择你最相信的一条判断，并只输出 JSON：\n"
    '{"ID":"...","location":"front|back|irrelevant"}\n'
    "要求：\n"
    "1) 只能输出 JSON，不要输出任何额外文字。\n"
    "2) 如果你认为所有条目都不可靠或都指向无关，可输出 location=irrelevant。\n"
)

def make_EF_user(pairs: List[Dict[str, Any]], interlude_id: str, interlude_range: str) -> str:
    return (
        f"插叙块 ID={interlude_id}, timeblock_range={interlude_range}\n"
        "下面是它与各判断文本的比较结果表（JSON 数组，每项含 judge_ID/location/reason/Credibility）：\n"
        f"{json.dumps(pairs, ensure_ascii=False, indent=2)}\n"
        "请只输出最终选择的 JSON。"
    )

# =========================================================
# Pair discovery
# =========================================================
def discover_file_pairs() -> List[Tuple[str, str, Path, Path]]:
    """
    返回所有可处理文件对：
    [(chapter_id, uuid, sentence_path, timeblock_path), ...]
    匹配依据：篇章id + uuid
    """
    sentence_map: Dict[Tuple[str, str], Path] = {}

    for sp in SENTENCE_DIR.glob("*_sentence.json"):
        try:
            chapter_id, uuid, kind = parse_filename(sp)
            if kind == "sentence":
                sentence_map[(chapter_id, uuid)] = sp
        except Exception as e:
            print(f"[yellow]跳过无法识别的 sentence 文件: {sp.name} | {e}[/yellow]")

    pairs: List[Tuple[str, str, Path, Path]] = []
    missing_sentence: List[Path] = []

    for tp in TIMEBLOCK_DIR.glob("*_timeblock.json"):
        try:
            chapter_id, uuid, kind = parse_filename(tp)
            if kind != "timeblock":
                continue
            key = (chapter_id, uuid)
            sp = sentence_map.get(key)
            if sp is None:
                missing_sentence.append(tp)
                continue
            pairs.append((chapter_id, uuid, sp, tp))
        except Exception as e:
            print(f"[yellow]跳过无法识别的 timeblock 文件: {tp.name} | {e}[/yellow]")

    if missing_sentence:
        print("[yellow]以下 timeblock 文件没有找到对应的 sentence 文件：[/yellow]")
        for p in missing_sentence:
            print(f"  - {p.name}")

    def sort_key(x):
        chapter_id, uuid, _, _ = x
        try:
            return (int(chapter_id), uuid)
        except (TypeError, ValueError):
            return (chapter_id, uuid)

    return sorted(pairs, key=sort_key)

# =========================================================
# Core pipeline for one file pair
# =========================================================
def process_one_pair(
    chapter_id: str,
    uuid: str,
    sentence_path: Path,
    timeblock_path: Path,
    model: str = CHAT_MODEL,
) -> None:
    sent_obj = load_json(sentence_path)
    tmb_obj = load_json(timeblock_path)

    sent_list = extract_sentence_list(sent_obj)
    tmb_list = extract_timeblock_list(tmb_obj)

    # -------------------------
    # sentence map
    # -------------------------
    sentence_map = {
        x["number"]: x["sentence"]
        for x in sent_list
        if isinstance(x, dict) and "number" in x and "sentence" in x
    }

    if not sentence_map:
        raise ValueError(f"sentence 文件中没有找到有效的 number/sentence: {sentence_path}")

    all_sentence_ids_sorted = sorted(sentence_map.keys(), key=id_key)

    # -------------------------
    # timeblock index
    # -------------------------
    by_id = {
        x["ID"]: x
        for x in tmb_list
        if isinstance(x, dict) and "ID" in x
    }

    if not by_id:
        raise ValueError(f"timeblock 文件中没有找到有效的 ID: {timeblock_path}")

    # -------------------------
    # Step 1: base sequence (non-interlude) + collect interludes
    # -------------------------
    sequence: List[str] = []
    interlude_blocks: List[Dict[str, Any]] = []

    for x in tmb_list:
        if not isinstance(x, dict):
            continue
        xid = x.get("ID")
        if not xid:
            continue

        if bool(x.get("Interlude")) is True:
            interlude_blocks.append(x)
        else:
            sequence.append(xid)

    # 初始序列：先放非插叙块
    seq_agent6 = list(sequence)

    # -------------------------
    # Step 2: insert each interlude
    # -------------------------
    for ib in tqdm(interlude_blocks, desc=f"[{chapter_id}_{uuid}] Interlude blocks"):
        interlude_id = ib["ID"]
        interlude_range = ib.get("timeblock_range") or ib.get("isorange") or ""
        if not interlude_range:
            print(f"[yellow]跳过插叙块（缺少 timeblock_range/isorange）: {interlude_id}[/yellow]")
            continue

        interlude_text = build_text_for_range(sentence_map, all_sentence_ids_sorted, interlude_range).strip()

        if not interlude_text:
            print(f"[yellow]插叙块文本为空，跳过: {interlude_id} | range={interlude_range}[/yellow]")
            continue

        pairs: List[Dict[str, Any]] = []

        for judge_id in tqdm(sequence, desc=f"[{chapter_id}_{uuid}] Compare {interlude_id}", leave=False):
            judge_obj = by_id.get(judge_id)
            if not judge_obj:
                continue

            judge_range = judge_obj.get("timeblock_range") or judge_obj.get("isorange") or ""
            if not judge_range:
                continue

            judge_text = build_text_for_range(sentence_map, all_sentence_ids_sorted, judge_range).strip()
            if not judge_text:
                continue

            try:
                ea = llm_json(
                    model=model,
                    instruction=EA_SYSTEM,
                    payload=make_EA_user(interlude_text, judge_id, judge_text),
                    schema_model=EA,
                )
            except Exception as exc:
                print(f"[yellow]EA 比较失败，按 irrelevant 继续: interlude={interlude_id} judge={judge_id} | {exc}[/yellow]")
                ea = EA(location="irrelevant", reason="LLM JSON parse failed; conservative fallback.", Credibility=1)

            pairs.append({
                "judge_ID": judge_id,
                "location": ea.location,
                "reason": ea.reason,
                "Credibility": ea.Credibility,
            })

        if not pairs:
            print(f"[yellow]没有生成可用比较结果，跳过插叙块: {interlude_id}[/yellow]")
            continue

        # 全 irrelevant -> 不插入
        if all(p["location"] == "irrelevant" for p in pairs):
            continue

        try:
            ef = llm_json(
                model=model,
                instruction=EF_SYSTEM,
                payload=make_EF_user(pairs, interlude_id, interlude_range),
                schema_model=EF,
            )
        except Exception as exc:
            print(f"[yellow]EF 裁决失败，使用最高可信度非 irrelevant 结果作为 fallback: interlude={interlude_id} | {exc}[/yellow]")
            ef = fallback_ef_from_pairs(pairs)

        if ef.location == "irrelevant" or ef.ID.lower() == "irrelevant":
            continue

        anchor_id = ef.ID
        location = ef.location

        try:
            idx = seq_agent6.index(anchor_id)
        except ValueError:
            print(f"[yellow]EF 选出的 anchor 不在当前序列中，跳过: {anchor_id}[/yellow]")
            continue

        if location == "front":
            insert_pos = idx
        elif location == "back":
            insert_pos = idx + 1
        else:
            continue

        if interlude_id not in seq_agent6:
            seq_agent6.insert(insert_pos, interlude_id)

    # -------------------------
    # Save final output
    # -------------------------
    out_path = OUT_DIR / f"{chapter_id}_{uuid}_sequence.json"
    save_json(out_path, seq_agent6)

    print(f"\n[green]✅ Done[/green] chapter={chapter_id}, uuid={uuid}")
    print(f"  saved -> {out_path}")

# =========================================================
# Batch run
# =========================================================
def process_all(model: str = CHAT_MODEL) -> None:
    pairs = discover_file_pairs()

    if not pairs:
        print("[red]没有发现可处理的 sentence/timeblock 配对文件。[/red]")
        print(f"sentence dir: {SENTENCE_DIR}")
        print(f"timeblock dir: {TIMEBLOCK_DIR}")
        return

    print(f"[cyan]共发现 {len(pairs)} 对可处理文件。[/cyan]")

    errors = []
    for chapter_id, uuid, sentence_path, timeblock_path in tqdm(pairs, desc="All files"):
        try:
            process_one_pair(
                chapter_id=chapter_id,
                uuid=uuid,
                sentence_path=sentence_path,
                timeblock_path=timeblock_path,
                model=model,
            )
        except Exception as e:
            print(f"[red]处理失败: {chapter_id}_{uuid}[/red]")
            print(f"[red]{type(e).__name__}: {e}[/red]")
            errors.append((chapter_id, uuid, type(e).__name__, str(e)))
    if errors:
        raise RuntimeError(f"Step8 failed for {len(errors)} file pair(s): {errors}")

def main() -> None:
    global RUN_ROOT, SENTENCE_DIR, TIMEBLOCK_DIR, OUT_DIR, client

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    SENTENCE_DIR = sentence_step_dir(RUN_ROOT, 5)
    TIMEBLOCK_DIR = timeblock_step_dir(RUN_ROOT, 7)
    OUT_DIR = sequence_step_dir(RUN_ROOT, 8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_step_logging(RUN_ROOT, "step_08_sequence_sorting")
    client = make_sync_chat_client()
    process_all(model=os.getenv("AIH_CHAT_MODEL", CHAT_MODEL))


if __name__ == "__main__":
    main()
