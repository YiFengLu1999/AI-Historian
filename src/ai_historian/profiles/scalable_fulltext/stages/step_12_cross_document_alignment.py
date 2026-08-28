import itertools
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import APIError, APITimeoutError, BadRequestError, RateLimitError
from pydantic import BaseModel, Field
from rich import print

from ai_historian.model_config import (
    CHAT_MODEL,
    EMBED_MODEL,
    create_chat_completion,
    make_embedding_client,
    make_sync_chat_client,
    validate_json_text,
)
from ai_historian.pipeline.logging import StepReporter, emit_log, setup_step_logging, step_tqdm
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir, timeblock_step_dir

# =========================================================
# 说明
# 1) 聊天模型默认走 llm_config.py 中配置的 OpenAI-compatible 服务
# 2) embedding 默认独立于聊天模型配置，方便在本地 / 服务器分阶段跑
# 3) issame 判定并发到最多 40 个请求
# 4) 支持 STEP12_MODE=full / retrieve_only / judge_only
# 5) 输入改为：
#       sentence/step5output
#       timeblock/step11output
# 6) 输出改为：
#       sentence/step12output
#    且输出文件名保持 sentence 原始 json 文件名不变
# =========================================================

LLM_MODEL = CHAT_MODEL
STEP12_MODE = (os.getenv("STEP12_MODE", "full").strip().lower() or "full")
TOP_K = 10
MAX_RETRIES = 3
MAX_WORKERS = 40
EMBED_BATCH_SIZE = 10
CHECKPOINT_EVERY = 100
EMBED_MAX_TEXT_CHARS = 5000
JUDGE_MAX_BACKGROUND_CHARS = 6000
EMBED_TIMEOUT_SECONDS = 90
EMBED_MAX_RETRIES = 4
STAGE1_PROGRESS_EVERY = 10
STEP12_JOB_ARTIFACT = "_step12_jobs.json"
STEP12_PAIR_LOG_ARTIFACT = "_step12_pair_logs.json"
STEP12_JUDGE_RESULTS_ARTIFACT = "_step12_judge_results.json"
JUDGE_RESULTS_SAVE_EVERY = 20

if STEP12_MODE not in {"full", "retrieve_only", "judge_only"}:
    raise ValueError(f"不支持的 STEP12_MODE: {STEP12_MODE}")

RUN_ROOT: Path
SENTENCE_DIR: Path
TIMEBLOCK_DIR: Path
OUTPUT_DIR: Path

_EMBED_CLIENT = None

# =========================
# Pydantic 输出控制
# =========================
class Sentencesame(BaseModel):
    isSame: bool = Field(..., description="如果是同一个事件则为 true，否则为 false")


# =========================
# 常量与缓存
# =========================
NEG_INF = (-10**9, 1, 1)
POS_INF = (10**9, 12, 31)
ISO_DATE_RE = re.compile(r"^([+-]?\d{4,})-(\d{2})-(\d{2})$")

# 同时兼容新旧命名，未来更稳一点
SENTENCE_FILE_SUFFIXES = ("_sentence", "_interlude")
TIMEBLOCK_FILE_SUFFIXES = ("_timeblock", "_timeblocks_updated")

embedding_cache: Dict[str, List[float]] = {}
judge_cache: Dict[Tuple[str, str, str], bool] = {}

embedding_lock = threading.Lock()
judge_lock = threading.Lock()


# =========================
# 读写工具
# =========================
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# 文件名解析
# 文件名规则：篇章id_uuid_文件属性.json
# 例如：
#   7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_sentence.json
#   7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b_timeblock.json
#
# 这里的 doc_key 统一取：
#   7_94d18bb5-29cc-51b5-b0c3-70afe2b6f85b
# =========================
def strip_known_suffix(stem: str, suffixes: Tuple[str, ...]) -> str:
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    raise ValueError(f"无法识别文件后缀: {stem}")

def extract_doc_key_from_sentence_file(path: Path) -> str:
    return strip_known_suffix(path.stem, SENTENCE_FILE_SUFFIXES)

def extract_doc_key_from_timeblock_file(path: Path) -> str:
    return strip_known_suffix(path.stem, TIMEBLOCK_FILE_SUFFIXES)


# =========================
# number 与 timeblock_range 解析
#
# 旧格式：
#   篇章id.段落id.句子id
# 新格式：
#   书籍uuid.篇章id.段落id.句子id
#
# 示例：
#   94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1
#
# 注意：
# uuid 里有 '-'，所以不能再粗暴用 split('-')
# =========================
def parse_sentence_number(number_str: str) -> Tuple[str, int, int, int]:
    """
    支持两种格式：
    1) uuid.chapter.paragraph.sentence
    2) chapter.paragraph.sentence   （兼容旧数据）
    """
    s = str(number_str).strip()
    parts = s.rsplit(".", 3)

    if len(parts) == 4:
        book_uuid, chapter_id, paragraph_id, sentence_id = parts
    elif len(parts) == 3:
        book_uuid = ""
        chapter_id, paragraph_id, sentence_id = parts
    else:
        raise ValueError(f"无法解析 number: {number_str}")

    return book_uuid, int(chapter_id), int(paragraph_id), int(sentence_id)


def split_sentence_range(range_str: str) -> Tuple[str, str]:
    """
    解析 timeblock_range，例如：
      94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.1-94d18bb5-29cc-51b5-b0c3-70afe2b6f85b.7.50.3

    由于 uuid 内本身有 '-'，不能直接 split('-')。
    这里采用稳健做法：逐个尝试每一个 '-' 作为真正分隔符，
    只要左右两边都能被 parse_sentence_number 成功解析，就接受。
    """
    s = str(range_str).strip()

    for i, ch in enumerate(s):
        if ch != "-":
            continue

        left = s[:i].strip()
        right = s[i + 1:].strip()

        if not left or not right:
            continue

        try:
            parse_sentence_number(left)
            parse_sentence_number(right)
            return left, right
        except Exception:
            continue

    raise ValueError(f"无法解析 timeblock_range: {range_str}")


def sentence_in_range(number_str: str, range_str: str) -> bool:
    """
    判断一个 sentence number 是否落在 timeblock_range 内。
    比较原则不变：仍然按 篇章id / 段落id / 句子id 做区间比较。
    uuid 只用于一致性校验，不参与数值排序逻辑。
    """
    num_uuid, num_chapter, num_paragraph, num_sentence = parse_sentence_number(number_str)
    start_str, end_str = split_sentence_range(range_str)
    start_uuid, start_chapter, start_paragraph, start_sentence = parse_sentence_number(start_str)
    end_uuid, end_chapter, end_paragraph, end_sentence = parse_sentence_number(end_str)

    # 只要出现多个不同 uuid，就说明不是同一文档范围，直接 False
    used_uuids = {u for u in (num_uuid, start_uuid, end_uuid) if u}
    if len(used_uuids) > 1:
        return False

    num_key = (num_chapter, num_paragraph, num_sentence)
    start_key = (start_chapter, start_paragraph, start_sentence)
    end_key = (end_chapter, end_paragraph, end_sentence)

    return start_key <= num_key <= end_key


# =========================
# ISO range 解析与重叠判断
# 这个跟 uuid 无关，仍按日期区间处理
# =========================
def parse_iso_boundary(token: str) -> Tuple[int, int, int]:
    """
    支持：
    - '-infinity'
    - '+infinity'
    - 'infinity'
    - 'inf' / '+inf' / '-inf'
    - 标准 ISO 日期，如 '-0209-07-01'
    """
    token = str(token).strip().lower()

    if token in {"-infinity", "-inf"}:
        return NEG_INF
    if token in {"infinity", "+infinity", "inf", "+inf"}:
        return POS_INF

    m = ISO_DATE_RE.match(token)
    if not m:
        raise ValueError(f"无法解析 ISO 日期边界: {token}")

    year, month, day = m.groups()
    return (int(year), int(month), int(day))


def parse_iso_range(iso_range: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    解析形如：
    - '-infinityto-0209-07-01'
    - '-0209-07-01to+infinity'
    - '-0209-07-01to-0244-10-01'
    """
    s = str(iso_range).strip()
    if "to" not in s:
        raise ValueError(f"非法 iso_range: {iso_range}")

    start_str, end_str = s.split("to", 1)
    return parse_iso_boundary(start_str), parse_iso_boundary(end_str)

def same_iso_range(range_a: str, range_b: str) -> bool:
    return parse_iso_range(range_a) == parse_iso_range(range_b)

def overlap_interval(range_a: str, range_b: str) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    a_start, a_end = parse_iso_range(range_a)
    b_start, b_end = parse_iso_range(range_b)

    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if start <= end:
        return start, end
    return None

def tuple_to_ordinalish(x: Tuple[int, int, int]) -> int:
    year, month, day = x
    return year * 372 + month * 31 + day


# =========================
# crossDocTransfer 规范化
# same_timeblock_id 统一成 list
# =========================
def normalize_crossdoc_transfer(sentence_obj: Dict[str, Any]) -> Dict[str, Any]:
    block = sentence_obj.get("crossDocTransfer")

    if not isinstance(block, dict):
        block = {
            "isSame": False,
            "same_timeblock_id": []
        }

    same_ids = block.get("same_timeblock_id", [])

    if isinstance(same_ids, str):
        same_ids = [same_ids] if same_ids.strip() else []
    elif same_ids is None:
        same_ids = []
    elif not isinstance(same_ids, list):
        same_ids = [str(same_ids)]

    block["same_timeblock_id"] = same_ids
    block["isSame"] = bool(block.get("isSame", False))
    sentence_obj["crossDocTransfer"] = block
    return block

def append_same_timeblock_id(sentence_obj: Dict[str, Any], timeblock_id: str) -> None:
    cross = normalize_crossdoc_transfer(sentence_obj)
    if timeblock_id not in cross["same_timeblock_id"]:
        cross["same_timeblock_id"].append(timeblock_id)
    cross["isSame"] = len(cross["same_timeblock_id"]) > 0


# =========================
# 句子聚合
# =========================
def get_sentences_by_timeblock_range(doc_sentences: List[Dict[str, Any]], range_str: str) -> List[Dict[str, Any]]:
    return [obj for obj in doc_sentences if sentence_in_range(obj["number"], range_str)]

def aggregate_sentences(sentences: List[Dict[str, Any]]) -> str:
    return "\n".join(str(obj.get("sentence", "")).strip() for obj in sentences if str(obj.get("sentence", "")).strip())

def is_sinking(sentence_obj: Dict[str, Any]) -> bool:
    return bool(sentence_obj.get("sink", {}).get("Is_it_sinking", False))


# =========================
# Embedding + cosine
# =========================
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def get_embedding_client():
    global _EMBED_CLIENT
    if _EMBED_CLIENT is None:
        _EMBED_CLIENT = make_embedding_client()
    return _EMBED_CLIENT


def normalize_text_for_model(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_text_for_embedding(text: str, max_chars: int = EMBED_MAX_TEXT_CHARS) -> List[str]:
    text = normalize_text_for_model(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts = re.split(r"(?<=[。！？；])|\n+", text)
    chunks: List[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(part), max_chars):
                chunk = part[i:i + max_chars].strip()
                if chunk:
                    chunks.append(chunk)
            continue

        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks or [text[:max_chars]]


def average_embeddings(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        raise ValueError("empty embedding vectors")
    if len(vectors) == 1:
        return vectors[0]

    dim = len(vectors[0])
    avg = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            avg[i] += value
    avg = [value / len(vectors) for value in avg]

    norm = math.sqrt(sum(value * value for value in avg))
    if norm > 0:
        avg = [value / norm for value in avg]
    return avg


def trim_text_for_judge(text: str, max_chars: int = JUDGE_MAX_BACKGROUND_CHARS) -> str:
    text = normalize_text_for_model(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def request_embeddings_batch(batch: List[str], model: str = EMBED_MODEL):
    last_error = None

    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            return get_embedding_client().embeddings.create(
                model=model,
                input=batch,
                timeout=EMBED_TIMEOUT_SECONDS,
            )

        except BadRequestError:
            raise

        except (APITimeoutError, RateLimitError, APIError) as e:
            last_error = e
            emit_log(
                f"Step12 | embedding retry={attempt}/{EMBED_MAX_RETRIES} "
                f"batch={len(batch)} reason={type(e).__name__}: {e}"
            )
            if attempt == EMBED_MAX_RETRIES:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

        except Exception as e:
            last_error = e
            emit_log(
                f"Step12 | embedding retry={attempt}/{EMBED_MAX_RETRIES} "
                f"batch={len(batch)} reason={type(e).__name__}: {e}"
            )
            if attempt == EMBED_MAX_RETRIES:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(
        f"embedding 请求连续失败：batch={len(batch)} timeout={EMBED_TIMEOUT_SECONDS}s last_error={repr(last_error)}"
    )


def get_embeddings(texts: List[str], model: str = EMBED_MODEL) -> Dict[str, List[float]]:
    unique_texts = list(dict.fromkeys([t for t in texts if isinstance(t, str) and t.strip()]))

    with embedding_lock:
        missing = [t for t in unique_texts if t not in embedding_cache]

    if missing:
        for text in missing:
            chunks = split_text_for_embedding(text)
            if not chunks:
                continue

            vectors: List[List[float]] = []
            for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch = chunks[i:i + EMBED_BATCH_SIZE]
                resp = request_embeddings_batch(batch, model=model)
                vectors.extend(item.embedding for item in resp.data)

            with embedding_lock:
                embedding_cache[text] = average_embeddings(vectors)

    with embedding_lock:
        return {t: embedding_cache[t] for t in unique_texts}

def retrieve_top_k_sentences(candidate_sentences: List[Dict[str, Any]], target_background: str, top_k: int = TOP_K) -> List[Tuple[float, Dict[str, Any]]]:
    if not candidate_sentences:
        return []

    candidate_texts = [obj["sentence"] for obj in candidate_sentences if obj.get("sentence", "").strip()]
    if not candidate_texts:
        return []

    text_to_vec = get_embeddings([target_background] + candidate_texts)
    query_vec = text_to_vec[target_background]

    scored = []
    for obj in candidate_sentences:
        sent = obj.get("sentence", "").strip()
        if not sent:
            continue
        sim = cosine_similarity(text_to_vec[sent], query_vec)
        scored.append((sim, obj))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# =========================
# AgentA: issame 判定
# =========================
AGENT_A_SYSTEM = """
你是一个严格的“事件同一性”判定器。

任务：
判断“文本一”所描述的事件，是否与“文本二背景信息”描述的是同一个具体事件或同一事件过程。

判定标准：
1. 事件是否相同，取决于它们是否指向同一具体事件或同一事件过程。
2. 不要求人物主体完全一致。
3. 只要两个文本描述的是：
   - 同一事件的不同参与者
   - 同一事件的不同阶段（如准备、发生、结果）
   - 同一事件的不同视角
   都应判定为同一个事件。
4. 事件描述可以有详略差异，只要本质上是同一个事件，就输出 true。
5. 只有在明显不是同一个事件时，才输出 false。
6. 你只能输出 JSON，格式严格为：
   {"isSame": true}
   或
   {"isSame": false}
7. 不要输出任何解释、前后缀、markdown、代码块。
""".strip()

def judge_same_event(text1: str, text1_background: str, text2_background: str, model: str = LLM_MODEL) -> bool:
    """
    单次 API 判定。
    为了线程安全和简单性，这里每个 worker 内部各自创建 client。
    """
    cache_key = (text1, text1_background, text2_background)

    with judge_lock:
        if cache_key in judge_cache:
            return judge_cache[cache_key]

    text1 = normalize_text_for_model(text1)
    text1_background = trim_text_for_judge(text1_background)
    text2_background = trim_text_for_judge(text2_background)

    user_prompt = f"""
请判断下面的“文本一”与“文本二背景信息”是否描述同一个事件。

【文本一】
{text1}

【文本一背景信息】
{text1_background}

【文本二背景信息】
{text2_background}
""".strip()

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            local_client = make_sync_chat_client()
            completion = create_chat_completion(
                local_client,
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": AGENT_A_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or ""
            parsed = validate_json_text(Sentencesame, content)
            result = bool(parsed.isSame)

            with judge_lock:
                judge_cache[cache_key] = result
            return result

        except Exception as e:
            last_error = e
            time.sleep(1.2 * attempt)

    raise RuntimeError(f"issame 判定连续失败：{last_error}")


# =========================
# 选择 source / target
# 规则：
#   文本二 = 交叉起点所在的 timeblock = start 更晚的那个
#   若 target 的 Granularity == "0"，直接跳过
#
# 若 start 相同：
#   1) 优先选 Granularity != '0' 的那个作为文本二
#   2) 若仍相同，再选时间跨度更短的那个作为文本二
# =========================
def pick_roles(doc1: str, tb1: Dict[str, Any], doc2: str, tb2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    start1, end1 = parse_iso_range(tb1["iso_range"])
    start2, end2 = parse_iso_range(tb2["iso_range"])

    if start1 > start2:
        source = {"doc_id": doc2, "tb": tb2}
        target = {"doc_id": doc1, "tb": tb1}
    elif start2 > start1:
        source = {"doc_id": doc1, "tb": tb1}
        target = {"doc_id": doc2, "tb": tb2}
    else:
        g1 = str(tb1.get("Granularity", ""))
        g2 = str(tb2.get("Granularity", ""))

        if g1 != "0" and g2 == "0":
            source = {"doc_id": doc2, "tb": tb2}
            target = {"doc_id": doc1, "tb": tb1}
        elif g2 != "0" and g1 == "0":
            source = {"doc_id": doc1, "tb": tb1}
            target = {"doc_id": doc2, "tb": tb2}
        else:
            span1 = tuple_to_ordinalish(end1) - tuple_to_ordinalish(start1)
            span2 = tuple_to_ordinalish(end2) - tuple_to_ordinalish(start2)

            if span1 <= span2:
                source = {"doc_id": doc2, "tb": tb2}
                target = {"doc_id": doc1, "tb": tb1}
            else:
                source = {"doc_id": doc1, "tb": tb1}
                target = {"doc_id": doc2, "tb": tb2}

    return source, target


# =========================
# 加载全部数据
# =========================
def build_data_index():
    if not SENTENCE_DIR.exists():
        raise FileNotFoundError(f"找不到目录: {SENTENCE_DIR}")
    if not TIMEBLOCK_DIR.exists():
        raise FileNotFoundError(f"找不到目录: {TIMEBLOCK_DIR}")

    sentence_files = sorted([
        p for p in SENTENCE_DIR.glob("*.json")
        if any(p.stem.endswith(suffix) for suffix in SENTENCE_FILE_SUFFIXES)
    ])
    timeblock_files = sorted([
        p for p in TIMEBLOCK_DIR.glob("*.json")
        if any(p.stem.endswith(suffix) for suffix in TIMEBLOCK_FILE_SUFFIXES)
    ])

    if not sentence_files:
        raise FileNotFoundError(f"{SENTENCE_DIR} 下没有 sentence json 文件")
    if not timeblock_files:
        raise FileNotFoundError(f"{TIMEBLOCK_DIR} 下没有 timeblock json 文件")

    sentence_map: Dict[str, List[Dict[str, Any]]] = {}
    sentence_filename_map: Dict[str, str] = {}

    for path in sentence_files:
        doc_id = extract_doc_key_from_sentence_file(path)
        data = load_json(path)

        if not isinstance(data, list):
            raise ValueError(f"{path} 不是句子 list JSON")

        for obj in data:
            normalize_crossdoc_transfer(obj)

        sentence_map[doc_id] = data
        sentence_filename_map[doc_id] = path.name

    timeblock_map: Dict[str, List[Dict[str, Any]]] = {}
    for path in timeblock_files:
        doc_id = extract_doc_key_from_timeblock_file(path)
        data = load_json(path)
        tmb = data.get("TMB", [])

        if not isinstance(tmb, list):
            raise ValueError(f"{path} 中的 TMB 不是 list")

        timeblock_map[doc_id] = tmb

    common_doc_ids = sorted(set(sentence_map.keys()) & set(timeblock_map.keys()))
    if not common_doc_ids:
        raise ValueError("step5output 和 step11output 没有共同 doc_id")

    only_sentence = sorted(set(sentence_map.keys()) - set(timeblock_map.keys()))
    only_timeblock = sorted(set(timeblock_map.keys()) - set(sentence_map.keys()))

    if only_sentence:
        print(f"[yellow]仅在 sentence 中存在，未参与处理：{only_sentence}[/yellow]")
    if only_timeblock:
        print(f"[yellow]仅在 timeblock 中存在，未参与处理：{only_timeblock}[/yellow]")

    return common_doc_ids, sentence_map, timeblock_map, sentence_filename_map


# =========================
# 收集所有 jobs（先不打 API）
# 每个 job 对应：
#   一个 source sentence
#   一个 target timeblock 背景
# =========================
def collect_jobs(
    sentence_map: Dict[str, List[Dict[str, Any]]],
    timeblock_map: Dict[str, List[Dict[str, Any]]],
    doc_ids: List[str],
    reporter: Optional[StepReporter] = None,
):
    jobs = []
    pair_logs = []
    seen_job_keys = set()
    retrieval_count = 0

    doc_pairs = list(itertools.combinations(doc_ids, 2))
    for pair_index, (doc1, doc2) in enumerate(step_tqdm(doc_pairs, desc="收集 timeblock 配对"), start=1):
        pair_jobs_before = len(jobs)
        if reporter is not None:
            reporter.info(f"阶段1 doc_pair={pair_index}/{len(doc_pairs)} {doc1} <-> {doc2}")
        tbs1 = timeblock_map[doc1]
        tbs2 = timeblock_map[doc2]

        for tb1 in tbs1:
            for tb2 in tbs2:
                r1 = tb1.get("iso_range", "")
                r2 = tb2.get("iso_range", "")
                if not r1 or not r2:
                    continue

                # 第一步：必须有交集
                ov = overlap_interval(r1, r2)
                if ov is None:
                    continue

                # 完全相同的 iso_range 不处理
                if same_iso_range(r1, r2):
                    continue

                # 根据规则选 source / target
                source, target = pick_roles(doc1, tb1, doc2, tb2)
                source_doc_id = source["doc_id"]
                target_doc_id = target["doc_id"]
                source_tb = source["tb"]
                target_tb = target["tb"]

                # target 不能是 Granularity == "0"
                if str(target_tb.get("Granularity", "")) == "0":
                    continue

                source_block_sentences = get_sentences_by_timeblock_range(
                    sentence_map[source_doc_id],
                    source_tb["timeblock_range"]
                )
                target_block_sentences = get_sentences_by_timeblock_range(
                    sentence_map[target_doc_id],
                    target_tb["timeblock_range"]
                )

                if not source_block_sentences or not target_block_sentences:
                    continue

                source_background = aggregate_sentences(source_block_sentences)
                target_background = aggregate_sentences(target_block_sentences)

                # 只保留 sinking = false 的句子
                candidate_sentences = [
                    obj for obj in source_block_sentences
                    if not is_sinking(obj)
                ]
                if not candidate_sentences:
                    continue

                # 用 target_background 去 source timeblock 中召回 top_k
                retrieval_count += 1
                emit_log(
                    f"Step12 | stage1 retrieval_start | pair={pair_index}/{len(doc_pairs)} "
                    f"source_tb={source_tb['ID']} target_tb={target_tb['ID']} "
                    f"source_sentences={len(source_block_sentences)} target_sentences={len(target_block_sentences)} "
                    f"candidates={len(candidate_sentences)}"
                )
                top_hits = retrieve_top_k_sentences(
                    candidate_sentences=candidate_sentences,
                    target_background=target_background,
                    top_k=TOP_K
                )
                emit_log(
                    f"Step12 | stage1 retrieval_done | pair={pair_index}/{len(doc_pairs)} "
                    f"source_tb={source_tb['ID']} target_tb={target_tb['ID']} hits={len(top_hits)}"
                )
                if reporter is not None and retrieval_count % STAGE1_PROGRESS_EVERY == 0:
                    reporter.info(
                        f"阶段1检索={retrieval_count} 累计jobs={len(jobs)} "
                        f"当前doc_pair={pair_index}/{len(doc_pairs)}"
                    )
                if not top_hits:
                    continue

                pair_logs.append({
                    "doc_pair": [doc1, doc2],
                    "source_doc_id": source_doc_id,
                    "source_timeblock_id": source_tb["ID"],
                    "source_timeblock_range": source_tb["timeblock_range"],
                    "source_iso_range": source_tb["iso_range"],
                    "target_doc_id": target_doc_id,
                    "target_timeblock_id": target_tb["ID"],
                    "target_timeblock_range": target_tb["timeblock_range"],
                    "target_iso_range": target_tb["iso_range"],
                    "source_candidates": len(candidate_sentences),
                    "top_k_used": len(top_hits),
                })

                # 为每个 top hit 创建一个独立判定 job
                for sim, sentence_obj in top_hits:
                    job_key = (
                        source_doc_id,
                        sentence_obj["number"],
                        target_doc_id,
                        target_tb["ID"]
                    )
                    if job_key in seen_job_keys:
                        continue
                    seen_job_keys.add(job_key)

                    jobs.append({
                        "source_doc_id": source_doc_id,
                        "source_sentence_number": sentence_obj["number"],
                        "source_sentence": sentence_obj["sentence"],
                        "source_sentence_obj": sentence_obj,  # 保留原对象引用，后面直接更新
                        "source_timeblock_id": source_tb["ID"],
                        "source_timeblock_range": source_tb["timeblock_range"],
                        "target_doc_id": target_doc_id,
                        "target_timeblock_id": target_tb["ID"],
                        "target_timeblock_range": target_tb["timeblock_range"],
                        "similarity": sim,
                        "text1": sentence_obj["sentence"],
                        "text1_background": source_background,
                        "text2_background": target_background,
                    })

        if reporter is not None:
            reporter.info(
                f"阶段1完成 doc_pair={pair_index}/{len(doc_pairs)} "
                f"新增jobs={len(jobs) - pair_jobs_before} 累计jobs={len(jobs)}"
            )

    return jobs, pair_logs, doc_pairs


# =========================
# 单个 job 的 worker
# 不在 worker 内直接写共享状态，只返回结果
# =========================
def run_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    same = judge_same_event(
        text1=job["text1"],
        text1_background=job["text1_background"],
        text2_background=job["text2_background"],
        model=LLM_MODEL
    )
    return {
        "same": same,
        "job": job
    }


# =========================
# checkpoint
# 只输出 sentence 结果文件
# 文件名保持 sentence 原始 json 文件名不变
# =========================
def save_checkpoint(sentence_map, sentence_filename_map):
    for doc_id, sentences in sentence_map.items():
        output_name = sentence_filename_map.get(doc_id, f"{doc_id}_sentence.json")
        save_json(sentences, OUTPUT_DIR / output_name)


def serialize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if key != "source_sentence_obj"
    }


def save_stage1_artifacts(jobs: List[Dict[str, Any]], pair_logs: List[Dict[str, Any]]) -> None:
    save_json(
        {
            "mode": STEP12_MODE,
            "job_count": len(jobs),
            "jobs": [serialize_job(job) for job in jobs],
        },
        OUTPUT_DIR / STEP12_JOB_ARTIFACT,
    )
    save_json(pair_logs, OUTPUT_DIR / STEP12_PAIR_LOG_ARTIFACT)


def build_job_key(job: Dict[str, Any]) -> str:
    return "||".join([
        str(job.get("source_doc_id", "")),
        str(job.get("source_sentence_number", "")),
        str(job.get("target_doc_id", "")),
        str(job.get("target_timeblock_id", "")),
    ])


def load_judge_results_cache() -> Dict[str, bool]:
    cache_path = OUTPUT_DIR / STEP12_JUDGE_RESULTS_ARTIFACT
    if not cache_path.exists():
        return {}

    payload = load_json(cache_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{cache_path} 不是 dict")

    results = payload.get("results", payload)
    if not isinstance(results, dict):
        raise ValueError(f"{cache_path} 中的 results 不是 dict")

    return {
        str(key): bool(value)
        for key, value in results.items()
    }


def save_judge_results_cache(results: Dict[str, bool]) -> None:
    save_json(
        {
            "result_count": len(results),
            "results": results,
        },
        OUTPUT_DIR / STEP12_JUDGE_RESULTS_ARTIFACT,
    )


def load_stage1_jobs() -> List[Dict[str, Any]]:
    artifact_path = OUTPUT_DIR / STEP12_JOB_ARTIFACT
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"未找到 Step12 阶段1产物：{artifact_path}。请先运行 STEP12_MODE=retrieve_only。"
        )

    payload = load_json(artifact_path)
    jobs = payload.get("jobs", payload) if isinstance(payload, dict) else payload
    if not isinstance(jobs, list):
        raise ValueError(f"{artifact_path} 中的 jobs 不是 list")
    return jobs


def attach_sentence_refs(
    jobs: List[Dict[str, Any]],
    sentence_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    sentence_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for doc_id, sentences in sentence_map.items():
        for sentence_obj in sentences:
            number = str(sentence_obj.get("number", "")).strip()
            if number:
                sentence_lookup[(doc_id, number)] = sentence_obj

    attached_jobs: List[Dict[str, Any]] = []
    for job in jobs:
        key = (str(job.get("source_doc_id", "")), str(job.get("source_sentence_number", "")))
        sentence_obj = sentence_lookup.get(key)
        if sentence_obj is None:
            raise KeyError(f"找不到 source sentence: doc_id={key[0]} number={key[1]}")

        enriched_job = dict(job)
        enriched_job["source_sentence_obj"] = sentence_obj
        attached_jobs.append(enriched_job)

    return attached_jobs


def run_stage2_jobs(
    jobs: List[Dict[str, Any]],
    sentence_map: Dict[str, List[Dict[str, Any]]],
    sentence_filename_map: Dict[str, str],
    reporter: StepReporter,
):
    match_logs = []
    finished_count = 0
    save_counter = 0
    cached_results = load_judge_results_cache()

    reporter.total = len(jobs)
    reporter.unit = "job"
    reporter.info(f"待判定 jobs={len(jobs)}")

    if not jobs:
        save_checkpoint(sentence_map, sentence_filename_map)
        reporter.finish(output_dir=OUTPUT_DIR, extra="无需判定")
        return sentence_map, match_logs

    cached_jobs = []
    pending_jobs = []
    for job in jobs:
        job_key = build_job_key(job)
        if job_key in cached_results:
            cached_jobs.append((job_key, job, cached_results[job_key]))
        else:
            pending_jobs.append(job)

    for job_key, job, same in cached_jobs:
        if same:
            append_same_timeblock_id(job["source_sentence_obj"], job["target_timeblock_id"])
            match_logs.append({
                "source_doc_id": job["source_doc_id"],
                "source_sentence_number": job["source_sentence_number"],
                "source_sentence": job["source_sentence"],
                "source_timeblock_id": job["source_timeblock_id"],
                "source_timeblock_range": job["source_timeblock_range"],
                "target_doc_id": job["target_doc_id"],
                "target_timeblock_id": job["target_timeblock_id"],
                "target_timeblock_range": job["target_timeblock_range"],
                "similarity": job["similarity"],
                "cached": True,
            })

    if cached_jobs:
        reporter.info(f"阶段2缓存命中={len(cached_jobs)} 待新判定={len(pending_jobs)}")

    if not pending_jobs:
        save_checkpoint(sentence_map, sentence_filename_map)
        reporter.finish(output_dir=OUTPUT_DIR, extra=f"jobs={len(jobs)} match_logs={len(match_logs)} cached={len(cached_jobs)}")
        return sentence_map, match_logs

    reporter.info(f"阶段2=并发 issame max_workers={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(run_one_job, job): job for job in pending_jobs}

        for future in step_tqdm(as_completed(future_to_job), total=len(future_to_job), desc="并发判定 issame"):
            try:
                result = future.result()
                same = result["same"]
                job = result["job"]
                job_key = build_job_key(job)
                cached_results[job_key] = same
                save_counter += 1

                if same:
                    append_same_timeblock_id(job["source_sentence_obj"], job["target_timeblock_id"])
                    match_logs.append({
                        "source_doc_id": job["source_doc_id"],
                        "source_sentence_number": job["source_sentence_number"],
                        "source_sentence": job["source_sentence"],
                        "source_timeblock_id": job["source_timeblock_id"],
                        "source_timeblock_range": job["source_timeblock_range"],
                        "target_doc_id": job["target_doc_id"],
                        "target_timeblock_id": job["target_timeblock_id"],
                        "target_timeblock_range": job["target_timeblock_range"],
                        "similarity": job["similarity"],
                    })

            except Exception as e:
                bad_job = future_to_job[future]
                match_logs.append({
                    "error": str(e),
                    "source_doc_id": bad_job["source_doc_id"],
                    "source_sentence_number": bad_job["source_sentence_number"],
                    "target_doc_id": bad_job["target_doc_id"],
                    "target_timeblock_id": bad_job["target_timeblock_id"],
                })

            finished_count += 1

            if save_counter and save_counter % JUDGE_RESULTS_SAVE_EVERY == 0:
                save_judge_results_cache(cached_results)

            if finished_count % CHECKPOINT_EVERY == 0:
                save_judge_results_cache(cached_results)
                save_checkpoint(sentence_map, sentence_filename_map)
                reporter.info(f"进度={finished_count}/{len(pending_jobs)} 匹配={len(match_logs)}")

    save_judge_results_cache(cached_results)
    save_checkpoint(sentence_map, sentence_filename_map)
    reporter.finish(output_dir=OUTPUT_DIR, extra=f"jobs={len(jobs)} match_logs={len(match_logs)}")

    return sentence_map, match_logs


# =========================
# 主流程
# =========================
def process_all_pairs_parallel():
    reporter = StepReporter("Step12")
    doc_ids, sentence_map, timeblock_map, sentence_filename_map = build_data_index()
    reporter.start(input_dir=SENTENCE_DIR, output_dir=OUTPUT_DIR, extra=f"文档={len(doc_ids)} mode={STEP12_MODE}")

    if STEP12_MODE == "judge_only":
        reporter.info("阶段2=加载阶段1产物")
        jobs = attach_sentence_refs(load_stage1_jobs(), sentence_map)
        sentence_map, match_logs = run_stage2_jobs(jobs, sentence_map, sentence_filename_map, reporter)
        return sentence_map, [], match_logs

    reporter.info("阶段1=收集候选 jobs")
    jobs, pair_logs, doc_pairs = collect_jobs(sentence_map, timeblock_map, doc_ids, reporter=reporter)
    save_stage1_artifacts(jobs, pair_logs)
    reporter.info(
        f"阶段1产物已写入 {OUTPUT_DIR / STEP12_JOB_ARTIFACT} "
        f"jobs={len(jobs)} 文档对={len(doc_pairs)}"
    )

    if STEP12_MODE == "retrieve_only":
        reporter.finish(output_dir=OUTPUT_DIR, extra=f"stage1_jobs={len(jobs)} 文档对={len(doc_pairs)}")
        return sentence_map, pair_logs, []

    sentence_map, match_logs = run_stage2_jobs(jobs, sentence_map, sentence_filename_map, reporter)
    return sentence_map, pair_logs, match_logs


def main() -> None:
    global RUN_ROOT, SENTENCE_DIR, TIMEBLOCK_DIR, OUTPUT_DIR

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    SENTENCE_DIR = sentence_step_dir(RUN_ROOT, 5)
    TIMEBLOCK_DIR = timeblock_step_dir(RUN_ROOT, 11)
    OUTPUT_DIR = sentence_step_dir(RUN_ROOT, 12)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_step_logging(RUN_ROOT, "step_12_cross_document_alignment")
    process_all_pairs_parallel()


if __name__ == "__main__":
    main()
