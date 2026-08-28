# ===========================================
# Step2: 批量人物判断（通用版）
# 人物集合、默认传主和别名表从 Step 1 manifest 动态生成。
# ===========================================

import asyncio
import json
import os
import random
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import APIError, APITimeoutError, AsyncOpenAI, BadRequestError, RateLimitError
from pydantic import ValidationError, create_model
from rich import print
from tqdm.auto import tqdm

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion_async,
    make_async_chat_client,
)
from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir

# ---------- 0. Runtime state ----------
RUN_ROOT: Path
INPUT_DIR: Path
OUTPUT_DIR: Path
MANIFEST_PATH: Path

# ---------- 1. 运行参数 ----------
MODEL_NAME = CHAT_MODEL
WINDOW = 12
CONCURRENCY = int(os.getenv("AIH_AGENT_CONCURRENCY", "40"))
MAX_RETRIES = 6
REQUEST_TIMEOUT = float(os.getenv("AIH_REQUEST_TIMEOUT", "300"))
SKIP_EXISTING = False
OVERWRITE_CHARACTERS = True
USE_LLM_ALIAS_GENERATION = True
ALIAS_MAX_RETRIES = 3

# ---------- 2. Chat client ----------
client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    if client is None:
        raise RuntimeError("Step 2 client has not been initialized")
    return client

# ---------- 3. 运行时状态 ----------
RUN_MANIFEST: Dict[str, Any] = {}
FILE_METADATA: Dict[str, Dict[str, Any]] = {}
ALL_PEOPLE: List[str] = []
ALIASES: Dict[str, List[str]] = {}
TITLE_SUFFIXES = [
    "本纪",
    "帝纪",
    "皇后传",
    "后妃传",
    "列传",
    "世家",
    "传",
    "纪",
]

# ---------- 4. 文件名解析 ----------
FILENAME_RE = re.compile(
    r"^(?P<chapter>\d+)_(?P<uuid>[0-9a-fA-F-]+)_sentence\.json$"
)


def parse_input_filename(path: Path) -> Tuple[int, str]:
    m = FILENAME_RE.match(path.name)
    if m:
        return int(m.group("chapter")), m.group("uuid")

    parts = path.stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return int(parts[0]), parts[1]

    raise ValueError(f"无法从文件名解析 chapter_id / book_uuid: {path.name}")


def sort_key_for_file(path: Path):
    try:
        chapter_id, book_uuid = parse_input_filename(path)
        return (chapter_id, book_uuid, path.name)
    except Exception:
        return (10**9, "", path.name)


# ---------- 5. 通用工具 ----------
def dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    results = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        results.append(value)
    return results


def load_step1_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}
    return data


def resolve_input_files(manifest: Dict[str, Any]) -> List[Path]:
    generated_files = manifest.get("generated_files")
    if isinstance(generated_files, list):
        resolved = []
        for name in generated_files:
            path = INPUT_DIR / str(name)
            if path.exists() and path.suffix == ".json":
                resolved.append(path)
        if resolved:
            return sorted(resolved, key=sort_key_for_file)

    return sorted(
        [path for path in INPUT_DIR.glob("*.json") if path.name != MANIFEST_PATH.name],
        key=sort_key_for_file,
    )


def collect_people_from_items(data: List[dict]) -> List[str]:
    people = []
    for item in data:
        characters = item.get("characters")
        if isinstance(characters, dict):
            people.extend(characters.keys())
    return dedupe_keep_order(people)


def infer_default_person_from_items(data: List[dict]) -> Optional[str]:
    for item in data:
        characters = item.get("characters")
        if not isinstance(characters, dict):
            continue
        for person, flag in characters.items():
            if bool(flag):
                return person
    return None


def resolve_people_for_file(file_name: str, data: List[dict]) -> Tuple[List[str], Optional[str]]:
    meta = FILE_METADATA.get(file_name, {})

    people = dedupe_keep_order(RUN_MANIFEST.get("folder_people", []))
    if not people:
        people = collect_people_from_items(data)

    default_person = meta.get("source_person")
    if not isinstance(default_person, str) or not default_person.strip():
        default_person = infer_default_person_from_items(data)
    else:
        default_person = default_person.strip()

    if default_person and default_person not in people:
        people.append(default_person)

    return people, default_person


# ---------- 6. 句子与编号处理 ----------
def get_sentence_text(item: dict) -> str:
    for key in ("sentence", "text", "content"):
        value = item.get(key)
        if isinstance(value, str):
            return value.strip()
    raise KeyError(
        f"记录里找不到句子字段。需要有 sentence / text / content 之一。当前 keys={list(item.keys())}"
    )


def normalize_number(raw_number, book_uuid: str, chapter_id: int, idx: int) -> str:
    raw = "" if raw_number is None else str(raw_number).strip()
    chapter_id_str = str(chapter_id)

    if raw:
        parts = raw.rsplit(".", 3)
        if len(parts) == 4:
            _, _, para_id, sent_id = parts
            return f"{book_uuid}.{chapter_id_str}.{para_id}.{sent_id}"

        if len(parts) == 3:
            _, para_id, sent_id = parts
            return f"{book_uuid}.{chapter_id_str}.{para_id}.{sent_id}"

        if len(parts) == 2:
            para_id, sent_id = parts
            return f"{book_uuid}.{chapter_id_str}.{para_id}.{sent_id}"

    return f"{book_uuid}.{chapter_id_str}.0.{idx+1}"


def parse_number_components(number: str) -> Tuple[str, int, str, str]:
    parts = str(number).rsplit(".", 3)
    if len(parts) == 4:
        book_uuid, chapter_id, para_id, sent_id = parts
    elif len(parts) == 3:
        book_uuid = ""
        chapter_id, para_id, sent_id = parts
    else:
        raise ValueError(f"number 不是 4 段格式: {number}")
    return book_uuid, int(chapter_id), para_id, sent_id


# ---------- 7. 别名表 ----------
def extract_title_aliases(title: str) -> List[str]:
    title = str(title).strip()
    if not title:
        return []

    aliases = [title]
    for suffix in TITLE_SUFFIXES:
        if title.endswith(suffix) and len(title) > len(suffix):
            base = title[: -len(suffix)].strip()
            if len(base) >= 2:
                aliases.append(base)
    return dedupe_keep_order(aliases)


def extract_person_aliases(person: str) -> List[str]:
    person = str(person).strip()
    aliases = [person]

    match = re.match(r"^(.*?[帝王汗后皇太子])(.*)$", person)
    if match:
        for part in match.groups():
            part = part.strip()
            if len(part) >= 2:
                aliases.append(part)

    return dedupe_keep_order(aliases)


def build_people_titles(people: List[str]) -> Dict[str, List[str]]:
    people_to_titles = {person: [] for person in people}
    for meta in FILE_METADATA.values():
        person = str(meta.get("source_person", "")).strip()
        title = str(meta.get("source_title") or meta.get("matched_title") or "").strip()
        if person in people_to_titles and title:
            people_to_titles[person].append(title)

    for person in people_to_titles:
        people_to_titles[person] = dedupe_keep_order(people_to_titles[person])
    return people_to_titles


def build_heuristic_alias_map(people: List[str]) -> Dict[str, List[str]]:
    people_to_titles = build_people_titles(people)
    alias_map = {}

    for person in people:
        aliases = extract_person_aliases(person)
        for title in people_to_titles.get(person, []):
            aliases.extend(extract_title_aliases(title))
        alias_map[person] = dedupe_keep_order(aliases)

    return alias_map


def merge_alias_maps(base: Dict[str, List[str]], extra: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = {}
    for person in dedupe_keep_order(list(base.keys()) + list(extra.keys())):
        merged[person] = dedupe_keep_order(base.get(person, []) + extra.get(person, []) + [person])
    return merged


# ---------- 8. 用 AI 生成自适应别名 ----------
def build_alias_messages(people_to_titles: Dict[str, List[str]]) -> List[dict]:
    lines = []
    for person, titles in people_to_titles.items():
        title_text = "、".join(titles) if titles else "无"
        lines.append(f"- {person}: {title_text}")

    developer_msg = """
你是历史人物别名整理助手。

你必须只输出一个 JSON 对象，格式为：
{"人物1": ["别名1", "别名2"], "人物2": ["别名1"]}

严格要求：
1. 键必须且只能是给定的人物名。
2. 每个值必须是字符串数组。
3. 每个人物的数组里必须保留其原名。
4. 只保留在该批文献中高把握会出现的称呼、帝号、谥号、简称。
5. 不要编造没有把握的别名；不确定时只保留原名。
6. 不要输出代词、关系词、泛化官职或群体称谓。
""".strip()

    user_msg = f"""
【书目候选】
{'、'.join(RUN_MANIFEST.get('candidate_books', [])) or '无'}

【人物与对应传记标题】
{chr(10).join(lines)}
""".strip()

    return [
        {"role": "system", "content": developer_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_json_object(text: str) -> str:
    if text is None:
        raise ValueError("模型返回为空")

    s = text.strip()

    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError(f"无法从模型输出中提取 JSON，对应内容前200字：{s[:200]}")
    return m.group(0)


async def generate_aliases_with_llm(people_to_titles: Dict[str, List[str]]) -> Dict[str, List[str]]:
    if not people_to_titles:
        return {}

    messages = build_alias_messages(people_to_titles)
    use_json_mode = True
    last_error = None

    for attempt in range(1, ALIAS_MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_completion_tokens=300,
                timeout=REQUEST_TIMEOUT,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp = await create_chat_completion_async(get_client(), **kwargs)
            text = resp.choices[0].message.content or ""
            payload = json.loads(extract_json_object(text))
            if not isinstance(payload, dict):
                raise ValueError("别名返回结果不是 JSON 对象")

            alias_map = {}
            for person in people_to_titles:
                raw_aliases = payload.get(person, [])
                if isinstance(raw_aliases, str):
                    raw_aliases = [raw_aliases]
                if not isinstance(raw_aliases, list):
                    raw_aliases = []
                cleaned = [person]
                for alias in raw_aliases:
                    alias = str(alias).strip()
                    if len(alias) >= 2:
                        cleaned.append(alias)
                alias_map[person] = dedupe_keep_order(cleaned)

            return alias_map

        except BadRequestError as e:
            last_error = e
            if use_json_mode:
                use_json_mode = False
                await asyncio.sleep(1.0)
                continue
            if attempt == ALIAS_MAX_RETRIES:
                break
            await asyncio.sleep(min(2 ** attempt + random.random(), 20))

        except (
            RateLimitError,
            APITimeoutError,
            APIError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            last_error = e
            if attempt == ALIAS_MAX_RETRIES:
                break
            await asyncio.sleep(min(2 ** attempt + random.random(), 20))

    print(f"[yellow]AI 别名生成失败，回退到启发式别名。错误：{repr(last_error)}[/yellow]")
    return {}


# ---------- 9. 显式别名规则 ----------
def strong_aliases(person_name: str) -> List[str]:
    aliases = ALIASES.get(person_name, [person_name])
    aliases = [alias for alias in aliases if len(alias) >= 2]
    if not aliases and len(person_name) >= 2:
        aliases = [person_name]
    return aliases


def split_explicit_true_and_unresolved(sentence: str, targets: List[str]) -> Tuple[List[str], List[str]]:
    explicit_true = []
    unresolved = []

    for person in targets:
        aliases = strong_aliases(person)
        if any(alias in sentence for alias in aliases):
            explicit_true.append(person)
        else:
            unresolved.append(person)

    return explicit_true, unresolved


# ---------- 10. 动态 Pydantic 校验 ----------
@lru_cache(maxsize=None)
def get_flags_model(targets_tuple: Tuple[str, ...]):
    fields = {name: (bool, False) for name in targets_tuple}
    model_name = "PeopleFlags_" + "_".join(targets_tuple)
    return create_model(model_name, **fields)


# ---------- 11. Prompt ----------
def build_messages(
    context: str,
    target_sentence: str,
    unresolved_targets: List[str],
    known_true_targets: List[str],
    chapter_id: int,
    default_person: Optional[str],
) -> List[dict]:
    target_keys_str = ", ".join([f'"{name}": bool' for name in unresolved_targets])
    unresolved_text = "、".join(unresolved_targets)
    known_true_text = "、".join(known_true_targets) if known_true_targets else "无"
    default_person_text = default_person if default_person else "无"

    alias_info = "\n".join(
        f"- {name}: {', '.join(ALIASES.get(name, [name]))}" for name in unresolved_targets
    )

    developer_msg = f"""
你是历史人物判定助手。

你必须只输出一个 JSON 对象，格式为：
{{{target_keys_str}}}

严格要求：
1. 只能输出 JSON，禁止输出解释、注释、代码块、前后缀文字。
2. 键必须且只能是：{", ".join(unresolved_targets)}
3. 每个值必须是 true 或 false
4. 同一句可以多人同时为 true

判定规则：
1. target 句若直接出现人物姓名、帝号、谥号或高把握别名，则该人物记为 true。
2. 若 target 句未直呼其名，但结合上文可知代词、称谓、省略主语、对话对象、行为承担者明确指向该人物，也记为 true。
3. 若只是背景提及、关系不明、推断不稳，则记为 false。
4. 本次只判断这些人物：{unresolved_text}
5. 已经由规则直接判定为 true 的人物：{known_true_text}
6. 当前传主（默认 true 人物）：{default_person_text}

别名表：
{alias_info}
""".strip()

    user_msg = f"""
【文件编号】
{chapter_id}

【背景】
{context}

【最后一句（target）】
{target_sentence}
""".strip()

    return [
        {"role": "system", "content": developer_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------- 12. 单次 LLM 调用 ----------
async def llm_call(
    context: str,
    target_sentence: str,
    unresolved_targets: List[str],
    known_true_targets: List[str],
    chapter_id: int,
    default_person: Optional[str],
) -> Dict[str, bool]:
    targets_tuple = tuple(unresolved_targets)
    FlagsModel = get_flags_model(targets_tuple)
    messages = build_messages(
        context=context,
        target_sentence=target_sentence,
        unresolved_targets=unresolved_targets,
        known_true_targets=known_true_targets,
        chapter_id=chapter_id,
        default_person=default_person,
    )

    use_json_mode = True
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_completion_tokens=120,
                timeout=REQUEST_TIMEOUT,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp = await create_chat_completion_async(get_client(), **kwargs)
            text = resp.choices[0].message.content or ""
            payload = json.loads(extract_json_object(text))

            normalized_payload = {
                name: bool(payload.get(name, False))
                for name in unresolved_targets
            }

            validated = FlagsModel(**normalized_payload)
            if hasattr(validated, "model_dump"):
                validated_payload = validated.model_dump()
            else:
                validated_payload = validated.dict()

            return {
                name: bool(validated_payload.get(name, False))
                for name in unresolved_targets
            }

        except BadRequestError as e:
            last_error = e
            if use_json_mode:
                use_json_mode = False
                await asyncio.sleep(1.0)
                continue

            if attempt == MAX_RETRIES:
                break
            await asyncio.sleep(min(2 ** attempt + random.random(), 20))

        except (
            RateLimitError,
            APITimeoutError,
            APIError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as e:
            last_error = e
            if use_json_mode and isinstance(e, (json.JSONDecodeError, ValueError)):
                use_json_mode = False
                await asyncio.sleep(1.0)
                continue
            if attempt == MAX_RETRIES:
                break
            await asyncio.sleep(min(2 ** attempt + random.random(), 20))

    print(
        "[yellow]LLM 判定失败，保守回退为 false | "
        f"chapter_id={chapter_id} | unresolved_targets={unresolved_targets} | error={repr(last_error)}[/yellow]"
    )
    return {name: False for name in unresolved_targets}


# ---------- 13. 构造上下文 ----------
def build_context(data: List[dict], idx: int, window: int = WINDOW) -> str:
    start = max(0, idx - window)
    lines = []
    for d in data[start: idx + 1]:
        lines.append(f"{d['number']} {get_sentence_text(d)}")
    return "\n".join(lines)


# ---------- 14. 单文件处理 ----------
async def annotate_one_file(file_path: Path) -> dict:
    chapter_id, book_uuid = parse_input_filename(file_path)
    file_meta = FILE_METADATA.get(file_path.name, {})
    chapter_text = file_meta.get("source_title") or file_meta.get("matched_title") or f"chapter_{chapter_id}"

    output_path = OUTPUT_DIR / file_path.name
    if SKIP_EXISTING and output_path.exists():
        print(f"[yellow]跳过已存在文件：{output_path.name}[/yellow]")
        return {
            "file": file_path.name,
            "status": "skipped",
            "total_sentences": None,
            "llm_calls": 0,
        }

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{file_path.name} 的顶层 JSON 不是 list，而是 {type(data)}")

    all_people, default_person = resolve_people_for_file(file_path.name, data)

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"{file_path.name} 第 {idx} 条记录不是 dict")

        _ = get_sentence_text(item)
        item["number"] = normalize_number(item.get("number"), book_uuid, chapter_id, idx)

        if OVERWRITE_CHARACTERS or "characters" not in item or not isinstance(item["characters"], dict):
            item["characters"] = {}

        item["characters"] = {person: False for person in all_people}
        if default_person is not None and default_person in item["characters"]:
            item["characters"][default_person] = True

    pending_jobs = []
    explicit_hits = 0

    for idx, item in enumerate(data):
        sentence = get_sentence_text(item)

        if default_person is not None and default_person in all_people:
            candidate_targets = [p for p in all_people if p != default_person]
            known_true_targets = [default_person]
        else:
            candidate_targets = all_people[:]
            known_true_targets = []

        explicit_true, unresolved = split_explicit_true_and_unresolved(sentence, candidate_targets)

        for person in explicit_true:
            item["characters"][person] = True

        known_true_targets = known_true_targets + explicit_true
        explicit_hits += len(explicit_true)

        if unresolved:
            pending_jobs.append(
                {
                    "idx": idx,
                    "unresolved_targets": unresolved,
                    "known_true_targets": known_true_targets,
                }
            )

    semaphore = asyncio.Semaphore(CONCURRENCY)
    llm_calls = len(pending_jobs)

    async def worker(job: dict):
        idx = job["idx"]
        unresolved_targets = job["unresolved_targets"]
        known_true_targets = job["known_true_targets"]

        async with semaphore:
            context = build_context(data, idx, window=WINDOW)
            target_sentence = get_sentence_text(data[idx])

            result = await llm_call(
                context=context,
                target_sentence=target_sentence,
                unresolved_targets=unresolved_targets,
                known_true_targets=known_true_targets,
                chapter_id=chapter_id,
                default_person=default_person,
            )
            return idx, result

    if llm_calls > 0:
        pbar = tqdm(total=llm_calls, desc=file_path.name, leave=True)
        tasks = [asyncio.create_task(worker(job)) for job in pending_jobs]

        try:
            for future in asyncio.as_completed(tasks):
                idx, result = await future
                for person, flag in result.items():
                    data[idx]["characters"][person] = flag
                pbar.update(1)
        finally:
            pbar.close()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"[green]完成[/green] {file_path.name} -> {output_path} | "
        f"传记={chapter_text} | 句子数={len(data)} | LLM调用数={llm_calls} | 显式命中数={explicit_hits}"
    )

    return {
        "file": file_path.name,
        "status": "done",
        "chapter_id": chapter_id,
        "chapter_name": chapter_text,
        "book_uuid": book_uuid,
        "total_sentences": len(data),
        "llm_calls": llm_calls,
        "explicit_hits": explicit_hits,
        "output": str(output_path),
    }


# ---------- 15. 主程序 ----------
async def main():
    global client, RUN_ROOT, INPUT_DIR, OUTPUT_DIR, MANIFEST_PATH
    global RUN_MANIFEST, FILE_METADATA, ALL_PEOPLE, ALIASES

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    INPUT_DIR = sentence_step_dir(RUN_ROOT, 1)
    OUTPUT_DIR = sentence_step_dir(RUN_ROOT, 2)
    MANIFEST_PATH = INPUT_DIR / "_collection_manifest.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = make_async_chat_client()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在：{INPUT_DIR}")

    RUN_MANIFEST = load_step1_manifest(MANIFEST_PATH)
    file_metadata = RUN_MANIFEST.get("files", {})
    FILE_METADATA = file_metadata if isinstance(file_metadata, dict) else {}

    input_files = resolve_input_files(RUN_MANIFEST)
    if not input_files:
        raise FileNotFoundError(f"{INPUT_DIR} 里没有找到 json 文件")

    manifest_people = RUN_MANIFEST.get("folder_people", [])
    if isinstance(manifest_people, list):
        ALL_PEOPLE = dedupe_keep_order(manifest_people)
    else:
        ALL_PEOPLE = []

    if not ALL_PEOPLE:
        for file_path in input_files:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                ALL_PEOPLE = dedupe_keep_order(ALL_PEOPLE + collect_people_from_items(data))

    ALIASES = build_heuristic_alias_map(ALL_PEOPLE)
    if USE_LLM_ALIAS_GENERATION and ALL_PEOPLE:
        llm_aliases = await generate_aliases_with_llm(build_people_titles(ALL_PEOPLE))
        ALIASES = merge_alias_maps(ALIASES, llm_aliases)

    print(f"[cyan]输入目录[/cyan]：{INPUT_DIR}")
    print(f"[cyan]输出目录[/cyan]：{OUTPUT_DIR}")
    print(f"[cyan]模型[/cyan]：{MODEL_NAME}")
    print(f"[cyan]并发数[/cyan]：{CONCURRENCY}")
    print(f"[cyan]文件数[/cyan]：{len(input_files)}")
    print(f"[cyan]当前人物集合[/cyan]：{ALL_PEOPLE}")
    print("[cyan]当前别名表[/cyan]：")
    for person in ALL_PEOPLE:
        print(f"  - {person}: {ALIASES.get(person, [person])}")

    summaries = []
    for file_path in input_files:
        summary = await annotate_one_file(file_path)
        summaries.append(summary)

    done_files = [x for x in summaries if x["status"] == "done"]
    total_sentences = sum(x["total_sentences"] for x in done_files if x["total_sentences"] is not None)
    total_llm_calls = sum(x["llm_calls"] for x in done_files)

    print("\n[bold green]全部处理完成[/bold green]")
    print(f"共处理文件数：{len(done_files)}")
    print(f"总句子数：{total_sentences}")
    print(f"总 LLM 调用数：{total_llm_calls}")

    return summaries


if __name__ == "__main__":
    asyncio.run(main())
