import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_historian.pipeline.input_manifest import (
    load_input_manifest,
    source_metadata,
    stable_collection_uuid,
)
from ai_historian.pipeline.logging import StepReporter, setup_step_logging
from ai_historian.pipeline.paths import (
    PROJECT_ROOT,
    derive_result_dir_name_from_input,
    resolve_run_root_from_input_dir,
    sentence_step_dir,
)
from ai_historian.resources import BOOK_CATALOG

# =========================
# 基础路径配置
# =========================
BASE_DIR = PROJECT_ROOT
CATALOG_PATH = Path(
    os.getenv("AIH_BOOK_CATALOG", str(BOOK_CATALOG))
)
DEFAULT_INPUT_DIR = BASE_DIR / "examples" / "input"
LEGACY_INPUT_DIR = DEFAULT_INPUT_DIR
LEGACY_FOLDER_BOOK_MAP = {
    "shiji_txt_文件": "史记",
}


def resolve_input_dir() -> Path:
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1])
        candidates = [raw, BASE_DIR / raw, BASE_DIR / "白话文" / raw]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return raw

    if DEFAULT_INPUT_DIR.exists():
        return DEFAULT_INPUT_DIR
    return LEGACY_INPUT_DIR


def derive_result_dir_name(input_dir: Path) -> str:
    return derive_result_dir_name_from_input(input_dir)


def resolve_output_dir(input_dir: Path) -> Path:
    output_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run_root = resolve_run_root_from_input_dir(input_dir, output_arg)
    return sentence_step_dir(run_root, 1)


INPUT_DIR: Path
OUTPUT_DIR: Path
MANIFEST_PATH: Path

# =========================
# 读取文本：兼容多种编码
# =========================
def read_text_auto(file_path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "big5"]
    last_error = None

    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"无法读取文件：{file_path}\n最后一次错误：{last_error}")


# =========================
# 通用工具
# =========================
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


def normalize_book_title(book_name: str) -> str:
    book_name = str(book_name).strip()
    if not book_name:
        return book_name
    if book_name.startswith("《") and book_name.endswith("》"):
        return book_name
    return f"《{book_name}》"


# =========================
# 标题规范化
# =========================
def normalize_chapter_title(title: str) -> str:
    title = title.strip()
    title = re.sub(r"\s+第[一二三四五六七八九十百千零〇两]+$", "", title)
    return title.strip()


def normalize_match_text(title: str) -> str:
    title = normalize_chapter_title(title)
    return re.sub(r"[\s_]+", "", title)


# =========================
# 句子切分
# =========================
def split_sentences(paragraph: str):
    end_punctuations = {"。", "！", "？"}

    open_quotes = {"“", "‘", '"', "「", "『"}
    close_quotes = {"”", "’", '"', "」", "』"}

    open_parentheses = {"（", "(", "[", "【", "〔"}
    close_parentheses = {"）", ")", "]", "】", "〕"}

    sentences = []
    current_sentence = []
    quote_depth = 0
    paren_depth = 0

    def flush_current():
        nonlocal current_sentence
        sentence = "".join(current_sentence).strip()
        if sentence:
            sentences.append(sentence)
        current_sentence = []

    for i, char in enumerate(paragraph):
        current_sentence.append(char)

        if char in open_quotes:
            quote_depth += 1

        elif char in close_quotes:
            if quote_depth > 0:
                quote_depth -= 1
            if i > 0 and paragraph[i - 1] in end_punctuations and quote_depth == 0 and paren_depth == 0:
                flush_current()

        elif char in open_parentheses:
            paren_depth += 1

        elif char in close_parentheses:
            if paren_depth > 0:
                paren_depth -= 1

        elif char in end_punctuations:
            if quote_depth == 0 and paren_depth == 0:
                flush_current()

    if current_sentence:
        flush_current()

    return sentences


# =========================
# 段落切分
# =========================
def split_paragraphs(text: str):
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    raw_paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = []

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        para = re.sub(r"\s*\n\s*", "", para)
        para = para.strip()

        if para:
            paragraphs.append(para)

    return paragraphs


# =========================
# 输入元数据
# =========================
def resolve_source_metadata(
    txt_path: Path, documents: Dict[str, Dict[str, str]]
) -> Dict[str, str]:
    return source_metadata(txt_path, documents)


# =========================
# 构造默认句子对象
# =========================
def build_sentence_record(
    number: str,
    sentence: str,
    people: List[str],
    default_true_person: Optional[str],
):
    resolved_people = dedupe_keep_order(people + ([default_true_person] if default_true_person else []))
    characters = {person: False for person in resolved_people}

    if default_true_person and default_true_person in characters:
        characters[default_true_person] = True

    return {
        "number": number,
        "sentence": sentence,
        "characters": characters,
        "Original_time_information": {
            "exist": False,
            "OTI": ""
        },
        "sink": {
            "Is_it_sinking": False,
            "reason": "..."
        },
        "Interlude": False,
        "crossDocTransfer": {
            "isSame": False,
            "same_timeblock_id": []
        }
    }


# =========================
# 目录信息加载
# =========================
def load_catalog_items(catalog_path: Path) -> List[dict]:
    raw = read_text_auto(catalog_path)
    obj = json.loads(raw)

    if not isinstance(obj, list):
        raise ValueError("史书_uuid.json 顶层不是 list。")
    return obj


def extract_candidate_book_titles(input_dir: Path) -> List[str]:
    legacy = LEGACY_FOLDER_BOOK_MAP.get(input_dir.name)
    if legacy:
        return [normalize_book_title(legacy)]

    folder_name = input_dir.name.strip()
    folder_name = re.sub(r"_txt_文件$", "", folder_name)
    parts = [part.strip() for part in folder_name.split("_") if part.strip()]

    return dedupe_keep_order([normalize_book_title(part) for part in parts])


def resolve_candidate_books(book_titles: List[str], catalog_items: List[dict]) -> List[dict]:
    title_to_item = {
        str(item.get("book", "")).strip(): item
        for item in catalog_items
        if isinstance(item, dict) and item.get("book")
    }

    books = []
    for title in book_titles:
        item = title_to_item.get(title)
        if item is not None:
            books.append(item)

    return books


def build_chapter_entries(candidate_books: List[dict]) -> List[dict]:
    entries = []
    for book in candidate_books:
        book_title = str(book.get("book", "")).strip()
        book_uuid = str(book.get("uuid", "")).strip()
        for chapter in book.get("chapters", []):
            sequence = chapter.get("sequence")
            title = str(chapter.get("title", "")).strip()
            if not title or sequence is None:
                continue
            entries.append(
                {
                    "book": book_title,
                    "book_uuid": book_uuid,
                    "sequence": int(sequence),
                    "title": title,
                    "core_title": normalize_chapter_title(title),
                }
            )
    return entries


# =========================
# 章节匹配
# =========================
def find_chapter_info_by_filename(
    source_title: str,
    file_stem: str,
    chapter_entries: List[dict],
) -> Optional[dict]:
    queries = dedupe_keep_order([
        normalize_match_text(source_title),
        normalize_match_text(file_stem),
    ])

    for query in queries:
        exact = []
        for entry in chapter_entries:
            normalized_title = normalize_match_text(entry["title"])
            normalized_core = normalize_match_text(entry["core_title"])
            if query == normalized_title or query == normalized_core:
                exact.append(entry)
        if len(exact) == 1:
            return exact[0]

    for query in queries:
        if len(query) < 3:
            continue
        fuzzy = []
        for entry in chapter_entries:
            normalized_title = normalize_match_text(entry["title"])
            normalized_core = normalize_match_text(entry["core_title"])
            title_match = (
                len(normalized_title) >= 4
                and (query in normalized_title or normalized_title in query)
            )
            core_match = (
                len(normalized_core) >= 4
                and (query in normalized_core or normalized_core in query)
            )
            if title_match or core_match:
                fuzzy.append(entry)
        if len(fuzzy) == 1:
            return fuzzy[0]

    return None


# =========================
# 运行前计划
# =========================
def build_folder_people(
    txt_files: List[Path], documents: Dict[str, Dict[str, str]]
) -> List[str]:
    people = []
    for txt_path in txt_files:
        file_info = resolve_source_metadata(txt_path, documents)
        people.append(file_info["source_person"])
    return dedupe_keep_order(people)


def build_file_plan(
    txt_files: List[Path],
    candidate_books: List[dict],
    input_dir: Path,
    documents: Dict[str, Dict[str, str]],
) -> List[dict]:
    chapter_entries = build_chapter_entries(candidate_books)
    fallback_book_uuid = (
        str(candidate_books[0].get("uuid", "")).strip()
        if len(candidate_books) == 1
        else stable_collection_uuid(txt_files, documents)
    )
    fallback_book_title = (
        str(candidate_books[0].get("book", "")).strip()
        if len(candidate_books) == 1
        else input_dir.name
    )
    fallback_sequence = max((entry["sequence"] for entry in chapter_entries), default=0) + 1

    plans = []
    seen_output_names = set()

    for txt_path in txt_files:
        file_info = resolve_source_metadata(txt_path, documents)
        chapter_info = find_chapter_info_by_filename(
            source_title=file_info["source_title"],
            file_stem=file_info["file_stem"],
            chapter_entries=chapter_entries,
        )

        if chapter_info is not None:
            chapter_id = chapter_info["sequence"]
            matched_title = chapter_info["title"]
            book_title = chapter_info["book"]
            book_uuid = chapter_info["book_uuid"]
            matched_catalog = True
        else:
            chapter_id = fallback_sequence
            fallback_sequence += 1
            matched_title = file_info["source_title"]
            book_title = fallback_book_title
            book_uuid = fallback_book_uuid
            matched_catalog = False

        output_file_name = f"{chapter_id}_{book_uuid}_sentence.json"
        if output_file_name in seen_output_names:
            original_chapter_id = chapter_id
            while True:
                chapter_id = fallback_sequence
                fallback_sequence += 1
                output_file_name = f"{chapter_id}_{book_uuid}_sentence.json"
                if output_file_name not in seen_output_names:
                    break
            matched_catalog = False
            print(
                "Step1 | duplicate_output_resolved | "
                f"source_file={txt_path.name} original_chapter_id={original_chapter_id} "
                f"new_chapter_id={chapter_id} output={output_file_name}"
            )
        seen_output_names.add(output_file_name)

        plans.append(
            {
                "source_file": txt_path.name,
                "source_path": txt_path,
                "source_person": file_info["source_person"],
                "source_title": file_info["source_title"],
                "book": book_title,
                "book_uuid": book_uuid,
                "chapter_id": chapter_id,
                "matched_title": matched_title,
                "matched_catalog": matched_catalog,
                "output_file_name": output_file_name,
            }
        )

    return plans


# =========================
# 处理单个 txt 文件
# =========================
def process_single_txt_file(file_plan: Dict[str, Any], folder_people: List[str], output_dir: Path):
    txt_path = Path(file_plan["source_path"])
    source_person = file_plan["source_person"]
    book_uuid = file_plan["book_uuid"]
    chapter_id = file_plan["chapter_id"]

    text = read_text_auto(txt_path)
    paragraphs = split_paragraphs(text)

    results = []

    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        sentences = split_sentences(paragraph)

        if not sentences:
            sentences = [paragraph]

        for sentence_index, sentence in enumerate(sentences, start=1):
            number = f"{book_uuid}.{chapter_id}.{paragraph_index}.{sentence_index}"
            record = build_sentence_record(
                number=number,
                sentence=sentence,
                people=folder_people,
                default_true_person=source_person,
            )
            results.append(record)

    output_path = output_dir / file_plan["output_file_name"]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "source_file": file_plan["source_file"],
        "source_person": source_person,
        "source_title": file_plan["source_title"],
        "matched_title": file_plan["matched_title"],
        "matched_catalog": file_plan["matched_catalog"],
        "chapter_id": chapter_id,
        "book": file_plan["book"],
        "book_uuid": book_uuid,
        "sentence_count": len(results),
        "output_file": file_plan["output_file_name"],
        "output_path": str(output_path),
    }


# =========================
# manifest 输出
# =========================
def write_manifest(
    manifest_path: Path,
    input_dir: Path,
    candidate_book_titles: List[str],
    folder_people: List[str],
    success_list: List[dict],
    failed_list: List[dict],
):
    manifest = {
        "input_dir": str(input_dir),
        "input_dir_name": input_dir.name,
        "candidate_books": candidate_book_titles,
        "folder_people": folder_people,
        "generated_files": [item["output_file"] for item in success_list],
        "files": {
            item["output_file"]: {
                "source_file": item["source_file"],
                "source_person": item["source_person"],
                "source_title": item["source_title"],
                "matched_title": item["matched_title"],
                "matched_catalog": item["matched_catalog"],
                "book": item["book"],
                "book_uuid": item["book_uuid"],
                "chapter_id": item["chapter_id"],
            }
            for item in success_list
        },
        "failed": failed_list,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# =========================
# 主函数：批量处理当前输入目录下全部 txt
# =========================
def main():
    global INPUT_DIR, OUTPUT_DIR, MANIFEST_PATH

    INPUT_DIR = resolve_input_dir()
    OUTPUT_DIR = resolve_output_dir(INPUT_DIR)
    MANIFEST_PATH = OUTPUT_DIR / "_collection_manifest.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_step_logging(OUTPUT_DIR.parent.parent, "step_01_text_preprocess")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在：{INPUT_DIR}")

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"未找到目录文件：{CATALOG_PATH}")

    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"在 {INPUT_DIR} 下没有找到任何 .txt 文件")

    documents = load_input_manifest(INPUT_DIR, txt_files)
    catalog_items = load_catalog_items(CATALOG_PATH)
    candidate_book_titles = extract_candidate_book_titles(INPUT_DIR)
    candidate_books = resolve_candidate_books(candidate_book_titles, catalog_items)
    folder_people = build_folder_people(txt_files, documents)
    file_plans = build_file_plan(txt_files, candidate_books, INPUT_DIR, documents)
    reporter = StepReporter("Step1", total=len(file_plans))
    reporter.start(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        extra=f"命中文献={len(candidate_books)} 人物={len(folder_people)}",
    )

    success_list = []
    failed_list = []

    for file_plan in file_plans:
        output_path = OUTPUT_DIR / file_plan["output_file_name"]
        if output_path.exists():
            sentence_count = None
            try:
                existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
                if isinstance(existing_payload, list):
                    sentence_count = len(existing_payload)
            except Exception:
                sentence_count = None

            success_list.append(
                {
                    "source_file": file_plan["source_file"],
                    "source_person": file_plan["source_person"],
                    "source_title": file_plan["source_title"],
                    "matched_title": file_plan["matched_title"],
                    "matched_catalog": file_plan["matched_catalog"],
                    "chapter_id": file_plan["chapter_id"],
                    "book": file_plan["book"],
                    "book_uuid": file_plan["book_uuid"],
                    "sentence_count": sentence_count,
                    "output_file": file_plan["output_file_name"],
                    "output_path": str(output_path),
                }
            )
            reporter.item_skip(
                file_plan["source_file"],
                detail="已存在输出",
            )
            continue

        try:
            result = process_single_txt_file(
                file_plan=file_plan,
                folder_people=folder_people,
                output_dir=OUTPUT_DIR,
            )
            success_list.append(result)
            reporter.item_ok(
                result["source_file"],
                detail=f"句子={result['sentence_count']} 传主={result['source_person']}",
            )

        except Exception as e:
            failed_item = {
                "source_file": file_plan["source_file"],
                "error": str(e),
            }
            failed_list.append(failed_item)
            reporter.item_fail(file_plan["source_file"], e)

    write_manifest(
        manifest_path=MANIFEST_PATH,
        input_dir=INPUT_DIR,
        candidate_book_titles=candidate_book_titles,
        folder_people=folder_people,
        success_list=success_list,
        failed_list=failed_list,
    )

    reporter.finish(output_dir=OUTPUT_DIR, extra=f"manifest={MANIFEST_PATH.name}")

    return {
        "success": success_list,
        "failed": failed_list,
        "manifest": str(MANIFEST_PATH),
    }


if __name__ == "__main__":
    main()
