#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "code" else SCRIPT_DIR
ROOT = PACKAGE_DIR.parents[1]
EXPERIMENT1_DIR = PACKAGE_DIR / "inputs" / "latex-cases"
OUT_DIR = PACKAGE_DIR / "outputs" / "current" / "generated_results"
PACKETS_DIR = OUT_DIR / "packets"
TABLES_DIR = OUT_DIR / "tables"
AGENT_RUN_DIR = OUT_DIR / "agent_run"
ARCHIVE_DIR = OUT_DIR / "archive"
AIH_AGENT_DIR = ROOT
AIH_WEB_AGENT_DIR = ROOT
AIH_SOURCE_DIR = ROOT / "src"
AIH_RESOURCE_DIR = AIH_SOURCE_DIR / "ai_historian" / "resources"
CROSSDOC_SCOPE_PATH = AGENT_RUN_DIR / "experiment1_crossdoc_scope.json"
AGENT_SCRIPT_OVERRIDE_DIR = (
    AIH_SOURCE_DIR / "ai_historian" / "profiles" / "evaluation" / "stages"
)

DOC_TITLES = {
    "7": "项羽本纪",
    "8": "高祖本纪",
    "53": "萧相国世家",
}

DOC_PERSONS = {
    "7": "项羽",
    "8": "刘邦",
    "53": "萧何",
}

CASE_FILES = {
    "H-C1": [EXPERIMENT1_DIR / "H-C1" / "H-C1.tex"],
    "H-C2": [EXPERIMENT1_DIR / "H-C2" / "H-C2.tex"],
    "H-C3": [EXPERIMENT1_DIR / "H-C3" / "H-C3.tex"],
    "H-C4": [EXPERIMENT1_DIR / "H-C4" / "H-C4.tex"],
    "H-C5": [
        EXPERIMENT1_DIR / "H-C5" / "H-C5a.tex",
        EXPERIMENT1_DIR / "H-C5" / "H-C5b.tex",
    ],
    "H-C6": [
        EXPERIMENT1_DIR / "H-C6" / "H-C6a.tex",
        EXPERIMENT1_DIR / "H-C6" / "H-C6b.tex",
    ],
}

CROSSDOC_CASES = {"H-C5", "H-C6"}
CROSSDOC_CASE_DOC_PAIRS = {
    "H-C5": ("7", "8"),
    "H-C6": ("8", "53"),
}
PDF_SUPPRESSED_BOUNDARY_LABELS = {
    "汉军退却时",
}
PDF_BOUNDARY_LABEL_REPLACEMENTS = {
}
ENV_FILES = [
    ROOT / ".env",
]

AGENT_PIPELINE_STEPS = [
    (1, "step_01_text_preprocess.py"),
    (2, "step_02_character_detection.py"),
    (3, "step_03_time_info_extraction.py"),
    (4, "step_04_description_detection.py"),
    (5, "step_05_interlude_detection.py"),
    (6, "step_06_timeblock_generation.py"),
    (7, "step_07_timeblock_conversion.py"),
    (8, "step_08_sequence_sorting.py"),
    (9, "step_09_granularity_classification.py"),
    (10, "step_10_tm_generation.py"),
    (10, "step_10b_cross_document_prealign.py"),
    (10, "step_10c_single_document_stabilize.py"),
    (10, "step_10d_crossdoc_temporal_graph.py"),
    (11, "step_11_iso_normalization.py"),
]


@dataclass(frozen=True)
class SentenceRef:
    case_id: str
    part_id: str
    doc_code: str
    short_id: str
    full_id: str
    tex_sentence: str = ""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env_files() -> list[str]:
    loaded = []
    explicit_env_file = os.getenv("ENV_FILE", "").strip()
    candidates = [Path(explicit_env_file).expanduser()] if explicit_env_file else ENV_FILES
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            value = os.path.expandvars(value)
            if key and key not in os.environ:
                os.environ[key] = value
        loaded.append(str(path))
    return loaded


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def strip_bce_year_parentheticals(value: Any) -> str:
    text = str(value if value is not None else "")
    return re.sub(r"（公元前\d+年[^）)]*[）)]", "", text)


def latex_highlight_text(text: Any, agent_oti_text: Any) -> tuple[str, bool]:
    text = str(text if text is not None else "")
    agent_oti_text = str(agent_oti_text if agent_oti_text is not None else "").strip()
    if not agent_oti_text:
        return latex_escape(text), False

    agent_oti_items = [agent_oti_text]
    if any(separator in agent_oti_text for separator in "、，,；;／/"):
        agent_oti_items.extend(re.split(r"[、，,；;／/]+", agent_oti_text))

    exact_agent_oti_items: list[str] = []
    for item in agent_oti_items:
        item = item.strip()
        if item and item in text and item not in exact_agent_oti_items:
            exact_agent_oti_items.append(item)
    if not exact_agent_oti_items:
        return latex_escape(text), False

    exact_agent_oti_items.sort(key=len, reverse=True)
    spans: list[tuple[int, int]] = []
    for candidate in exact_agent_oti_items:
        start = 0
        while True:
            index = text.find(candidate, start)
            if index == -1:
                break
            end = index + len(candidate)
            if not any(index < old_end and end > old_start for old_start, old_end in spans):
                spans.append((index, end))
            start = end
    spans.sort()

    chunks: list[str] = []
    cursor = 0
    for start, end in spans:
        chunks.append(latex_escape(text[cursor:start]))
        chunks.append(rf"\textcolor{{AIHRed}}{{{latex_escape(text[start:end])}}}")
        cursor = end
    chunks.append(latex_escape(text[cursor:]))
    return "".join(chunks), True


def truncate_for_pdf(value: Any, limit: int = 46) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def short_doc_id(short_id: str) -> str:
    return short_id.split(".", 1)[0]


def short_to_full_id(short_id: str, full_ids_by_short: dict[str, str], doc_prefixes: dict[str, str]) -> str:
    if short_id in full_ids_by_short:
        return full_ids_by_short[short_id]
    doc_code = short_doc_id(short_id)
    if doc_code in doc_prefixes:
        return f"{doc_prefixes[doc_code]}.{short_id}"
    raise KeyError(f"Cannot resolve sentence id {short_id}")


def load_shiji_uuid() -> str:
    catalog_path = AIH_RESOURCE_DIR / "book_catalog.json"
    catalog = read_json(catalog_path)
    for item in catalog:
        if str(item.get("book", "")).strip() == "《史记》":
            return str(item["uuid"])
    raise KeyError(f"Cannot find 《史记》 uuid in {catalog_path}")


def extract_short_ids_from_tex(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    ids = re.findall(r"【(\d+\.\d+\.\d+)】", text)
    deduped: list[str] = []
    seen = set()
    for sid in ids:
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(sid)
    return deduped


def extract_items_from_tex(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    chunks = re.findall(r"\\selectfont\s+(.*?)\}\};", text)
    items: list[tuple[str, str]] = []
    current_id = ""
    current_text_parts: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", chunk)
        chunk = chunk.replace(r"\_", "_").strip()
        if not chunk or chunk.startswith("人机实验1") or chunk.startswith("用时"):
            continue
        match = re.match(r"【(\d+\.\d+\.\d+)】\s*(.*)", chunk)
        if match:
            if current_id:
                items.append((current_id, "".join(current_text_parts).strip()))
            current_id = match.group(1)
            current_text_parts = [match.group(2)]
        elif current_id:
            current_text_parts.append(chunk)
    if current_id:
        items.append((current_id, "".join(current_text_parts).strip()))

    deduped: list[tuple[str, str]] = []
    seen = set()
    for short_id, sentence in items:
        if short_id in seen:
            continue
        seen.add(short_id)
        deduped.append((short_id, sentence))
    return deduped


def load_sentences(run_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    by_full: dict[str, dict[str, Any]] = {}
    full_ids_by_short: dict[str, str] = {}
    doc_prefixes: dict[str, str] = {}
    # The formal Experiment 1 pipeline ends at Step11. Load Step1 as the
    # complete sentence index, then overlay Step5 sink/interlude annotations.
    for step_dir in ["step1output", "step5output"]:
        sentence_root = run_root / "sentence" / step_dir
        for path in sorted(sentence_root.glob("*_sentence.json")):
            for item in read_json(path):
                full_id = str(item["number"])
                short_id = ".".join(full_id.split(".")[-3:])
                item["_short_id"] = short_id
                item["_doc_code"] = short_doc_id(short_id)
                by_full[full_id] = item
                full_ids_by_short[short_id] = full_id
                doc_prefixes.setdefault(short_doc_id(short_id), ".".join(full_id.split(".")[:-3]))
    return by_full, full_ids_by_short, doc_prefixes


def parse_number_tail(full_id: str) -> tuple[int, int, int]:
    tail = full_id.split(".")[-3:]
    return tuple(int(part) for part in tail)  # type: ignore[return-value]


def split_timeblock_range(range_text: str) -> tuple[str, str]:
    # Full sentence ids begin with a UUID, so a naive split("-") breaks on
    # UUID hyphens. The actual range separator is between two full ids.
    match = re.match(r"^(.+\.\d+\.\d+\.\d+)-(.+\.\d+\.\d+\.\d+)$", range_text)
    if not match:
        raise ValueError(f"Bad timeblock_range: {range_text}")
    return match.group(1), match.group(2)


def in_timeblock(full_id: str, range_text: str) -> bool:
    start, end = split_timeblock_range(range_text)
    doc = ".".join(full_id.split(".")[:-3])
    if not start.startswith(doc) or not end.startswith(doc):
        return False
    n = parse_number_tail(full_id)
    return parse_number_tail(start) <= n <= parse_number_tail(end)


def load_timeblocks(run_root: Path, timeblock_output_step: int = 14) -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    timeblock_root = run_root / "timeblock" / f"step{timeblock_output_step}output"
    if not timeblock_root.exists():
        raise FileNotFoundError(f"Cannot find timeblock output dir: {timeblock_root}")
    for path in sorted(timeblock_root.glob("*_timeblock.json")):
        data = read_json(path)
        blocks = data.get("TMB", [])
        if not isinstance(blocks, list):
            raise TypeError(f"{path} does not contain a TMB list")
        for block in blocks:
            doc_code = ".".join(str(block["ID"]).split(".")[-3:-2])
            by_doc.setdefault(doc_code, []).append(block)
    return by_doc


def find_timeblock(full_id: str, timeblocks_by_doc: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    doc_code = full_id.split(".")[-3]
    candidates = timeblocks_by_doc.get(doc_code, [])
    hits = [block for block in candidates if in_timeblock(full_id, str(block.get("timeblock_range", "")))]
    if not hits:
        return None
    hits.sort(key=lambda block: parse_number_tail(split_timeblock_range(str(block["timeblock_range"]))[0]), reverse=True)
    return hits[0]


def timeblock_order_key(block: dict[str, Any]) -> tuple[int, int, int]:
    range_text = str(block.get("timeblock_range", "") or "")
    if range_text:
        try:
            start, _end = split_timeblock_range(range_text)
            return parse_number_tail(start)
        except Exception:
            pass
    return parse_number_tail(str(block.get("ID", "")))


def nonzero_granularity(block: dict[str, Any]) -> bool:
    return str(block.get("Granularity", "") or "").strip() != "0"


def timeblock_conversion(block: dict[str, Any]) -> dict[str, Any]:
    conversion = block.get("Conversion information") or block.get("Conversion_information") or {}
    return conversion if isinstance(conversion, dict) else {}


def split_time_marker_items(value: Any) -> list[str]:
    text = strip_bce_year_parentheticals(value).strip()
    if not text:
        return []
    candidates = [text]
    if any(separator in text for separator in "、，,；;／/"):
        candidates.extend(re.split(r"[、，,；;／/]+", text))
    items: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in items:
            items.append(candidate)
    return items


def looks_like_display_time_marker(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(
        re.search(
            r"(?:"
            r"\d+年|\d+月|\d+日|"
            r"[一二三四五六七八九十元正闰]+年|"
            r"[一二三四五六七八九十正闰]+月|"
            r"[一二三四五六七八九十初廿三]+日|"
            r"春天|夏天|秋天|冬天|春|夏|秋|冬|"
            r"早晨|中午|夜间|夜里|夜晚|上午|下午|傍晚|"
            r"明天|次日|翌日"
            r")",
            text,
        )
    )


def timeblock_text(block: dict[str, Any], sentences_by_full: dict[str, dict[str, Any]]) -> str:
    range_text = str(block.get("timeblock_range", "") or "")
    if not range_text:
        return ""
    chunks: list[str] = []
    for full_id, sentence in sentences_by_full.items():
        try:
            if in_timeblock(full_id, range_text):
                chunks.append(str(sentence.get("sentence", "") or ""))
        except Exception:
            continue
    return "\n".join(chunks)


def display_boundary_marker(block: dict[str, Any], sentences_by_full: dict[str, dict[str, Any]]) -> str:
    original = timeblock_conversion(block).get("time_information_original", "")
    block_text = timeblock_text(block, sentences_by_full)
    exact_items = [
        item
        for item in split_time_marker_items(original)
        if (
            item in block_text
            and looks_like_display_time_marker(item)
            and item not in PDF_SUPPRESSED_BOUNDARY_LABELS
        )
    ]
    return exact_items[0] if exact_items else ""


def display_time_label(value: Any) -> str:
    label = strip_bce_year_parentheticals(value).strip()
    if label in PDF_SUPPRESSED_BOUNDARY_LABELS:
        return ""
    return PDF_BOUNDARY_LABEL_REPLACEMENTS.get(label, label)


def pdf_boundary_label(block: dict[str, Any], sentences_by_full: dict[str, dict[str, Any]]) -> str:
    tm = display_time_label(block.get("TM", ""))
    if tm:
        return tm
    marker = display_boundary_marker(block, sentences_by_full)
    return display_time_label(marker)


def timeblock_boundary_tm_by_doc(
    timeblocks_by_doc: dict[str, list[dict[str, Any]]],
    sentences_by_full: dict[str, dict[str, Any]],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Build PDF display intervals from Agent TimeBlock boundaries."""
    labels_by_doc: dict[str, dict[str, tuple[str, str]]] = {}
    for doc_code, blocks in timeblocks_by_doc.items():
        ordered_blocks = sorted(blocks, key=timeblock_order_key)
        boundary_labels = [
            pdf_boundary_label(block, sentences_by_full)
            for block in ordered_blocks
        ]
        prev_anchor_pos: list[int] = []
        last_anchor = -1
        for index, block in enumerate(ordered_blocks):
            prev_anchor_pos.append(last_anchor)
            if boundary_labels[index]:
                last_anchor = index

        next_anchor_pos = [-1] * len(ordered_blocks)
        next_anchor = -1
        for index in range(len(ordered_blocks) - 1, -1, -1):
            next_anchor_pos[index] = next_anchor
            if boundary_labels[index]:
                next_anchor = index

        block_labels: dict[str, tuple[str, str]] = {}
        for index, block in enumerate(ordered_blocks):
            if boundary_labels[index]:
                start_tm = boundary_labels[index]
            else:
                prev_index = prev_anchor_pos[index]
                start_tm = (
                    boundary_labels[prev_index]
                    if prev_index != -1
                    else "-infinity"
                )
            next_index = next_anchor_pos[index]
            end_tm = (
                boundary_labels[next_index]
                if next_index != -1
                else "+infinity"
            )
            block_labels[source_short_id(str(block.get("ID", "")))] = (start_tm, end_tm)
        labels_by_doc[doc_code] = block_labels
    return labels_by_doc


def parse_iso_ym(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if value == "-infinity":
        return "-infinity"
    if value in {"infinity", "+infinity"}:
        return "+infinity"
    match = re.match(r"(-?\d{4,})-(\d{2})", value)
    if not match:
        return value
    return f"{match.group(1)}-{match.group(2)}"


def iso_range_to_ym(iso_range: str) -> tuple[str, str, str]:
    iso_range = str(iso_range or "").strip()
    if not iso_range:
        return "", "", "1"
    if "to" in iso_range:
        start, end = iso_range.split("to", 1)
        return parse_iso_ym(start), parse_iso_ym(end), ""
    return parse_iso_ym(iso_range), parse_iso_ym(iso_range), ""


def source_short_id(full_id: str | None) -> str:
    if not full_id:
        return ""
    parts = str(full_id).split(".")
    if len(parts) < 3:
        return str(full_id)
    return ".".join(parts[-3:])


def sentence_crossdoc_sources(sentence: dict[str, Any]) -> list[str]:
    cross = sentence.get("crossDocTransfer") or {}
    if not isinstance(cross, dict) or not cross.get("isSame"):
        return []
    sources = cross.get("same_timeblock_id") or []
    if not isinstance(sources, list):
        return []
    return [source_short_id(str(item)) for item in sources]


def allowed_crossdoc_source_docs(case_id: str, doc_code: str) -> set[str]:
    pair = CROSSDOC_CASE_DOC_PAIRS.get(case_id)
    if not pair or doc_code not in pair:
        return set()
    return {code for code in pair if code != doc_code}


def crossdoc_source_allowed(case_id: str, doc_code: str, source_id: str) -> bool:
    source_doc = short_doc_id(source_id)
    return source_doc in allowed_crossdoc_source_docs(case_id, doc_code)


def collect_case_sentence_refs(full_ids_by_short: dict[str, str], doc_prefixes: dict[str, str]) -> dict[str, list[SentenceRef]]:
    result: dict[str, list[SentenceRef]] = {}
    for case_id, tex_paths in CASE_FILES.items():
        refs: list[SentenceRef] = []
        for tex_path in tex_paths:
            part_id = tex_path.stem
            for short_id, tex_sentence in extract_items_from_tex(tex_path):
                doc_code = short_doc_id(short_id)
                refs.append(
                    SentenceRef(
                        case_id=case_id,
                        part_id=part_id,
                        doc_code=doc_code,
                        short_id=short_id,
                        full_id=short_to_full_id(short_id, full_ids_by_short, doc_prefixes),
                        tex_sentence=tex_sentence,
                    )
                )
        result[case_id] = refs
    return result


def collect_case_sentence_refs_from_latex() -> dict[str, list[SentenceRef]]:
    shiji_uuid = load_shiji_uuid()
    result: dict[str, list[SentenceRef]] = {}
    for case_id, tex_paths in CASE_FILES.items():
        refs: list[SentenceRef] = []
        for tex_path in tex_paths:
            part_id = tex_path.stem
            for short_id, tex_sentence in extract_items_from_tex(tex_path):
                doc_code = short_doc_id(short_id)
                refs.append(
                    SentenceRef(
                        case_id=case_id,
                        part_id=part_id,
                        doc_code=doc_code,
                        short_id=short_id,
                        full_id=f"{shiji_uuid}.{short_id}",
                        tex_sentence=tex_sentence,
                    )
                )
        result[case_id] = refs
    return result


def standard_agent_sentence_record(ref: SentenceRef, case_people: list[str]) -> dict[str, Any]:
    default_true_person = DOC_PERSONS.get(ref.doc_code, "")
    characters = {person: False for person in case_people}
    if default_true_person:
        characters.setdefault(default_true_person, False)
        characters[default_true_person] = True
    return {
        "number": ref.full_id,
        "sentence": ref.tex_sentence,
        "characters": characters,
        "Original_time_information": {
            "exist": False,
            "OTI": "",
        },
        "sink": {
            "Is_it_sinking": False,
            "reason": "...",
        },
        "Interlude": False,
        "crossDocTransfer": {
            "isSame": False,
            "same_timeblock_id": [],
        },
        "experiment_case": {
            "case_id": ref.case_id,
            "part_id": ref.part_id,
            "short_id": ref.short_id,
            "source_text": DOC_TITLES.get(ref.doc_code, ref.doc_code),
            "crossdoc_enabled_by_design": ref.case_id in CROSSDOC_CASES,
        },
    }


def case_refs_to_json(refs_by_case: dict[str, list[SentenceRef]]) -> dict[str, Any]:
    cases = []
    for case_id, refs in refs_by_case.items():
        items = []
        for idx, ref in enumerate(refs, 1):
            items.append(
                {
                    "case_id": ref.case_id,
                    "part_id": ref.part_id,
                    "item_no": idx,
                    "sentence_id": ref.short_id,
                    "full_sentence_id": ref.full_id,
                    "doc_code": ref.doc_code,
                    "source_text": DOC_TITLES.get(ref.doc_code, ref.doc_code),
                    "sentence": ref.tex_sentence,
                    "crossdoc_enabled_by_design": ref.case_id in CROSSDOC_CASES,
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "crossdoc_enabled_by_design": case_id in CROSSDOC_CASES,
                "items": items,
            }
        )
    return {
        "schema": "AIH_experiment1_cases_from_latex.v1",
        "source": "experiments/experiment-1/inputs/latex-cases/",
        "agent_compatible": True,
        "cases": cases,
    }


def write_agent_case_inputs(refs_by_case: dict[str, list[SentenceRef]]) -> Path:
    input_dir = AGENT_RUN_DIR / "agent_case_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for case_id, refs in refs_by_case.items():
        case_dir = input_dir / case_id
        step1_dir = case_dir / "sentence" / "step1output"
        step1_dir.mkdir(parents=True, exist_ok=True)
        case_people = [
            DOC_PERSONS[doc_code]
            for doc_code in sorted({ref.doc_code for ref in refs}, key=int)
            if doc_code in DOC_PERSONS
        ]
        combined_payload = []
        records_by_doc: dict[str, list[dict[str, Any]]] = {}
        for ref in refs:
            record = standard_agent_sentence_record(ref, case_people)
            combined_payload.append(record)
            records_by_doc.setdefault(ref.doc_code, []).append(record)
        write_json(input_dir / f"{case_id}_sentence.json", combined_payload)
        write_json(case_dir / f"{case_id}_sentence.json", combined_payload)
        for doc_code, records in records_by_doc.items():
            book_uuid = records[0]["number"].split(".", 1)[0]
            write_json(step1_dir / f"{doc_code}_{book_uuid}_sentence.json", records)
    return input_dir


def write_crossdoc_scope_config_from_latex() -> Path:
    cases: dict[str, Any] = {}
    for case_id in sorted(CROSSDOC_CASES):
        doc_pair = CROSSDOC_CASE_DOC_PAIRS[case_id]
        allowed_by_doc = {doc_code: [] for doc_code in doc_pair}
        for tex_path in CASE_FILES[case_id]:
            for short_id, _sentence in extract_items_from_tex(tex_path):
                doc_code = short_doc_id(short_id)
                if doc_code in allowed_by_doc:
                    allowed_by_doc[doc_code].append(short_id)
        cases[case_id] = {
            "doc_pair": list(doc_pair),
            "allowed_short_ids_by_doc": {
                doc_code: sorted(set(short_ids), key=parse_number_tail)
                for doc_code, short_ids in allowed_by_doc.items()
            },
        }
    payload = {
        "schema": "AIH_experiment1_crossdoc_scope.v1",
        "source": "experiments/experiment-1/inputs/latex-cases/H-C5,H-C6/",
        "cases": cases,
    }
    write_json(CROSSDOC_SCOPE_PATH, payload)
    return CROSSDOC_SCOPE_PATH


def build_prefill_rows(
    refs: list[SentenceRef],
    sentences_by_full: dict[str, dict[str, Any]],
    timeblocks_by_doc: dict[str, list[dict[str, Any]]],
    microiou_boundary_source: str = "display_tm",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boundary_tm_by_doc = timeblock_boundary_tm_by_doc(timeblocks_by_doc, sentences_by_full)
    for idx, ref in enumerate(refs, 1):
        sentence = sentences_by_full.get(ref.full_id, {"sentence": ref.tex_sentence})
        block = find_timeblock(ref.full_id, timeblocks_by_doc)
        block_id = "" if block is None else source_short_id(str(block.get("ID", "")))
        block_start_tm, block_end_tm = ("", "")
        if block is not None:
            block_start_tm, block_end_tm = boundary_tm_by_doc.get(ref.doc_code, {}).get(
                block_id,
                (str(block.get("TM", "") or ""), str(block.get("TM", "") or "")),
            )
        iso_range = "" if block is None else str(block.get("iso_range", ""))
        start_ym, end_ym, unknown = iso_range_to_ym(iso_range)
        if microiou_boundary_source == "iso_range" and block is not None:
            block_start_tm, block_end_tm = start_ym, end_ym
        oti = sentence.get("Original_time_information") or {}
        if not isinstance(oti, dict):
            oti = {}
        oti_text = str(oti.get("OTI", "") or "").strip()
        oti_exists = bool(oti.get("exist") or oti_text)
        sink = sentence.get("sink") or {}
        if not isinstance(sink, dict):
            sink = {}
        sink_exists = bool(sink.get("Is_it_sinking"))
        sink_reason = strip_bce_year_parentheticals(sink.get("reason", "")).strip() if sink_exists else ""
        interlude_exists = bool(sentence.get("Interlude"))
        interlude_reason = (
            str(sentence.get("Interlude_reason", "") or sentence.get("interlude_reason", "") or "").strip()
            if interlude_exists
            else ""
        )
        has_valid_microiou_range = bool(iso_range and not unknown)
        if has_valid_microiou_range:
            # MicroIoU is a time-range metric. When Step11 has produced a usable
            # propagated range, do not let a sentence-level state flag erase it.
            sink_exists = False
            sink_reason = ""
            interlude_exists = False
            interlude_reason = ""
        tb_update = "" if block is None else str(block.get("TB_Update", "") or "")
        tb_update_short = source_short_id(tb_update)
        tb_update_allowed = (
            tb_update_short
            if ref.case_id in CROSSDOC_CASES and crossdoc_source_allowed(ref.case_id, ref.doc_code, tb_update_short)
            else ""
        )
        crossdoc_evidence_sources: list[str] = []
        if block is not None and ref.case_id in CROSSDOC_CASES:
            evidence = block.get("crossdoc_time_evidence")
            if isinstance(evidence, dict) and evidence.get("applied"):
                evidence_source = source_short_id(str(evidence.get("source_timeblock_id", "") or ""))
                if evidence_source and crossdoc_source_allowed(ref.case_id, ref.doc_code, evidence_source):
                    crossdoc_evidence_sources.append(evidence_source)
        crossdoc_evidence_sources = list(dict.fromkeys(crossdoc_evidence_sources))
        raw_cross_sources = sentence_crossdoc_sources(sentence)
        cross_sources = [
            source for source in raw_cross_sources
            if crossdoc_source_allowed(ref.case_id, ref.doc_code, source)
        ]
        filtered_cross_sources = [
            source for source in raw_cross_sources
            if source and not crossdoc_source_allowed(ref.case_id, ref.doc_code, source)
        ]
        crossdoc_used = bool(ref.case_id in CROSSDOC_CASES and (tb_update_allowed or cross_sources or crossdoc_evidence_sources))
        note_bits = []
        if block is None:
            note_bits.append("未在 Agent TimeBlock 输出中找到对应块。")
        else:
            note_bits.append(f"Agent TimeBlock={source_short_id(str(block.get('ID', '')))}; TM={block.get('TM', '')}; Granularity={block.get('Granularity', '')}.")
        if ref.case_id in CROSSDOC_CASES:
            if crossdoc_used:
                note_bits.append("本 case 按规划启用跨文本 Agent；时间来自或参考跨文本传播结果。")
            else:
                note_bits.append("本 case 按规划检查跨文本 Agent；该句未检测到具体跨文本传播源。")
            if filtered_cross_sources:
                note_bits.append(f"已过滤非本 case 跨文本来源：{';'.join(filtered_cross_sources)}。")
            if crossdoc_evidence_sources:
                note_bits.append(f"Step11 已应用跨文本事件时间证据：{';'.join(crossdoc_evidence_sources)}。")
        rows.append(
            {
                "case_id": ref.case_id,
                "part_id": ref.part_id,
                "item_no": idx,
                "sentence_id": ref.short_id,
                "source_text": DOC_TITLES.get(ref.doc_code, ref.doc_code),
                "sentence": strip_bce_year_parentheticals(sentence.get("sentence", "") or ref.tex_sentence),
                "ai_start_ym": start_ym,
                "ai_end_ym": end_ym,
                "ai_unknown": unknown,
                "ai_tm": "" if block is None else display_time_label(block.get("TM", "")),
                "ai_timeblock_id": block_id,
                "ai_timeblock_start_tm": block_start_tm,
                "ai_timeblock_end_tm": block_end_tm,
                "ai_iso_range": iso_range,
                "ai_crossdoc_used": "1" if crossdoc_used else "",
                "ai_crossdoc_source_timeblock": tb_update_allowed or ";".join(cross_sources or crossdoc_evidence_sources),
                "ai_oti_exists": "1" if oti_exists else "",
                "ai_oti_text": oti_text,
                "ai_sink": "1" if sink_exists else "",
                "ai_sink_reason": sink_reason,
                "ai_interlude": "1" if interlude_exists else "",
                "ai_interlude_reason": interlude_reason,
                "ai_agent_note": " ".join(note_bits),
                "participant_start_ym": start_ym,
                "participant_end_ym": end_ym,
                "participant_unknown": unknown,
                "participant_notes": "",
                "accepted_ai_without_edit": "",
            }
        )
    return rows


def maybe_refine_rows_with_chat(rows: list[dict[str, Any]], model: str, case_id: str = "") -> list[dict[str, Any]]:
    from ai_historian.model_config import create_chat_completion, resolve_chat_config

    config = resolve_chat_config()

    try:
        from openai import AuthenticationError, OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package before using --chat-refine.") from exc

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=float(os.getenv("AIH_REQUEST_TIMEOUT", "60")),
        max_retries=int(os.getenv("AIH_REFINE_MAX_RETRIES", "1")),
    )
    refined: list[dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, 1):
        prefix = f"{case_id} " if case_id else ""
        print(
            f"[ChatRefine] {prefix}{index}/{total} sentence_id={row.get('sentence_id', '')}",
            flush=True,
        )
        prompt = {
            "task": "Review an AIHAgent sentence-level time range draft for a Human+AI experiment. Return only JSON.",
            "required_schema": {
                "ai_start_ym": "YYYY-MM or empty",
                "ai_end_ym": "YYYY-MM or empty",
                "ai_unknown": "1 if unknown else empty string",
                "ai_agent_note": "short Chinese note, preserve cross-document evidence if relevant",
            },
            "sentence_id": row["sentence_id"],
            "source_text": row["source_text"],
            "sentence": row["sentence"],
            "agent_draft": {
                "ai_start_ym": row["ai_start_ym"],
                "ai_end_ym": row["ai_end_ym"],
                "ai_unknown": row["ai_unknown"],
                "ai_tm": row["ai_tm"],
                "ai_iso_range": row["ai_iso_range"],
                "ai_crossdoc_used": row["ai_crossdoc_used"],
                "ai_crossdoc_source_timeblock": row["ai_crossdoc_source_timeblock"],
                "ai_agent_note": row["ai_agent_note"],
            },
        }
        try:
            response = create_chat_completion(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful assistant for Chinese historical chronology annotation. Output strict JSON only."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0,
            )
        except AuthenticationError as exc:
            raise RuntimeError(
                f"{config.provider} authentication failed. Re-export a valid "
                f"{config.api_key_env}, then rerun the command."
            ) from exc
        content = response.choices[0].message.content or "{}"
        try:
            patch = json.loads(content)
        except json.JSONDecodeError:
            patch = {}
        new_row = dict(row)
        for key in ["ai_start_ym", "ai_end_ym", "ai_unknown", "ai_agent_note"]:
            if key in patch:
                new_row[key] = patch[key]
        new_row["participant_start_ym"] = new_row["ai_start_ym"]
        new_row["participant_end_ym"] = new_row["ai_end_ym"]
        new_row["participant_unknown"] = new_row["ai_unknown"]
        refined.append(new_row)
        print(
            f"[ChatRefine] {prefix}{index}/{total} done start={new_row.get('ai_start_ym', '')} end={new_row.get('ai_end_ym', '')} unknown={new_row.get('ai_unknown', '')}",
            flush=True,
        )
    return refined


def pdf_time_group_key(row: dict[str, Any]) -> tuple[str, str]:
    start_tm = str(row.get("ai_timeblock_start_tm", "") or "").strip()
    end_tm = str(row.get("ai_timeblock_end_tm", "") or "").strip()
    if start_tm or end_tm:
        return ("boundary", f"{start_tm}\u241f{end_tm}")
    if row.get("ai_unknown"):
        return ("unknown", "")
    timeblock_id = str(row.get("ai_timeblock_id", "") or "")
    if timeblock_id:
        return ("timeblock", timeblock_id)
    return (
        "marker",
        str(row.get("ai_tm", "") or ""),
    )


def pdf_group_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[str, str] | None = None
    for row in rows:
        key = pdf_time_group_key(row)
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(row)
        current_key = key
    if current:
        groups.append(current)
    return groups


def pdf_time_marker_label(group: list[dict[str, Any]]) -> str:
    if not group:
        return "（无）"

    first = group[0]
    start = strip_bce_year_parentheticals(first.get("ai_timeblock_start_tm", "")).strip()
    end = strip_bce_year_parentheticals(first.get("ai_timeblock_end_tm", "")).strip()
    if start or end:
        if not start:
            return f"（{end}）"
        if not end:
            return f"（{start}）"
        return f"（{start}，{end}）"

    marker = strip_bce_year_parentheticals(first.get("ai_tm", "") or first.get("ai_oti_text", "")).strip()
    return f"（{marker or '无'}）"


def pdf_note_latex(row: dict[str, Any]) -> str:
    notes: list[str] = []
    if row.get("ai_crossdoc_used"):
        crossdoc_note = "跨文本TimeBlock"
        if row.get("ai_crossdoc_source_timeblock"):
            crossdoc_note = f"{crossdoc_note}：{row['ai_crossdoc_source_timeblock']}".strip()
        notes.append(latex_escape(crossdoc_note))
    if row.get("ai_sink"):
        sink_reason = str(row.get("ai_sink_reason", "") or "").strip()
        sink_note = "下沉" if not sink_reason else f"下沉：{sink_reason}"
        notes.append(rf"\textbf{{{latex_escape(sink_note)}}}")
    return r"；".join(notes)


def pdf_sentence_prefix_latex(row: dict[str, Any]) -> str:
    labels: list[str] = []
    if row.get("ai_sink"):
        labels.append("【下沉】")
    if row.get("ai_interlude"):
        labels.append("【插叙】")
    if not labels:
        return ""
    return rf"\textbf{{{latex_escape(''.join(labels))}}}"


def build_agent_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "AIH_CROSSDOC_DOC_PAIRS",
        ";".join(",".join(pair) for pair in CROSSDOC_CASE_DOC_PAIRS.values()),
    )
    env.setdefault("AIH_DISABLE_EMBEDDING", "1")
    if CROSSDOC_SCOPE_PATH.exists():
        env.setdefault("AIH_CROSSDOC_SCOPE_FILE", str(CROSSDOC_SCOPE_PATH))
    pythonpath_parts = [
        str(AIH_SOURCE_DIR),
    ]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def masked_env_snapshot(env: dict[str, str]) -> dict[str, str]:
    keys = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "AIH_COMPATIBLE_API_KEY",
        "AIH_COMPATIBLE_BASE_URL",
        "AIH_CHAT_PROVIDER",
        "AIH_CHAT_MODEL",
        "AIH_AGENT_CONCURRENCY",
        "AIH_AGENT_BATCH_SIZE",
        "AIH_AGENT_MAX_WORKERS",
        "AIH_EMBED_API_KEY",
        "AIH_EMBED_BASE_URL",
        "AIH_EMBED_MODEL",
        "AIH_EMBED_BATCH_SIZE",
        "AIH_CROSSDOC_DOC_PAIRS",
        "AIH_CROSSDOC_SCOPE_FILE",
    ]
    snapshot: dict[str, str] = {}
    for key in keys:
        value = env.get(key, "")
        if "KEY" in key and value:
            value = f"{value[:6]}...{value[-4:]}" if len(value) > 12 else "***"
        snapshot[key] = value
    return snapshot


def validate_api_env(env: dict[str, str], needs_embedding: bool) -> None:
    from ai_historian.model_config import resolve_chat_config, resolve_embedding_config

    chat = resolve_chat_config(env)
    keys_to_check = [(chat.api_key_env, chat.api_key)]
    if needs_embedding:
        embedding = resolve_embedding_config(env, required=True)
        keys_to_check.append(("AIH_EMBED_API_KEY", embedding.api_key))
    for key, value in keys_to_check:
        if not value or re.fullmatch(r"\$\{[^}]+\}", value):
            raise RuntimeError(f"{key} is required for this pipeline run.")
        if "${" in value or "}" in value:
            raise RuntimeError(
                f"{key} still contains an unresolved placeholder. "
                "Open the API settings page and save a real key for the selected provider."
            )
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                f"{key} contains non-ASCII characters. It looks like a placeholder or invalid key; "
                f"re-export the real API key without Chinese text or extra spaces."
            ) from exc


def step_needs_chat_api(step_num: int) -> bool:
    return step_num in {2, 3, 4, 5, 7, 9, 11, 12, 14}


def compact_api_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", "")
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 500:
        message = message[:500] + "..."
    return f"status={status_code} {message}".strip()


def preflight_api_clients(env: dict[str, str], needs_embedding: bool) -> list[dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package before running the AIHAgent pipeline.") from exc
    from ai_historian.model_config import (
        create_chat_completion,
        resolve_chat_config,
        resolve_embedding_config,
    )

    checks: list[dict[str, Any]] = []

    chat = resolve_chat_config(env)
    chat_check = {
        "name": "chat",
        "provider": chat.provider,
        "model": chat.model,
        "base_url": chat.base_url,
        "status": "running",
    }
    checks.append(chat_check)
    try:
        chat_client = OpenAI(
            api_key=chat.api_key,
            base_url=chat.base_url,
            timeout=float(env.get("AIH_PREFLIGHT_TIMEOUT", "180")),
            max_retries=0,
        )
        create_chat_completion(
            chat_client,
            model=chat.model,
            messages=[{"role": "user", "content": "Return OK."}],
            temperature=0,
            max_tokens=16,
        )
        chat_check["status"] = "ok"
    except Exception as exc:
        chat_check["status"] = "failed"
        chat_check["error"] = compact_api_error(exc)
        return checks

    if needs_embedding:
        embedding = resolve_embedding_config(env, required=True)
        embed_check = {
            "name": "embedding",
            "model": embedding.model,
            "base_url": embedding.base_url,
            "status": "running",
        }
        checks.append(embed_check)
        try:
            embedding_client = OpenAI(api_key=embedding.api_key, base_url=embedding.base_url)
            embedding_client.embeddings.create(model=embedding.model, input=["ping"])
            embed_check["status"] = "ok"
        except Exception as exc:
            embed_check["status"] = "failed"
            embed_check["error"] = compact_api_error(exc)
            return checks

    return checks


def run_agent_pipeline(
    run_root: Path,
    input_dir: Path,
    skip_crossdoc_presteps: bool = False,
    log_root: Path | None = None,
    start_step: int = 3,
    end_step: int = 11,
) -> Path:
    steps = [(num, step) for num, step in AGENT_PIPELINE_STEPS if start_step <= num <= end_step]
    if skip_crossdoc_presteps:
        steps = [(num, step) for num, step in steps if step not in {
            "step_10b_cross_document_prealign.py",
            "step_10d_crossdoc_temporal_graph.py",
        }]
    if not steps:
        raise RuntimeError(f"No AIHAgent steps selected for start_step={start_step}, end_step={end_step}.")

    env = build_agent_env()
    needs_chat = any(step_needs_chat_api(num) for num, _ in steps)
    needs_embedding = False
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (log_root or (OUT_DIR / "logs")) / f"run_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "input_dir": str(input_dir),
        "skip_crossdoc_presteps": skip_crossdoc_presteps,
        "start_step": start_step,
        "end_step": end_step,
        "steps": [],
        "preflight": [],
        "env": masked_env_snapshot(env),
    }
    write_json(summary_path, summary)
    try:
        if needs_chat or needs_embedding:
            validate_api_env(env, needs_embedding=needs_embedding)
    except Exception as exc:
        summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        summary["status"] = "preflight_failed"
        summary["preflight_error"] = str(exc)
        write_json(summary_path, summary)
        raise

    print("[AIHAgent] running structured Agent pipeline", flush=True)
    print(f"[AIHAgent] input_dir={input_dir}", flush=True)
    print(f"[AIHAgent] run_root={run_root}", flush=True)
    from ai_historian.model_config import resolve_chat_config

    chat = resolve_chat_config(env)
    print(f"[AIHAgent] provider={chat.provider}", flush=True)
    print(f"[AIHAgent] model={chat.model}", flush=True)
    print(f"[AIHAgent] base_url={chat.base_url}", flush=True)
    print(f"[AIHAgent] logs={log_dir}", flush=True)
    if needs_embedding:
        print(f"[AIHAgent] embedding_model={env.get('AIH_EMBED_MODEL', '')}", flush=True)
        print("[AIHAgent] Agent9 requires an embedding-capable endpoint.", flush=True)
    if needs_chat or needs_embedding:
        print("[AIHAgent] preflight start: validating chat API" + (" and embedding API" if needs_embedding else ""), flush=True)
        try:
            summary["preflight"] = preflight_api_clients(env, needs_embedding=needs_embedding)
        except Exception as exc:
            summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
            summary["status"] = "preflight_failed"
            summary["preflight_error"] = str(exc)
            write_json(summary_path, summary)
            raise
        failed_preflight = next((item for item in summary["preflight"] if item.get("status") == "failed"), None)
        if failed_preflight:
            summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
            summary["status"] = "preflight_failed"
            summary["preflight_error"] = (
                f"{failed_preflight['name']} API failed for model={failed_preflight.get('model')}, "
                f"base_url={failed_preflight.get('base_url')}: {failed_preflight.get('error')}"
            )
            write_json(summary_path, summary)
            raise RuntimeError(summary["preflight_error"])
        write_json(summary_path, summary)
        print("[AIHAgent] preflight ok", flush=True)

    scripts_dir = AGENT_SCRIPT_OVERRIDE_DIR
    for idx, (agent_step_num, step) in enumerate(steps, 1):
        override_script = AGENT_SCRIPT_OVERRIDE_DIR / step
        script = override_script if override_script.exists() else scripts_dir / step
        step_started = time.time()
        step_record: dict[str, Any] = {
            "index": idx,
            "agent_step_num": agent_step_num,
            "step": step,
            "script": str(script),
            "log": str(log_dir / f"{idx:02d}_agent{agent_step_num:02d}_{Path(step).stem}.log"),
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "status": "running",
        }
        summary["steps"].append(step_record)
        write_json(summary_path, summary)
        print(f"[AIHAgent] {idx}/{len(steps)} start {step}", flush=True)
        if agent_step_num == 1:
            command = [sys.executable, str(script), str(input_dir), str(run_root)]
        else:
            command = [sys.executable, str(script), str(run_root)]
        with Path(step_record["log"]).open("w", encoding="utf-8") as log_file:
            log_file.write(f"# step={step}\n")
            log_file.write(f"# started_at={step_record['started_at']}\n")
            log_file.write(f"# command={' '.join(command)}\n\n")
            log_file.flush()
            proc = subprocess.Popen(
                command,
                cwd=AIH_AGENT_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
            returncode = proc.wait()

        step_record["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        step_record["elapsed_seconds"] = round(time.time() - step_started, 2)
        step_record["returncode"] = returncode
        step_record["status"] = "ok" if returncode == 0 else "failed"
        write_json(summary_path, summary)
        if returncode != 0:
            summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
            summary["status"] = "failed"
            write_json(summary_path, summary)
            raise subprocess.CalledProcessError(returncode, [sys.executable, str(script), str(run_root)])
        print(f"[AIHAgent] {idx}/{len(steps)} done {step}", flush=True)
    summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    summary["status"] = "ok"
    write_json(summary_path, summary)
    return log_dir


def run_case_pipelines(
    refs_by_case: dict[str, list[SentenceRef]],
    agent_case_input_dir: Path,
    skip_crossdoc_presteps: bool,
    log_root: Path,
    start_step: int,
    end_step: int,
) -> tuple[dict[str, Path], dict[str, str]]:
    case_run_roots: dict[str, Path] = {}
    case_log_dirs: dict[str, str] = {}
    first_step = max(2, start_step)
    for case_id in refs_by_case:
        case_run_root = agent_case_input_dir / case_id
        if not (case_run_root / "sentence" / "step1output").is_dir():
            raise FileNotFoundError(f"Missing case step1output: {case_run_root / 'sentence' / 'step1output'}")
        print(f"[Experiment1] running {case_id} as an isolated AIHAgent case", flush=True)
        case_log_dir = run_agent_pipeline(
            case_run_root,
            input_dir=case_run_root,
            skip_crossdoc_presteps=skip_crossdoc_presteps,
            log_root=log_root / case_id,
            start_step=first_step,
            end_step=end_step,
        )
        case_run_roots[case_id] = case_run_root
        case_log_dirs[case_id] = str(case_log_dir)
    return case_run_roots, case_log_dirs


def write_readme(manifest: dict[str, Any]) -> None:
    lines = [
        "# 实验1_AI辅助",
        "",
        "本 README 描述当前结果目录的生成方式和内容。",
        "",
        "本目录按仓库中的实验一规范与 Human+AI 条件生成：",
        "",
        "- 复用实验1的 H-C1 到 H-C6 case packet。",
        "- 右侧答题表预填 AIHAgent 给出的句子级月份范围。",
        "- 参与者可以接受、修改开始/结束年月，或改成未知。",
        "- `accepted_ai_without_edit` 用来后续计算 AI acceptance rate；人工修改后可与 AI 原答案比较 edit outcome direction。",
        "",
        "Agent 使用说明：",
        "",
        "- H-C1 到 H-C4 不启用跨文本 Agent。",
        "- H-C5 到 H-C6 的跨文档证据由 Step10B–10D 生成，并由 Step11 直接写入最终 `iso_range`。",
        "- 默认流程先把实验一 case packet 转换成专属 Agent 输入，再在本结果目录 `agent_run/agent_case_input/H-C*` 中分别执行仓库内的 Agent 实现。",
        "- `--use-existing-agent-output` 用于继续使用本结果目录 `agent_run/agent_case_input/H-C*` 中已经生成的阶段输出。",
        "- 运行 Agent 流水线时会自动写日志到 `agent_run/logs/run_YYYYMMDD_HHMMSS/`。每个 step 一个日志文件，并生成 `run_summary.json` 记录模型、endpoint、并发设置、耗时和退出码。",
        "- 正式 Experiment 1 Agent 流程在 Step11 结束，不运行旧版 Step12/Step13，也不需要 embedding endpoint。",
        "- `--chat-refine` 只是可选的表格逐句后处理，不是正式 Agent 流水线。",
        "- PDF 按 Agent 推理出的 TimeBlock 边界区间分组；`Granularity=0` 不会阻止原文时间标志物显示为边界。",
        "- PDF 组标题仍显示时间范围；范围前后两个边界都优先显示对应 TimeBlock 的 Agent 最终转换 `TM`，只有该边界 `TM` 为空时，才退回显示能精确落回正文的 Agent 原文时间标志物；开放边界保留 `-infinity` / `+infinity`；已知原文错误边界标签不显示。",
        "- 下沉句子和插叙句子分别在句子正文前标记 `【下沉】`、`【插叙】`；下沉原因仍保留在备注栏。",
        "- PDF 里的 AI 内容只来自 Agent 输出和实验1原始句子；脚本不人工补写、不推断替代时间标志物。",
        "- 时间标志物红字只来自 `Original_time_information.OTI`；若 Agent 在 OTI 中列出多个原文片段，则逐项精确匹配句子正文，无法精确匹配时不显示 fallback 文案。",
        "",
        "复现命令：",
        "",
        "```bash",
        "python experiments/experiment-1/code/generate_experiment1_ai_assisted.py --use-existing-agent-output",
        "```",
        "",
        "正式重新运行实验专属 Agent 流水线时去掉 `--use-existing-agent-output`；本地 API key 放在仓库根目录 `.env`，不要提交。",
        "",
        "结果目录结构：",
        "",
        "- `packets/`：每个 H-C packet 的 PDF、LaTeX、包内 CSV。",
        "- `tables/`：每个 case 的汇总 CSV 和全量汇总 CSV。",
        "- `agent_run/`：Agent case input、跨文本 scope、从 LaTeX 抽出的 case JSON、运行日志。",
        "- `archive/`：旧版合并 PDF 等非正式归档产物。",
        "- `manifest.json`：生成配置和每个 case 的统计。",
        "",
        "生成文件：",
    ]
    for item in manifest["cases"]:
        lines.append(f"- `{item['csv']}`：{item['case_id']}，{item['sentence_count']} 句。")
        for part in item.get("parts", []):
            lines.append(
                f"- `{part['pdf']}`：{part['part_id']} assisted 条件答题 PDF，{part['sentence_count']} 句。"
            )
    lines.extend(
        [
            "- `tables/all_cases_ai_assisted_prefill.csv`：所有 assisted 条件合并表。",
            "- `manifest.json`：生成配置和每个 case 的统计。",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def split_rows_by_part(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_part: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_part.setdefault(str(row.get("part_id", "")), []).append(row)
    return rows_by_part


def build_assisted_pdf(part_id: str, rows: list[dict[str, Any]], out_dir: Path) -> Path:
    tex_path = out_dir / f"{part_id}.tex"
    pdf_path = out_dir / f"{part_id}.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[paperwidth=297mm,paperheight=210mm,margin=12mm]{geometry}",
        r"\usepackage{fontspec}",
        r"\usepackage{xeCJK}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{booktabs}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.35}",
        r"\definecolor{AIHHeader}{HTML}{D8E4F0}",
        r"\definecolor{AIHRed}{HTML}{C62828}",
        r"\IfFontExistsTF{TeX Gyre Termes}{\setmainfont{TeX Gyre Termes}}{\setmainfont{Times New Roman}}",
        r"\IfFontExistsTF{Noto Serif CJK SC}{\setCJKmainfont{Noto Serif CJK SC}}{\setCJKmainfont{Songti SC}}",
        r"\begin{document}",
        r"\begin{center}",
        rf"{{\Large\bfseries 实验1 AI辅助答题表 {latex_escape(part_id)}}}\\[0.5em]",
        r"{\normalsize 参与者可直接接受 AI 标注，也可在人工栏修正。}\\[0.8em]",
        r"\end{center}",
    ]
    lines.extend(
        [
            r"\scriptsize",
            r"\begin{longtable}{>{\raggedright\arraybackslash}p{9mm} >{\raggedright\arraybackslash}p{18mm} >{\raggedright\arraybackslash}p{22mm} >{\raggedright\arraybackslash}p{151mm} >{\raggedright\arraybackslash}p{34mm} >{\centering\arraybackslash}p{16mm} >{\raggedright\arraybackslash}p{20mm}}",
            r"\toprule",
            r"序号 & 句ID & 来源 & 句子 & 人工时间 & 接受 & 备注 \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"序号 & 句ID & 来源 & 句子 & 人工时间 & 接受 & 备注 \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for group_index, group in enumerate(pdf_group_rows(rows)):
        label = pdf_time_marker_label(group)
        lines.append(
            rf"\rowcolor{{AIHHeader}}\multicolumn{{7}}{{p{{271mm}}}}{{\bfseries AI建议时间：{latex_escape(label)}}} \\"
        )
        for row in group:
            sentence_latex, _oti_rendered = latex_highlight_text(row.get("sentence", ""), row.get("ai_oti_text", ""))
            prefix_latex = pdf_sentence_prefix_latex(row)
            if prefix_latex:
                sentence_latex = f"{prefix_latex}{sentence_latex}"
            note_latex = pdf_note_latex(row)
            values = [
                latex_escape(row.get("item_no", "")),
                latex_escape(row.get("sentence_id", "")),
                latex_escape(row.get("source_text", "")),
                sentence_latex,
                "",
                "",
                "",
            ]
            lines.append(" & ".join(values) + r" \\")
            if note_latex:
                lines.append(
                    rf"\multicolumn{{7}}{{p{{271mm}}}}{{\textbf{{备注：}}{note_latex}}} \\"
                )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    lines.extend([r"\end{document}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    xelatex = shutil.which("xelatex")
    if not xelatex:
        raise RuntimeError(f"xelatex not found; wrote LaTeX source only: {tex_path}")
    for _ in range(2):
        subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=out_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if not pdf_path.exists():
        raise RuntimeError(f"PDF was not created: {pdf_path}")
    return pdf_path


def build_assisted_pdfs(rows_by_case: dict[str, list[dict[str, Any]]], fieldnames: list[str]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for case_id, rows in rows_by_case.items():
        case_dir = PACKETS_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for part_id, part_rows in split_rows_by_part(rows).items():
            part_csv = case_dir / f"{part_id}_ai_assisted_prefill.csv"
            write_csv(part_csv, part_rows, fieldnames)
            pdf_path = build_assisted_pdf(part_id, part_rows, case_dir)
            outputs.append(
                {
                    "case_id": case_id,
                    "part_id": part_id,
                    "csv": str(part_csv.relative_to(OUT_DIR)),
                    "tex": str((case_dir / f"{part_id}.tex").relative_to(OUT_DIR)),
                    "pdf": str(pdf_path.relative_to(OUT_DIR)),
                    "sentence_count": len(part_rows),
                }
            )
    return outputs


def main() -> None:
    global OUT_DIR, PACKETS_DIR, TABLES_DIR, AGENT_RUN_DIR, ARCHIVE_DIR, CROSSDOC_SCOPE_PATH, AGENT_SCRIPT_OVERRIDE_DIR
    loaded_env_files = load_env_files()
    parser = argparse.ArgumentParser(description="Generate Human+AI prefill sheets for AIH experiment 1.")
    parser.add_argument(
        "--use-existing-agent-output",
        action="store_true",
        help="Reuse experiment-local <output-dir>/agent_run/agent_case_input/H-C* outputs instead of rerunning the Agent steps.",
    )
    parser.add_argument("--log-dir", default="", help="Directory for AIHAgent run logs. Defaults to <output-dir>/agent_run/logs.")
    parser.add_argument("--start-step", type=int, default=1, help="First AIHAgent step to run.")
    parser.add_argument("--end-step", type=int, default=11, choices=range(1, 12), help="Last formal AIHAgent step to run. The formal pipeline ends at Step11.")
    parser.add_argument("--skip-crossdoc-presteps", action="store_true", help="Skip Step10b/Step10d cross-document pre-steps. Use for H-C1..H-C4.")
    parser.add_argument("--chat-refine", action="store_true", help="Optional row-level postprocess using the selected chat provider; not the formal Agent pipeline.")
    parser.add_argument("--refine-model", default=os.getenv("AIH_CHAT_MODEL", ""))
    parser.add_argument("--no-pdf", action="store_true", help="Skip per-packet PDF generation.")
    parser.add_argument(
        "--timeblock-output-step",
        type=int,
        default=11,
        choices=[11],
        help="Formal Experiment 1 exports always read Step11 timeblocks.",
    )
    parser.add_argument(
        "--microiou-boundary-source",
        choices=["display_tm", "iso_range"],
        default="display_tm",
        help="Source for ai_timeblock_start_tm/end_tm in exported CSV. Use iso_range for MicroIoU scoring so display labels cannot override Agent ISO ranges.",
    )
    parser.add_argument("--prepare-case-json-only", action="store_true", help="Only convert experiment 1 case packets into standard AIHAgent sentence JSON files.")
    parser.add_argument("--case-ids", default="", help="Comma-separated case ids to run, e.g. H-C3 or H-C5,H-C6. Defaults to all cases.")
    parser.add_argument(
        "--output-dir",
        default=str(OUT_DIR),
        help="Directory for generated outputs. Defaults to outputs/current/generated_results.",
    )
    parser.add_argument(
        "--agent-script-override-dir",
        default=str(AGENT_SCRIPT_OVERRIDE_DIR),
        help="Directory containing experiment-specific Agent step override scripts.",
    )
    args = parser.parse_args()

    OUT_DIR = Path(args.output_dir).expanduser()
    if not OUT_DIR.is_absolute():
        OUT_DIR = (Path.cwd() / OUT_DIR).resolve()
    PACKETS_DIR = OUT_DIR / "packets"
    TABLES_DIR = OUT_DIR / "tables"
    AGENT_RUN_DIR = OUT_DIR / "agent_run"
    ARCHIVE_DIR = OUT_DIR / "archive"
    CROSSDOC_SCOPE_PATH = AGENT_RUN_DIR / "experiment1_crossdoc_scope.json"
    log_root = Path(args.log_dir).expanduser() if args.log_dir else AGENT_RUN_DIR / "logs"
    if not log_root.is_absolute():
        log_root = (Path.cwd() / log_root).resolve()
    AGENT_SCRIPT_OVERRIDE_DIR = Path(args.agent_script_override_dir).expanduser()
    if not AGENT_SCRIPT_OVERRIDE_DIR.is_absolute():
        AGENT_SCRIPT_OVERRIDE_DIR = (Path.cwd() / AGENT_SCRIPT_OVERRIDE_DIR).resolve()

    for path in [OUT_DIR, PACKETS_DIR, TABLES_DIR, AGENT_RUN_DIR, ARCHIVE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    crossdoc_scope_path = write_crossdoc_scope_config_from_latex()
    refs_by_case = collect_case_sentence_refs_from_latex()
    if args.case_ids.strip():
        wanted_cases = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        unknown_cases = wanted_cases - set(refs_by_case)
        if unknown_cases:
            raise SystemExit(f"Unknown case id(s): {', '.join(sorted(unknown_cases))}")
        refs_by_case = {
            case_id: refs
            for case_id, refs in refs_by_case.items()
            if case_id in wanted_cases
        }
    write_json(AGENT_RUN_DIR / "experiment1_cases_from_latex.json", case_refs_to_json(refs_by_case))
    agent_case_input_dir = write_agent_case_inputs(refs_by_case)
    if args.prepare_case_json_only:
        print(f"Wrote standard AIHAgent case JSON to {agent_case_input_dir}")
        return

    case_agent_run_roots: dict[str, Path] = {
        case_id: agent_case_input_dir / case_id
        for case_id in refs_by_case
    }
    case_agent_log_dirs: dict[str, str] = {}
    if not args.use_existing_agent_output:
        try:
            case_agent_run_roots, case_agent_log_dirs = run_case_pipelines(
                refs_by_case=refs_by_case,
                agent_case_input_dir=agent_case_input_dir,
                skip_crossdoc_presteps=args.skip_crossdoc_presteps,
                log_root=log_root,
                start_step=args.start_step,
                end_step=args.end_step,
            )
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"[AIHAgent] {exc}", file=sys.stderr, flush=True)
            raise SystemExit(1) from exc

    fieldnames = [
        "case_id",
        "part_id",
        "item_no",
        "sentence_id",
        "source_text",
        "sentence",
        "ai_start_ym",
        "ai_end_ym",
        "ai_unknown",
        "ai_tm",
        "ai_timeblock_id",
        "ai_timeblock_start_tm",
        "ai_timeblock_end_tm",
        "ai_iso_range",
        "ai_crossdoc_used",
        "ai_crossdoc_source_timeblock",
        "ai_oti_exists",
        "ai_oti_text",
        "ai_sink",
        "ai_sink_reason",
        "ai_interlude",
        "ai_interlude_reason",
        "ai_agent_note",
        "participant_start_ym",
        "participant_end_ym",
        "participant_unknown",
        "participant_notes",
        "accepted_ai_without_edit",
    ]

    all_rows: list[dict[str, Any]] = []
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    case_manifest_items: dict[str, dict[str, Any]] = {}
    manifest = {
        "source_plan": str(ROOT / "docs" / "reproducibility-guide.md"),
        "source_experiment1_dir": str(EXPERIMENT1_DIR),
        "agent_structure_dir": str(AIH_AGENT_DIR),
        "agent_result_source_policy": "Use the Agent scripts/structure from agent_structure_dir, but use only experiment-local outputs under <output-dir>/agent_run/agent_case_input/H-C* as answer data.",
        "loaded_env_files": loaded_env_files,
        "experiment_agent_input_dir": str(agent_case_input_dir),
        "agent_run_roots_by_case": {
            case_id: str(path)
            for case_id, path in case_agent_run_roots.items()
        },
        "reused_experiment_local_agent_output": bool(args.use_existing_agent_output),
        "latex_case_json": str(AGENT_RUN_DIR / "experiment1_cases_from_latex.json"),
        "crossdoc_scope": str(crossdoc_scope_path),
        "agent_case_input_dir": str(agent_case_input_dir),
        "deepseek_refine": bool(args.deepseek_refine),
        "deepseek_model": args.deepseek_model if args.deepseek_refine else "",
        "timeblock_output_step": args.timeblock_output_step,
        "microiou_boundary_source": args.microiou_boundary_source,
        "agent_pipeline_log_dirs_by_case": case_agent_log_dirs,
        "pdf_content_policy": {
            "sentence_text": "experiment1 original LaTeX sentence, overlaid with Agent sentence text when present",
            "time_grouping": f"Experiment-local Agent-inferred TimeBlock boundary interval from <output-dir>/agent_run/agent_case_input/H-C*/timeblock/step{args.timeblock_output_step}output is the PDF grouping key; same ISO range alone is not a merge key.",
            "time_header": "PDF group headers display a start/end range. Each boundary label prioritizes the corresponding TimeBlock's Agent-converted TM; if that boundary TM is empty, the label falls back to Agent original calendar-like boundary markers that exactly appear in the TimeBlock sentence text. Open boundaries keep -infinity/+infinity; known source-text error boundary labels are suppressed; ISO values are never displayed as header labels.",
            "oti_highlight": "Experiment-local Agent sentence/step5output Original_time_information.OTI only; multi-item OTI strings are split only by list punctuation and each item is rendered only on exact substring match",
            "sink_marking": "Experiment-local Agent sentence/step5output sink.Is_it_sinking and sink.reason",
            "crossdoc_note": f"Experiment-local Agent timeblock/step{args.timeblock_output_step}output crossdoc evidence fields",
            "no_manual_pdf_content_edits": True,
        },
        "cases": [],
    }

    for case_id, refs in refs_by_case.items():
        sentences_by_full, _full_ids_by_short, _doc_prefixes = load_sentences(case_agent_run_roots[case_id])
        timeblocks_by_doc = load_timeblocks(case_agent_run_roots[case_id], args.timeblock_output_step)
        rows = build_prefill_rows(refs, sentences_by_full, timeblocks_by_doc, args.microiou_boundary_source)
        if args.chat_refine:
            print(f"[ChatRefine] refining {case_id}: {len(rows)} rows", flush=True)
            rows = maybe_refine_rows_with_chat(rows, args.refine_model, case_id=case_id)
        rows_by_case[case_id] = rows
        csv_name = f"{case_id}_ai_assisted_prefill.csv"
        csv_path = TABLES_DIR / csv_name
        write_csv(csv_path, rows, fieldnames)
        all_rows.extend(rows)
        case_manifest_items[case_id] = {
            "case_id": case_id,
            "csv": str(csv_path.relative_to(OUT_DIR)),
            "sentence_count": len(rows),
            "crossdoc_enabled_by_design": case_id in CROSSDOC_CASES,
            "crossdoc_rows": sum(1 for row in rows if row["ai_crossdoc_used"]),
            "parts": [],
        }

    write_csv(TABLES_DIR / "all_cases_ai_assisted_prefill.csv", all_rows, fieldnames)
    if not args.no_pdf:
        pdf_outputs = build_assisted_pdfs(rows_by_case, fieldnames)
        manifest["pdfs"] = pdf_outputs
        for output in pdf_outputs:
            case_manifest_items[output["case_id"]]["parts"].append(output)
    manifest["cases"] = list(case_manifest_items.values())
    write_json(OUT_DIR / "manifest.json", manifest)
    write_readme(manifest)

    print(f"Wrote {len(all_rows)} assisted rows to {OUT_DIR}")
    if not args.no_pdf:
        print(f"Wrote {len(manifest.get('pdfs', []))} packet PDFs under {OUT_DIR}")


if __name__ == "__main__":
    main()
