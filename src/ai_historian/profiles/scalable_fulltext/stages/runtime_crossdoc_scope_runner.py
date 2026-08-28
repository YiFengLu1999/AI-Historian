#!/usr/bin/env python3
"""Run final 10B on scoped episode windows without changing agent_impl.

The final Experiment1 agent succeeds by running cross-document alignment on
small case scopes. This wrapper recreates that runtime shape for full-text
three-document folders:

1. Build lexical episode windows from the existing full step10output.
2. Create temporary mini run roots that contain only a document pair/window.
3. Invoke the unmodified step_10b_cross_document_prealign.py on each mini root.
4. Merge accepted crossdoc_context_evidence back into the original step10output.

It intentionally lives outside scripts/agent_impl so the final Agent structure
stays unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_historian.model_config import (
    EMBED_MODEL,
    make_embedding_client,
)
from ai_historian.pipeline.paths import PROJECT_ROOT, resolve_run_root

SENTENCE_SUFFIXES = ("_sentence", "_interlude")
TIMEBLOCK_SUFFIXES = ("_timeblock", "_timeblocks_updated")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
ALNUM_RE = re.compile(r"[A-Za-z0-9_]+")
TEMPORAL_ANCHOR_RE = re.compile(
    r"(?:公元前|前)?\d{1,4}年|(?:元|[一二三四五六七八九十百廿卅]+)年|[正一二三四五六七八九十冬腊]+月|春季?|夏季?|秋季?|冬季?"
)
ACTION_TERMS = {
    "攻", "围", "败", "破", "杀", "降", "追", "立", "封", "迁", "奔", "入关",
    "渡河", "驻军", "救", "归附", "背叛", "投降", "进军", "出兵", "战败",
    "平定", "夺取", "攻打", "坑杀", "逃走", "会合", "防守", "挑战", "谢罪",
}
HIGH_SIGNAL_ENTITY_TERMS = {
    "刘邦", "沛公", "汉王", "汉高祖",
    "项羽", "项籍", "项王", "项梁",
    "萧何", "张良", "韩信", "樊哙", "曹参",
    "楚怀王", "义帝", "秦二世", "赵高", "章邯", "陈余",
    "彭越", "曹咎", "司马欣", "董翳", "雍齿", "子婴",
    "丰邑", "沛县", "咸阳", "关中", "成皋", "荥阳", "外黄", "陈留", "鸿门",
}
STOP_TERMS = {
    "于是", "这个", "那个", "时候", "将军", "大王", "军队", "百姓", "天下",
    "诸侯", "后来", "现在", "可以", "不能", "没有", "已经", "原来", "自己",
    "一个", "一次", "一起", "他们", "这里", "那里", "东西", "南北", "前后",
}
CROSSDOC_FIELD_KEYS = (
    "crossdoc_boundary",
    "crossdoc_prealign",
    "crossdoc_event_links",
    "crossdoc_time_evidence",
    "crossdoc_interval_evidence",
    "crossdoc_context_evidence",
    "event_cluster_id",
)
CHINESE_NUM = {
    "元": 1, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10,
}


@dataclass(frozen=True)
class DocFile:
    doc_id: str
    sentence_path: Path
    timeblock_path: Path
    sequence_path: Path | None


@dataclass(frozen=True)
class Window:
    doc_id: str
    index: int
    start: int
    end: int
    short_ids: tuple[str, ...]
    text: str
    vector: Counter[str]


@dataclass(frozen=True)
class ScopeCandidate:
    case_id: str
    score: float
    left_doc: str
    right_doc: str
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    detail: str
    selected_pairs: tuple[dict[str, Any], ...] = ()
    allowed_short_ids_by_doc: dict[str, set[str]] | None = None
    candidate_focus: dict[str, Any] | None = None


@dataclass(frozen=True)
class PairSpec:
    case_id: str
    left_doc: str
    right_doc: str
    allowed_short_ids_by_doc: dict[str, set[str]] | None = None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_suffix(stem: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def doc_code(doc_id: str) -> str:
    return str(doc_id).split("_", 1)[0]


def short_id(number: str) -> str:
    parts = str(number or "").split(".")
    return ".".join(parts[-3:]) if len(parts) >= 3 else str(number or "")


def parse_number(number: str) -> tuple[str, int, int, int]:
    parts = str(number).strip().rsplit(".", 3)
    if len(parts) != 4:
        raise ValueError(f"bad number: {number}")
    return parts[0], int(parts[1]), int(parts[2]), int(parts[3])


def order_key(number: str) -> tuple[int, int, int]:
    _uuid, chapter, paragraph, sentence = parse_number(number)
    return chapter, paragraph, sentence


def split_range(range_text: str) -> tuple[str, str]:
    match = re.match(r"^(.+\.\d+\.\d+\.\d+)-(.+\.\d+\.\d+\.\d+)$", str(range_text).strip())
    if not match:
        raise ValueError(f"bad timeblock_range: {range_text}")
    return match.group(1), match.group(2)


def in_range(number: str, range_text: str) -> bool:
    start, end = split_range(range_text)
    number_uuid = parse_number(number)[0]
    return (
        parse_number(start)[0] == number_uuid
        and parse_number(end)[0] == number_uuid
        and order_key(start) <= order_key(number) <= order_key(end)
    )


def timeblocks_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("TMB"), list):
        return payload["TMB"]
    if isinstance(payload, list):
        return payload
    return []


def set_timeblocks_payload(payload: Any, blocks: list[dict[str, Any]]) -> Any:
    if isinstance(payload, dict):
        out = dict(payload)
        out["TMB"] = blocks
        return out
    return {"TMB": blocks}


def sentence_numbers_for_block(block: dict[str, Any], sentences: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    range_text = str(block.get("timeblock_range", "") or "")
    if not range_text:
        block_id = str(block.get("ID", "") or "")
        return {block_id} if block_id else set()
    for row in sentences:
        number = str(row.get("number", "") or "")
        try:
            if number and in_range(number, range_text):
                out.add(number)
        except Exception:
            continue
    return out


def text_for_block(block: dict[str, Any], sentences: list[dict[str, Any]]) -> str:
    numbers = sentence_numbers_for_block(block, sentences)
    chunks = [
        str(row.get("sentence", "") or "")
        for row in sentences
        if str(row.get("number", "") or "") in numbers
    ]
    if chunks:
        return "\n".join(chunks)
    return str(block.get("summary", "") or block.get("TM", "") or "")


def tokens(text: str) -> Counter[str]:
    text = re.sub(r"\s+", "", str(text or ""))
    values: list[str] = []
    for match in CJK_RE.finditer(text):
        run = match.group(0)
        if len(run) == 1:
            values.append(run)
        else:
            values.extend(run[i : i + 2] for i in range(len(run) - 1))
            values.extend(run[i : i + 3] for i in range(max(0, len(run) - 2)))
    values.extend(match.group(0).lower() for match in ALNUM_RE.finditer(text))
    return Counter(v for v in values if len(v) >= 2)


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[k] * right[k] for k in common)
    lnorm = math.sqrt(sum(v * v for v in left.values()))
    rnorm = math.sqrt(sum(v * v for v in right.values()))
    if not lnorm or not rnorm:
        return 0.0
    return dot / (lnorm * rnorm)


def vector_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    lnorm = math.sqrt(sum(a * a for a in left))
    rnorm = math.sqrt(sum(b * b for b in right))
    if not lnorm or not rnorm:
        return 0.0
    return dot / (lnorm * rnorm)


def discover_doc_files(run_root: Path) -> dict[str, DocFile]:
    sentence_dir = run_root / "sentence" / "step5output"
    timeblock_dir = run_root / "timeblock" / "step10output"
    sequence_dir = run_root / "sequence" / "step8output"
    if not sentence_dir.is_dir():
        raise FileNotFoundError(f"missing {sentence_dir}")
    if not timeblock_dir.is_dir():
        raise FileNotFoundError(f"missing {timeblock_dir}")

    sentence_by_doc = {
        strip_suffix(path.stem, SENTENCE_SUFFIXES): path
        for path in sentence_dir.glob("*.json")
        if any(path.stem.endswith(suffix) for suffix in SENTENCE_SUFFIXES)
    }
    sequence_by_doc = {
        path.stem.replace("_sequence", ""): path
        for path in sequence_dir.glob("*_sequence.json")
    } if sequence_dir.is_dir() else {}

    out: dict[str, DocFile] = {}
    for path in timeblock_dir.glob("*.json"):
        if not any(path.stem.endswith(suffix) for suffix in TIMEBLOCK_SUFFIXES):
            continue
        doc_id = strip_suffix(path.stem, TIMEBLOCK_SUFFIXES)
        sentence_path = sentence_by_doc.get(doc_id)
        if sentence_path:
            out[doc_id] = DocFile(doc_id, sentence_path, path, sequence_by_doc.get(doc_id))
    return out


def requested_pair_specs(docs: dict[str, DocFile]) -> list[PairSpec]:
    path = os.getenv("AIH_CROSSDOC_SCOPE_FILE", "").strip()
    if not path:
        doc_ids = sorted(docs)
        return [
            PairSpec("", left, right, None)
            for i, left in enumerate(doc_ids)
            for right in doc_ids[i + 1 :]
        ]

    scope_path = Path(path)
    if not scope_path.exists():
        raise FileNotFoundError(f"AIH_CROSSDOC_SCOPE_FILE not found: {scope_path}")
    payload = load_json(scope_path)
    cases = payload.get("cases", {}) if isinstance(payload, dict) else {}
    if not isinstance(cases, dict):
        cases = {}

    by_code: dict[str, list[str]] = {}
    for doc_id in sorted(docs):
        by_code.setdefault(doc_code(doc_id), []).append(doc_id)

    specs: list[PairSpec] = []
    for case_id, case_scope in cases.items():
        if not isinstance(case_scope, dict):
            continue
        raw_pair = case_scope.get("doc_pair", [])
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            continue
        allowed_raw = case_scope.get("allowed_short_ids_by_doc", {})
        allowed: dict[str, set[str]] = {}
        if isinstance(allowed_raw, dict):
            for code, values in allowed_raw.items():
                if isinstance(values, list):
                    allowed[str(code)] = {str(value) for value in values}
        for left in by_code.get(str(raw_pair[0]), []):
            for right in by_code.get(str(raw_pair[1]), []):
                if left != right:
                    specs.append(PairSpec(str(case_id), left, right, allowed or None))
    if not specs:
        raise RuntimeError(f"No doc_pair in {scope_path} matches docs: {sorted(by_code)}")
    print(
        "scope | external scope file | "
        f"{scope_path} | pairs="
        + ",".join(f"{doc_code(spec.left_doc)}-{doc_code(spec.right_doc)}" for spec in specs)
    )
    return specs


def build_windows(
    doc_id: str,
    blocks: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    window_size: int,
    overlap: int,
) -> list[Window]:
    step = max(1, window_size - overlap)
    windows: list[Window] = []
    for index, start in enumerate(range(0, len(blocks), step), 1):
        end = min(len(blocks), start + window_size)
        if end <= start:
            continue
        selected = blocks[start:end]
        text = "\n".join(text_for_block(block, sentences) for block in selected)
        vec = tokens(text)
        if not vec:
            continue
        windows.append(
            Window(
                doc_id=doc_id,
                index=index,
                start=start,
                end=end,
                short_ids=tuple(short_id(str(block.get("ID", "") or "")) for block in selected),
                text=text,
                vector=vec,
            )
        )
        if end == len(blocks):
            break
    return windows


def build_block_units(
    doc_id: str,
    blocks: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
) -> list[Window]:
    units: list[Window] = []
    for index, block in enumerate(blocks):
        text = text_for_block(block, sentences)
        vec = tokens(text)
        if not vec:
            continue
        units.append(
            Window(
                doc_id=doc_id,
                index=index + 1,
                start=index,
                end=index + 1,
                short_ids=(short_id(str(block.get("ID", "") or "")),),
                text=text,
                vector=vec,
            )
        )
    return units


def select_window_pairs(
    left_windows: list[Window],
    right_windows: list[Window],
    top_k: int,
    min_score: float,
    selector: str = "lexical",
    embeddings: dict[tuple[str, int], list[float]] | None = None,
) -> list[tuple[float, Window, Window]]:
    scored: list[tuple[float, Window, Window]] = []
    for left in left_windows:
        for right in right_windows:
            lexical_score = cosine(left.vector, right.vector)
            if selector == "embedding":
                score = vector_cosine(
                    (embeddings or {}).get((left.doc_id, left.index), []),
                    (embeddings or {}).get((right.doc_id, right.index), []),
                )
            elif selector == "hybrid":
                embedding_score = vector_cosine(
                    (embeddings or {}).get((left.doc_id, left.index), []),
                    (embeddings or {}).get((right.doc_id, right.index), []),
                )
                score = 0.75 * embedding_score + 0.25 * lexical_score
            else:
                score = lexical_score
            if score >= min_score:
                scored.append((score, left, right))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[float, Window, Window]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for score, left, right in scored:
        # Prefer coverage over many near-duplicate overlapping windows.
        if left.index in used_left and right.index in used_right:
            continue
        selected.append((score, left, right))
        used_left.add(left.index)
        used_right.add(right.index)
        if len(selected) >= top_k:
            break
    return selected


def score_window_pair(
    left: Window,
    right: Window,
    selector: str,
    embeddings: dict[tuple[str, int], list[float]] | None = None,
) -> float:
    lexical_score = cosine(left.vector, right.vector)
    if selector == "embedding":
        return vector_cosine(
            (embeddings or {}).get((left.doc_id, left.index), []),
            (embeddings or {}).get((right.doc_id, right.index), []),
        )
    if selector == "hybrid":
        embedding_score = vector_cosine(
            (embeddings or {}).get((left.doc_id, left.index), []),
            (embeddings or {}).get((right.doc_id, right.index), []),
        )
        return 0.75 * embedding_score + 0.25 * lexical_score
    return lexical_score


def is_nonanchor_context_block(block: dict[str, Any]) -> bool:
    granularity = str(block.get("Granularity", "") or "").strip()
    tm = str(block.get("TM", "") or "").strip()
    return granularity == "0" and not tm


def is_concrete_anchor_block(block: dict[str, Any]) -> bool:
    granularity = str(block.get("Granularity", "") or "").strip()
    tm = str(block.get("TM", "") or "").strip()
    return bool(tm and granularity != "0" and TEMPORAL_ANCHOR_RE.search(tm))


def is_nonanchor_block(block: dict[str, Any]) -> bool:
    return not is_concrete_anchor_block(block)


def block_sid(block: dict[str, Any]) -> str:
    return short_id(str(block.get("ID", "") or ""))


def block_tm(block: dict[str, Any]) -> str:
    return str(block.get("TM", "") or "").strip()


def expand_indices(
    blocks: list[dict[str, Any]],
    indices: set[int],
    pad: int,
    pre_anchor_backfill: int = 1,
) -> tuple[int, ...]:
    count = len(blocks)
    expanded: set[int] = set()
    for index in indices:
        start = max(0, index - pad)
        end = min(count, index + pad + 1)
        expanded.update(range(start, end))
    # Experiment1 scopes often include a broad non-anchor setup block just
    # before the selected dated block. Keep that runtime shape without changing
    # the final 10B adjudicator.
    for index in list(expanded):
        cursor = index - 1
        steps = 0
        while cursor >= 0 and steps < max(0, pre_anchor_backfill):
            if not is_nonanchor_context_block(blocks[cursor]):
                break
            expanded.add(cursor)
            cursor -= 1
            steps += 1
    return tuple(sorted(expanded))


def nearest_index(
    blocks: list[dict[str, Any]],
    start: int,
    direction: int,
    predicate,
    max_distance: int,
) -> int | None:
    pos = start + direction
    distance = 0
    while 0 <= pos < len(blocks) and distance < max_distance:
        if predicate(blocks[pos]):
            return pos
        pos += direction
        distance += 1
    return None


def episode_packet_indices(
    blocks: list[dict[str, Any]],
    seed_index: int,
    context_pad: int,
    anchor_search: int,
    require_nonanchor: bool = True,
) -> tuple[int, ...]:
    if not blocks:
        return ()
    count = len(blocks)
    seed_index = max(0, min(seed_index, count - 1))
    selected: set[int] = set(range(max(0, seed_index - context_pad), min(count, seed_index + context_pad + 1)))

    before_anchor = nearest_index(blocks, seed_index, -1, is_concrete_anchor_block, anchor_search)
    after_anchor = nearest_index(blocks, seed_index, 1, is_concrete_anchor_block, anchor_search)
    if before_anchor is not None:
        selected.add(before_anchor)
    if after_anchor is not None:
        selected.add(after_anchor)
    if is_concrete_anchor_block(blocks[seed_index]):
        selected.add(seed_index)

    if require_nonanchor and not any(is_nonanchor_block(blocks[index]) for index in selected):
        nonanchor_search = min(anchor_search, max(2, context_pad + 2))
        before_nonanchor = nearest_index(blocks, seed_index, -1, is_nonanchor_block, nonanchor_search)
        after_nonanchor = nearest_index(blocks, seed_index, 1, is_nonanchor_block, nonanchor_search)
        if before_nonanchor is not None:
            selected.add(before_nonanchor)
        if after_nonanchor is not None:
            selected.add(after_nonanchor)

    return tuple(sorted(selected))


def salient_terms(text: str) -> set[str]:
    value = re.sub(r"\s+", "", str(text or ""))
    terms = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        for n in (2, 3):
            for idx in range(max(0, len(run) - n + 1)):
                token = run[idx : idx + n]
                if token in STOP_TERMS:
                    continue
                if any(action in token for action in ACTION_TERMS):
                    continue
                terms.add(token)
    return terms


def entity_terms(text: str) -> set[str]:
    value = re.sub(r"\s+", "", str(text or ""))
    return {term for term in HIGH_SIGNAL_ENTITY_TERMS if term in value}


def action_terms(text: str) -> set[str]:
    value = str(text or "")
    return {term for term in ACTION_TERMS if len(term) >= 2 and term in value}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def overlap_coefficient(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def parse_era_year(tm: str) -> tuple[str, int] | None:
    value = str(tm or "")
    for era in ("秦二世", "汉高祖", "汉王", "汉", "秦"):
        if era not in value:
            continue
        match = re.search(rf"{era}([元一二三四五六七八九十])年", value)
        if match:
            return era, CHINESE_NUM.get(match.group(1), 0)
    return None


def temporal_compatibility(left_quality: dict[str, Any], right_quality: dict[str, Any]) -> dict[str, Any]:
    left_values = [
        parse_era_year(str(left_quality.get("before_anchor_tm", ""))),
        parse_era_year(str(left_quality.get("after_anchor_tm", ""))),
    ]
    right_values = [
        parse_era_year(str(right_quality.get("before_anchor_tm", ""))),
        parse_era_year(str(right_quality.get("after_anchor_tm", ""))),
    ]
    left_values = [item for item in left_values if item]
    right_values = [item for item in right_values if item]
    if not left_values or not right_values:
        return {"score": 0.0, "penalty": 0.0, "reason": "unknown"}
    left_eras = {era for era, _year in left_values}
    right_eras = {era for era, _year in right_values}
    if left_eras & right_eras:
        left_years = [year for era, year in left_values if era in right_eras]
        right_years = [year for era, year in right_values if era in left_eras]
        distance = max(0, max(min(right_years) - max(left_years), min(left_years) - max(right_years)))
        if distance == 0:
            return {"score": 0.18, "penalty": 0.0, "reason": "overlap"}
        if distance == 1:
            return {"score": 0.08, "penalty": 0.0, "reason": "adjacent"}
        return {"score": 0.0, "penalty": min(0.35, 0.12 * distance), "reason": f"year_distance_{distance}"}
    # Different era labels can still be adjacent across the Qin/Han transition,
    # but should not dominate semantically stronger evidence.
    return {"score": 0.02, "penalty": 0.08, "reason": "different_era"}


def event_pair_features(left: Window, right: Window, left_quality: dict[str, Any], right_quality: dict[str, Any]) -> dict[str, Any]:
    left_entities = entity_terms(left.text)
    right_entities = entity_terms(right.text)
    entity_overlap = overlap_coefficient(left_entities, right_entities)
    lexical_overlap = jaccard(salient_terms(left.text), salient_terms(right.text))
    action_overlap = jaccard(action_terms(left.text), action_terms(right.text))
    temporal = temporal_compatibility(left_quality, right_quality)
    penalty = float(temporal["penalty"])
    max_text_chars = max(len(left.text), len(right.text))
    width_penalty = 0.0
    if max_text_chars > 6000:
        width_penalty = 0.16
    elif max_text_chars > 3500:
        width_penalty = 0.08
    penalty += width_penalty
    if entity_overlap < 0.15 and action_overlap < 0.35:
        penalty += 0.10
    if entity_overlap == 0.0 and action_overlap < 0.20:
        penalty += 0.08
    score = 0.34 * entity_overlap + 0.12 * lexical_overlap + 0.16 * action_overlap + float(temporal["score"]) - penalty
    return {
        "score": round(score, 6),
        "entity_overlap": round(entity_overlap, 4),
        "lexical_overlap": round(lexical_overlap, 4),
        "action_overlap": round(action_overlap, 4),
        "shared_entities": sorted(left_entities & right_entities)[:20],
        "temporal": temporal,
        "penalty": round(penalty, 6),
        "width_penalty": round(width_penalty, 6),
        "max_text_chars": max_text_chars,
    }


def episode_packet_quality(
    blocks: list[dict[str, Any]],
    indices: tuple[int, ...],
    seed_index: int,
) -> dict[str, Any]:
    anchors = [idx for idx in indices if is_concrete_anchor_block(blocks[idx])]
    nonanchors = [idx for idx in indices if not is_concrete_anchor_block(blocks[idx])]
    before = [idx for idx in anchors if idx <= seed_index]
    after = [idx for idx in anchors if idx > seed_index]
    before_anchor = max(before) if before else None
    after_anchor = min(after) if after else None
    has_before = before_anchor is not None
    has_after = after_anchor is not None
    has_nonanchor = bool(nonanchors)
    seed_is_nonanchor = not is_concrete_anchor_block(blocks[seed_index])
    score = 0.0
    score += 0.35 if has_before else -0.25
    score += 0.25 if has_after else 0.0
    score += 0.25 if has_nonanchor else -0.20
    score += 0.12 if seed_is_nonanchor else 0.0
    score += 0.10 if seed_index in indices else 0.0
    score += max(0.0, 0.10 - 0.01 * max(0, len(indices) - 5))
    return {
        "score": round(score, 6),
        "anchor_count": len(anchors),
        "nonanchor_count": len(nonanchors),
        "before_anchor_id": block_sid(blocks[before_anchor]) if before_anchor is not None else "",
        "before_anchor_tm": block_tm(blocks[before_anchor]) if before_anchor is not None else "",
        "after_anchor_id": block_sid(blocks[after_anchor]) if after_anchor is not None else "",
        "after_anchor_tm": block_tm(blocks[after_anchor]) if after_anchor is not None else "",
        "seed_id": block_sid(blocks[seed_index]),
        "seed_is_nonanchor": seed_is_nonanchor,
        "anchor_ids": [block_sid(blocks[idx]) for idx in anchors],
        "nonanchor_ids": [block_sid(blocks[idx]) for idx in nonanchors],
        "has_before_anchor": has_before,
        "has_after_anchor": has_after,
        "has_nonanchor": has_nonanchor,
    }


def neighboring_short_ids(
    sentences: list[dict[str, Any]],
    focus_ids: set[str],
    radius: int,
    max_ids: int,
) -> set[str]:
    if not focus_ids:
        return set()
    rows: list[tuple[str, tuple[int, int, int] | tuple[int, str], int]] = []
    for index, row in enumerate(sentences):
        sid = short_id(str(row.get("number", "") or ""))
        if sid:
            rows.append((sid, short_id_sort_key(sid), index))
    by_sid = {sid: index for index, (sid, _key, _row_index) in enumerate(rows)}
    selected: set[str] = set()
    for sid in sorted_short_ids(focus_ids):
        pos = by_sid.get(sid)
        if pos is None:
            continue
        start = max(0, pos - max(0, radius))
        end = min(len(rows), pos + max(0, radius) + 1)
        selected.update(rows[i][0] for i in range(start, end))
    if len(selected) <= max_ids:
        return selected
    ranked: list[str] = []
    for sid in sorted_short_ids(focus_ids):
        if sid in selected and sid not in ranked:
            ranked.append(sid)
    for sid in sorted_short_ids(selected):
        if sid not in ranked:
            ranked.append(sid)
    return set(ranked[:max_ids])


def episode_focus_short_ids(
    quality: dict[str, Any],
    window: Window,
    sentences: list[dict[str, Any]],
    radius: int = 1,
    max_ids: int = 28,
) -> set[str]:
    focus_ids: set[str] = set()
    for key in ("seed_id", "before_anchor_id", "after_anchor_id"):
        value = str(quality.get(key, "") or "")
        if value:
            focus_ids.add(value)
    for key in ("anchor_ids", "nonanchor_ids"):
        for value in quality.get(key, []) or []:
            if str(value or ""):
                focus_ids.add(str(value))
    focus_ids.update(str(value) for value in window.short_ids if str(value))
    allowed = neighboring_short_ids(sentences, focus_ids, radius=radius, max_ids=max_ids)
    return allowed or focus_ids


def build_candidate_focus(
    left_doc: str,
    right_doc: str,
    left: Window,
    right: Window,
    left_quality: dict[str, Any],
    right_quality: dict[str, Any],
    retrieval_score: float,
    selector: str,
) -> dict[str, Any]:
    return {
        "strategy": "episode_packet_anchor_quality",
        "selector": selector,
        "retrieval_score": round(retrieval_score, 6),
        "left_doc": doc_code(left_doc),
        "right_doc": doc_code(right_doc),
        "seed_pair": {
            doc_code(left_doc): left.short_ids[0] if left.short_ids else "",
            doc_code(right_doc): right.short_ids[0] if right.short_ids else "",
        },
        "focus_by_doc": {
            doc_code(left_doc): {
                "seed_id": left_quality.get("seed_id", ""),
                "recommended_before_anchor_id": left_quality.get("before_anchor_id", ""),
                "recommended_before_anchor_tm": left_quality.get("before_anchor_tm", ""),
                "recommended_after_anchor_id": left_quality.get("after_anchor_id", ""),
                "recommended_after_anchor_tm": left_quality.get("after_anchor_tm", ""),
                "anchor_quality_score": left_quality.get("score", 0),
                "anchor_count": left_quality.get("anchor_count", 0),
                "nonanchor_count": left_quality.get("nonanchor_count", 0),
                "anchor_ids": left_quality.get("anchor_ids", []),
                "nonanchor_ids": left_quality.get("nonanchor_ids", []),
                "seed_text": re.sub(r"\s+", "", left.text)[:220],
            },
            doc_code(right_doc): {
                "seed_id": right_quality.get("seed_id", ""),
                "recommended_before_anchor_id": right_quality.get("before_anchor_id", ""),
                "recommended_before_anchor_tm": right_quality.get("before_anchor_tm", ""),
                "recommended_after_anchor_id": right_quality.get("after_anchor_id", ""),
                "recommended_after_anchor_tm": right_quality.get("after_anchor_tm", ""),
                "anchor_quality_score": right_quality.get("score", 0),
                "anchor_count": right_quality.get("anchor_count", 0),
                "nonanchor_count": right_quality.get("nonanchor_count", 0),
                "anchor_ids": right_quality.get("anchor_ids", []),
                "nonanchor_ids": right_quality.get("nonanchor_ids", []),
                "seed_text": re.sub(r"\s+", "", right.text)[:220],
            },
        },
    }


def indices_for_allowed_short_ids(
    blocks: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    allowed_short_ids: set[str],
) -> tuple[int, ...]:
    wanted = {str(value) for value in allowed_short_ids if str(value)}
    if not wanted:
        return ()
    out: list[int] = []
    for index, block in enumerate(blocks):
        numbers = sentence_numbers_for_block(block, sentences)
        if not numbers:
            block_id = str(block.get("ID", "") or "")
            numbers = {block_id} if block_id else set()
        if any(short_id(number) in wanted for number in numbers):
            out.append(index)
    return tuple(out)


def scope_candidates_from_block_sets(
    units_by_doc: dict[str, list[Window]],
    original_blocks: dict[str, list[dict[str, Any]]],
    sentence_cache: dict[str, list[dict[str, Any]]],
    pair_specs: list[PairSpec],
    top_k_per_pair: int,
    max_cases: int,
    min_score: float,
    selector: str,
    embeddings: dict[tuple[str, int], list[float]],
    context_pad: int,
    pre_anchor_backfill: int,
) -> list[ScopeCandidate]:
    candidates: list[ScopeCandidate] = []
    for pair_spec in pair_specs:
        left_doc = pair_spec.left_doc
        right_doc = pair_spec.right_doc
        if pair_spec.allowed_short_ids_by_doc:
            left_allowed = pair_spec.allowed_short_ids_by_doc.get(doc_code(left_doc), set())
            right_allowed = pair_spec.allowed_short_ids_by_doc.get(doc_code(right_doc), set())
            left_indices = indices_for_allowed_short_ids(
                original_blocks[left_doc],
                sentence_cache[left_doc],
                left_allowed,
            )
            right_indices = indices_for_allowed_short_ids(
                original_blocks[right_doc],
                sentence_cache[right_doc],
                right_allowed,
            )
            if not left_indices or not right_indices:
                print(
                    f"scope | pair {doc_code(left_doc)}-{doc_code(right_doc)} | "
                    f"external allowed ids matched left={len(left_indices)} right={len(right_indices)}"
                )
                continue
            case_prefix = pair_spec.case_id or f"scope_{len(candidates) + 1:03d}"
            case_id = f"{case_prefix}_{doc_code(left_doc)}_set_{doc_code(right_doc)}_set"
            detail = (
                f"{doc_code(left_doc)} blocks={len(left_indices)} exact_allowed={len(left_allowed)} "
                f"{doc_code(right_doc)} blocks={len(right_indices)} exact_allowed={len(right_allowed)} "
                "strategy=external_allowed_short_ids"
            )
            candidates.append(
                ScopeCandidate(
                    case_id=case_id,
                    score=1.0,
                    left_doc=left_doc,
                    right_doc=right_doc,
                    left_indices=left_indices,
                    right_indices=right_indices,
                    detail=detail,
                    selected_pairs=(
                        {
                            "score": 1.0,
                            "left_id": "external_allowed_short_ids",
                            "right_id": "external_allowed_short_ids",
                            "left_text": "",
                            "right_text": "",
                        },
                    ),
                    allowed_short_ids_by_doc={
                        doc_code(left_doc): set(left_allowed),
                        doc_code(right_doc): set(right_allowed),
                    },
                )
            )
            print(
                f"scope | pair {doc_code(left_doc)}-{doc_code(right_doc)} | "
                f"external allowed ids | {detail}"
            )
            continue
        scored: list[tuple[float, Window, Window]] = []
        for left in units_by_doc[left_doc]:
            for right in units_by_doc[right_doc]:
                score = score_window_pair(left, right, selector, embeddings)
                if score >= min_score:
                    scored.append((score, left, right))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, Window, Window]] = []
        seen_left: set[int] = set()
        seen_right: set[int] = set()
        for item in scored:
            _score, left, right = item
            if left.start not in seen_left or right.start not in seen_right:
                selected.append(item)
                seen_left.add(left.start)
                seen_right.add(right.start)
            if len(selected) >= max(1, top_k_per_pair):
                break
        if len(selected) < max(1, top_k_per_pair):
            selected_keys = {(left.start, right.start) for _score, left, right in selected}
            for item in scored:
                _score, left, right = item
                key = (left.start, right.start)
                if key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
                if len(selected) >= max(1, top_k_per_pair):
                    break
        if not selected:
            print(
                f"scope | pair {doc_code(left_doc)}-{doc_code(right_doc)} | "
                f"blocks={len(units_by_doc[left_doc])}x{len(units_by_doc[right_doc])} | selected=0"
            )
            continue
        left_indices = {item[1].start for item in selected}
        right_indices = {item[2].start for item in selected}
        expanded_left = expand_indices(original_blocks[left_doc], left_indices, context_pad, pre_anchor_backfill)
        expanded_right = expand_indices(original_blocks[right_doc], right_indices, context_pad, pre_anchor_backfill)
        mean_score = sum(item[0] for item in selected) / len(selected)
        top_score = selected[0][0]
        selected_pair_rows = tuple(
            {
                "score": round(score, 6),
                "left_id": left.short_ids[0] if left.short_ids else "",
                "right_id": right.short_ids[0] if right.short_ids else "",
                "left_text": re.sub(r"\s+", "", left.text)[:120],
                "right_text": re.sub(r"\s+", "", right.text)[:120],
            }
            for score, left, right in selected
        )
        case_prefix = pair_spec.case_id or f"scope_{len(candidates) + 1:03d}"
        case_id = f"{case_prefix}_{doc_code(left_doc)}_set_{doc_code(right_doc)}_set"
        detail = (
            f"{doc_code(left_doc)} blocks={len(expanded_left)} selected={len(left_indices)} "
            f"{doc_code(right_doc)} blocks={len(expanded_right)} selected={len(right_indices)} "
            f"top={top_score:.4f} mean={mean_score:.4f} "
            f"strategy=diverse_block_coverage pre_anchor_backfill={pre_anchor_backfill}"
        )
        candidates.append(
            ScopeCandidate(
                case_id=case_id,
                score=top_score,
                left_doc=left_doc,
                right_doc=right_doc,
                left_indices=expanded_left,
                right_indices=expanded_right,
                detail=detail,
                selected_pairs=selected_pair_rows,
            )
        )
        print(
            f"scope | pair {doc_code(left_doc)}-{doc_code(right_doc)} | "
            f"blocks={len(units_by_doc[left_doc])}x{len(units_by_doc[right_doc])} | "
            f"selected_pairs={len(selected)} | {detail}"
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    if max_cases > 0:
        candidates = candidates[:max_cases]
    return candidates


def scope_candidates_from_episode_packets(
    units_by_doc: dict[str, list[Window]],
    original_blocks: dict[str, list[dict[str, Any]]],
    sentence_cache: dict[str, list[dict[str, Any]]],
    pair_specs: list[PairSpec],
    top_k_per_pair: int,
    max_cases: int,
    min_score: float,
    selector: str,
    embeddings: dict[tuple[str, int], list[float]],
    context_pad: int,
    anchor_search: int,
) -> list[ScopeCandidate]:
    candidates: list[ScopeCandidate] = []
    for pair_spec in pair_specs:
        if pair_spec.allowed_short_ids_by_doc:
            continue
        left_doc = pair_spec.left_doc
        right_doc = pair_spec.right_doc
        scored: list[tuple[float, Window, Window]] = []
        for left in units_by_doc[left_doc]:
            for right in units_by_doc[right_doc]:
                score = score_window_pair(left, right, selector, embeddings)
                if score >= min_score:
                    scored.append((score, left, right))
        scored.sort(key=lambda item: item[0], reverse=True)

        selected: list[tuple[float, float, Window, Window, tuple[int, ...], tuple[int, ...], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        seen_left: set[int] = set()
        seen_right: set[int] = set()
        quality_scored: list[tuple[float, float, Window, Window, tuple[int, ...], tuple[int, ...], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for score, left, right in scored:
            left_indices = episode_packet_indices(
                original_blocks[left_doc],
                left.start,
                context_pad=context_pad,
                anchor_search=anchor_search,
            )
            right_indices = episode_packet_indices(
                original_blocks[right_doc],
                right.start,
                context_pad=context_pad,
                anchor_search=anchor_search,
            )
            if not left_indices or not right_indices:
                continue
            left_quality = episode_packet_quality(original_blocks[left_doc], left_indices, left.start)
            right_quality = episode_packet_quality(original_blocks[right_doc], right_indices, right.start)
            anchor_quality = float(left_quality.get("score", 0) or 0) + float(right_quality.get("score", 0) or 0)
            event_features = event_pair_features(left, right, left_quality, right_quality)
            # Retrieval still drives topical relevance; anchor quality makes the
            # packet legally usable; event/temporal features avoid weak contexts.
            final_score = score + 0.18 * anchor_quality + float(event_features.get("score", 0) or 0)
            quality_scored.append((final_score, score, left, right, left_indices, right_indices, left_quality, right_quality, event_features))
        quality_scored.sort(key=lambda item: item[0], reverse=True)
        for item in quality_scored:
            _final_score, _score, left, right, _left_indices, _right_indices, _left_quality, _right_quality, _event_features = item
            if left.start in seen_left and right.start in seen_right:
                continue
            selected.append(item)
            seen_left.add(left.start)
            seen_right.add(right.start)
            if len(selected) >= max(1, top_k_per_pair):
                break

        for local_index, (final_score, score, left, right, left_indices, right_indices, left_quality, right_quality, event_features) in enumerate(selected, 1):
            case_prefix = pair_spec.case_id or f"episode_{len(candidates) + 1:03d}"
            case_id = (
                f"{case_prefix}_{doc_code(left_doc)}_{left.short_ids[0] if left.short_ids else local_index}"
                f"_{doc_code(right_doc)}_{right.short_ids[0] if right.short_ids else local_index}"
            ).replace(".", "_")
            left_anchor_count = sum(is_concrete_anchor_block(original_blocks[left_doc][idx]) for idx in left_indices)
            right_anchor_count = sum(is_concrete_anchor_block(original_blocks[right_doc][idx]) for idx in right_indices)
            left_nonanchor_count = len(left_indices) - left_anchor_count
            right_nonanchor_count = len(right_indices) - right_anchor_count
            left_allowed = episode_focus_short_ids(left_quality, left, sentence_cache[left_doc])
            right_allowed = episode_focus_short_ids(right_quality, right, sentence_cache[right_doc])
            detail = (
                f"{doc_code(left_doc)} seed={left.short_ids[0] if left.short_ids else ''} "
                f"blocks={len(left_indices)} anchors={left_anchor_count} nonanchors={left_nonanchor_count} allowed={len(left_allowed)} | "
                f"{doc_code(right_doc)} seed={right.short_ids[0] if right.short_ids else ''} "
                f"blocks={len(right_indices)} anchors={right_anchor_count} nonanchors={right_nonanchor_count} allowed={len(right_allowed)} | "
                f"score={score:.4f} final={final_score:.4f} "
                f"anchor_quality={left_quality.get('score', 0)}+{right_quality.get('score', 0)} "
                f"event_score={event_features.get('score', 0)} "
                f"strategy=episode_packet anchor_search={anchor_search}"
            )
            candidate_focus = build_candidate_focus(
                left_doc,
                right_doc,
                left,
                right,
                left_quality,
                right_quality,
                score,
                selector,
            )
            candidates.append(
                ScopeCandidate(
                    case_id=case_id,
                    score=final_score,
                    left_doc=left_doc,
                    right_doc=right_doc,
                    left_indices=left_indices,
                    right_indices=right_indices,
                    detail=detail,
                    allowed_short_ids_by_doc={
                        doc_code(left_doc): left_allowed,
                        doc_code(right_doc): right_allowed,
                    },
                    candidate_focus=candidate_focus,
                    selected_pairs=(
                        {
                            "score": round(score, 6),
                            "final_score": round(final_score, 6),
                            "left_id": left.short_ids[0] if left.short_ids else "",
                            "right_id": right.short_ids[0] if right.short_ids else "",
                            "left_text": re.sub(r"\s+", "", left.text)[:180],
                            "right_text": re.sub(r"\s+", "", right.text)[:180],
                            "left_anchor_count": left_anchor_count,
                            "right_anchor_count": right_anchor_count,
                            "left_nonanchor_count": left_nonanchor_count,
                            "right_nonanchor_count": right_nonanchor_count,
                            "left_allowed_count": len(left_allowed),
                            "right_allowed_count": len(right_allowed),
                            "left_anchor_quality": left_quality,
                            "right_anchor_quality": right_quality,
                            "event_pair_features": event_features,
                            "candidate_focus": candidate_focus,
                        },
                    ),
                )
            )
        print(
            f"scope | pair {doc_code(left_doc)}-{doc_code(right_doc)} | "
            f"episode_packets={len(selected)}"
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    if max_cases > 0:
        candidates = candidates[:max_cases]
    return candidates


def scope_candidates_from_windows(
    windows: dict[str, list[Window]],
    pair_specs: list[PairSpec],
    top_k_per_pair: int,
    max_cases: int,
    min_score: float,
    selector: str,
    embeddings: dict[tuple[str, int], list[float]],
    context_pad: int,
    original_blocks: dict[str, list[dict[str, Any]]],
) -> list[ScopeCandidate]:
    out: list[ScopeCandidate] = []
    raw: list[tuple[float, Window, Window]] = []
    for pair_spec in pair_specs:
        left_doc = pair_spec.left_doc
        right_doc = pair_spec.right_doc
        pair_candidates = select_window_pairs(
            windows[left_doc],
            windows[right_doc],
            top_k=top_k_per_pair,
            min_score=min_score,
            selector=selector,
            embeddings=embeddings,
        )
        raw.extend(pair_candidates)
        print(
            f"scope | pair {doc_code(left_doc)}-{doc_code(right_doc)} | "
            f"windows={len(windows[left_doc])}x{len(windows[right_doc])} | selected={len(pair_candidates)}"
        )
    raw.sort(key=lambda item: item[0], reverse=True)
    if max_cases > 0:
        raw = raw[:max_cases]
    for index, (score, left_win, right_win) in enumerate(raw, 1):
        left_indices = tuple(range(max(0, left_win.start - context_pad), min(len(original_blocks[left_win.doc_id]), left_win.end + context_pad)))
        right_indices = tuple(range(max(0, right_win.start - context_pad), min(len(original_blocks[right_win.doc_id]), right_win.end + context_pad)))
        out.append(
            ScopeCandidate(
                case_id=f"scope_{index:03d}_{doc_code(left_win.doc_id)}_{left_win.index:03d}_{doc_code(right_win.doc_id)}_{right_win.index:03d}",
                score=score,
                left_doc=left_win.doc_id,
                right_doc=right_win.doc_id,
                left_indices=left_indices,
                right_indices=right_indices,
                detail=(
                    f"{doc_code(left_win.doc_id)}[{left_win.start}:{left_win.end}] "
                    f"{doc_code(right_win.doc_id)}[{right_win.start}:{right_win.end}]"
                ),
            )
        )
    return out


def embed_windows(
    windows_by_doc: dict[str, list[Window]],
    batch_size: int,
    text_chars: int,
) -> dict[tuple[str, int], list[float]]:
    all_windows = [window for windows in windows_by_doc.values() for window in windows]
    if not all_windows:
        return {}

    client = make_embedding_client()
    out: dict[tuple[str, int], list[float]] = {}
    print(
        f"scope | embedding selector | model={EMBED_MODEL} | "
        f"windows={len(all_windows)} | batch={batch_size}"
    )
    def compact_text(text: str, limit: int) -> str:
        return re.sub(r"\s+", "\n", text).strip()[: max(200, limit)]

    def embed_batch(batch: list[Window], limit: int) -> list[list[float]]:
        inputs = [compact_text(window.text, limit) for window in batch]
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=inputs,
                timeout=120,
            )
            vectors = [item.embedding for item in resp.data]
            if len(vectors) != len(batch):
                raise RuntimeError(f"embedding count mismatch: expected={len(batch)} got={len(vectors)}")
            return [[float(value) for value in vector] for vector in vectors]
        except Exception as exc:
            message = str(exc)
            too_long = "maximum input length" in message or "max" in message.lower() and "token" in message.lower()
            if len(batch) > 1:
                vectors: list[list[float]] = []
                for window in batch:
                    vectors.extend(embed_batch([window], limit))
                return vectors
            if too_long and limit > 500:
                next_limit = max(500, limit // 2)
                print(
                    f"scope | embedding selector | shrink window "
                    f"{doc_code(batch[0].doc_id)}#{batch[0].index} chars={limit}->{next_limit}"
                )
                return embed_batch(batch, next_limit)
            raise

    for start in range(0, len(all_windows), max(1, batch_size)):
        batch = all_windows[start : start + max(1, batch_size)]
        vectors = embed_batch(batch, text_chars)
        for window, vector in zip(batch, vectors):
            out[(window.doc_id, window.index)] = vector
        print(f"scope | embedding selector | embedded={min(start + len(batch), len(all_windows))}/{len(all_windows)}")
    return out


def expanded_slice(blocks: list[dict[str, Any]], window: Window, pad: int) -> list[dict[str, Any]]:
    start = max(0, window.start - pad)
    end = min(len(blocks), window.end + pad)
    return [dict(block) for block in blocks[start:end]]


def filter_sentences_for_blocks(blocks: list[dict[str, Any]], sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted: set[str] = set()
    for block in blocks:
        wanted.update(sentence_numbers_for_block(block, sentences))
    return [row for row in sentences if str(row.get("number", "") or "") in wanted]


def short_id_sort_key(value: str) -> tuple[int, int, int] | tuple[int, str]:
    parts = str(value).split(".")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return int(parts[0]), int(parts[1]), int(parts[2])
    return (999999, str(value))


def sorted_short_ids(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=short_id_sort_key)


def filter_sentences_for_short_ids(sentences: list[dict[str, Any]], allowed_short_ids: set[str]) -> list[dict[str, Any]]:
    wanted = {str(value) for value in allowed_short_ids if str(value)}
    return [
        row
        for row in sentences
        if short_id(str(row.get("number", "") or "")) in wanted
    ]


def filter_blocks_for_short_ids(
    blocks: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    allowed_short_ids: set[str],
) -> list[dict[str, Any]]:
    wanted = {str(value) for value in allowed_short_ids if str(value)}
    out: list[dict[str, Any]] = []
    for block in blocks:
        numbers = sentence_numbers_for_block(block, sentences)
        if not numbers:
            block_id = str(block.get("ID", "") or "")
            numbers = {block_id} if block_id else set()
        if any(short_id(number) in wanted for number in numbers):
            out.append(dict(block))
    return out


def allowed_short_ids_for_blocks(blocks: list[dict[str, Any]], sentences: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        numbers = sorted(sentence_numbers_for_block(block, sentences), key=order_key)
        if not numbers:
            numbers = [str(block.get("ID", "") or "")]
        for number in numbers:
            sid = short_id(number)
            if sid and sid not in seen:
                seen.add(sid)
                values.append(sid)
    return values


def write_mini_run(
    base_run_root: Path,
    temp_root: Path,
    case_id: str,
    left_doc: str,
    right_doc: str,
    left_blocks: list[dict[str, Any]],
    right_blocks: list[dict[str, Any]],
    docs: dict[str, DocFile],
    sentence_cache: dict[str, list[dict[str, Any]]],
    allowed_short_ids_by_doc: dict[str, set[str]] | None = None,
    candidate_focus: dict[str, Any] | None = None,
    rerun_step10_from_step9: bool = False,
) -> Path:
    run_root = temp_root / case_id
    if run_root.exists():
        shutil.rmtree(run_root)

    left_allowed = set((allowed_short_ids_by_doc or {}).get(doc_code(left_doc), set()))
    right_allowed = set((allowed_short_ids_by_doc or {}).get(doc_code(right_doc), set()))
    if not left_allowed:
        left_allowed = set(allowed_short_ids_for_blocks(left_blocks, sentence_cache[left_doc]))
    if not right_allowed:
        right_allowed = set(allowed_short_ids_for_blocks(right_blocks, sentence_cache[right_doc]))

    scope = {
        "schema": "AIH_runtime_crossdoc_scope.v1",
        "cases": {
            case_id: {
                "doc_pair": [doc_code(left_doc), doc_code(right_doc)],
                "allowed_short_ids_by_doc": {
                    doc_code(left_doc): sorted_short_ids(left_allowed),
                    doc_code(right_doc): sorted_short_ids(right_allowed),
                },
                "candidate_focus": candidate_focus or {},
            }
        },
    }
    save_json(run_root / "timeblock" / "runtime_crossdoc_scope.json", scope)

    for doc_id, blocks in ((left_doc, left_blocks), (right_doc, right_blocks)):
        doc = docs[doc_id]
        allowed = left_allowed if doc_id == left_doc else right_allowed
        filtered_sentences = filter_sentences_for_short_ids(sentence_cache[doc_id], allowed)
        save_json(run_root / "sentence" / "step5output" / doc.sentence_path.name, filtered_sentences)
        if rerun_step10_from_step9:
            step9_path = base_run_root / "timeblock" / "step9output" / doc.timeblock_path.name
            if not step9_path.exists():
                raise FileNotFoundError(f"missing step9output for mini Step10 rerun: {step9_path}")
            step9_payload = load_json(step9_path)
            step9_blocks = filter_blocks_for_short_ids(timeblocks_from_payload(step9_payload), sentence_cache[doc_id], allowed)
            if not step9_blocks:
                raise RuntimeError(f"no step9 blocks matched allowed ids for {doc_id}")
            save_json(
                run_root / "timeblock" / "step9output" / doc.timeblock_path.name,
                set_timeblocks_payload(step9_payload, step9_blocks),
            )
        else:
            payload = load_json(doc.timeblock_path)
            save_json(
                run_root / "timeblock" / "step10output" / doc.timeblock_path.name,
                set_timeblocks_payload(payload, blocks),
            )
            sequence = [str(block.get("ID", "") or "") for block in blocks if str(block.get("ID", "") or "")]
            save_json(run_root / "sequence" / "step8output" / f"{doc_id}_sequence.json", sequence)

    return run_root


def rebuild_sequence_from_step10(mini_root: Path) -> None:
    step10_dir = mini_root / "timeblock" / "step10output"
    sequence_dir = mini_root / "sequence" / "step8output"
    for path in sorted(step10_dir.glob("*_timeblock.json")):
        doc_id = strip_suffix(path.stem, TIMEBLOCK_SUFFIXES)
        payload = load_json(path)
        sequence = [
            str(block.get("ID", "") or "")
            for block in timeblocks_from_payload(payload)
            if str(block.get("ID", "") or "")
        ]
        save_json(sequence_dir / f"{doc_id}_sequence.json", sequence)


def run_step10(mini_root: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "ai_historian.profiles.scalable_fulltext.agent_stages.step_10_tm_generation",
        str(mini_root),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ.copy(), check=True)
    rebuild_sequence_from_step10(mini_root)


def run_step10b(mini_root: Path, scope_file: Path) -> None:
    env = os.environ.copy()
    env["AIH_CROSSDOC_SCOPE_FILE"] = str(scope_file)
    env["AIH_CROSSDOC_SCOPE_INTERNAL"] = "1"
    env["AIH_CROSSDOC_SCOPE_STRATEGY"] = ""
    cmd = [
        sys.executable,
        "-m",
        "ai_historian.profiles.scalable_fulltext.agent_stages.step_10b_cross_document_prealign",
        str(mini_root),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def evidence_target_key(evidence: dict[str, Any]) -> tuple[str, str]:
    return (
        str(evidence.get("target_doc_id", "") or ""),
        short_id(str(evidence.get("target_timeblock_id", "") or "")),
    )


def evidence_rank(evidence: dict[str, Any]) -> tuple[float, float, float, float, float]:
    relation = str(evidence.get("relation", "") or "")
    strong_relation = relation in {"episode_context", "same_sequence_phase"}
    relaxed = bool(evidence.get("recall_relation_relaxed"))
    confidence = float(evidence.get("confidence", 0) or 0)
    warnings = evidence.get("quality_warnings", [])
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    quote_score = float(evidence.get("source_quote_score", 0) or 0) + float(evidence.get("target_quote_score", 0) or 0)
    return (
        1.0 if strong_relation else 0.0,
        0.0 if relaxed else 1.0,
        confidence,
        -float(warning_count),
        quote_score,
    )


def dedupe_evidence_by_target(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for evidence in evidence_items:
        key = evidence_target_key(evidence)
        if not key[0] or not key[1]:
            continue
        old = best.get(key)
        if old is None or evidence_rank(evidence) > evidence_rank(old):
            best[key] = evidence
    return list(best.values())


def merge_evidence(
    original_payloads: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> int:
    by_doc_and_id: dict[tuple[str, str], dict[str, Any]] = {}
    for doc_id, payload in original_payloads.items():
        for block in timeblocks_from_payload(payload):
            block_id = str(block.get("ID", "") or "")
            if block_id:
                by_doc_and_id[(doc_id, block_id)] = block
                by_doc_and_id[(doc_id, short_id(block_id))] = block

    changed = 0
    for evidence in dedupe_evidence_by_target(evidence_items):
        target_doc = str(evidence.get("target_doc_id", "") or "")
        target_id = str(evidence.get("target_timeblock_id", "") or "")
        block = by_doc_and_id.get((target_doc, target_id)) or by_doc_and_id.get((target_doc, short_id(target_id)))
        if not block:
            continue
        old = block.get("crossdoc_context_evidence")
        old_conf = float(old.get("confidence", 0) or 0) if isinstance(old, dict) else -1.0
        new_conf = float(evidence.get("confidence", 0) or 0)
        if not isinstance(old, dict) or new_conf >= old_conf:
            block["crossdoc_context_evidence"] = evidence
            block.pop("iso_range", None)
            changed += 1
    return changed


def clear_existing_crossdoc_context(original_payloads: dict[str, Any]) -> int:
    cleared = 0
    for payload in original_payloads.values():
        for block in timeblocks_from_payload(payload):
            if isinstance(block, dict) and any(key in block for key in CROSSDOC_FIELD_KEYS):
                for key in CROSSDOC_FIELD_KEYS:
                    block.pop(key, None)
                block.pop("iso", None)
                block.pop("iso_range", None)
                cleared += 1
    return cleared


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scoped 10B windows for full-text crossdoc alignment.")
    parser.add_argument("run_root", help="Existing result/result_* run root.")
    parser.add_argument(
        "--scope-mode",
        choices=["block_set", "episode_packet", "window"],
        default=os.getenv("AIH_CROSSDOC_SCOPE_MODE", "block_set"),
        help="block_set merges related timeblocks; episode_packet builds small anchor-bracketed candidates; window keeps the older contiguous-window behavior.",
    )
    parser.add_argument("--window-size", type=int, default=28)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--context-pad", type=int, default=4)
    parser.add_argument(
        "--anchor-search",
        type=int,
        default=int(os.getenv("AIH_CROSSDOC_SCOPE_ANCHOR_SEARCH", "8")),
        help="For episode_packet mode, search this many blocks in each direction for source/target anchors.",
    )
    parser.add_argument(
        "--pre-anchor-backfill",
        type=int,
        default=int(os.getenv("AIH_CROSSDOC_SCOPE_PRE_ANCHOR_BACKFILL", "1")),
        help="For block_set mode, include this many preceding non-anchor context blocks after padding.",
    )
    parser.add_argument("--top-k-per-pair", type=int, default=6)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--max-cases", type=int, default=18)
    parser.add_argument(
        "--selector",
        choices=["embedding", "lexical", "hybrid"],
        default=os.getenv("AIH_SCOPE_SELECTOR", "embedding"),
        help="How to select window pairs before calling the unmodified 10B.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-text-chars", type=int, default=6000)
    parser.add_argument(
        "--fallback-lexical-on-embedding-error",
        action="store_true",
        help="Fallback to lexical selection if embedding API fails.",
    )
    parser.add_argument(
        "--rerun-step10-from-step9",
        action="store_true",
        default=os.getenv("AIH_CROSSDOC_SCOPE_RERUN_STEP10_FROM_STEP9", "0").strip().lower() in {"1", "true", "yes"},
        help="Write scoped step9output into the mini root and rerun Step10 before the unmodified 10B.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument(
        "--clear-existing-crossdoc",
        action="store_true",
        default=os.getenv("AIH_CROSSDOC_CLEAR_EXISTING", "0").strip().lower() in {"1", "true", "yes", "on"},
        help="Before merging this scoped run, remove existing crossdoc_context_evidence from the original step10output.",
    )
    args = parser.parse_args()

    run_root = resolve_run_root(args.run_root).resolve()
    docs = discover_doc_files(run_root)
    if len(docs) < 2:
        raise SystemExit(f"Need at least 2 docs with sentence/timeblock outputs under {run_root}")

    sentence_cache = {doc_id: load_json(doc.sentence_path) for doc_id, doc in docs.items()}
    original_payloads = {doc_id: load_json(doc.timeblock_path) for doc_id, doc in docs.items()}
    original_blocks = {doc_id: timeblocks_from_payload(payload) for doc_id, payload in original_payloads.items()}

    if args.scope_mode in {"block_set", "episode_packet"}:
        windows = {
            doc_id: build_block_units(doc_id, original_blocks[doc_id], sentence_cache[doc_id])
            for doc_id in docs
        }
    else:
        windows = {
            doc_id: build_windows(doc_id, original_blocks[doc_id], sentence_cache[doc_id], args.window_size, args.overlap)
            for doc_id in docs
        }

    selector = args.selector
    selector_error = ""
    embeddings: dict[tuple[str, int], list[float]] = {}
    if selector in {"embedding", "hybrid"}:
        try:
            embeddings = embed_windows(
                windows,
                batch_size=args.embedding_batch_size,
                text_chars=args.embedding_text_chars,
            )
        except Exception as exc:
            if not args.fallback_lexical_on_embedding_error:
                raise
            selector_error = f"{type(exc).__name__}: {exc}"
            print(f"scope | embedding selector failed, fallback lexical | {selector_error}")
            selector = "lexical"

    pair_specs = requested_pair_specs(docs)
    if args.scope_mode == "block_set":
        candidates = scope_candidates_from_block_sets(
            windows,
            original_blocks,
            sentence_cache,
            pair_specs,
            top_k_per_pair=args.top_k_per_pair,
            max_cases=args.max_cases,
            min_score=args.min_score,
            selector=selector,
            embeddings=embeddings,
            context_pad=args.context_pad,
            pre_anchor_backfill=args.pre_anchor_backfill,
        )
    elif args.scope_mode == "episode_packet":
        candidates = scope_candidates_from_episode_packets(
            windows,
            original_blocks,
            sentence_cache,
            pair_specs,
            top_k_per_pair=args.top_k_per_pair,
            max_cases=args.max_cases,
            min_score=args.min_score,
            selector=selector,
            embeddings=embeddings,
            context_pad=args.context_pad,
            anchor_search=args.anchor_search,
        )
    else:
        candidates = scope_candidates_from_windows(
            windows,
            pair_specs,
            top_k_per_pair=args.top_k_per_pair,
            max_cases=args.max_cases,
            min_score=args.min_score,
            selector=selector,
            embeddings=embeddings,
            context_pad=args.context_pad,
            original_blocks=original_blocks,
        )

    temp_root = run_root / "timeblock" / "runtime_crossdoc_scope_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    all_evidence: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        print(
            f"scope | {index}/{len(candidates)} | score={candidate.score:.4f} | "
            f"{candidate.detail} | {candidate.case_id}"
        )
        if args.dry_run:
            left_allowed = (
                sorted_short_ids(candidate.allowed_short_ids_by_doc.get(doc_code(candidate.left_doc), set()))
                if candidate.allowed_short_ids_by_doc
                else allowed_short_ids_for_blocks(
                    [original_blocks[candidate.left_doc][i] for i in candidate.left_indices],
                    sentence_cache[candidate.left_doc],
                )
            )
            right_allowed = (
                sorted_short_ids(candidate.allowed_short_ids_by_doc.get(doc_code(candidate.right_doc), set()))
                if candidate.allowed_short_ids_by_doc
                else allowed_short_ids_for_blocks(
                    [original_blocks[candidate.right_doc][i] for i in candidate.right_indices],
                    sentence_cache[candidate.right_doc],
                )
            )
            case_reports.append({
                "case_id": candidate.case_id,
                "score": round(candidate.score, 6),
                "scope_mode": args.scope_mode,
                "left_doc": candidate.left_doc,
                "right_doc": candidate.right_doc,
                "left_short_ids": left_allowed,
                "right_short_ids": right_allowed,
                "selected_pairs": list(candidate.selected_pairs),
                "candidate_focus": candidate.candidate_focus or {},
                "dry_run": True,
            })
            continue

        left_blocks = [dict(original_blocks[candidate.left_doc][i]) for i in candidate.left_indices]
        right_blocks = [dict(original_blocks[candidate.right_doc][i]) for i in candidate.right_indices]
        mini_root = write_mini_run(
            run_root,
            temp_root,
            candidate.case_id,
            candidate.left_doc,
            candidate.right_doc,
            left_blocks,
            right_blocks,
            docs,
            sentence_cache,
            allowed_short_ids_by_doc=candidate.allowed_short_ids_by_doc,
            candidate_focus=candidate.candidate_focus,
            rerun_step10_from_step9=args.rerun_step10_from_step9,
        )
        scope_file = mini_root / "timeblock" / "runtime_crossdoc_scope.json"
        try:
            if args.rerun_step10_from_step9:
                run_step10(mini_root)
            run_step10b(mini_root, scope_file)
            report_path = mini_root / "timeblock" / "step10b_crossdoc_prealign_report.json"
            report = load_json(report_path)
            evidence = report.get("context_evidence", []) if isinstance(report, dict) else []
            all_evidence.extend(evidence)
            left_report_ids = (
                sorted_short_ids(candidate.allowed_short_ids_by_doc.get(doc_code(candidate.left_doc), set()))
                if candidate.allowed_short_ids_by_doc
                else allowed_short_ids_for_blocks(left_blocks, sentence_cache[candidate.left_doc])
            )
            right_report_ids = (
                sorted_short_ids(candidate.allowed_short_ids_by_doc.get(doc_code(candidate.right_doc), set()))
                if candidate.allowed_short_ids_by_doc
                else allowed_short_ids_for_blocks(right_blocks, sentence_cache[candidate.right_doc])
            )
            case_reports.append({
                "case_id": candidate.case_id,
                "score": round(candidate.score, 6),
                "scope_mode": args.scope_mode,
                "left_doc": candidate.left_doc,
                "right_doc": candidate.right_doc,
                "left_short_ids": left_report_ids,
                "right_short_ids": right_report_ids,
                "selected_pairs": list(candidate.selected_pairs),
                "candidate_focus": candidate.candidate_focus or {},
                "mini_root": str(mini_root),
                "contexts": len(evidence),
                "accepted": sum(s.get("accepted", 0) for s in report.get("context_stats", [])),
                "quality_warning_accepts": sum(s.get("quality_warning_accepts", 0) for s in report.get("context_stats", [])),
                "calls": sum(s.get("calls", 0) for s in report.get("context_stats", [])),
                "rejected": sum(s.get("rejected", 0) for s in report.get("context_stats", [])),
                "invalid_anchor": sum(s.get("invalid_anchor", 0) for s in report.get("context_stats", [])),
                "failed_calls": sum(s.get("failed_calls", 0) for s in report.get("context_stats", [])),
                "api_errors": [
                    error
                    for s in report.get("context_stats", [])
                    for error in s.get("api_errors", [])
                    if isinstance(s.get("api_errors", []), list)
                ],
                "mini_context_stats": report.get("context_stats", []),
                "report": report,
            })
        except subprocess.CalledProcessError as exc:
            case_reports.append({"case_id": candidate.case_id, "score": round(candidate.score, 6), "scope_mode": args.scope_mode, "error": str(exc)})

    deduped_evidence = dedupe_evidence_by_target(all_evidence)
    cleared_existing_crossdoc = 0
    changed = 0
    if not args.dry_run:
        if args.clear_existing_crossdoc:
            cleared_existing_crossdoc = clear_existing_crossdoc_context(original_payloads)
        changed = merge_evidence(original_payloads, deduped_evidence)
        for doc_id, payload in original_payloads.items():
            save_json(docs[doc_id].timeblock_path, payload)

    aggregate = {
        "schema": "AIH_runtime_crossdoc_scope_runner.v1",
        "run_root": str(run_root),
        "method": "runtime_scoped_unmodified_step10b",
        "selector": selector,
        "scope_mode": args.scope_mode,
        "requested_selector": args.selector,
        "selector_error": selector_error,
        "embedding_model": EMBED_MODEL if selector in {"embedding", "hybrid"} else "",
        "window_size": args.window_size,
        "overlap": args.overlap,
        "context_pad": args.context_pad,
        "pre_anchor_backfill": args.pre_anchor_backfill,
        "anchor_search": args.anchor_search,
        "rerun_step10_from_step9": args.rerun_step10_from_step9,
        "top_k_per_pair": args.top_k_per_pair,
        "min_score": args.min_score,
        "candidate_cases": len(candidates),
        "raw_contexts": len(all_evidence),
        "contexts": len(deduped_evidence),
        "merged_contexts": changed,
        "cleared_existing_crossdoc": cleared_existing_crossdoc,
        "quality_gate_mode": os.getenv("AIH_CROSSDOC_QUALITY_GATE_MODE", ""),
        "recall_accept_weak_context": os.getenv("AIH_CROSSDOC_RECALL_ACCEPT_WEAK_CONTEXT", ""),
        "min_weak_context_confidence": os.getenv("AIH_CROSSDOC_WEAK_CONTEXT_MIN_CONF", ""),
        "context_stats": [
            {
                "case_id": item.get("case_id"),
                "accepted": item.get("accepted", 0),
                "quality_warning_accepts": item.get("quality_warning_accepts", 0),
                "calls": item.get("calls", 0),
                "rejected": item.get("rejected", 0),
                "invalid_anchor": item.get("invalid_anchor", 0),
                "failed_calls": item.get("failed_calls", 0),
                "api_errors": item.get("api_errors", []),
                "contexts": item.get("contexts", 0),
                "error": item.get("error", ""),
                "mini_context_stats": item.get("mini_context_stats", []),
            }
            for item in case_reports
        ],
        "case_reports": case_reports,
        "raw_context_evidence": all_evidence,
        "context_evidence": deduped_evidence,
    }
    save_json(run_root / "timeblock" / "runtime_crossdoc_scope_report.json", aggregate)
    if not args.dry_run:
        save_json(run_root / "timeblock" / "step10b_crossdoc_prealign_report.json", {
            "schema": "AIH_experiment1_crossdoc_anchor_boundary.final",
            "run_root": str(run_root),
            "method": "runtime_scoped_quote_verified_episode_context_boundary_only",
            "selector": selector,
            "scope_mode": args.scope_mode,
            "requested_selector": args.selector,
            "selector_error": selector_error,
            "embedding_model": EMBED_MODEL if selector in {"embedding", "hybrid"} else "",
            "candidate_cases": len(candidates),
            "merged_contexts": changed,
            "cleared_existing_crossdoc": cleared_existing_crossdoc,
            "quality_gate_mode": os.getenv("AIH_CROSSDOC_QUALITY_GATE_MODE", ""),
            "recall_accept_weak_context": os.getenv("AIH_CROSSDOC_RECALL_ACCEPT_WEAK_CONTEXT", ""),
            "min_weak_context_confidence": os.getenv("AIH_CROSSDOC_WEAK_CONTEXT_MIN_CONF", ""),
            "runtime_scope_report": "runtime_crossdoc_scope_report.json",
            "matches": [],
            "interval_evidence": [],
            "raw_contexts": len(all_evidence),
            "context_evidence": deduped_evidence,
            "context_stats": [
                {
                    "case_id": item.get("case_id"),
                    "accepted": item.get("accepted", 0),
                    "quality_warning_accepts": item.get("quality_warning_accepts", 0),
                    "calls": item.get("calls", 0),
                    "rejected": item.get("rejected", 0),
                    "invalid_anchor": item.get("invalid_anchor", 0),
                    "failed_calls": item.get("failed_calls", 0),
                    "api_errors": item.get("api_errors", []),
                    "contexts": item.get("contexts", 0),
                    "error": item.get("error", ""),
                    "mini_context_stats": item.get("mini_context_stats", []),
                }
                for item in case_reports
            ],
        })

    print(
        f"scope | done | cases={len(candidates)} | contexts={len(deduped_evidence)} raw_contexts={len(all_evidence)} | "
        f"merged={changed} | report={run_root / 'timeblock' / 'runtime_crossdoc_scope_report.json'}"
    )

    if not args.keep_temp and not args.dry_run:
        # Keep mini outputs only when they are useful for debugging. The aggregate
        # report keeps enough metadata for normal inspection.
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
