from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from ai_historian.model_config import (
    CHAT_MODEL,
    create_chat_completion,
    make_sync_chat_client,
    validate_json_text,
)
from ai_historian.pipeline.logging import StepReporter, setup_step_logging
from ai_historian.pipeline.paths import (
    PROJECT_ROOT,
    resolve_run_root,
    sentence_step_dir,
    sequence_step_dir,
    timeblock_step_dir,
)

TIMEBLOCK_STEP = int(os.getenv("AIH_CROSSDOC_PREALIGN_TIMEBLOCK_STEP", "10"))
RUN_ROOT: Path
SENTENCE_DIR: Path
SEQUENCE_DIR: Path
TIMEBLOCK_DIR: Path


SENTENCE_FILE_SUFFIXES = ("_sentence", "_interlude")
TIMEBLOCK_FILE_SUFFIXES = ("_timeblock", "_timeblocks_updated")
MODEL = os.getenv("AIH_CHAT_MODEL", CHAT_MODEL)
STEP_LABEL = os.getenv("AIH_CROSSDOC_PREALIGN_STEP_LABEL", "Step10b")
REPORT_NAME = os.getenv(
    "AIH_CROSSDOC_PREALIGN_REPORT_NAME",
    "step10b_crossdoc_prealign_report.json",
)
CLIENT = None
CONCURRENCY = max(1, int(os.getenv("AIH_PIPELINE_CONCURRENCY", os.getenv("AIH_AGENT_CONCURRENCY", "4"))))
TOP_K = max(1, int(os.getenv("AIH_CROSSDOC_PREALIGN_TOP_K", "6")))
MAX_VERIFICATION_JOBS = max(1, int(os.getenv("AIH_CROSSDOC_PREALIGN_MAX_VERIFY", "80")))
MAX_INTERVAL_ANCHOR_SPAN = max(1, int(os.getenv("AIH_CROSSDOC_INTERVAL_MAX_ANCHOR_SPAN", "3")))
MIN_RETRIEVAL_SCORE = float(os.getenv("AIH_CROSSDOC_PREALIGN_MIN_SCORE", "0.08"))
MAX_BLOCK_TEXT_CHARS = int(os.getenv("AIH_CROSSDOC_PREALIGN_MAX_TEXT_CHARS", "1200"))
MAX_SCHEMA_TEXT_CHARS = int(os.getenv("AIH_CROSSDOC_SCHEMA_MAX_TEXT_CHARS", "6000"))
MIN_EPISODE_CONFIDENCE = float(os.getenv("AIH_CROSSDOC_PREALIGN_MIN_EPISODE_CONF", "0.60"))
MIN_QUOTE_SUPPORT_SCORE = float(os.getenv("AIH_CROSSDOC_QUOTE_MIN_SCORE", "0.50"))
MIN_ATOMIC_TIME_EVIDENCE_CONFIDENCE = float(os.getenv("AIH_CROSSDOC_TIME_EVIDENCE_MIN_CONF", "0.82"))
QUALITY_GATE_MODE = os.getenv("AIH_CROSSDOC_QUALITY_GATE_MODE", "warn").strip().lower()
RECALL_ACCEPT_WEAK_CONTEXT = os.getenv("AIH_CROSSDOC_RECALL_ACCEPT_WEAK_CONTEXT", "0").strip().lower() in {"1", "true", "yes", "on"}
MIN_WEAK_CONTEXT_CONFIDENCE = float(os.getenv("AIH_CROSSDOC_WEAK_CONTEXT_MIN_CONF", "0.40"))
GENERIC_DYNASTY_RE = re.compile(r"^[\u4e00-\u9fff]{1,4}朝(?:时期|期|时代)?$")
LOCATION_SUFFIX_RE = re.compile(r"[\u4e00-\u9fff]{1,6}(?:县|郡|关|城|宫|府库)")
ACTIVE_ALIAS: Dict[str, str] = {}
ACTIVE_LOCATIONS: Set[str] = set()
ACTIVE_ACTION_TERMS: Set[str] = set()
CROSSDOC_FIELD_KEYS = (
    "crossdoc_boundary",
    "crossdoc_prealign",
    "crossdoc_event_links",
    "crossdoc_time_evidence",
    "crossdoc_interval_evidence",
    "crossdoc_context_evidence",
    "event_cluster_id",
)
RUNTIME_SCOPE_STRATEGIES = {
    "",
    "runtime",
    "runtime_scope",
    "episode_packet",
    "runtime_episode_packet",
    "embedding_window",
    "runtime_embedding_window",
}
LEGACY_SCOPE_STRATEGIES = {"legacy", "full_text", "fulltext", "plain", "off", "none", "disabled"}


class EventSignature(BaseModel):
    event_type: str = Field("", description="通用事件类型，例如 military, travel, appointment, governance, speech, record, unclear")
    participants: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    time_hint: str = ""
    event_summary: str = ""
    is_atomic_event: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class EventEquivalence(BaseModel):
    relation: str = Field(
        ...,
        pattern=r"^(same_atomic_event|same_event_different_granularity|subevent_of|summary_of_phase|temporal_neighbor|background_context|same_event|broader_same_episode|causal_neighbor|unrelated)$",
    )
    same_event: bool
    transferable_anchor: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""


class IntervalDecision(BaseModel):
    relation: str = Field(
        ...,
        pattern=r"^(contained_in_source_interval|overlaps_source_interval|source_interval_too_narrow|target_too_broad|unrelated)$",
    )
    transferable_interval: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""


class CrossdocInducedSchema(BaseModel):
    entity_alias_groups: List[List[str]] = Field(
        default_factory=list,
        description="Only aliases/coreferent names observed in the provided input texts.",
    )
    location_terms: List[str] = Field(
        default_factory=list,
        description="Location or geopolitical terms observed in the provided input texts.",
    )
    action_terms: List[str] = Field(
        default_factory=list,
        description="Event trigger words or short verb phrases observed in the provided input texts.",
    )
    notes: str = ""


class EpisodeContextItem(BaseModel):
    target_timeblock_id: Union[str, None] = ""
    episode_label: Union[str, None] = ""
    relation: str = Field(
        "episode_context",
        pattern=r"^(episode_context|same_sequence_phase|weak_context|unrelated)$",
    )
    source_anchor_before_timeblock_id: Union[str, None] = ""
    source_anchor_after_timeblock_id: Union[str, None] = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    supporting_source_timeblock_ids: List[Union[str, None]] = Field(default_factory=list)
    supporting_source_quote: Union[str, None] = ""
    supporting_target_quote: Union[str, None] = ""
    reason: Union[str, None] = ""


class EpisodeContextResponse(BaseModel):
    contexts: List[EpisodeContextItem] = Field(default_factory=list)


def normalize_episode_context_response(response: EpisodeContextResponse) -> EpisodeContextResponse:
    cleaned: List[EpisodeContextItem] = []
    for item in response.contexts or []:
        item.target_timeblock_id = str(item.target_timeblock_id or "").strip()
        item.episode_label = str(item.episode_label or "").strip()
        item.source_anchor_before_timeblock_id = str(item.source_anchor_before_timeblock_id or "").strip()
        item.source_anchor_after_timeblock_id = str(item.source_anchor_after_timeblock_id or "").strip()
        item.supporting_source_quote = str(item.supporting_source_quote or "").strip()
        item.supporting_target_quote = str(item.supporting_target_quote or "").strip()
        item.reason = str(item.reason or "").strip()
        item.supporting_source_timeblock_ids = [
            str(value).strip()
            for value in item.supporting_source_timeblock_ids or []
            if str(value or "").strip()
        ]
        cleaned.append(item)
    response.contexts = cleaned
    return response


def compact_schema_list(values: List[str], limit: int = 80) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        item = re.sub(r"\s+", "", str(value or "").strip())
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item[:40])
        if len(out) >= limit:
            break
    return out


def normalize_induced_schema(schema: CrossdocInducedSchema) -> CrossdocInducedSchema:
    groups: List[List[str]] = []
    seen_groups: Set[Tuple[str, ...]] = set()
    for group in schema.entity_alias_groups or []:
        cleaned = compact_schema_list(group, limit=12)
        if len(cleaned) < 2:
            continue
        key = tuple(sorted(cleaned))
        if key in seen_groups:
            continue
        seen_groups.add(key)
        groups.append(cleaned)
        if len(groups) >= 40:
            break
    return CrossdocInducedSchema(
        entity_alias_groups=groups,
        location_terms=compact_schema_list(schema.location_terms, limit=120),
        action_terms=compact_schema_list(schema.action_terms, limit=120),
        notes=str(schema.notes or "").strip()[:300],
    )


def set_active_schema(schema: CrossdocInducedSchema) -> None:
    global ACTIVE_ALIAS, ACTIVE_LOCATIONS, ACTIVE_ACTION_TERMS
    alias: Dict[str, str] = {}
    for group in schema.entity_alias_groups:
        canonical = sorted(group, key=len, reverse=True)[0]
        for name in group:
            alias[name] = canonical
    ACTIVE_ALIAS = alias
    ACTIVE_LOCATIONS = set(schema.location_terms)
    ACTIVE_ACTION_TERMS = set(schema.action_terms)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_suffix(stem: str, suffixes: Tuple[str, ...]) -> str:
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    raise ValueError(f"无法识别文件后缀: {stem}")


def doc_key_from_sentence(path: Path) -> str:
    return strip_suffix(path.stem, SENTENCE_FILE_SUFFIXES)


def doc_key_from_timeblock(path: Path) -> str:
    return strip_suffix(path.stem, TIMEBLOCK_FILE_SUFFIXES)


def doc_code(doc_id: str) -> str:
    return doc_id.split("_", 1)[0]


def short_id(number: str) -> str:
    parts = str(number).split(".")
    return ".".join(parts[-3:]) if len(parts) >= 3 else str(number)


def parse_number(number: str) -> Tuple[str, int, int, int]:
    parts = str(number).strip().rsplit(".", 3)
    if len(parts) != 4:
        raise ValueError(f"无法解析 number: {number}")
    return parts[0], int(parts[1]), int(parts[2]), int(parts[3])


def order_key(number: str) -> Tuple[int, int, int]:
    _uuid, chapter, paragraph, sentence = parse_number(number)
    return chapter, paragraph, sentence


def split_range(range_text: str) -> Tuple[str, str]:
    match = re.match(r"^(.+\.\d+\.\d+\.\d+)-(.+\.\d+\.\d+\.\d+)$", str(range_text).strip())
    if not match:
        raise ValueError(f"无法解析 timeblock_range: {range_text}")
    return match.group(1), match.group(2)


def in_range(number: str, range_text: str) -> bool:
    start, end = split_range(range_text)
    number_uuid = parse_number(number)[0]
    if parse_number(start)[0] != number_uuid or parse_number(end)[0] != number_uuid:
        return False
    return order_key(start) <= order_key(number) <= order_key(end)


def load_crossdoc_scope() -> Dict[str, Any]:
    path = os.getenv("AIH_CROSSDOC_SCOPE_FILE", "").strip()
    if not path:
        return {}
    scope_path = Path(path)
    if not scope_path.exists():
        return {}
    payload = load_json(scope_path)
    cases = payload.get("cases", {})
    return cases if isinstance(cases, dict) else {}


def build_case_doc_pairs(doc_ids: List[str]) -> List[Tuple[str, str, str, Optional[Dict[str, Any]]]]:
    scope = load_crossdoc_scope()
    if not scope:
        return [("", left, right, None) for i, left in enumerate(doc_ids) for right in doc_ids[i + 1 :]]

    by_code: Dict[str, List[str]] = {}
    for doc_id in doc_ids:
        by_code.setdefault(doc_code(doc_id), []).append(doc_id)

    pairs: List[Tuple[str, str, str, Optional[Dict[str, Any]]]] = []
    for case_id, case_scope in scope.items():
        raw_pair = case_scope.get("doc_pair", [])
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            continue
        for left in by_code.get(str(raw_pair[0]), []):
            for right in by_code.get(str(raw_pair[1]), []):
                if left != right:
                    pairs.append((str(case_id), left, right, case_scope))
    return pairs


def allowed_short_ids(case_scope: Optional[Dict[str, Any]], doc_id: str) -> Optional[Set[str]]:
    if not case_scope:
        return None
    by_doc = case_scope.get("allowed_short_ids_by_doc", {})
    if not isinstance(by_doc, dict):
        return None
    values = by_doc.get(doc_code(doc_id), [])
    return {str(item) for item in values} if isinstance(values, list) else set()


def candidate_focus_for_direction(
    case_scope: Optional[Dict[str, Any]],
    source_doc: str,
    target_doc: str,
) -> Dict[str, Any]:
    if not case_scope:
        return {}
    focus = case_scope.get("candidate_focus", {})
    if not isinstance(focus, dict):
        return {}
    by_doc = focus.get("focus_by_doc", {})
    if not isinstance(by_doc, dict):
        by_doc = {}
    source_focus = by_doc.get(doc_code(source_doc), {})
    target_focus = by_doc.get(doc_code(target_doc), {})
    return {
        "strategy": focus.get("strategy", ""),
        "selector": focus.get("selector", ""),
        "retrieval_score": focus.get("retrieval_score", 0),
        "source_doc": doc_code(source_doc),
        "target_doc": doc_code(target_doc),
        "source_focus": source_focus if isinstance(source_focus, dict) else {},
        "target_focus": target_focus if isinstance(target_focus, dict) else {},
    }


def text_for_block(block: Dict[str, Any], sentences: List[Dict[str, Any]]) -> str:
    range_text = str(block.get("timeblock_range", "") or "")
    chunks = []
    for sentence in sentences:
        number = str(sentence.get("number", "") or "")
        try:
            if number and in_range(number, range_text):
                chunks.append(str(sentence.get("sentence", "") or ""))
        except Exception:
            continue
    return "\n".join(chunks)


def sentence_numbers_for_block(block: Dict[str, Any], sentences: List[Dict[str, Any]]) -> List[str]:
    numbers = []
    for sentence in sentences:
        number = str(sentence.get("number", "") or "")
        try:
            if number and in_range(number, str(block.get("timeblock_range", "") or "")):
                numbers.append(number)
        except Exception:
            continue
    return numbers


def has_explicit_temporal_anchor(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    return bool(
        re.search(r"(?:公元前|前)?\d{1,4}年", value)
        or re.search(r"(?:元|[一二三四五六七八九十百廿卅]+)年", value)
        or re.search(r"[正一二三四五六七八九十冬腊]+月", value)
        or re.search(r"(春季?|夏季?|秋季?|冬季?)", value)
    )


def is_concrete_anchor(block: Dict[str, Any]) -> bool:
    tm = str(block.get("TM", "") or "").strip()
    gran = str(block.get("Granularity", "") or "").strip()
    if gran == "0" or not tm:
        return False
    return has_explicit_temporal_anchor(tm)


def update_sequence_file(path: Path, replacements: Dict[str, List[str]]) -> int:
    if not path.exists() or not replacements:
        return 0
    data = load_json(path)
    if not isinstance(data, list):
        return 0
    changed = 0
    out: List[str] = []
    for item in data:
        value = str(item)
        if value in replacements:
            out.extend(replacements[value])
            changed += 1
        else:
            out.append(value)
    if changed:
        save_json(path, out)
    return changed


def split_non_anchor_blocks(
    blocks: List[Dict[str, Any]],
    sentences: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], int]:
    replacements: Dict[str, List[str]] = {}
    split_count = 0
    result: List[Dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        tm = str(block.get("TM", "") or "")
        sentence_numbers = sentence_numbers_for_block(block, sentences)
        # Keep any generated TM as a single timeblock. Sentence-level splitting is
        # only for long narrative spans that still have no temporal marker.
        should_split = (
            len(sentence_numbers) > 1
            and str(block.get("Granularity", "") or "").strip() == "0"
            and not tm.strip()
        )
        if not should_split:
            result.append(block)
            continue

        new_ids: List[str] = []
        for number in sentence_numbers:
            new_block = dict(block)
            new_block["ID"] = number
            new_block["timeblock_range"] = f"{number}-{number}"
            new_block["Granularity"] = "0"
            new_block.pop("iso", None)
            new_block.pop("iso_range", None)
            new_block["crossdoc_prealign_split_from"] = block.get("ID", "")
            result.append(new_block)
            new_ids.append(number)
        replacements[str(block.get("ID", ""))] = new_ids
        split_count += 1

    result.sort(key=lambda obj: order_key(str(obj.get("ID", ""))))
    return result, replacements, split_count


def clean_list(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for value in values or []:
        item = canonical_entity(re.sub(r"\s+", "", str(value or "").strip()))
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item[:40])
    return out[:12]


def canonical_entity(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip())
    if not text:
        return ""
    return ACTIVE_ALIAS.get(text, text)


def canonicalize_text(text: str) -> str:
    compact = compact_text(text)
    for alias, canonical in sorted(ACTIVE_ALIAS.items(), key=lambda item: len(item[0]), reverse=True):
        compact = compact.replace(alias, canonical)
    return compact


def alias_terms(text: str, sig: Optional[EventSignature] = None) -> Set[str]:
    terms: Set[str] = set()
    compact = compact_text(text)
    for alias, canonical in ACTIVE_ALIAS.items():
        if alias and alias in compact:
            terms.add(canonical)
    if isinstance(sig, EventSignature):
        for field in ("participants", "objects"):
            for item in getattr(sig, field, []) or []:
                canonical = canonical_entity(str(item))
                if canonical:
                    terms.add(canonical)
    return terms


def action_terms(text: str, sig: Optional[EventSignature] = None) -> Set[str]:
    compact = compact_text(text)
    terms = {term for term in ACTIVE_ACTION_TERMS if term and term in compact}
    if isinstance(sig, EventSignature):
        action_blob = compact_text("".join(sig.actions) + sig.event_summary)
        terms.update(term for term in ACTIVE_ACTION_TERMS if term and term in action_blob)
        if sig.event_type and sig.event_type != "unclear":
            terms.add(f"type:{sig.event_type}")
    return terms


def location_terms(text: str, sig: Optional[EventSignature] = None) -> Set[str]:
    compact = compact_text(text)
    terms = {match.group(0) for match in LOCATION_SUFFIX_RE.finditer(compact)}
    terms.update(location for location in ACTIVE_LOCATIONS if location and location in compact)
    if isinstance(sig, EventSignature):
        terms.update(canonical_entity(str(item)) for item in sig.locations if str(item).strip())
    return {term for term in terms if term}


def event_feature_terms(text: str, sig: Optional[EventSignature] = None) -> Set[str]:
    terms = set()
    terms.update(f"person:{item}" for item in alias_terms(text, sig))
    terms.update(f"action:{item}" for item in action_terms(text, sig))
    terms.update(f"loc:{item}" for item in location_terms(text, sig))
    if isinstance(sig, EventSignature):
        terms.update(f"object:{canonical_entity(str(item))}" for item in sig.objects if str(item).strip())
    return {term for term in terms if term and not term.endswith(":")}


def fallback_signature(text: str) -> EventSignature:
    compact = re.sub(r"\s+", "", text or "")
    return EventSignature(
        event_type="unclear",
        event_summary=compact[:120],
        is_atomic_event=len(compact) <= 160,
        confidence=0.2 if compact else 0.0,
    )


def signature_is_informative(sig: EventSignature) -> bool:
    if not isinstance(sig, EventSignature):
        return False
    role_count = (
        len(sig.participants or [])
        + len(sig.locations or [])
        + len(sig.actions or [])
        + len(sig.objects or [])
    )
    has_type = bool(str(sig.event_type or "").strip() and str(sig.event_type or "").strip() != "unclear")
    has_summary = bool(str(sig.event_summary or "").strip())
    return bool(role_count or has_type or has_summary)


def doc_text_for_schema(doc_id: str, sentences: Dict[str, List[Dict[str, Any]]]) -> str:
    chunks = []
    for row in sentences.get(doc_id, []):
        text = str(row.get("sentence", "") or "").strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks)[:MAX_SCHEMA_TEXT_CHARS]


def schema_prompt(left_doc_code: str, left_text: str, right_doc_code: str, right_text: str) -> str:
    return f"""
你要为两个历史叙事输入文本归纳跨文本事件共指所需的 schema。只能使用下面两篇输入文本中出现的信息，不要补充外部知识，不要参考标准答案。

输出 JSON：
{{
  "entity_alias_groups": [["同一人物或集团在文本中出现的不同名称"]],
  "location_terms": ["文本中出现的地点、区域、政权空间"],
  "action_terms": ["文本中出现的事件触发词或短动宾短语"],
  "notes": "简短说明"
}}

严格要求：
- entity_alias_groups 只放你能从输入文本判断为同一实体的别名；不确定就不要合并。
- location_terms 只放文本内出现的地名/区域/政权空间，不要扩展。
- action_terms 只放文本内出现的核心事件动词或短语，用于后续候选召回。
- 这是召回 schema，不是答案；不要输出任何 ISO、年份推断或最终时间范围。

文档 A ({left_doc_code}):
{left_text}

文档 B ({right_doc_code}):
{right_text}
""".strip()


def schema_is_informative(schema: CrossdocInducedSchema) -> bool:
    return bool(schema.entity_alias_groups or schema.location_terms or schema.action_terms)


def is_fatal_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    fatal_markers = (
        "authenticationerror",
        "incorrect api key",
        "invalid api key",
        "permission denied",
        "unauthorized",
        "401",
        "quota",
        "billing",
    )
    return any(marker in text for marker in fatal_markers)


def signature_candidates_for_schema(
    left_doc: str,
    right_doc: str,
    block_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, List[str]]:
    participants: Set[str] = set()
    locations: Set[str] = set()
    actions: Set[str] = set()
    objects: Set[str] = set()
    summaries: List[str] = []
    for (doc_id, _block_id), item in block_index.items():
        if doc_id not in {left_doc, right_doc}:
            continue
        sig = item.get("signature")
        if not isinstance(sig, EventSignature):
            continue
        participants.update(str(value).strip() for value in sig.participants if str(value).strip())
        locations.update(str(value).strip() for value in sig.locations if str(value).strip())
        actions.update(str(value).strip() for value in sig.actions if str(value).strip())
        objects.update(str(value).strip() for value in sig.objects if str(value).strip())
        if sig.event_summary:
            summaries.append(str(sig.event_summary).strip())
    return {
        "participants": compact_schema_list(sorted(participants), limit=120),
        "locations": compact_schema_list(sorted(locations), limit=120),
        "actions": compact_schema_list(sorted(actions), limit=120),
        "objects": compact_schema_list(sorted(objects), limit=120),
        "summaries": compact_schema_list(summaries, limit=80),
    }


def schema_candidate_prompt(left_doc_code: str, right_doc_code: str, candidates: Dict[str, List[str]]) -> str:
    return f"""
你要根据两个历史文本中已经抽取出的事件签名候选，归纳跨文本事件共指所需 schema。只能使用候选列表中的词，不要补充外部知识，不要参考标准答案。

输出 JSON：
{{
  "entity_alias_groups": [["同一人物或集团在候选中出现的不同名称"]],
  "location_terms": ["候选中出现的地点、区域、政权空间"],
  "action_terms": ["候选中出现的事件触发词或短动宾短语"],
  "notes": "简短说明"
}}

要求：
- entity_alias_groups 只合并明显同一实体的名称；不确定就不要合并。
- location_terms 从 participants/locations/objects/summaries 中挑出地名或区域。
- action_terms 从 actions/summaries 中挑出核心动作词或短语。
- 输出空 schema 没有帮助；如果候选中有可用地点或动作，必须填入。
- 不要输出 ISO、年份推断或最终时间范围。

doc_pair: {left_doc_code} <-> {right_doc_code}
participants: {json.dumps(candidates.get("participants", []), ensure_ascii=False)}
locations: {json.dumps(candidates.get("locations", []), ensure_ascii=False)}
actions: {json.dumps(candidates.get("actions", []), ensure_ascii=False)}
objects: {json.dumps(candidates.get("objects", []), ensure_ascii=False)}
summaries: {json.dumps(candidates.get("summaries", []), ensure_ascii=False)}
""".strip()


def fallback_schema_from_signature_candidates(candidates: Dict[str, List[str]], note: str) -> CrossdocInducedSchema:
    return CrossdocInducedSchema(
        entity_alias_groups=[],
        location_terms=compact_schema_list(candidates.get("locations", []) + candidates.get("objects", []), limit=120),
        action_terms=compact_schema_list(candidates.get("actions", []), limit=120),
        notes=note,
    )


def induce_crossdoc_schema(
    left_doc: str,
    right_doc: str,
    sentences: Dict[str, List[Dict[str, Any]]],
    block_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> CrossdocInducedSchema:
    left_text = doc_text_for_schema(left_doc, sentences)
    right_text = doc_text_for_schema(right_doc, sentences)
    if not left_text and not right_text:
        return CrossdocInducedSchema()
    try:
        response = create_chat_completion(
            CLIENT,
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是严谨的跨文本事件 schema 归纳器，只输出 JSON。"},
                {"role": "user", "content": schema_prompt(doc_code(left_doc), left_text, doc_code(right_doc), right_text)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1200,
        )
        content = response.choices[0].message.content or "{}"
        schema = normalize_induced_schema(validate_json_text(CrossdocInducedSchema, content))
        if schema_is_informative(schema):
            return schema
        first_note = "full_text_schema_empty"
    except Exception as exc:
        if is_fatal_llm_error(exc):
            raise
        first_note = f"full_text_schema_failed: {type(exc).__name__}: {str(exc)[:160]}"

    candidates = signature_candidates_for_schema(left_doc, right_doc, block_index)
    try:
        response = create_chat_completion(
            CLIENT,
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是严谨的跨文本事件 schema 归纳器，只输出 JSON。"},
                {"role": "user", "content": schema_candidate_prompt(doc_code(left_doc), doc_code(right_doc), candidates)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1200,
        )
        content = response.choices[0].message.content or "{}"
        schema = normalize_induced_schema(validate_json_text(CrossdocInducedSchema, content))
        schema.notes = "; ".join(part for part in [first_note, schema.notes] if part)
        if schema_is_informative(schema):
            return schema
        return fallback_schema_from_signature_candidates(candidates, f"{first_note}; candidate_schema_empty; fallback_from_event_signatures")
    except Exception as exc:
        if is_fatal_llm_error(exc):
            raise
        return fallback_schema_from_signature_candidates(candidates, f"{first_note}; candidate_schema_failed: {type(exc).__name__}: {str(exc)[:160]}; fallback_from_event_signatures")


def signature_prompt(text: str, tm: str) -> str:
    return f"""
你要为跨文本历史叙事对齐生成通用事件签名。不要依赖特定语料词表；只根据文本语义抽取。

输出 JSON，字段固定为：
{{
  "event_type": "military|travel|appointment|governance|speech|record|death|meeting|unclear 等通用类型",
  "participants": ["人物、集团、军队、官职主体"],
  "locations": ["地点、区域、政权空间"],
  "actions": ["核心动作，使用短动词或动宾短语"],
  "objects": ["动作对象、制度对象、物资对象"],
  "time_hint": "文本内出现的时间提示；没有则空",
  "event_summary": "一句话概括该 TimeBlock 的中心事件",
  "is_atomic_event": true,
  "confidence": 0.0
}}

判断原则：
- 抽取通用语义角色，不要补充文本没有的信息。
- 如果该块包含多个事件，summary 写最中心事件，is_atomic_event=false。
- 时间提示只记录文本内有的表达，不要推理 ISO。

TM: {tm}
文本:
{text[:MAX_BLOCK_TEXT_CHARS]}
""".strip()


def call_event_signature(text: str, tm: str) -> EventSignature:
    if not str(text or "").strip():
        return fallback_signature(text)
    prompts = [
        signature_prompt(text, tm),
        signature_prompt(text, tm)
        + "\n\n上一轮如果抽取为空，必须重新阅读文本。只要文本中有事件，就至少填写 event_summary、participants、actions；不确定的字段可以少填，但不要返回全空 JSON。",
    ]
    for prompt in prompts:
        try:
            response = create_chat_completion(
                CLIENT,
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是严谨的历史事件抽取器，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=700,
            )
            content = response.choices[0].message.content or "{}"
            sig = validate_json_text(EventSignature, content)
            sig.participants = clean_list(sig.participants)
            sig.locations = clean_list(sig.locations)
            sig.actions = clean_list(sig.actions)
            sig.objects = clean_list(sig.objects)
            sig.event_summary = str(sig.event_summary or "").strip()[:220]
            sig.time_hint = str(sig.time_hint or "").strip()[:80]
            if signature_is_informative(sig):
                return sig
        except Exception as exc:
            if is_fatal_llm_error(exc):
                raise
            continue
    return fallback_signature(text)


def char_bigrams(text: str) -> Set[str]:
    compact = re.sub(r"\s+", "", text or "")
    return {
        compact[i : i + 2]
        for i in range(max(0, len(compact) - 1))
        if re.search(r"[\u4e00-\u9fff]{2}", compact[i : i + 2])
    }


def compact_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text or "")


def is_generic_dynasty_text(text: str) -> bool:
    return bool(GENERIC_DYNASTY_RE.match(re.sub(r"\s+", "", str(text or ""))))


def jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def raw_text_similarity(left_text: str, right_text: str) -> float:
    left = canonicalize_text(left_text)
    right = canonicalize_text(right_text)
    if not left or not right:
        return 0.0
    bigram_score = jaccard(char_bigrams(left), char_bigrams(right))
    sequence_score = SequenceMatcher(None, left, right).ratio()
    return max(bigram_score, sequence_score)


def field_set(sig: EventSignature, field: str) -> Set[str]:
    value = getattr(sig, field)
    if isinstance(value, list):
        return {str(item) for item in value if str(item).strip()}
    return set()


def semantic_overlap(left_sig: EventSignature, right_sig: EventSignature, left_text: str, right_text: str) -> float:
    score = 0.0
    score += 0.30 * jaccard(action_terms(left_text, left_sig), action_terms(right_text, right_sig))
    score += 0.25 * jaccard(alias_terms(left_text, left_sig), alias_terms(right_text, right_sig))
    score += 0.18 * jaccard(location_terms(left_text, left_sig), location_terms(right_text, right_sig))
    score += 0.12 * jaccard(field_set(left_sig, "actions"), field_set(right_sig, "actions"))
    score += 0.05 * jaccard(field_set(left_sig, "objects"), field_set(right_sig, "objects"))
    score += 0.10 * raw_text_similarity(left_text, right_text)
    return score


def retrieval_score(left_text: str, left_sig: EventSignature, right_text: str, right_sig: EventSignature) -> float:
    left_blob = " ".join([
        left_sig.event_summary,
        " ".join(left_sig.participants),
        " ".join(left_sig.locations),
        " ".join(left_sig.actions),
        " ".join(left_sig.objects),
    ]) or left_text
    right_blob = " ".join([
        right_sig.event_summary,
        " ".join(right_sig.participants),
        " ".join(right_sig.locations),
        " ".join(right_sig.actions),
        " ".join(right_sig.objects),
    ]) or right_text
    left_features = event_feature_terms(left_text, left_sig)
    right_features = event_feature_terms(right_text, right_sig)
    score = 0.24 * raw_text_similarity(left_text, right_text)
    score += 0.18 * jaccard(char_bigrams(canonicalize_text(left_blob)), char_bigrams(canonicalize_text(right_blob)))
    score += 0.20 * jaccard(alias_terms(left_text, left_sig), alias_terms(right_text, right_sig))
    score += 0.16 * jaccard(action_terms(left_text, left_sig), action_terms(right_text, right_sig))
    score += 0.10 * jaccard(location_terms(left_text, left_sig), location_terms(right_text, right_sig))
    score += 0.08 * jaccard(left_features, right_features)
    score += 0.03 * jaccard(field_set(left_sig, "objects"), field_set(right_sig, "objects"))
    if left_sig.event_type and left_sig.event_type == right_sig.event_type and left_sig.event_type != "unclear":
        score += 0.01
    return score


def retrieval_bucket_scores(left_text: str, left_sig: EventSignature, right_text: str, right_sig: EventSignature) -> Dict[str, float]:
    buckets: Dict[str, float] = {}
    raw = raw_text_similarity(left_text, right_text)
    if raw >= 0.12:
        buckets["raw_text"] = raw
    alias_score = jaccard(alias_terms(left_text, left_sig), alias_terms(right_text, right_sig))
    if alias_score > 0:
        buckets["participant"] = alias_score
    action_score = jaccard(action_terms(left_text, left_sig), action_terms(right_text, right_sig))
    if action_score > 0:
        buckets["action"] = action_score
    location_score = jaccard(location_terms(left_text, left_sig), location_terms(right_text, right_sig))
    if location_score > 0:
        buckets["location"] = location_score
    object_score = jaccard(field_set(left_sig, "objects"), field_set(right_sig, "objects"))
    if object_score > 0:
        buckets["object"] = object_score
    if left_sig.event_type and left_sig.event_type == right_sig.event_type and left_sig.event_type != "unclear":
        buckets["event_type"] = 1.0
    semantic = semantic_overlap(left_sig, right_sig, left_text, right_text)
    if semantic >= 0.10:
        buckets["semantic"] = semantic
    return buckets


def is_lexical_same_event(left_text: str, right_text: str) -> bool:
    return raw_text_similarity(left_text, right_text) >= 0.58


def equivalence_prompt(
    source_text: str,
    source_tm: str,
    source_sig: EventSignature,
    target_text: str,
    target_tm: str,
    target_sig: EventSignature,
) -> str:
    source_signature_json = json.dumps(source_sig.model_dump(), ensure_ascii=False)
    target_signature_json = json.dumps(target_sig.model_dump(), ensure_ascii=False)
    source_features_json = json.dumps(sorted(event_feature_terms(source_text, source_sig)), ensure_ascii=False)
    target_features_json = json.dumps(sorted(event_feature_terms(target_text, target_sig)), ensure_ascii=False)
    return f"""
你要判断两个 TimeBlock 的跨文本事件共指关系，并决定是否可以把 source 的时间锚点作为 target 的事件时间证据。

只输出 JSON：
{{
  "relation": "same_atomic_event|same_event_different_granularity|subevent_of|summary_of_phase|temporal_neighbor|background_context|unrelated",
  "same_event": true,
  "transferable_anchor": true,
  "confidence": 0.0,
  "reason": "简短理由"
}}

严格规则：
- same_atomic_event：两个 mention 指向同一原子事件，例如同一场封王、同一次攻取、同一次死亡、同一次会盟。
- same_event_different_granularity：一个 mention 是另一个同一事件的详细/概括表述，但仍然是同一事件，不是整个阶段。
- subevent_of：source 是 target 阶段中的一个子事件，或 target 是 source 所属阶段，不可直接传播锚点。
- summary_of_phase：target 是阶段摘要、履历阶段、战役阶段或长期状态，不可直接传播单个 source 锚点。
- temporal_neighbor：前后相邻、因果相邻或同地相邻，但不是同一事件。
- background_context：只共享人物、政权、朝代、地点或主题背景。
- unrelated：无明确事件关系。
- 只有 same_atomic_event 和高置信 same_event_different_granularity 可以 transferable_anchor=true。
- subevent_of、summary_of_phase、temporal_neighbor、background_context、unrelated 必须 transferable_anchor=false。
- 不要因为二者属于同一篇章、同一历史时期、同一人物生涯或主题相近就传播锚点。
- 人名可能有别名，例如同一人物在不同文本中以名、字、爵号、官称出现；可以参考 canonical event features，但最终仍要根据事件语义判定。

source TM: {source_tm}
source event signature: {source_signature_json}
source canonical event features: {source_features_json}
source text:
{source_text[:MAX_BLOCK_TEXT_CHARS]}

target TM: {target_tm}
target event signature: {target_signature_json}
target canonical event features: {target_features_json}
target text:
{target_text[:MAX_BLOCK_TEXT_CHARS]}
""".strip()


def call_event_equivalence(
    source_text: str,
    source_tm: str,
    source_sig: EventSignature,
    target_text: str,
    target_tm: str,
    target_sig: EventSignature,
) -> EventEquivalence:
    prompt = equivalence_prompt(source_text, source_tm, source_sig, target_text, target_tm, target_sig)
    messages = [
        {"role": "system", "content": "你是严谨的跨文本事件对齐审查器，只输出 JSON。"},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            response = create_chat_completion(
                CLIENT,
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=520,
            )
            content = response.choices[0].message.content or "{}"
            return validate_json_text(EventEquivalence, content)
        except Exception as exc:
            last_error = exc
            messages = [
                {"role": "system", "content": "你必须只输出完整 JSON，不能输出空对象。"},
                {
                    "role": "user",
                    "content": prompt
                    + "\n\n上一次输出无法解析。请严格输出包含 relation、same_event、transferable_anchor、confidence、reason 的 JSON。",
                },
            ]
    return EventEquivalence(
        relation="unrelated",
        same_event=False,
        transferable_anchor=False,
        confidence=0.0,
        reason=(
            "LLM equivalence judgment failed after retries: "
            f"{type(last_error).__name__ if last_error else 'Unknown'}: "
            f"{str(last_error)[:240] if last_error else ''}"
        )
    )


def is_transferable_decision(decision: EventEquivalence) -> bool:
    if not decision.transferable_anchor:
        return False
    if decision.relation == "same_atomic_event" and decision.same_event:
        return True
    if decision.relation == "same_event_different_granularity" and decision.same_event and decision.confidence >= MIN_ATOMIC_TIME_EVIDENCE_CONFIDENCE:
        return True
    # Backward compatibility for older prompt/cache outputs.
    if decision.relation == "same_event" and decision.same_event and decision.confidence >= MIN_ATOMIC_TIME_EVIDENCE_CONFIDENCE:
        return True
    return False


def interval_prompt(target_text: str, target_tm: str, interval: Dict[str, Any]) -> str:
    return f"""
你要判断 target TimeBlock 的叙事是否可以被 source 文档的一个时间轴区间覆盖。

这是通用跨文本时间轴定位任务，不是标准答案匹配。只根据 source interval 和 target text 的语义判断。

只输出 JSON：
{{
  "relation": "contained_in_source_interval|overlaps_source_interval|source_interval_too_narrow|target_too_broad|unrelated",
  "transferable_interval": true,
  "confidence": 0.0,
  "reason": "简短理由"
}}

严格规则：
- contained_in_source_interval：target 的事件、阶段或压缩叙述整体落在 source interval 覆盖的时间段内，可以传播 interval。
- overlaps_source_interval：target 与 source interval 有部分重叠，但不能确定完整覆盖，不可传播。
- source_interval_too_narrow：source interval 只是 target 的一个子事件或片段，不可传播。
- target_too_broad：target 跨越 source interval 之外的更长阶段，不可传播。
- unrelated：只共享人物、朝代、地点、主题背景，或没有明确语义关系。
- 只有 contained_in_source_interval 且置信度高时 transferable_interval=true。
- 不要因为同一人物/同一朝代/同一篇章就传播区间。

source interval:
- start_tm: {interval.get('start_tm', '')}
- end_tm: {interval.get('end_tm', '')}
- start_block_id: {interval.get('start_block_id', '')}
- end_block_id: {interval.get('end_block_id', '')}
- summary: {interval.get('summary', '')}
source interval text:
{str(interval.get('text', ''))[:MAX_BLOCK_TEXT_CHARS]}

target TM: {target_tm}
target text:
{target_text[:MAX_BLOCK_TEXT_CHARS]}
""".strip()


def call_interval_decision(target_text: str, target_tm: str, interval: Dict[str, Any]) -> IntervalDecision:
    messages = [
        {"role": "system", "content": "你是严谨的跨文本时间轴区间定位审查器，只输出 JSON。"},
        {"role": "user", "content": interval_prompt(target_text, target_tm, interval)},
    ]
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            response = create_chat_completion(
                CLIENT,
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=520,
            )
            content = response.choices[0].message.content or "{}"
            return validate_json_text(IntervalDecision, content)
        except Exception as exc:
            last_error = exc
            messages = [
                {"role": "system", "content": "你必须只输出完整 JSON，不能输出空对象。"},
                {
                    "role": "user",
                    "content": interval_prompt(target_text, target_tm, interval)
                    + "\n\n上一次输出无法解析。请严格输出包含 relation、transferable_interval、confidence、reason 的 JSON。",
                },
            ]
    return IntervalDecision(
        relation="unrelated",
        transferable_interval=False,
        confidence=0.0,
        reason=f"LLM interval judgment failed after retries: {type(last_error).__name__ if last_error else 'Unknown'}: {str(last_error)[:240] if last_error else ''}",
    )


def build_block_index(
    timeblocks: Dict[str, List[Dict[str, Any]]],
    sentences: Dict[str, List[Dict[str, Any]]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for doc_id, blocks in timeblocks.items():
        doc_sentences = sentences.get(doc_id, [])
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("ID", "") or "")
            text = text_for_block(block, doc_sentences)
            sentence_rows = []
            for sentence in doc_sentences:
                number = str(sentence.get("number", "") or "")
                try:
                    if number and in_range(number, str(block.get("timeblock_range", "") or "")):
                        sentence_rows.append(sentence)
                except Exception:
                    continue
            index[(doc_id, block_id)] = {
                "doc_id": doc_id,
                "block": block,
                "block_id": block_id,
                "text": text,
                "sentences": sentence_rows,
                "sentence_numbers": sentence_numbers_for_block(block, doc_sentences),
            }
    return index


def ordered_doc_items(doc_id: str, timeblocks: Dict[str, List[Dict[str, Any]]], block_index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    items_by_id: Dict[str, Dict[str, Any]] = {}
    for block in timeblocks.get(doc_id, []):
        block_id = str(block.get("ID", "") or "")
        key = (doc_id, block_id)
        if key in block_index:
            items_by_id[block_id] = block_index[key]

    sequence_path = SEQUENCE_DIR / f"{doc_id}_sequence.json"
    if sequence_path.exists():
        try:
            sequence_ids = [str(item).strip() for item in load_json(sequence_path) if str(item).strip()]
        except Exception:
            sequence_ids = []
        ordered = [items_by_id[item_id] for item_id in sequence_ids if item_id in items_by_id]
        seen = {str(item.get("block_id", "")) for item in ordered}
        ordered.extend(item for block_id, item in items_by_id.items() if block_id not in seen)
        return ordered

    items = []
    for block in timeblocks.get(doc_id, []):
        block_id = str(block.get("ID", "") or "")
        if block_id in items_by_id:
            items.append(items_by_id[block_id])
    return items


def source_timeline_intervals(doc_id: str, timeblocks: Dict[str, List[Dict[str, Any]]], block_index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = ordered_doc_items(doc_id, timeblocks, block_index)
    anchor_indices = [idx for idx, item in enumerate(items) if is_concrete_anchor(item["block"])]
    intervals: List[Dict[str, Any]] = []
    if not anchor_indices:
        return intervals
    for pos, start_idx in enumerate(anchor_indices):
        for span in range(1, MAX_INTERVAL_ANCHOR_SPAN + 1):
            end_anchor_pos = pos + span
            next_idx = anchor_indices[end_anchor_pos] if end_anchor_pos < len(anchor_indices) else -1
            start_item = items[start_idx]
            end_item = items[next_idx] if next_idx != -1 else None
            span_end = next_idx if next_idx != -1 else len(items)
            if span_end <= start_idx:
                continue
            span_items = items[start_idx:span_end]
            chunks = [str(item.get("text", "") or "").strip() for item in span_items if str(item.get("text", "") or "").strip()]
            text = "\n".join(chunks)
            if not text:
                continue
            signature_text = text[:MAX_BLOCK_TEXT_CHARS]
            signature = call_event_signature(signature_text, str(start_item["block"].get("TM", "") or ""))
            intervals.append({
                "source_doc_id": doc_id,
                "start_block_id": start_item["block"].get("ID", ""),
                "end_block_id": end_item["block"].get("ID", "") if end_item else "",
                "start_tm": str(start_item["block"].get("TM", "") or "").strip(),
                "end_tm": str(end_item["block"].get("TM", "") or "").strip() if end_item else "",
                "open_end": end_item is None,
                "anchor_span": span,
                "text": text,
                "summary": signature.event_summary,
                "signature": signature,
            })
            if next_idx == -1:
                break
    return intervals


def interval_retrieval_score(target: Dict[str, Any], interval: Dict[str, Any]) -> float:
    target_text = str(target.get("text", ""))
    target_sig = target.get("signature") if isinstance(target.get("signature"), EventSignature) else ensure_item_signature(target)
    interval_sig = interval.get("signature") if isinstance(interval.get("signature"), EventSignature) else fallback_signature(str(interval.get("text", "")))
    interval_text = str(interval.get("text", ""))
    score = 0.25 * raw_text_similarity(target_text, interval_text)
    score += 0.35 * semantic_overlap(interval_sig, target_sig, interval_text, target_text)
    score += 0.15 * jaccard(char_bigrams(canonicalize_text(str(interval.get("summary", "")))), char_bigrams(canonicalize_text(target_text)))
    score += 0.15 * jaccard(alias_terms(interval_text, interval_sig), alias_terms(target_text, target_sig))
    score += 0.10 * jaccard(action_terms(interval_text, interval_sig), action_terms(target_text, target_sig))
    return score


def apply_crossdoc_intervals(
    case_id: str,
    left_doc: str,
    right_doc: str,
    case_scope: Optional[Dict[str, Any]],
    timeblocks: Dict[str, List[Dict[str, Any]]],
    block_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "case_id": case_id,
        "doc_pair": [doc_code(left_doc), doc_code(right_doc)],
        "source_intervals": 0,
        "target_items": 0,
        "retrieval_pairs_scored": 0,
        "verification_jobs": 0,
        "verification_jobs_before_limit": 0,
        "max_verification_jobs": MAX_VERIFICATION_JOBS,
        "accepted": 0,
        "rejected_by_relation": {},
        "top_candidates": [],
        "accepted_samples": [],
        "rejected_samples": [],
    }
    jobs_by_target: Dict[Tuple[str, str], List[Tuple[float, Dict[str, Any], Dict[str, Any]]]] = {}
    for source_doc, target_doc in [(left_doc, right_doc), (right_doc, left_doc)]:
        allowed = allowed_short_ids(case_scope, target_doc)
        intervals = source_timeline_intervals(source_doc, timeblocks, block_index)
        targets = [
            block_index[(target_doc, str(block.get("ID", "") or ""))]
            for block in timeblocks[target_doc]
            if (target_doc, str(block.get("ID", "") or "")) in block_index
            and not is_concrete_anchor(block)
            and not isinstance(block.get("crossdoc_time_evidence"), dict)
            and not isinstance(block.get("crossdoc_interval_evidence"), dict)
            and (allowed is None or short_id(str(block.get("ID", "") or "")) in allowed)
        ]
        stats["source_intervals"] += len(intervals)
        stats["target_items"] += len(targets)
        for target in targets:
            if is_generic_dynasty_text(str(target["block"].get("TM", "") or "")):
                # Generic dynasty labels are usually display/context labels; let the text drive interval matching.
                pass
            scored = []
            for interval in intervals:
                score = interval_retrieval_score(target, interval)
                stats["retrieval_pairs_scored"] += 1
                scored.append((score, interval))
            for score, interval in sorted(scored, key=lambda pair: pair[0], reverse=True)[:TOP_K]:
                if len(stats["top_candidates"]) < 40:
                    stats["top_candidates"].append({
                        "score": round(score, 4),
                        "target_timeblock_id": target["block"].get("ID", ""),
                        "target_tm": target["block"].get("TM", ""),
                        "target_text": str(target.get("text", ""))[:180],
                        "source_start_tm": interval.get("start_tm", ""),
                        "source_end_tm": interval.get("end_tm", ""),
                        "source_anchor_span": interval.get("anchor_span", ""),
                        "source_summary": str(interval.get("summary", ""))[:180],
                        "source_text": str(interval.get("text", ""))[:240],
                    })
                target_key = (str(target["doc_id"]), str(target["block_id"]))
                jobs_by_target.setdefault(target_key, []).append((score, target, interval))

    per_target_jobs: List[List[Tuple[float, Dict[str, Any], Dict[str, Any]]]] = [
        sorted(items, key=lambda item: item[0], reverse=True)[:TOP_K]
        for items in jobs_by_target.values()
    ]
    stats["verification_jobs_before_limit"] = sum(len(items) for items in per_target_jobs)
    jobs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for rank in range(TOP_K):
        round_items = [items[rank] for items in per_target_jobs if rank < len(items)]
        round_items.sort(key=lambda item: item[0], reverse=True)
        for item in round_items:
            if len(jobs) < MAX_VERIFICATION_JOBS:
                jobs.append(item)
    if len(jobs) < MAX_VERIFICATION_JOBS:
        already = {(str(target["doc_id"]), str(target["block_id"]), str(interval.get("start_block_id", "")), str(interval.get("end_block_id", ""))) for _score, target, interval in jobs}
        leftovers = []
        for items in per_target_jobs:
            for item in items:
                _score, target, interval = item
                key = (str(target["doc_id"]), str(target["block_id"]), str(interval.get("start_block_id", "")), str(interval.get("end_block_id", "")))
                if key not in already:
                    leftovers.append(item)
        for item in sorted(leftovers, key=lambda item: item[0], reverse=True):
            if len(jobs) >= MAX_VERIFICATION_JOBS:
                break
            jobs.append(item)
    stats["verification_jobs"] = len(jobs)
    best_by_target: Dict[Tuple[str, str], Tuple[float, Dict[str, Any], Dict[str, Any], IntervalDecision]] = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(
                call_interval_decision,
                str(target.get("text", "")),
                str(target["block"].get("TM", "") or ""),
                interval,
            ): (score, target, interval)
            for score, target, interval in jobs
        }
        for future in as_completed(futures):
            score, target, interval = futures[future]
            decision = future.result()
            accepted = (
                decision.transferable_interval
                and decision.relation == "contained_in_source_interval"
                and decision.confidence >= MIN_EPISODE_CONFIDENCE
            )
            if not accepted:
                stats["rejected_by_relation"][decision.relation] = stats["rejected_by_relation"].get(decision.relation, 0) + 1
                if len(stats["rejected_samples"]) < 20:
                    stats["rejected_samples"].append({
                        "score": round(score, 4),
                        "target_timeblock_id": target["block"].get("ID", ""),
                        "target_tm": target["block"].get("TM", ""),
                        "target_text": str(target.get("text", ""))[:180],
                        "source_start_tm": interval.get("start_tm", ""),
                        "source_end_tm": interval.get("end_tm", ""),
                        "source_anchor_span": interval.get("anchor_span", ""),
                        "source_summary": str(interval.get("summary", ""))[:180],
                        "relation": decision.relation,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                    })
                continue
            target_key = (str(target["doc_id"]), str(target["block_id"]))
            rank_score = score + decision.confidence
            old = best_by_target.get(target_key)
            if old is None or rank_score > old[0]:
                best_by_target[target_key] = (rank_score, target, interval, decision)

    for (_target_doc, target_id), (rank_score, target, interval, decision) in best_by_target.items():
        block = target["block"]
        evidence = {
            "source_doc_id": interval.get("source_doc_id", ""),
            "start_source_timeblock_id": interval.get("start_block_id", ""),
            "end_source_timeblock_id": interval.get("end_block_id", ""),
            "start_tm": interval.get("start_tm", ""),
            "end_tm": interval.get("end_tm", ""),
            "anchor_span": interval.get("anchor_span", ""),
            "target_doc_id": target["doc_id"],
            "target_timeblock_id": target_id,
            "target_tm_before": block.get("TM", ""),
            "method": "generic_timeline_interval_evidence",
            "relation": decision.relation,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "retrieval_plus_confidence": round(rank_score, 4),
            "source_summary": interval.get("summary", ""),
        }
        block["crossdoc_interval_evidence"] = evidence
        block.pop("iso", None)
        block.pop("iso_range", None)
        matches.append(evidence)
        if len(stats["accepted_samples"]) < 20:
            stats["accepted_samples"].append(evidence)
    stats["accepted"] = len(matches)
    return matches, stats


def compact_episode_text(text: str, limit: int = 220) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())[:limit]


def quote_norm(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text or ""))


def quote_support_score(quote: str, text: str) -> float:
    q = quote_norm(quote)
    t = quote_norm(text)
    if not q or not t:
        return 0.0
    if len(q) >= 4 and q in t:
        return 1.0
    q_bigrams = {q[i : i + 2] for i in range(max(0, len(q) - 1))}
    t_bigrams = {t[i : i + 2] for i in range(max(0, len(t) - 1))}
    if not q_bigrams or not t_bigrams:
        return 0.0
    overlap = len(q_bigrams & t_bigrams)
    return overlap / max(1, len(q_bigrams))


def best_supported_quote(candidate_quote: str, support_text: str, min_chars: int = 24, max_chars: int = 110) -> Tuple[str, float]:
    """Return a real contiguous support-text span closest to a model quote.

    The LLM sometimes returns a faithful paraphrase instead of an exact source
    substring. For recall-first runs we repair only the quote string, never the
    anchors or relation, and keep the repaired span traceable to the input text.
    """
    support = re.sub(r"\s+", "", str(support_text or ""))
    if not support:
        return str(candidate_quote or ""), 0.0
    quote = re.sub(r"\s+", "", str(candidate_quote or ""))
    if quote and quote in support:
        return quote, 1.0

    q_norm = quote_norm(quote)
    if len(q_norm) < 6:
        return quote, quote_support_score(quote, support)

    lengths = sorted({max(min_chars, min(max_chars, len(q_norm) + delta)) for delta in (-20, 0, 20, 40)})
    best_quote = quote
    best_score = quote_support_score(quote, support)
    for length in lengths:
        if length <= 0:
            continue
        step = max(8, length // 3)
        if len(support) <= length:
            spans = [support]
        else:
            spans = [support[start : start + length] for start in range(0, len(support) - length + 1, step)]
            if support[-length:] not in spans:
                spans.append(support[-length:])
        for span in spans:
            score = quote_support_score(quote, span)
            if score > best_score:
                best_quote = span
                best_score = score
    return best_quote, best_score


def maybe_repair_quote(
    quote: str,
    support_text: str,
    current_score: float,
    *,
    min_score: float = 0.42,
) -> Tuple[str, float, bool]:
    if current_score >= MIN_QUOTE_SUPPORT_SCORE:
        return quote, current_score, False
    repaired, repaired_score = best_supported_quote(quote, support_text)
    if repaired_score >= max(min_score, current_score + 0.08):
        return repaired, repaired_score, True
    return quote, current_score, False


def quote_in_text(quote: str, text: str) -> bool:
    q = quote_norm(quote)
    if len(q) < 4:
        return False
    return quote_support_score(quote, text) >= MIN_QUOTE_SUPPORT_SCORE


def quote_in_any_source_item(quote: str, items: List[Dict[str, Any]]) -> bool:
    return any(quote_in_text(quote, str(item.get("text", ""))) for item in items)


STRONG_EVENT_TERMS = {
    "攻", "围", "败", "破", "杀", "降", "追", "立", "封", "迁", "奔", "入关",
    "渡河", "驻军", "救", "归附", "背叛", "投降", "进军", "出兵", "战败",
    "平定", "夺取", "攻打", "坑杀", "逃走", "会合", "防守",
}
GENERIC_SPEECH_RE = re.compile(r"(?:说|曰|问|答|告|历数|谢罪|劝告)[：:，,。]?$")


def quote_cjk_len(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", str(text or "")))


def contains_strong_event_term(text: str) -> bool:
    value = str(text or "")
    return any(term in value for term in STRONG_EVENT_TERMS)


def generic_speech_intro(text: str) -> bool:
    value = re.sub(r"\s+", "", str(text or ""))
    if not value:
        return True
    if GENERIC_SPEECH_RE.search(value):
        return True
    return quote_cjk_len(value) < 18 and any(term in value for term in ("说", "曰", "问", "答", "告", "谢罪", "历数"))


def named_overlap_score(left: str, right: str) -> float:
    terms = set()
    for text in (left, right):
        for token in re.findall(r"[\u4e00-\u9fff]{2,4}", str(text or "")):
            if token in {"于是", "这个", "那个", "时候", "将军", "大王", "军队", "百姓", "天下", "诸侯"}:
                continue
            terms.add(token)
    left_terms = {term for term in terms if term in str(left or "")}
    right_terms = {term for term in terms if term in str(right or "")}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


def phase_evidence_quality(item: EpisodeContextItem, source_quote: str, target_quote: str) -> Tuple[bool, List[str], float, Dict[str, Any]]:
    reasons: List[str] = []
    source_len = quote_cjk_len(source_quote)
    target_len = quote_cjk_len(target_quote)
    source_has_event = contains_strong_event_term(source_quote)
    target_has_event = contains_strong_event_term(target_quote)
    overlap = named_overlap_score(source_quote, target_quote)

    if source_len < 18:
        reasons.append("thin_source_quote_evidence")
    if target_len < 18:
        reasons.append("thin_target_quote_evidence")
    if generic_speech_intro(source_quote):
        reasons.append("generic_source_speech_intro_quote")
    if generic_speech_intro(target_quote):
        reasons.append("generic_target_speech_intro_quote")
    if not source_has_event:
        reasons.append("weak_source_event_action")
    if not target_has_event:
        reasons.append("weak_target_event_action")

    if item.relation == "same_sequence_phase":
        if overlap < 0.12:
            reasons.append("weak_event_overlap")
        if ("同一战争进程" in item.reason or "同一历史阶段" in item.reason) and not any(
            term in item.reason for term in ("成皋", "外黄", "垓下", "章邯", "鸿门", "入关", "追击", "东征", "荥阳")
        ):
            reasons.append("same_sequence_phase_too_broad")

    score = 0.0
    score += min(0.20, source_len / 120)
    score += min(0.20, target_len / 120)
    score += 0.18 if source_has_event else 0.0
    score += 0.18 if target_has_event else 0.0
    score += min(0.20, overlap * 0.35)
    score += 0.04 if item.relation == "episode_context" else 0.0
    quality = {
        "score": round(score, 4),
        "source_quote_cjk_len": source_len,
        "target_quote_cjk_len": target_len,
        "source_has_event_action": source_has_event,
        "target_has_event_action": target_has_event,
        "event_overlap_score": round(overlap, 4),
        "reasons": reasons,
    }
    return not reasons, reasons, score, quality


def source_support_text_for_context(item: EpisodeContextItem, source_by_short: Dict[str, Dict[str, Any]], source_items: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for raw_id in item.supporting_source_timeblock_ids or []:
        source_item = source_by_short.get(short_id(str(raw_id or "")))
        if source_item:
            chunks.append(str(source_item.get("text", "") or ""))
    if not chunks:
        chunks = [str(source_item.get("text", "") or "") for source_item in source_items]
    return "".join(chunks)


def episode_neighbor_text(items: List[Dict[str, Any]], index: int, radius: int = 1) -> str:
    chunks: List[str] = []
    start = max(0, index - radius)
    end = min(len(items), index + radius + 1)
    for item in items[start:end]:
        text = str(item.get("text", "") or "").strip()
        if text:
            chunks.append(text)
    return compact_episode_text("".join(chunks), 260)


def anchor_context(items: List[Dict[str, Any]], index: int, direction: int) -> Dict[str, Any]:
    pos = index + direction
    while 0 <= pos < len(items):
        block = items[pos]["block"]
        if is_concrete_anchor(block):
            return {
                "id": short_id(str(block.get("ID", ""))),
                "tm": str(block.get("TM", "") or ""),
                "text": compact_episode_text(str(items[pos].get("text", "") or ""), 140),
            }
        pos += direction
    return {}


def episode_alignment_prompt(
    source_doc: str,
    target_doc: str,
    source_items: List[Dict[str, Any]],
    target_items: List[Dict[str, Any]],
    target_context_items: Optional[List[Dict[str, Any]]] = None,
    candidate_focus: Optional[Dict[str, Any]] = None,
    require_diagnostic: bool = False,
) -> str:
    target_context_items = target_context_items or target_items
    target_context_pos = {
        str(item.get("block_id", "") or ""): idx
        for idx, item in enumerate(target_context_items)
    }
    source_rows = []
    for idx, item in enumerate(source_items):
        block = item["block"]
        source_rows.append({
            "id": short_id(str(block.get("ID", ""))),
            "tm": str(block.get("TM", "") or ""),
            "is_anchor": is_concrete_anchor(block),
            "text": compact_episode_text(str(item.get("text", ""))),
            "neighbor_text": episode_neighbor_text(source_items, idx),
            "prev_anchor": anchor_context(source_items, idx, -1),
            "next_anchor": anchor_context(source_items, idx, 1),
        })
    target_rows = []
    for item in target_items:
        block = item["block"]
        context_idx = target_context_pos.get(str(item.get("block_id", "") or ""), 0)
        target_rows.append({
            "id": short_id(str(block.get("ID", ""))),
            "tm": str(block.get("TM", "") or ""),
            "text": compact_episode_text(str(item.get("text", ""))),
            "neighbor_text": episode_neighbor_text(target_context_items, context_idx),
            "prev_anchor": anchor_context(target_context_items, context_idx, -1),
            "next_anchor": anchor_context(target_context_items, context_idx, 1),
        })
    diagnostic_instruction = ""
    if require_diagnostic:
        diagnostic_instruction = """
- 本次是 candidate_focus 诊断重试：禁止返回空 contexts。
- 如果不能建立可接受的 episode_context，请为 candidate_focus.target_focus.nonanchor_ids 中最相关的一个 target 输出 relation=weak_context 或 unrelated，confidence 不超过 0.65，并在 reason 中说明失败原因。
- 诊断项仍必须使用真实 target_timeblock_id；如果 source anchor 不足以支持，请留空或使用你检查过的推荐 anchor，并在 reason 中说明不支持。
""".strip()
    return f"""
你要做跨文本叙事 episode 对齐，用于给 target TimeBlock 提供 source 文档时间轴上下文。

输入只包含当前 case 的 source/target 文本，不要使用外部知识，不要参考标准答案，不要输出 ISO 或最终时间范围。

输出 JSON：
{{
  "contexts": [
    {{
      "target_timeblock_id": "target id",
      "episode_label": "简短 episode 名称",
      "relation": "episode_context|same_sequence_phase|weak_context|unrelated",
      "source_anchor_before_timeblock_id": "source 中真实存在且 is_anchor=true 的 id",
      "source_anchor_after_timeblock_id": "source 中真实存在且 is_anchor=true 的 id；如果没有后锚点可空",
      "confidence": 0.0,
      "supporting_source_timeblock_ids": ["source ids"],
      "supporting_source_quote": "source_blocks 中支持该 episode 对齐的连续原文片段",
      "supporting_target_quote": "target_blocks 中支持该 episode 对齐的连续原文片段",
      "reason": "简短理由"
    }}
  ]
}}

规则：
- 这是 episode/time-context 对齐，不是 same-event 判断。
- 请逐一检查 target_blocks 中的每个 target；能建立 episode context 的输出 context，不能建立的可以输出 weak_context/unrelated，也可以省略。
- 只有当 target 的事件/阶段明显落在 source 文档两个时间锚点之间，或从 source 某个锚点开始的一段叙事中，才输出 context。
- source_anchor_before_timeblock_id 和 source_anchor_after_timeblock_id 必须从 source_blocks 里选择，不能编造。
- 只允许选择 is_anchor=true 的 source block 作为 before/after anchor。
- 如果只是共享人物、朝代、地点，relation=weak_context 或 unrelated，confidence 不得高于 0.65。
- 如果 target 是 source 一段连续叙事中的相邻事件、阶段摘要或同一政治/军事进程，可用 episode_context 或 same_sequence_phase。
- target_blocks/source_blocks 的 prev_anchor、next_anchor、neighbor_text 是局部叙事约束；不要只因为共享人物或“起事/入关/为王”等关键词就跳到远处锚点。
- 如果 target 的局部前后锚点显示它处在某段叙事之前/之后，选择的 source before/after anchor 必须与该局部顺序兼容；不兼容时输出 weak_context 或省略。
- supporting_source_quote 必须是 source_blocks.text 中出现的连续片段，不要使用省略号、概括改写或拼接不连续片段。
- supporting_target_quote 必须是对应 target block text 中出现的连续片段，不要使用省略号、概括改写或拼接不连续片段。
- 如果找不到可引用的原文片段，不要输出该 context。
- 不要输出 starttoend，不要输出 ISO。
- candidate_focus 是检索层给出的只读候选焦点；它不是答案。你可以优先检查其中推荐的 source anchors 和 target seed，但仍必须遵守上面的 quote 和 is_anchor 规则。
- 如果 candidate_focus.target_focus.nonanchor_ids 非空，优先判断这些 target non-anchor 是否落在 candidate_focus.source_focus 推荐的 source anchor 之后或之间。
- 如果 candidate_focus 非空且你认为没有可接受的 episode_context，请至少为 candidate_focus.target_focus.nonanchor_ids 中最相关的一个 target 输出 weak_context 或 unrelated，并写明 reason。这个诊断项不会被 accepted，但能帮助定位候选包为什么失败。
{diagnostic_instruction}

candidate_focus={json.dumps(candidate_focus or {}, ensure_ascii=False)}

source_doc={doc_code(source_doc)}
source_blocks={json.dumps(source_rows, ensure_ascii=False)}

target_doc={doc_code(target_doc)}
target_blocks={json.dumps(target_rows, ensure_ascii=False)}
""".strip()


def focus_wants_diagnostic(candidate_focus: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(candidate_focus, dict):
        return False
    target_focus = candidate_focus.get("target_focus", {})
    return isinstance(target_focus, dict) and bool(target_focus.get("nonanchor_ids"))


def call_episode_context_alignment(
    source_doc: str,
    target_doc: str,
    source_items: List[Dict[str, Any]],
    target_items: List[Dict[str, Any]],
    target_context_items: Optional[List[Dict[str, Any]]] = None,
    candidate_focus: Optional[Dict[str, Any]] = None,
) -> Tuple[EpisodeContextResponse, Optional[Dict[str, Any]]]:
    if not source_items or not target_items:
        return EpisodeContextResponse(), None
    prompt = episode_alignment_prompt(source_doc, target_doc, source_items, target_items, target_context_items, candidate_focus)
    messages = [
        {"role": "system", "content": "你是严谨的跨文本历史叙事 episode 对齐器，只输出 JSON。"},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            response = create_chat_completion(
                CLIENT,
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1800,
            )
            content = response.choices[0].message.content or "{}"
            parsed = normalize_episode_context_response(validate_json_text(EpisodeContextResponse, content))
            if parsed.contexts or not focus_wants_diagnostic(candidate_focus):
                return parsed, None
            diagnostic_prompt = episode_alignment_prompt(
                source_doc,
                target_doc,
                source_items,
                target_items,
                target_context_items,
                candidate_focus,
                require_diagnostic=True,
            )
            diagnostic_response = create_chat_completion(
                CLIENT,
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是严谨的跨文本历史叙事 episode 对齐诊断器，只输出 JSON。"},
                    {"role": "user", "content": diagnostic_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1800,
            )
            diagnostic_content = diagnostic_response.choices[0].message.content or "{}"
            return normalize_episode_context_response(validate_json_text(EpisodeContextResponse, diagnostic_content)), None
        except Exception as exc:
            last_error = exc
            messages = [
                {"role": "system", "content": "你必须只输出一个完整 JSON 对象，不能截断，不能省略逗号，不能输出 Markdown。"},
                {
                    "role": "user",
                    "content": prompt
                    + "\n\n上一次输出不是合法 JSON。请重新输出严格 JSON："
                    + '{"contexts":[{"target_timeblock_id":"","episode_label":"","relation":"weak_context","source_anchor_before_timeblock_id":"","source_anchor_after_timeblock_id":"","confidence":0,"supporting_source_timeblock_ids":[],"supporting_source_quote":"","supporting_target_quote":"","reason":""}]}',
                },
            ]
    error = {
        "type": type(last_error).__name__ if last_error else "Unknown",
        "message": str(last_error)[:500] if last_error else "",
    }
    print(f"Episode context alignment failed after retries: {error['type']}: {error['message'][:240]}")
    return EpisodeContextResponse(), error


def apply_episode_context_alignment(
    case_id: str,
    left_doc: str,
    right_doc: str,
    case_scope: Optional[Dict[str, Any]],
    timeblocks: Dict[str, List[Dict[str, Any]]],
    block_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    stats = {
        "case_id": case_id,
        "doc_pair": [doc_code(left_doc), doc_code(right_doc)],
        "calls": 0,
        "accepted": 0,
        "quality_warning_accepts": 0,
        "rejected": 0,
        "invalid_anchor": 0,
        "failed_calls": 0,
        "api_errors": [],
        "accepted_samples": [],
        "rejected_samples": [],
    }
    for source_doc, target_doc in [(left_doc, right_doc), (right_doc, left_doc)]:
        allowed = allowed_short_ids(case_scope, target_doc)
        direction_focus = candidate_focus_for_direction(case_scope, source_doc, target_doc)
        source_items = ordered_doc_items(source_doc, timeblocks, block_index)
        target_context_items = ordered_doc_items(target_doc, timeblocks, block_index)
        target_items = [
            block_index[(target_doc, str(block.get("ID", "") or ""))]
            for block in timeblocks[target_doc]
            if (target_doc, str(block.get("ID", "") or "")) in block_index
            and not is_concrete_anchor(block)
            and (allowed is None or short_id(str(block.get("ID", "") or "")) in allowed)
        ]
        if not source_items or not target_items:
            continue
        source_by_short = {short_id(str(item["block"].get("ID", ""))): item for item in source_items}
        target_by_short = {short_id(str(item["block"].get("ID", ""))): item for item in target_items}
        stats["calls"] += 1
        response, call_error = call_episode_context_alignment(source_doc, target_doc, source_items, target_items, target_context_items, direction_focus)
        if call_error:
            stats["failed_calls"] += 1
            if len(stats["api_errors"]) < 10:
                stats["api_errors"].append({
                    "source_doc": doc_code(source_doc),
                    "target_doc": doc_code(target_doc),
                    **call_error,
                })
            continue
        for item in response.contexts:
            target_short = short_id(item.target_timeblock_id)
            target_item = target_by_short.get(target_short)
            before_item = source_by_short.get(short_id(item.source_anchor_before_timeblock_id))
            after_item = source_by_short.get(short_id(item.source_anchor_after_timeblock_id)) if item.source_anchor_after_timeblock_id else None
            source_support_text = source_support_text_for_context(item, source_by_short, source_items)
            source_quote_score = quote_support_score(item.supporting_source_quote, source_support_text)
            target_quote_score = quote_support_score(item.supporting_target_quote, str(target_item.get("text", "")) if target_item else "")
            source_quote_repaired = False
            target_quote_repaired = False
            source_quote = item.supporting_source_quote
            target_quote = item.supporting_target_quote
            strong_relation_for_repair = item.relation in {"episode_context", "same_sequence_phase"} and item.confidence >= MIN_EPISODE_CONFIDENCE
            weak_relation_for_repair = (
                RECALL_ACCEPT_WEAK_CONTEXT
                and item.relation == "weak_context"
                and item.confidence >= MIN_WEAK_CONTEXT_CONFIDENCE
            )
            if strong_relation_for_repair or weak_relation_for_repair:
                source_quote, source_quote_score, source_quote_repaired = maybe_repair_quote(
                    item.supporting_source_quote,
                    source_support_text,
                    source_quote_score,
                )
                target_quote, target_quote_score, target_quote_repaired = maybe_repair_quote(
                    item.supporting_target_quote,
                    str(target_item.get("text", "")) if target_item else "",
                    target_quote_score,
                )
            source_quote_ok = source_quote_score >= MIN_QUOTE_SUPPORT_SCORE
            target_quote_ok = target_quote_score >= MIN_QUOTE_SUPPORT_SCORE
            quality_ok, quality_reasons, quality_score, quality_detail = phase_evidence_quality(
                item,
                source_quote,
                target_quote,
            )
            strong_relation = item.relation in {"episode_context", "same_sequence_phase"}
            weak_relation_recalled = (
                RECALL_ACCEPT_WEAK_CONTEXT
                and item.relation == "weak_context"
                and item.confidence >= MIN_WEAK_CONTEXT_CONFIDENCE
            )
            accepted = (
                target_item is not None
                and before_item is not None
                and is_concrete_anchor(before_item["block"])
                and (after_item is None or is_concrete_anchor(after_item["block"]))
                and (
                    (strong_relation and item.confidence >= MIN_EPISODE_CONFIDENCE)
                    or weak_relation_recalled
                )
                and source_quote_ok
                and target_quote_ok
                and (quality_ok or QUALITY_GATE_MODE != "strict")
            )
            if not accepted:
                reject_reasons: List[str] = []
                if target_item is None:
                    reject_reasons.append("missing_target")
                if before_item is None:
                    reject_reasons.append("missing_before_anchor")
                elif not is_concrete_anchor(before_item["block"]):
                    reject_reasons.append("before_anchor_not_concrete")
                if item.source_anchor_after_timeblock_id and after_item is None:
                    reject_reasons.append("missing_after_anchor")
                elif after_item is not None and not is_concrete_anchor(after_item["block"]):
                    reject_reasons.append("after_anchor_not_concrete")
                if not strong_relation and not weak_relation_recalled:
                    reject_reasons.append("unsupported_relation")
                if strong_relation and item.confidence < MIN_EPISODE_CONFIDENCE:
                    reject_reasons.append("low_confidence")
                if item.relation == "weak_context" and RECALL_ACCEPT_WEAK_CONTEXT and item.confidence < MIN_WEAK_CONTEXT_CONFIDENCE:
                    reject_reasons.append("low_weak_context_confidence")
                if not source_quote_ok:
                    reject_reasons.append("source_quote_failed")
                if not target_quote_ok:
                    reject_reasons.append("target_quote_failed")
                reject_reasons.extend(reason for reason in quality_reasons if reason not in reject_reasons)
                if before_item is None or (after_item is None and item.source_anchor_after_timeblock_id) or "before_anchor_not_concrete" in reject_reasons or "after_anchor_not_concrete" in reject_reasons:
                    stats["invalid_anchor"] += 1
                else:
                    stats["rejected"] += 1
                if len(stats["rejected_samples"]) < 20:
                    rejected = item.model_dump()
                    rejected["reject_reasons"] = reject_reasons or ["unknown"]
                    rejected["candidate_focus"] = direction_focus
                    rejected["source_quote_ok"] = source_quote_ok
                    rejected["target_quote_ok"] = target_quote_ok
                    rejected["source_quote_repaired"] = source_quote_repaired
                    rejected["target_quote_repaired"] = target_quote_repaired
                    rejected["repaired_source_quote"] = source_quote if source_quote_repaired else ""
                    rejected["repaired_target_quote"] = target_quote if target_quote_repaired else ""
                    rejected["source_quote_score"] = round(source_quote_score, 4)
                    rejected["target_quote_score"] = round(target_quote_score, 4)
                    rejected["phase_evidence_quality"] = quality_detail
                    stats["rejected_samples"].append(rejected)
                continue
            target_block = target_item["block"]
            before_block = before_item["block"]
            after_block = after_item["block"] if after_item else {}
            evidence = {
                "source_doc_id": source_doc,
                "target_doc_id": target_doc,
                "target_timeblock_id": target_block.get("ID", ""),
                "target_tm_before": target_block.get("TM", ""),
                "episode_label": item.episode_label,
                "relation": item.relation,
                "confidence": item.confidence,
                "source_anchor_before_timeblock_id": before_block.get("ID", ""),
                "source_anchor_before_tm": before_block.get("TM", ""),
                "source_anchor_before_granularity": str(before_block.get("Granularity", "") or "1"),
                "source_anchor_after_timeblock_id": after_block.get("ID", "") if after_block else "",
                "source_anchor_after_tm": after_block.get("TM", "") if after_block else "",
                "source_anchor_after_granularity": str(after_block.get("Granularity", "") or "1") if after_block else "",
                "supporting_source_timeblock_ids": item.supporting_source_timeblock_ids,
                "supporting_source_quote": source_quote,
                "supporting_target_quote": target_quote,
                "quote_verified": True,
                "source_quote_repaired": source_quote_repaired,
                "target_quote_repaired": target_quote_repaired,
                "source_quote_score": round(source_quote_score, 4),
                "target_quote_score": round(target_quote_score, 4),
                "phase_evidence_quality": quality_detail,
                "quality_gate_mode": QUALITY_GATE_MODE,
                "quality_warnings": quality_reasons,
                "recall_relation_relaxed": weak_relation_recalled,
                "method": "crossdoc_episode_context_alignment",
                "reason": item.reason,
            }
            old = target_block.get("crossdoc_context_evidence")
            if not isinstance(old, dict) or float(old.get("confidence", 0) or 0) < item.confidence:
                target_block["crossdoc_context_evidence"] = evidence
                target_block.pop("iso_range", None)
                contexts.append(evidence)
            stats["accepted"] += 1
            if quality_reasons:
                stats["quality_warning_accepts"] += 1
            if len(stats["accepted_samples"]) < 20:
                stats["accepted_samples"].append(evidence)
    return contexts, stats


def clear_stale_crossdoc_fields(timeblocks: Dict[str, List[Dict[str, Any]]]) -> int:
    cleared = 0
    for blocks in timeblocks.values():
        for block in blocks:
            if not isinstance(block, dict):
                continue
            had_crossdoc = False
            for key in CROSSDOC_FIELD_KEYS:
                if key in block:
                    block.pop(key, None)
                    had_crossdoc = True
            if had_crossdoc:
                block.pop("iso", None)
                block.pop("iso_range", None)
                cleared += 1
    return cleared


def count_existing_crossdoc_fields(timeblocks: Dict[str, List[Dict[str, Any]]]) -> int:
    return sum(
        1
        for blocks in timeblocks.values()
        for block in blocks
        if isinstance(block, dict) and any(key in block for key in CROSSDOC_FIELD_KEYS)
    )


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def enrich_signatures(block_index: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    items = [
        item for item in block_index.values()
        if str(item.get("text", "")).strip()
    ]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(
                call_event_signature,
                str(item["text"]),
                str(item["block"].get("TM", "") or ""),
            ): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                item["signature"] = future.result()
            except Exception:
                item["signature"] = fallback_signature(str(item.get("text", "")))


def source_event_items_for_block(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    block = item["block"]
    sentence_rows = item.get("sentences") if isinstance(item.get("sentences"), list) else []
    if len(sentence_rows) <= 1:
        return [item]

    source_items = []
    for row in sentence_rows:
        number = str(row.get("number", "") or "")
        text = str(row.get("sentence", "") or "").strip()
        if not number or not text:
            continue
        source_items.append({
            "doc_id": item["doc_id"],
            "block": block,
            "block_id": f"{item['block_id']}::{number}",
            "source_timeblock_id": item["block_id"],
            "text": text,
            "sentences": [row],
            "sentence_numbers": [number],
            "signature": None,
        })
    return source_items or [item]


def ensure_item_signature(item: Dict[str, Any]) -> EventSignature:
    sig = item.get("signature")
    if isinstance(sig, EventSignature) and signature_is_informative(sig):
        return sig
    sig = call_event_signature(str(item.get("text", "")), str(item["block"].get("TM", "") or ""))
    item["signature"] = sig
    return sig


def apply_crossdoc_prealign(
    case_id: str,
    left_doc: str,
    right_doc: str,
    case_scope: Optional[Dict[str, Any]],
    timeblocks: Dict[str, List[Dict[str, Any]]],
    block_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    verification_jobs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    jobs_by_target: Dict[Tuple[str, str], List[Tuple[float, Dict[str, Any], Dict[str, Any], str]]] = {}
    stats: Dict[str, Any] = {
        "case_id": case_id,
        "doc_pair": [doc_code(left_doc), doc_code(right_doc)],
        "source_event_items": 0,
        "target_items": 0,
        "retrieval_pairs_scored": 0,
        "verification_jobs": 0,
        "verification_jobs_before_limit": 0,
        "max_verification_jobs": MAX_VERIFICATION_JOBS,
        "accepted": 0,
        "accepted_by_lexical_similarity": 0,
        "time_evidence": 0,
        "rejected_by_relation": {},
        "top_candidates": [],
        "rejected_samples": [],
    }

    for source_doc, target_doc in [(left_doc, right_doc), (right_doc, left_doc)]:
        allowed = allowed_short_ids(case_scope, target_doc)
        source_items: List[Dict[str, Any]] = []
        for block in timeblocks[source_doc]:
            key = (source_doc, str(block.get("ID", "") or ""))
            if key not in block_index or not is_concrete_anchor(block):
                continue
            source_items.extend(source_event_items_for_block(block_index[key]))
        target_items = [
            block_index[(target_doc, str(block.get("ID", "") or ""))]
            for block in timeblocks[target_doc]
            if (target_doc, str(block.get("ID", "") or "")) in block_index
            and not is_concrete_anchor(block)
            and (allowed is None or short_id(str(block.get("ID", "") or "")) in allowed)
        ]
        stats["source_event_items"] += len(source_items)
        stats["target_items"] += len(target_items)

        for source_item in source_items:
            source_sig = ensure_item_signature(source_item)
            for target_item in target_items:
                target_sig = ensure_item_signature(target_item)
                score = retrieval_score(
                    str(source_item["text"]),
                    source_sig,
                    str(target_item["text"]),
                    target_sig,
                )
                bucket_scores = retrieval_bucket_scores(
                    str(source_item["text"]),
                    source_sig,
                    str(target_item["text"]),
                    target_sig,
                )
                stats["retrieval_pairs_scored"] += 1
                if score >= MIN_RETRIEVAL_SCORE or bucket_scores:
                    target_key = (str(target_item["doc_id"]), str(target_item["block_id"]))
                    if score >= MIN_RETRIEVAL_SCORE:
                        jobs_by_target.setdefault(target_key, []).append((score, source_item, target_item, "overall"))
                    for bucket, bucket_score in bucket_scores.items():
                        combined_score = max(score, MIN_RETRIEVAL_SCORE) + 0.03 * min(bucket_score, 1.0)
                        jobs_by_target.setdefault(target_key, []).append((combined_score, source_item, target_item, bucket))
                    if len(stats["top_candidates"]) < 30:
                        source_features = event_feature_terms(str(source_item["text"]), source_sig)
                        target_features = event_feature_terms(str(target_item["text"]), target_sig)
                        stats["top_candidates"].append({
                            "score": round(score, 4),
                            "retrieval_buckets": sorted(bucket_scores),
                            "raw_text_similarity": round(raw_text_similarity(str(source_item["text"]), str(target_item["text"])), 4),
                            "feature_overlap": round(jaccard(source_features, target_features), 4),
                            "shared_features": sorted(source_features & target_features)[:12],
                            "source_doc_id": source_item["doc_id"],
                            "source_timeblock_id": source_item.get("source_timeblock_id", source_item["block"].get("ID", "")),
                            "source_event_id": source_item["block_id"],
                            "source_tm": source_item["block"].get("TM", ""),
                            "source_text": str(source_item["text"])[:160],
                            "target_doc_id": target_item["doc_id"],
                            "target_timeblock_id": target_item["block"].get("ID", ""),
                            "target_tm": target_item["block"].get("TM", ""),
                            "target_text": str(target_item["text"])[:160],
                        })
    stats["verification_jobs_before_limit"] = len(verification_jobs)
    per_target_jobs: List[List[Tuple[float, Dict[str, Any], Dict[str, Any]]]] = []
    for items in jobs_by_target.values():
        selected: List[Tuple[float, Dict[str, Any], Dict[str, Any], str]] = []
        seen_source_events: Set[str] = set()
        for bucket in ("raw_text", "action", "location", "event_type", "object", "semantic", "participant", "overall"):
            bucket_items = sorted((item for item in items if item[3] == bucket), key=lambda item: item[0], reverse=True)
            for item in bucket_items[:2]:
                source_event_id = str(item[1].get("block_id", ""))
                if source_event_id in seen_source_events:
                    continue
                selected.append(item)
                seen_source_events.add(source_event_id)
        if len(selected) < TOP_K:
            for item in sorted(items, key=lambda item: item[0], reverse=True):
                source_event_id = str(item[1].get("block_id", ""))
                if source_event_id in seen_source_events:
                    continue
                selected.append(item)
                seen_source_events.add(source_event_id)
                if len(selected) >= TOP_K:
                    break
        per_target_jobs.append([(score, source, target) for score, source, target, _bucket in selected[:TOP_K]])
    stats["verification_jobs_before_limit"] = sum(len(items) for items in per_target_jobs)
    verification_jobs = []
    for rank in range(TOP_K):
        round_items = [items[rank] for items in per_target_jobs if rank < len(items)]
        round_items.sort(key=lambda item: item[0], reverse=True)
        for item in round_items:
            if len(verification_jobs) < MAX_VERIFICATION_JOBS:
                verification_jobs.append(item)
    if len(verification_jobs) < MAX_VERIFICATION_JOBS:
        seen = {(str(source["block_id"]), str(target["block_id"])) for _score, source, target in verification_jobs}
        leftovers = []
        for items in per_target_jobs:
            for item in items:
                _score, source, target = item
                key = (str(source["block_id"]), str(target["block_id"]))
                if key not in seen:
                    leftovers.append(item)
                    seen.add(key)
        for item in sorted(leftovers, key=lambda item: item[0], reverse=True):
            if len(verification_jobs) >= MAX_VERIFICATION_JOBS:
                break
            verification_jobs.append(item)
    stats["verification_jobs"] = len(verification_jobs)

    best_by_target: Dict[Tuple[str, str], Tuple[float, Dict[str, Any], Dict[str, Any], EventEquivalence]] = {}
    for score, source_item, target_item in verification_jobs:
        if not is_lexical_same_event(str(source_item["text"]), str(target_item["text"])):
            continue
        target_key = (str(target_item["doc_id"]), str(target_item["block_id"]))
        decision = EventEquivalence(
            relation="same_atomic_event",
            same_event=True,
            transferable_anchor=True,
            confidence=raw_text_similarity(str(source_item["text"]), str(target_item["text"])),
            reason="accepted by generic raw-text similarity before LLM equivalence",
        )
        old = best_by_target.get(target_key)
        rank_score = score + decision.confidence
        if old is None or rank_score > old[0]:
            best_by_target[target_key] = (rank_score, source_item, target_item, decision)
            stats["accepted_by_lexical_similarity"] += 1

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {}
        for score, source_item, target_item in verification_jobs:
            if is_lexical_same_event(str(source_item["text"]), str(target_item["text"])):
                continue
            futures[
                executor.submit(
                    call_event_equivalence,
                    str(source_item["text"]),
                    str(source_item["block"].get("TM", "") or ""),
                    source_item["signature"],
                    str(target_item["text"]),
                    str(target_item["block"].get("TM", "") or ""),
                    target_item["signature"],
                )
            ] = (score, source_item, target_item)

        for future in as_completed(futures):
            score, source_item, target_item = futures[future]
            decision = future.result()
            if not is_transferable_decision(decision):
                stats["rejected_by_relation"][decision.relation] = stats["rejected_by_relation"].get(decision.relation, 0) + 1
                if len(stats["rejected_samples"]) < 30:
                    source_features = event_feature_terms(str(source_item["text"]), source_item["signature"])
                    target_features = event_feature_terms(str(target_item["text"]), target_item["signature"])
                    stats["rejected_samples"].append({
                        "score": round(score, 4),
                        "feature_overlap": round(jaccard(source_features, target_features), 4),
                        "shared_features": sorted(source_features & target_features)[:12],
                        "relation": decision.relation,
                        "same_event": decision.same_event,
                        "transferable_anchor": decision.transferable_anchor,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                        "source_event_id": source_item["block_id"],
                        "source_tm": source_item["block"].get("TM", ""),
                        "source_text": str(source_item["text"])[:160],
                        "target_timeblock_id": target_item["block"].get("ID", ""),
                        "target_text": str(target_item["text"])[:160],
                    })
                continue
            target_key = (str(target_item["doc_id"]), str(target_item["block_id"]))
            old = best_by_target.get(target_key)
            rank_score = score + decision.confidence
            if old is None or rank_score > old[0]:
                best_by_target[target_key] = (rank_score, source_item, target_item, decision)

    for (_target_doc, target_id), (rank_score, source_item, target_item, decision) in best_by_target.items():
        source_block = source_item["block"]
        target_block = target_item["block"]
        if is_concrete_anchor(target_block):
            continue
        before_tm = str(target_block.get("TM", "") or "")
        source_tm = str(source_block.get("TM", "") or "").strip()
        cluster_id = f"{case_id or 'crossdoc'}::{short_id(str(source_block.get('ID', '')))}::{short_id(target_id)}"
        target_block["event_cluster_id"] = cluster_id
        event_link = {
            "source_doc_id": source_item["doc_id"],
            "source_timeblock_id": source_block.get("ID", ""),
            "source_tm": source_tm,
            "source_granularity": str(source_block.get("Granularity", "") or "2"),
            "target_doc_id": target_item["doc_id"],
            "target_timeblock_id": target_id,
            "target_tm_before": before_tm,
            "method": "generic_event_coreference_time_evidence",
            "source_event_id": source_item["block_id"],
            "retrieval_plus_confidence": round(rank_score, 4),
            "relation": decision.relation,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "raw_text_similarity": round(raw_text_similarity(str(source_item["text"]), str(target_item["text"])), 4),
            "shared_features": sorted(
                event_feature_terms(str(source_item["text"]), source_item["signature"])
                & event_feature_terms(str(target_item["text"]), target_item["signature"])
            )[:12],
            "source_event_signature": source_item["signature"].model_dump(),
            "target_event_signature": target_item["signature"].model_dump(),
        }
        target_block["crossdoc_event_links"] = [event_link]
        if is_transferable_decision(decision) and source_tm and decision.confidence >= MIN_ATOMIC_TIME_EVIDENCE_CONFIDENCE:
            target_block["crossdoc_time_evidence"] = {
                "anchor_tm": source_tm,
                "anchor_granularity": str(source_block.get("Granularity", "") or "2"),
                "source_doc_id": source_item["doc_id"],
                "source_timeblock_id": source_block.get("ID", ""),
                "source_event_id": source_item["block_id"],
                "relation": decision.relation,
                "confidence": decision.confidence,
                "method": "crossdoc_event_coreference_anchor",
                "reason": decision.reason,
            }
            target_block.pop("iso", None)
            target_block.pop("iso_range", None)
            stats["time_evidence"] += 1
        matches.append(event_link)
    stats["accepted"] = len(matches)
    stats["top_candidates"] = sorted(stats["top_candidates"], key=lambda row: row["score"], reverse=True)[:30]
    return matches, stats


def run_runtime_scope_strategy() -> bool:
    if os.getenv("AIH_CROSSDOC_SCOPE_INTERNAL", "").strip() == "1":
        return False

    raw_strategy = os.getenv("AIH_CROSSDOC_SCOPE_STRATEGY")
    strategy = (raw_strategy if raw_strategy is not None else "runtime_episode_packet").strip().lower()
    if strategy in LEGACY_SCOPE_STRATEGIES:
        return False
    if strategy not in RUNTIME_SCOPE_STRATEGIES:
        raise ValueError(
            "未知 AIH_CROSSDOC_SCOPE_STRATEGY="
            f"{strategy!r}; use runtime_episode_packet or legacy."
        )

    args = [
        sys.executable,
        "-m",
        "ai_historian.profiles.scalable_fulltext.stages.runtime_crossdoc_scope_runner",
        str(RUN_ROOT),
        "--scope-mode",
        os.getenv("AIH_CROSSDOC_SCOPE_MODE", "episode_packet"),
        "--window-size",
        os.getenv("AIH_CROSSDOC_SCOPE_WINDOW_SIZE", "28"),
        "--overlap",
        os.getenv("AIH_CROSSDOC_SCOPE_OVERLAP", "8"),
        "--context-pad",
        os.getenv("AIH_CROSSDOC_SCOPE_CONTEXT_PAD", "1"),
        "--anchor-search",
        os.getenv("AIH_CROSSDOC_SCOPE_ANCHOR_SEARCH", "16"),
        "--pre-anchor-backfill",
        os.getenv("AIH_CROSSDOC_SCOPE_PRE_ANCHOR_BACKFILL", "1"),
        "--top-k-per-pair",
        os.getenv("AIH_CROSSDOC_SCOPE_TOP_K_PER_PAIR", "30"),
        "--max-cases",
        os.getenv("AIH_CROSSDOC_SCOPE_MAX_CASES", "60"),
        "--min-score",
        os.getenv("AIH_CROSSDOC_SCOPE_MIN_SCORE", "0.00"),
        "--selector",
        os.getenv("AIH_CROSSDOC_SCOPE_SELECTOR", "lexical"),
        "--embedding-batch-size",
        os.getenv("AIH_CROSSDOC_SCOPE_EMBEDDING_BATCH_SIZE", "16"),
        "--embedding-text-chars",
        os.getenv("AIH_CROSSDOC_SCOPE_EMBEDDING_TEXT_CHARS", "2000"),
    ]
    if os.getenv("AIH_CROSSDOC_SCOPE_KEEP_TEMP", "0").strip().lower() in {"1", "true", "yes"}:
        args.append("--keep-temp")
    if os.getenv("AIH_CROSSDOC_SCOPE_FALLBACK_LEXICAL", "0").strip().lower() in {"1", "true", "yes"}:
        args.append("--fallback-lexical-on-embedding-error")
    if os.getenv("AIH_CROSSDOC_SCOPE_RERUN_STEP10_FROM_STEP9", "0").strip().lower() in {"1", "true", "yes"}:
        args.append("--rerun-step10-from-step9")
    if env_flag("AIH_CROSSDOC_CLEAR_EXISTING", "1"):
        args.append("--clear-existing-crossdoc")

    reporter = StepReporter(STEP_LABEL, total=1)
    reporter.start(input_dir=TIMEBLOCK_DIR, output_dir=TIMEBLOCK_DIR)
    reporter.info(
        "runtime_scope "
        f"strategy={strategy} "
        f"mode={os.getenv('AIH_CROSSDOC_SCOPE_MODE', 'episode_packet')} "
        f"selector={os.getenv('AIH_CROSSDOC_SCOPE_SELECTOR', 'lexical')} "
        f"window={os.getenv('AIH_CROSSDOC_SCOPE_WINDOW_SIZE', '28')} "
        f"overlap={os.getenv('AIH_CROSSDOC_SCOPE_OVERLAP', '8')} "
        f"anchor_search={os.getenv('AIH_CROSSDOC_SCOPE_ANCHOR_SEARCH', '16')} "
        f"pre_anchor_backfill={os.getenv('AIH_CROSSDOC_SCOPE_PRE_ANCHOR_BACKFILL', '1')} "
        f"rerun_step10_from_step9={os.getenv('AIH_CROSSDOC_SCOPE_RERUN_STEP10_FROM_STEP9', '0')} "
        f"max_cases={os.getenv('AIH_CROSSDOC_SCOPE_MAX_CASES', '60')} "
        f"clear_existing={os.getenv('AIH_CROSSDOC_CLEAR_EXISTING', '1')}"
    )
    env = os.environ.copy()
    env["AIH_CROSSDOC_SCOPE_INTERNAL"] = "1"
    subprocess.run(args, cwd=PROJECT_ROOT, env=env, check=True)
    reporter.item_ok("crossdoc_prealign", detail="method=runtime_scope")
    reporter.finish()
    return True


def main() -> None:
    global RUN_ROOT, SENTENCE_DIR, SEQUENCE_DIR, TIMEBLOCK_DIR, CLIENT

    RUN_ROOT = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    SENTENCE_DIR = sentence_step_dir(RUN_ROOT, 5)
    SEQUENCE_DIR = sequence_step_dir(RUN_ROOT, 8)
    TIMEBLOCK_DIR = timeblock_step_dir(RUN_ROOT, TIMEBLOCK_STEP)
    setup_step_logging(RUN_ROOT, "step_10b_cross_document_prealign")

    if run_runtime_scope_strategy():
        return

    CLIENT = make_sync_chat_client()

    if not TIMEBLOCK_DIR.exists():
        raise FileNotFoundError(f"找不到 timeblock step10output: {TIMEBLOCK_DIR}")
    if not SENTENCE_DIR.exists():
        raise FileNotFoundError(f"找不到 sentence step5output: {SENTENCE_DIR}")

    reporter = StepReporter(STEP_LABEL, total=1)
    reporter.start(input_dir=TIMEBLOCK_DIR, output_dir=TIMEBLOCK_DIR)

    sentence_files = sorted(path for path in SENTENCE_DIR.glob("*.json") if any(path.stem.endswith(s) for s in SENTENCE_FILE_SUFFIXES))
    timeblock_files = sorted(path for path in TIMEBLOCK_DIR.glob("*.json") if any(path.stem.endswith(s) for s in TIMEBLOCK_FILE_SUFFIXES))
    sequence_files = {path.stem.replace("_sequence", ""): path for path in SEQUENCE_DIR.glob("*_sequence.json")} if SEQUENCE_DIR.exists() else {}

    sentences: Dict[str, List[Dict[str, Any]]] = {doc_key_from_sentence(path): load_json(path) for path in sentence_files}
    timeblock_payloads: Dict[str, Dict[str, Any]] = {doc_key_from_timeblock(path): load_json(path) for path in timeblock_files}
    timeblocks: Dict[str, List[Dict[str, Any]]] = {
        doc_id: payload.get("TMB", []) if isinstance(payload.get("TMB", []), list) else []
        for doc_id, payload in timeblock_payloads.items()
    }
    existing_crossdoc_count = count_existing_crossdoc_fields(timeblocks)
    if (
        existing_crossdoc_count
        and os.getenv("AIH_CROSSDOC_SCOPE_INTERNAL", "").strip() != "1"
        and not env_flag("AIH_ALLOW_LEGACY_CROSSDOC_CLEAR", "0")
    ):
        raise RuntimeError(
            "普通全文 10B 会清除已有 crossdoc evidence；当前检测到 "
            f"{existing_crossdoc_count} 个已有 crossdoc 字段。"
            "请使用默认 runtime scope，或在确认要低召回重算时设置 "
            "AIH_CROSSDOC_SCOPE_STRATEGY=legacy AIH_ALLOW_LEGACY_CROSSDOC_CLEAR=1。"
        )
    stale_crossdoc_cleared = clear_stale_crossdoc_fields(timeblocks)
    all_doc_ids = sorted(set(sentences) & set(timeblocks))
    case_doc_pairs = build_case_doc_pairs(all_doc_ids)

    if not case_doc_pairs:
        report = {
            "schema": "AIH_experiment1_crossdoc_prealign.v2",
            "run_root": str(RUN_ROOT),
            "method": "generic_event_signature_llm_equivalence",
            "split_non_anchor_blocks": 0,
            "sequence_files_updated": 0,
            "matches": [],
            "skipped": "no_crossdoc_doc_pair",
        }
        save_json(RUN_ROOT / "timeblock" / REPORT_NAME, report)
        reporter.item_ok("crossdoc_prealign", detail="skipped=no_crossdoc_doc_pair")
        reporter.finish()
        return

    # V24 keeps split conservative and generic: only long non-anchor blocks with
    # no explicit temporal anchor are split to sentence-grain candidates.
    # This prevents one broad Granularity=0 block from forcing a single range
    # over multiple discourse phases.
    split_total = 0
    sequence_updates = 0
    for doc_id, blocks in list(timeblocks.items()):
        if doc_id in sentences:
            new_blocks, replacements, split_count = split_non_anchor_blocks(blocks, sentences[doc_id])
            timeblocks[doc_id] = new_blocks
            timeblock_payloads[doc_id]["TMB"] = new_blocks
            split_total += split_count
            sequence_updates += update_sequence_file(sequence_files.get(doc_id, Path()), replacements)

    block_index = build_block_index(timeblocks, sentences)
    enrich_signatures(block_index)

    all_matches: List[Dict[str, Any]] = []
    all_intervals: List[Dict[str, Any]] = []
    all_contexts: List[Dict[str, Any]] = []
    pair_stats: List[Dict[str, Any]] = []
    interval_stats: List[Dict[str, Any]] = []
    context_stats: List[Dict[str, Any]] = []
    induced_schemas: List[Dict[str, Any]] = []
    for case_id, left_doc, right_doc, case_scope in case_doc_pairs:
        if left_doc not in timeblocks or right_doc not in timeblocks:
            continue
        induced_schemas.append({
            "case_id": case_id,
            "doc_pair": [doc_code(left_doc), doc_code(right_doc)],
            "schema": {},
            "skipped": "final_context_boundary_only",
        })
        pair_stats.append({
            "case_id": case_id,
            "doc_pair": [doc_code(left_doc), doc_code(right_doc)],
            "skipped": "final_disables_same_event_anchor_transfer",
        })
        interval_stats.append({
            "case_id": case_id,
            "doc_pair": [doc_code(left_doc), doc_code(right_doc)],
            "skipped": "final_disables_direct_interval_range_transfer",
        })
        context_matches, cstats = apply_episode_context_alignment(case_id, left_doc, right_doc, case_scope, timeblocks, block_index)
        all_contexts.extend(context_matches)
        context_stats.append(cstats)

    for path in timeblock_files:
        doc_id = doc_key_from_timeblock(path)
        save_json(path, timeblock_payloads[doc_id])

    report = {
        "schema": "AIH_experiment1_crossdoc_anchor_boundary.final",
        "run_root": str(RUN_ROOT),
        "method": "quote_verified_episode_context_boundary_only",
        "model": MODEL,
        "top_k": TOP_K,
        "max_verification_jobs": MAX_VERIFICATION_JOBS,
        "min_retrieval_score": MIN_RETRIEVAL_SCORE,
        "min_episode_confidence": MIN_EPISODE_CONFIDENCE,
        "min_quote_support_score": MIN_QUOTE_SUPPORT_SCORE,
        "quality_gate_mode": QUALITY_GATE_MODE,
        "recall_accept_weak_context": RECALL_ACCEPT_WEAK_CONTEXT,
        "min_weak_context_confidence": MIN_WEAK_CONTEXT_CONFIDENCE,
        "stale_crossdoc_cleared": stale_crossdoc_cleared,
        "split_non_anchor_blocks": split_total,
        "sequence_files_updated": sequence_updates,
        "induced_schemas": induced_schemas,
        "pair_stats": pair_stats,
        "interval_stats": interval_stats,
        "context_stats": context_stats,
        "matches": all_matches,
        "interval_evidence": all_intervals,
        "context_evidence": all_contexts,
    }
    save_json(RUN_ROOT / "timeblock" / REPORT_NAME, report)
    reporter.item_ok("crossdoc_prealign", detail=f"splits={split_total} matches={len(all_matches)} intervals={len(all_intervals)} contexts={len(all_contexts)} method=final_anchor_boundary")
    reporter.finish()


if __name__ == "__main__":
    main()
