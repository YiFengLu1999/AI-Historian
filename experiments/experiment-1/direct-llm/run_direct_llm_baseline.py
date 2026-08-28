#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from direct_llm_agent_postprocess import postprocess_rows_with_agent_step11

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parents[1]
RULES_TEX = PACKAGE_DIR / "inputs" / "config" / "direct-llm-rules.tex"
CASES_JSON = PACKAGE_DIR / "inputs" / "cases" / "experiment-1-cases.json"

CSV_FIELDS = [
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


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    text: str


@dataclass(frozen=True)
class CasePacket:
    case_id: str
    part_id: str
    source_text: str
    tex_path: Path
    sentences: list[Sentence]


def load_env_defaults() -> None:
    env_file = os.getenv("ENV_FILE")
    candidates = [Path(env_file).expanduser()] if env_file else [REPO_ROOT / ".env"]
    for path in candidates:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clean_latex_text(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = text.replace(r"\textbf", "")
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda m: m.group(1) or "", text)
    text = re.sub(r"[{}]", "", text)
    text = text.replace(r"\%", "%").replace(r"\_", "_")
    return re.sub(r"\s+", " ", text).strip()


def normalize_time_text(text: Any) -> str:
    value = str(text or "").strip()
    value = value.replace(" ", "")
    value = value.replace("，", "").replace(",", "")
    value = value.replace("（", "(").replace("）", ")")
    value = re.sub(r"^(约|大约|大概|约在|大约在|大概在)", "", value)
    value = re.sub(r"(左右|前后|之间|期间)$", "", value)
    value = value.replace("漢", "汉")
    value = value.replace("高帝", "高祖")
    value = value.replace("汉高帝", "汉高祖")
    value = re.sub(r"^汉高祖(?=[元一二三四五六七八九十百零〇0-9])", "汉", value)
    return value.strip()


def read_rules_text() -> str:
    text = RULES_TEX.read_text(encoding="utf-8")
    body = re.sub(r"\\documentclass.*?\\begin\{document\}", "", text, flags=re.S)
    body = re.sub(r"\\end\{document\}.*", "", body, flags=re.S)
    return clean_latex_text(body)


def extract_node_texts(tex: str) -> list[str]:
    pieces = []
    for match in re.finditer(r"\\selectfont(?:\\bfseries)?\s+(.+?)\}\};", tex):
        value = clean_latex_text(match.group(1))
        if value and not value.startswith("用时：") and not value.startswith("人机实验1"):
            pieces.append(value)
    return pieces


def parse_case_tex(path: Path, case_id: str, part_id: str) -> CasePacket:
    tex = path.read_text(encoding="utf-8")
    title_match = re.search(r"\\newcommand\{\\QuestionTitle\}\{([^{}]+)\}", tex)
    source_text = ""
    if title_match:
        title = clean_latex_text(title_match.group(1))
        if "：" in title:
            source_text = title.rsplit("：", 1)[-1].strip()
    sentences: list[Sentence] = []
    current_id = ""
    current_text: list[str] = []
    for piece in extract_node_texts(tex):
        match = re.match(r"^【([^】]+)】\s*(.*)$", piece)
        if match:
            if current_id:
                sentences.append(Sentence(current_id, "".join(current_text).strip()))
            current_id = match.group(1).strip()
            current_text = [match.group(2).strip()]
        elif current_id:
            current_text.append(piece.strip())
    if current_id:
        sentences.append(Sentence(current_id, "".join(current_text).strip()))
    if not sentences:
        raise ValueError(f"No sentences parsed from {path}")
    return CasePacket(case_id=case_id, part_id=part_id, source_text=source_text, tex_path=path, sentences=sentences)


def case_packets(case_ids: list[str]) -> list[CasePacket]:
    payload = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    requested = set(case_ids)
    packets: list[CasePacket] = []
    for case in payload.get("cases", []):
        case_id = str(case.get("case_id", ""))
        if case_id not in requested:
            continue
        grouped: dict[tuple[str, str], list[Sentence]] = {}
        for item in case.get("items", []):
            key = (str(item.get("part_id", "")), str(item.get("source_text", "")))
            grouped.setdefault(key, []).append(
                Sentence(str(item.get("sentence_id", "")), str(item.get("sentence", "")))
            )
        for (part_id, source_text), sentences in grouped.items():
            packets.append(
                CasePacket(
                    case_id=case_id,
                    part_id=part_id,
                    source_text=source_text,
                    tex_path=CASES_JSON,
                    sentences=sentences,
                )
            )
    missing = requested - {packet.case_id for packet in packets}
    if missing:
        raise KeyError(f"Unknown case IDs: {', '.join(sorted(missing))}")
    return packets


def make_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Please install the OpenAI Python SDK before running this baseline.") from exc

    from ai_historian.model_config import resolve_chat_config

    config = resolve_chat_config()
    kwargs: dict[str, Any] = {
        "api_key": config.api_key,
        "base_url": config.base_url,
        "timeout": float(os.getenv("AIH_DIRECT_LLM_TIMEOUT", "180")),
    }
    return OpenAI(**kwargs)


def model_name() -> str:
    from ai_historian.model_config import resolve_chat_config

    return resolve_chat_config().model


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return payload


def group_packets_by_case(packets: list[CasePacket]) -> dict[str, list[CasePacket]]:
    grouped: dict[str, list[CasePacket]] = {}
    for packet in packets:
        grouped.setdefault(packet.case_id, []).append(packet)
    return grouped


def prompt_for_case(rules_text: str, case_id: str, packets: list[CasePacket]) -> list[dict[str, str]]:
    material_blocks = []
    for packet in packets:
        sentence_lines = "\n".join(f"{packet.part_id}\t{s.sentence_id}\t{s.text}" for s in packet.sentences)
        material_blocks.append(
            f"part_id={packet.part_id}\nsource_text={packet.source_text}\n句子格式：part_id<TAB>sentence_id<TAB>sentence\n{sentence_lines}"
        )
    materials = "\n\n".join(material_blocks)
    user_prompt = f"""
你现在完成一个人类参与者同款的时间范围标注任务。请严格依据作答规则和给出的文本，不使用任何外部资料。

【参与者作答规则】
{rules_text}

【自动评分补充】
1. 你的主要任务仍然是给每一句输出人类可读的 start_time/end_time/unknown/sink/interlude。
2. 不要输出 ISO，不要把中文纪年换算成公元纪年；ISO 转换会由统一后处理完成。
3. 请像人类参与者一样写中文时间表达，例如“汉十一年”“汉十二年秋”“汉元年十二月”。
4. 如果是下沉句，sink=true；如果是插叙，interlude=true。未知则 unknown=true。

【输出 JSON 格式】
只输出 JSON，不要解释。格式：
{{
    "rows": [
    {{
      "part_id": "H-C1",
      "sentence_id": "53.10.1",
      "start_time": "汉十一年",
      "end_time": "汉十二年秋",
      "unknown": false,
      "sink": false,
      "sink_reason": "",
      "interlude": false,
      "interlude_reason": "",
      "notes": "简短说明"
    }}
  ]
}}

【材料】
case_id={case_id}

【句子】
{materials}
""".strip()
    return [
        {
            "role": "system",
            "content": "你是严谨的中文历史时间标注员。你必须逐句输出完整 JSON，不能遗漏句子。",
        },
        {"role": "user", "content": user_prompt},
    ]


def call_llm(client: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    from ai_historian.model_config import create_chat_completion

    model = model_name()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(os.getenv("AIH_DIRECT_LLM_TEMPERATURE", "0")),
    }
    max_tokens = int(os.getenv("AIH_DIRECT_LLM_MAX_TOKENS", "0"))
    if max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    kwargs["response_format"] = {"type": "json_object"}
    attempts = max(1, int(os.getenv("AIH_DIRECT_LLM_RETRIES", "3")))
    for attempt in range(1, attempts + 1):
        try:
            response = create_chat_completion(client, **kwargs)
            content = response.choices[0].message.content or ""
            return parse_json_object(content)
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(30, 5 * attempt)
            print(
                f"[DirectLLM] transient request/response failure "
                f"({attempt}/{attempts}): {exc}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "是"}


def rows_from_response(packets: list[CasePacket], payload: dict[str, Any]) -> list[dict[str, str]]:
    by_part_and_id: dict[tuple[str, str], dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for raw_row in payload.get("rows", []):
        if not isinstance(raw_row, dict):
            continue
        part_id = str(raw_row.get("part_id", "")).strip()
        sentence_id = str(raw_row.get("sentence_id", "")).strip()
        if part_id and sentence_id:
            by_part_and_id[(part_id, sentence_id)] = raw_row
        if sentence_id and sentence_id not in by_id:
            by_id[sentence_id] = raw_row
    rows: list[dict[str, str]] = []
    for packet in packets:
        for index, sentence in enumerate(packet.sentences, start=1):
            item = by_part_and_id.get((packet.part_id, sentence.sentence_id)) or by_id.get(sentence.sentence_id, {})
            start_time = normalize_time_text(item.get("start_time"))
            end_time = normalize_time_text(item.get("end_time"))
            unknown = truthy(item.get("unknown"))
            sink = truthy(item.get("sink"))
            interlude = truthy(item.get("interlude"))
            row = {field: "" for field in CSV_FIELDS}
            row.update(
                {
                    "case_id": packet.case_id,
                    "part_id": packet.part_id,
                    "item_no": str(index),
                    "sentence_id": sentence.sentence_id,
                    "source_text": packet.source_text,
                    "sentence": sentence.text,
                    "ai_start_ym": "",
                    "ai_end_ym": "",
                    "ai_unknown": "1" if unknown else "",
                    "ai_tm": start_time,
                    "ai_timeblock_id": "",
                    "ai_timeblock_start_tm": start_time,
                    "ai_timeblock_end_tm": end_time,
                    "ai_iso_range": "",
                    "ai_crossdoc_used": "",
                    "ai_oti_exists": "",
                    "ai_sink": "1" if sink else "",
                    "ai_sink_reason": str(item.get("sink_reason") or ""),
                    "ai_interlude": "1" if interlude else "",
                    "ai_interlude_reason": str(item.get("interlude_reason") or ""),
                    "ai_agent_note": (
                        "Direct LLM baseline raw output; ISO will be post-processed by AIHAgent Step11; "
                        f"start_time={start_time}; end_time={end_time}; "
                        f"notes={item.get('notes') or ''}"
                    ),
                }
            )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_once(
    client: Any,
    rules_text: str,
    packets: list[CasePacket],
    run_dir: Path,
    reuse_raw: bool = False,
) -> dict[str, Any]:
    raw_rows: list[dict[str, str]] = []
    timings: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, str]] = []
    raw_dir = run_dir / "raw_llm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    active_path = run_dir / "direct_llm_active.json"
    for case_id, case_packets_for_run in group_packets_by_case(packets).items():
        active_path.write_text(json.dumps({
            "case_id": case_id,
            "stage": "模型生成或读取缓存",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        start = time.time()
        raw_path = raw_dir / f"{case_id}.json"
        if reuse_raw and raw_path.is_file():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            print(f"[DirectLLM] Reusing raw response: {raw_path}", flush=True)
        else:
            payload = call_llm(client, prompt_for_case(rules_text, case_id, case_packets_for_run))
        llm_elapsed = time.time() - start
        raw_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows = rows_from_response(case_packets_for_run, payload)
        active_path.write_text(json.dumps({
            "case_id": case_id,
            "stage": "Step11 后处理",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        postprocess = postprocess_rows_with_agent_step11(rows, run_dir / "postprocess" / case_id)
        post_rows = postprocess["rows"]
        unresolved_rows.extend(postprocess["unresolved_rows"])
        raw_rows.extend(post_rows)
        timings.append(
            {
                "case_id": case_id,
                "part_id": "+".join(packet.part_id for packet in case_packets_for_run),
                "rows": len(post_rows),
                "llm_wall_seconds": round(llm_elapsed, 3),
                "postprocess_wall_seconds": postprocess["step11_seconds"],
                "total_wall_seconds": round(llm_elapsed + postprocess["step11_seconds"], 3),
                "unresolved_boundaries": len(
                    [row for row in postprocess["unresolved_rows"] if row.get("case_id") == case_id]
                ),
                "model": model_name(),
            }
        )
        (run_dir / f"{case_id}_step11.log").write_text(postprocess["step11_log"], encoding="utf-8")
        print(
            f"[DirectLLM] {case_id} rows={len(post_rows)} llm={llm_elapsed:.1f}s post={postprocess['step11_seconds']:.1f}s",
            flush=True,
        )
    active_path.unlink(missing_ok=True)
    table_path = run_dir / "tables" / "all_cases_direct_llm_prefill.csv"
    write_csv(table_path, raw_rows)
    unresolved_csv = run_dir / "tables" / "direct_llm_unresolved_boundaries.csv"
    unresolved_csv.parent.mkdir(parents=True, exist_ok=True)
    with unresolved_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "part_id",
            "item_no",
            "sentence_id",
            "boundary_type",
            "raw_time_text",
            "canonical_time_text",
            "synthetic_timeblock_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unresolved_rows)
    (run_dir / "direct_llm_unresolved_boundaries.json").write_text(
        json.dumps(unresolved_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "direct_llm_timing_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "part_id",
                "rows",
                "llm_wall_seconds",
                "postprocess_wall_seconds",
                "total_wall_seconds",
                "unresolved_boundaries",
                "model",
            ],
        )
        writer.writeheader()
        writer.writerows(timings)
    return {
        "run_dir": str(run_dir),
        "table_path": str(table_path),
        "rows": len(raw_rows),
        "timings": timings,
        "unresolved_csv": str(unresolved_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct LLM baseline for Experiment 1.")
    parser.add_argument("--case-ids", default="H-C1,H-C2,H-C3,H-C4,H-C5,H-C6")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-dir", default=str(PACKAGE_DIR / "outputs" / "current" / f"generated_results_direct_llm_{datetime.now():%Y%m%d_%H%M%S}"))
    parser.add_argument("--label", default="direct_llm_baseline")
    parser.add_argument("--reuse-raw", action="store_true", help="Reuse saved raw_llm case JSON when present")
    args = parser.parse_args()

    load_env_defaults()
    output_dir = Path(args.output_dir).resolve()
    packets = case_packets([case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()])
    rules_text = read_rules_text()
    client = make_client()

    summaries = []
    for run_index in range(1, args.runs + 1):
        run_dir = output_dir if args.runs == 1 else output_dir / "runs" / f"run_{run_index:02d}"
        summaries.append(run_once(client, rules_text, packets, run_dir, reuse_raw=args.reuse_raw))

    if args.runs > 1:
        first_table = Path(summaries[0]["table_path"])
        final_table = output_dir / "tables" / "all_cases_direct_llm_prefill.csv"
        final_table.parent.mkdir(parents=True, exist_ok=True)
        final_table.write_text(first_table.read_text(encoding="utf-8"), encoding="utf-8")
        first_unresolved = Path(summaries[0]["unresolved_csv"])
        (output_dir / "tables" / "direct_llm_unresolved_boundaries.csv").write_text(
            first_unresolved.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        first_unresolved_json = first_unresolved.parent.parent / "direct_llm_unresolved_boundaries.json"
        if first_unresolved_json.exists():
            (output_dir / "direct_llm_unresolved_boundaries.json").write_text(
                first_unresolved_json.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        first_timing = Path(summaries[0]["run_dir"]) / "direct_llm_timing_summary.csv"
        if first_timing.exists():
            (output_dir / "direct_llm_timing_summary.csv").write_text(
                first_timing.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    summary = {
        "label": args.label,
        "output_dir": str(output_dir),
        "runs": summaries,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Direct LLM baseline: participant rules + case text -> Chinese sentence-level ranges; boundary-to-ISO post-processing uses the same AIHAgent Step11 canonicalizer and normalization logic.",
    }
    (output_dir / "direct_llm_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
