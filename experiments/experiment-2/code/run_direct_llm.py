#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parents[1]
EXPERIMENT2_DIR = PACKAGE_DIR / "inputs" / "forms"
STANDARD_JSON = PACKAGE_DIR / "inputs" / "scoring" / "experiment-2-standard-answers.json"

CSV_FIELDS = [
    "form_id",
    "block",
    "question_id",
    "agent",
    "source_id",
    "doc",
    "answer_time_span",
    "answer_sink_yes_no",
    "answer_interlude_yes_no",
    "answer_choice",
    "confidence",
    "notes",
]


CONCEPT_CONTEXT = """
核心概念定义：

1. 时间信息原文
- 时间信息原文是句子中能够明确指示事件发生或动作施行时间点、时间段或时间顺序的名词性/介词性短语。
- 如果句子中只有“最初”“后来”“曾经”“当时”等非常模糊、不能独立定位当前事件时间的表达，通常不算明确时间信息原文。
- 如果没有明确时间信息原文，填写“无”。
- 保留题面原文表达，不要转成 ISO，不要自行改写成公元纪年。

2. 下沉句
- 下沉句是没有具体事件发生、主要用于人物身份、背景、评价、状态、说明来源等描述性语句。
- 但如果描述性句子里有明确时间信息原文，例如“某年春天”“汉三年”“这时”，则通常不按下沉处理。

3. 插叙 / 倒叙
- 插叙指连续叙述过程中，暂时中断当前时间线，插入与当前叙述时间不一致的内容。
- 插叙可能回溯过去、提前描述未来，也可能补充背景或人物经历；关键是叙事时间结构短暂偏离当前主线。
- 直接引语中的内容不作为插叙判定依据；“某人说：……”内部内容不要单独判成插叙。

4. 直接引语 / 说话内容处理
- 凡是“某人说：……”“某人告诉……：……”“某人问……：……”这类结构，默认只把“说、告诉、问、答、劝、命令”等叙述层说话行为当作可判定对象。
- 引号或冒号之后的说话内容，默认不进入当前题目的时间信息、下沉句、插叙、事件顺序或跨文本对齐判断范围，除非题目明确要求分析这段引语本身。
- 因此，不要把引语内部的未来计划、威胁、回忆、判断、命令，当成当前叙事层已经发生的事件或当前句的时间锚点。

5. 纪年规则
- 汉初纪年注意十月为一年开始。
- 对于 TimeBlock 补全，只在题目给出的背景和材料内推理，不使用标准答案。
- 不要把题面原文改写成公元纪年；只在 A/B/C/D 选项之间选择。
- 如果题目要求补全年月，先找材料内部最近且可延续的明确纪年锚点，再处理“正月、四月、八月、十一月”等月名。

6. 跨文本对齐
- 判断两个文本片段是否描述同一具体事件或同一紧密事件链时，优先看核心参与者、动作、对象、地点、因果触发和直接结果。
- 只处于同一大历史阶段、同一人物生涯阶段、同一战争背景，不能直接算同一事件。
- 如果一个选项只共享人物或时代背景，但核心动作不同，应排在共享核心动作链的选项之后。
""".strip()


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


def make_client():
    from openai import OpenAI

    from ai_historian.model_config import resolve_chat_config

    config = resolve_chat_config(require_model=False)
    return OpenAI(api_key=config.api_key, base_url=config.base_url), config.base_url


def strip_latex(text: str) -> str:
    def label_enumerate(match: re.Match[str]) -> str:
        body = match.group(1)
        parts = re.split(r"\\item\s+", body)
        labeled = []
        labels = ["A", "B", "C", "D", "E", "F"]
        for i, part in enumerate(parts[1:]):
            labeled.append(f"{labels[i]}. {part.strip()}")
        return "\n".join(labeled)

    text = text.replace("\\par", "\n")
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\begin\{enumerate\}\[[^\]]*\\Alph[^\]]*\](.*?)\\end\{enumerate\}", label_enumerate, text, flags=re.S)
    text = re.sub(r"\\textcolor\{[^{}]*\}\{\\textbf\{([^{}]*)\}\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\paragraph\{([^{}]*)\}", lambda m: "\n### " + m.group(1) + "\n", text)
    text = re.sub(r"\\section\{([^{}]*)\}", lambda m: "\n## " + m.group(1) + "\n", text)
    text = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}", "", text)
    text = re.sub(r"\\item\s+", "- ", text)
    text = re.sub(r"\\choicebox\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\answerline\{[^{}]*\}", "____", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda m: m.group(1) or "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("~", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_heading(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.replace("\\quad", " ")).strip()


def extract_question_sections(form_id: str) -> dict[str, str]:
    tex_path = EXPERIMENT2_DIR / form_id / f"{form_id}_questions.tex"
    text = tex_path.read_text(encoding="utf-8")
    pattern = re.compile(r"\\paragraph\{([^{}]+)\}(.*?)(?=\\paragraph\{|\\end\{document\})", re.S)
    sections: dict[str, str] = {}
    for match in pattern.finditer(text):
        heading = normalize_heading(match.group(1))
        qid_match = re.match(r"([ABC]\S+)", heading)
        if not qid_match:
            continue
        question_id = qid_match.group(1)
        body = strip_latex(match.group(2))
        sections[question_id] = f"{heading}\n\n{body}"
    return sections


def load_questions() -> list[dict[str, Any]]:
    questions = []
    for form_id in ["T1", "T2", "T3"]:
        response_path = EXPERIMENT2_DIR / form_id / f"{form_id}_response_sheet.csv"
        sections = extract_question_sections(form_id)
        with response_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                qid = row["question_id"]
                if qid not in sections:
                    raise KeyError(f"Missing question section for {form_id} {qid}")
                questions.append({**row, "question_text": sections[qid]})
    return questions


def response_schema(block: str) -> dict[str, Any]:
    if block == "A":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "time_span": {"type": "string"},
                "sink_yes_no": {"type": "string", "enum": ["是", "否", "不确定"]},
                "interlude_yes_no": {"type": "string", "enum": ["是", "否", "不确定"]},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["time_span", "sink_yes_no", "interlude_yes_no", "confidence", "reason"],
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "choice": {"type": "string", "enum": ["A", "B", "C", "D"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["choice", "confidence", "reason"],
    }


def build_prompt(question: dict[str, Any]) -> list[dict[str, str]]:
    block = question["block"]
    if block == "A":
        task = """
你现在扮演句子级诊断 Agent。请对一道 Block A 题作答：
- 填写“时间信息原文”：如果没有明确时间信息原文，填“无”。
- 判断“是否为下沉句”：只能答“是”“否”或“不确定”。
- 判断“是否为插叙/倒叙”：只能答“是”“否”或“不确定”。
""".strip()
        output = """
只返回 JSON：
{
  "time_span": "原文短语或无",
  "sink_yes_no": "是/否/不确定",
  "interlude_yes_no": "是/否/不确定",
  "confidence": 0.0到1.0,
  "reason": "一句话理由"
}
""".strip()
    elif question["agent"].strip() == "Agent 5":
        task = """
你现在扮演 TimeBlock 时间标志物补全 Agent。请根据题面背景、材料和选项，从 A/B/C/D 中选择最合适的一项。
作答步骤：
1. 先识别材料中要补全的时间表达本身，例如“正月”“四月”“八月”“十一月间”。
2. 再找材料内部最近、可延续到该表达的明确纪年锚点；不要因为熟悉历史事实而跳到题面之外的年份。
3. 汉初十月为岁首：同一汉年内部月份顺序是十月、十一月、十二月、正月、二月……九月。
4. 最后逐项核对 A/B/C/D，选择和补全结果完全一致的选项。
""".strip()
        output = '只返回 JSON：{"choice":"A/B/C/D","confidence":0.0到1.0,"reason":"一句话理由"}'
    elif question["agent"].strip() == "Agent 6":
        task = """
你现在扮演 TimeBlock 顺序判断 Agent。请根据题面材料 A、B 和选项，从 A/B/C/D 中选择最合适的一项。
作答步骤：
1. 分别抽取材料 A 和材料 B 的核心事件：参与者、动作、对象、地点、结果。
2. 如果 A/B 描述同一事件或同一不可拆分的连续事件链，不要强行排序，应选择题面中表示“同一/无法区分/同时段”的选项。
3. 如果二者不是同一事件，再根据材料内部时间词和事件因果判断先后。
4. 必须把你的判断映射回题面 A/B/C/D 选项，不要只回答“A材料早于B材料”。
""".strip()
        output = '只返回 JSON：{"choice":"A/B/C/D","confidence":0.0到1.0,"reason":"一句话理由"}'
    else:
        task = """
你现在扮演跨文本事件对齐 Agent。请判断 target 与哪个 source 最可能描述同一具体事件或同一紧密事件链，从 A/B/C/D 中选择最合适的一项。
作答步骤：
1. 先抽取 target 的核心事件：参与者、动作、对象、地点、触发原因、直接结果。
2. 对每个 source 选项做同样抽取。
3. 优先选择核心动作链一致的选项；人物相同、时代相同、战争背景相同但动作不同，不足以判为同一事件。
4. 如果 target 是任命、受封、进军、败亡、围城、入关、守关、杀/俘/降等具体事件，source 也必须覆盖同一类核心动作或其直接前后链。
5. 必须在 A/B/C/D 中选择最合适的一项。
""".strip()
        output = '只返回 JSON：{"choice":"A/B/C/D","confidence":0.0到1.0,"reason":"一句话理由"}'

    user = f"""
{CONCEPT_CONTEXT}

任务：
{task}

题目信息：
form_id: {question['form_id']}
block: {question['block']}
question_id: {question['question_id']}
agent: {question['agent']}
source_id: {question['source_id']}
doc: {question.get('doc') or ''}

题面：
{question['question_text']}

{output}
不要输出 markdown，不要解释 JSON 之外的任何内容。
""".strip()
    return [
        {"role": "system", "content": "你是严谨的历史文本诊断 Agent，只根据题面作答。"},
        {"role": "user", "content": user},
    ]


def parse_json_text(text: str) -> dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm(client, model: str, question: dict[str, Any], max_retries: int = 3) -> dict[str, Any]:
    from ai_historian.model_config import create_chat_completion

    messages = build_prompt(question)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = create_chat_completion(client, **kwargs)
            content = response.choices[0].message.content or "{}"
            return parse_json_text(content)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError("unreachable")


def norm_text(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .replace(" ", "")
        .replace("，", "")
        .replace(",", "")
        .replace("。", "")
        .replace("；", "")
        .replace("：", "")
        .replace("、", "")
    )


def norm_yes_no(value: Any) -> str:
    v = norm_text(value)
    if v in {"是", "yes", "true", "1"}:
        return "是"
    if v in {"否", "no", "false", "0"}:
        return "否"
    if v in {"不确定", "不知道", "无法判断", "unknown"}:
        return "不确定"
    return v


def norm_choice(value: Any) -> str:
    v = str(value or "").strip().upper()
    match = re.search(r"[ABCD]", v)
    return match.group(0) if match else ""


def majority(values: list[Any], normalizer=lambda x: str(x or "").strip()) -> Any:
    keyed = [(normalizer(v), v) for v in values]
    keyed = [(k, v) for k, v in keyed if k]
    if not keyed:
        return ""
    counts = Counter(k for k, _ in keyed)
    top_count = counts.most_common(1)[0][1]
    winners = {k for k, count in counts.items() if count == top_count}
    for key, value in keyed:
        if key in winners:
            return value
    return keyed[0][1]


def raw_to_row(question: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    block = question["block"]
    if block == "A":
        return {
            "form_id": question["form_id"],
            "block": block,
            "question_id": question["question_id"],
            "agent": question["agent"],
            "source_id": question["source_id"],
            "doc": question.get("doc", ""),
            "answer_time_span": str(raw.get("time_span") or raw.get("answer_time_span") or "").strip() or "无",
            "answer_sink_yes_no": norm_yes_no(raw.get("sink_yes_no") or raw.get("answer_sink_yes_no")),
            "answer_interlude_yes_no": norm_yes_no(raw.get("interlude_yes_no") or raw.get("answer_interlude_yes_no")),
            "answer_choice": "",
            "confidence": raw.get("confidence", ""),
            "notes": raw.get("reason", ""),
        }
    return {
        "form_id": question["form_id"],
        "block": block,
        "question_id": question["question_id"],
        "agent": question["agent"],
        "source_id": question["source_id"],
        "doc": question.get("doc", ""),
        "answer_time_span": "",
        "answer_sink_yes_no": "",
        "answer_interlude_yes_no": "",
        "answer_choice": norm_choice(raw.get("choice") or raw.get("answer_choice")),
        "confidence": raw.get("confidence", ""),
        "notes": raw.get("reason", ""),
    }


def consensus_rows(run_rows_by_question: dict[str, list[dict[str, Any]]], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for question in questions:
        qid = question["question_id"]
        rows = run_rows_by_question[qid]
        base = {field: rows[0].get(field, "") for field in CSV_FIELDS}
        if question["block"] == "A":
            base["answer_time_span"] = majority([r["answer_time_span"] for r in rows], norm_text)
            base["answer_sink_yes_no"] = majority([r["answer_sink_yes_no"] for r in rows], norm_yes_no)
            base["answer_interlude_yes_no"] = majority([r["answer_interlude_yes_no"] for r in rows], norm_yes_no)
        else:
            base["answer_choice"] = majority([r["answer_choice"] for r in rows], norm_choice)
        confidences = []
        for r in rows:
            try:
                confidences.append(float(r.get("confidence") or 0))
            except ValueError:
                pass
        base["confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else ""
        base["notes"] = " | ".join(f"run{i+1}:{r.get('notes','')}" for i, r in enumerate(rows))
        out.append(base)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def load_gold() -> dict[str, dict[str, Any]]:
    standard = json.loads(STANDARD_JSON.read_text(encoding="utf-8"))
    gold = {}
    for form_id in ["T1", "T2", "T3"]:
        for row in standard["units"][form_id]["rows"]:
            gold[f"{form_id}::{row['questionId']}"] = row
    return gold


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = load_gold()
    detail = []
    for row in rows:
        key = f"{row['form_id']}::{row['question_id']}"
        g = gold[key]
        if row["block"] == "A":
            checks = [
                ("time_span", norm_text(row["answer_time_span"]), norm_text(g.get("correctTimeSpan"))),
                ("sink", norm_yes_no(row["answer_sink_yes_no"]), norm_yes_no(g.get("correctSinkYesNo"))),
                ("interlude", norm_yes_no(row["answer_interlude_yes_no"]), norm_yes_no(g.get("correctInterludeYesNo"))),
            ]
        else:
            checks = [("choice", norm_choice(row["answer_choice"]), norm_choice(g.get("correctChoice")))]
        for field, answer, correct in checks:
            detail.append(
                {
                    "form_id": row["form_id"],
                    "block": row["block"],
                    "question_id": row["question_id"],
                    "field": field,
                    "answer": answer,
                    "correct": correct,
                    "is_correct": int(answer == correct),
                }
            )
    component_correct = sum(d["is_correct"] for d in detail)
    by_question = defaultdict(list)
    for d in detail:
        by_question[(d["form_id"], d["block"], d["question_id"])].append(d["is_correct"])
    strict_correct = sum(1 for vals in by_question.values() if vals and all(vals))
    return {
        "component_accuracy": component_correct / len(detail) if detail else None,
        "component_correct": component_correct,
        "component_total": len(detail),
        "row_strict_accuracy": strict_correct / len(by_question) if by_question else None,
        "row_strict_correct": strict_correct,
        "row_strict_total": len(by_question),
        "detail": detail,
        "by_block": summarize_detail(detail, "block"),
        "by_form": summarize_detail(detail, "form_id"),
        "by_field": summarize_detail(detail, "field"),
    }


def summarize_detail(detail: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups = defaultdict(lambda: [0, 0])
    for row in detail:
        groups[row[key]][0] += int(row["is_correct"])
        groups[row[key]][1] += 1
    return [
        {"group": group, "correct": vals[0], "total": vals[1], "accuracy": vals[0] / vals[1] if vals[1] else None}
        for group, vals in sorted(groups.items())
    ]


def main() -> None:
    load_env_defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=os.getenv("AIH_CHAT_MODEL", ""))
    parser.add_argument("--forms", default="T1,T2,T3")
    parser.add_argument("--limit", type=int, default=0, help="debug: only run first N questions")
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("AIH_CHAT_MODEL or --model is required.")
    client, base_url = make_client()
    selected_forms = {x.strip() for x in args.forms.split(",") if x.strip()}
    questions = [q for q in load_questions() if q["form_id"] in selected_forms]
    if args.limit:
        questions = questions[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PACKAGE_DIR / "outputs" / "current" / f"direct_llm_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_run_rows: list[list[dict[str, Any]]] = []
    timing = []

    print(f"Experiment2 LLM subagents | questions={len(questions)} runs={args.runs} model={args.model} base_url={base_url}")
    for run_index in range(1, args.runs + 1):
        run_rows = []
        raw_records = []
        run_start = time.time()
        print(f"[run {run_index}/{args.runs}] start")
        for idx, question in enumerate(questions, start=1):
            t0 = time.time()
            raw = call_llm(client, args.model, question)
            elapsed = time.time() - t0
            row = raw_to_row(question, raw)
            run_rows.append(row)
            raw_records.append({"question": question, "raw_response": raw, "row": row, "elapsed_seconds": elapsed})
            print(f"[run {run_index}] {idx}/{len(questions)} {question['question_id']} {elapsed:.1f}s")
        run_seconds = time.time() - run_start
        timing.append({"run": run_index, "seconds": run_seconds})
        run_dir = output_dir / f"run_{run_index:02d}"
        write_csv(run_dir / "experiment2_llm_subagents_response.csv", run_rows)
        (run_dir / "raw_responses.json").write_text(json.dumps(raw_records, ensure_ascii=False, indent=2), encoding="utf-8")
        all_run_rows.append(run_rows)

    run_rows_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_rows in all_run_rows:
        for row in run_rows:
            run_rows_by_question[row["question_id"]].append(row)
    consensus = consensus_rows(run_rows_by_question, questions)
    write_csv(output_dir / "experiment-2-llm-subagents-consensus-response.csv", consensus)
    score = score_rows(consensus)
    (output_dir / "experiment-2-llm-subagents-score.json").write_text(
        json.dumps(
            {
                "label": "experiment2_llm_subagents_consensus",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "runs": args.runs,
                "questions": len(questions),
                "timing": timing,
                "total_seconds": sum(x["seconds"] for x in timing),
                "consensus_method": "per-field majority vote across runs; ties keep earliest run value; no gold labels used in voting",
                **score,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "experiment-2-llm-subagents-score-detail.csv",
        score["detail"],
        ["form_id", "block", "question_id", "field", "answer", "correct", "is_correct"],
    )
    latest = PACKAGE_DIR / "outputs" / "latest_direct_llm_output_dir.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(output_dir), encoding="utf-8")
    print(json.dumps({k: score[k] for k in ["component_accuracy", "component_correct", "component_total", "row_strict_accuracy", "row_strict_correct", "row_strict_total"]}, ensure_ascii=False, indent=2))
    print(f"OUT={output_dir}")


if __name__ == "__main__":
    main()
