#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]
LLM_SCRIPT = PACKAGE_DIR / "code" / "run_direct_llm.py"

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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


llm = load_module(LLM_SCRIPT, "experiment2_llm_subagents")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def short_id(value: str) -> str:
    parts = str(value or "").split(".")
    return ".".join(parts[-3:]) if len(parts) >= 3 else str(value or "")


def id_doc(value: str) -> str:
    sid = short_id(value)
    return sid.split(".", 1)[0] if "." in sid else ""


def norm_choice(value: Any) -> str:
    v = str(value or "").strip().upper()
    match = re.search(r"[ABCD]", v)
    return match.group(0) if match else ""


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


def response_schema() -> dict[str, Any]:
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


def call_llm(client, model: str, messages: list[dict[str, str]], max_retries: int = 3) -> dict[str, Any]:
    from ai_historian.model_config import create_chat_completion

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = create_chat_completion(client, **kwargs)
            return parse_json_text(response.choices[0].message.content or "{}")
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError("unreachable")


def sentence_observation(sentence_index: dict[str, dict[str, Any]], sid: str) -> dict[str, Any]:
    item = sentence_index.get(sid, {})
    oti = item.get("Original_time_information") or {}
    sink = item.get("sink") or {}
    return {
        "id": sid,
        "sentence": item.get("sentence", ""),
        "original_time_information": oti,
        "sink": sink,
        "interlude": item.get("Interlude", ""),
    }


def timeblock_observation(timeblock_by_sentence: dict[str, dict[str, Any]], sid: str) -> dict[str, Any]:
    block = timeblock_by_sentence.get(sid, {})
    return {
        "sentence_id": sid,
        "timeblock_id": short_id(block.get("ID", "")),
        "timeblock_range": block.get("timeblock_range", ""),
        "TM": block.get("TM", ""),
        "granularity": block.get("Granularity", ""),
        "iso": block.get("iso", ""),
        "iso_range": block.get("iso_range", ""),
        "time_anchor": block.get("time_anchor", {}),
        "conversion_information": block.get("Conversion information", {}),
    }


def load_crossdoc_evidence(agent_root: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for path in agent_root.glob("**/timeblock/step10b_crossdoc_prealign_report.json"):
        if "/runs/" in str(path):
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        for stats in data.get("context_stats", []):
            for sample in stats.get("accepted_samples", []):
                key = (
                    short_id(sample.get("target_timeblock_id", "")),
                    tuple(sorted(short_id(x) for x in sample.get("supporting_source_timeblock_ids", []))),
                    str(sample.get("episode_label", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                item = {
                    "case": path.parts[-4] if len(path.parts) >= 4 else "",
                    "target_timeblock_id": short_id(sample.get("target_timeblock_id", "")),
                    "supporting_source_timeblock_ids": [short_id(x) for x in sample.get("supporting_source_timeblock_ids", [])],
                    "episode_label": sample.get("episode_label", ""),
                    "confidence": sample.get("confidence", ""),
                    "supporting_source_quote": sample.get("supporting_source_quote", ""),
                    "supporting_target_quote": sample.get("supporting_target_quote", ""),
                    "reason": sample.get("reason", ""),
                }
                evidence.append(item)
    return evidence


def relevant_crossdoc_evidence(evidence: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    out = []
    docs = {id_doc(x) for x in ids if id_doc(x)}
    for item in evidence:
        item_ids = {short_id(item.get("target_timeblock_id", ""))}
        item_ids.update(short_id(x) for x in item.get("supporting_source_timeblock_ids", []))
        item_docs = {id_doc(x) for x in item_ids if id_doc(x)}
        quote = f"{item.get('supporting_source_quote','')} {item.get('supporting_target_quote','')}"
        if ids & item_ids or (docs and docs & item_docs) or any(sid in quote for sid in ids):
            out.append(item)
    return out[:8]


def build_agent_context(
    question: dict[str, Any],
    sentence_index: dict[str, dict[str, Any]],
    timeblock_by_sentence: dict[str, dict[str, Any]],
    crossdoc_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    ids: set[str] = set()
    for sid in str(question.get("source_id", "")).split("|"):
        if sid.strip():
            ids.add(short_id(sid.strip()))
    for candidate_ids in question.get("choice_ids", {}).values():
        ids.update(short_id(x) for x in candidate_ids)
    target_id = question.get("target_id")
    if target_id:
        ids.add(short_id(target_id))
    return {
        "sentence_observations": [sentence_observation(sentence_index, sid) for sid in sorted(ids)],
        "timeblock_observations": [timeblock_observation(timeblock_by_sentence, sid) for sid in sorted(ids)],
        "crossdoc_evidence": relevant_crossdoc_evidence(crossdoc_evidence, ids),
    }


def build_prompt(
    question: dict[str, Any],
    evidence_mode: str,
    agent_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if question["block"] == "A":
        role = "你是 AIHAgent 的 Agent 2/4/5：句子级时间信息、下沉句、插叙联合判定模块。"
        task = """
请只根据题面里人类可见的句子与上下文，回答：
1. 时间信息原文
2. 是否下沉句
3. 是否插叙/倒叙

规则：
1. 只能依据题面可见文本，不能假设你看过任何 TM、iso、iso_range 或内部标注。
2. 时间信息原文必须保持题面原文，不要改写成公元纪年。
3. 引号中的未来计划、命令、威胁，不自动算当前叙事时间锚点。
4. 只返回题面直接支持的判断；不确定时可答“不确定”。
""".strip()
        user = f"""
{llm.CONCEPT_CONTEXT}

{task}

题目信息：
form_id: {question['form_id']}
block: {question['block']}
question_id: {question['question_id']}
agent: {question['agent']}
source_id: {question['source_id']}

题面：
{llm.strip_latex(question.get('question_section', ''))}

只返回 JSON：
{{"time_span":"原文短语或无","sink_yes_no":"是/否/不确定","interlude_yes_no":"是/否/不确定","confidence":0.0到1.0,"reason":"一句话说明依据"}}
不要输出 markdown 或 JSON 之外的文字。
""".strip()
        return [{"role": "system", "content": role}, {"role": "user", "content": user}]

    if question["agent"].strip() == "Agent 5":
        role = "你是 AIHAgent 的 Agent 5：TimeBlock 时间表达补全与选项判定模块。"
        task = """
请根据题面和规则，在 A/B/C/D 中选择最合适的时间补全。
规则：
1. 先识别题面要求补全的局部时间表达。
2. 只能依据题面可见文本，不能假设你看过任何 TM、iso、iso_range 或内部时间轴。
3. 以题面材料内部的连续叙事为主证据；如果题面明确给出前后锚点，必须优先服从题面锚点。
3. 如果原文只写“四月”“正月”“十一月间”等局部表达，必须先确定它挂靠的是哪一个王年，再判断月份。
4. 汉初纪年以十月为岁首；“汉二年冬 -> 春 -> 四月”这类序列，要按同一叙事链延续，不要被别处同句号的 TM 带偏。
5. 只输出选项，不输出 ISO。
""".strip()
    elif question["agent"].strip() == "Agent 6":
        role = "你是 AIHAgent 的 Agent 6：TimeBlock 事件顺序判定模块。"
        task = """
请判断材料 A、B 的关系，并映射到 A/B/C/D。
规则：
1. 先判断是否同一具体事件或同一不可拆分事件链；如果是，选择题面中对应“同一/大致同时”的选项。
2. 若不是同一事件，再结合题面时间词、承接词、因果顺序判断先后。
3. 只能依据题面可见材料，不能假设你看过任何 TM、iso、iso_range 或内部时间轴。
4. 只有当题面材料本身不足以支持稳定排序时，才选择“无法判断”。
5. 不要因为都发生在同一年就误判为同一事件。
5. 先做事件级判断，再做时间级判断。
""".strip()
    else:
        role = "你是 AIHAgent 的 Agent 9：跨文本事件共指与同一历史过程对齐模块。"
        task = """
请从 A/B/C/D 中选择最可能与 target 描述同一具体事件或同一紧密事件链的 source。
规则：
1. 抽取 target 和每个 source 的参与者、动作、对象、地点、触发原因、直接结果。
2. 优先选择核心动作链一致的选项；同一人物、同一时代、时间接近但核心动作不同，不足以判同一事件。
3. 如果 target 是“起兵、受封、入关、守关、围城、战败、任命、迁都、攻破、投降”等具体事件，source 也必须覆盖同类核心动作或直接前后链。
4. 只能依据题面可见 target 和候选 source，不能假设你看过任何 crossdoc 报告或内部对齐结果。
5. 宽泛背景、后果总结、同一战争阶段说明，优先级低于直接叙述同一事件的候选。
""".strip()
    examples = """
泛化示例：
- 示例1：target 写“某王即位后次年春，命某将北伐”；source A 只说“该王在位期间国势转强”，source B 说“次年春命某将北伐赵地”。应选 B，因为核心动作链一致。
- 示例2：target 写“四月大败敌军”；上文先给出“某王二年冬”“春天”并继续同一战事。即使别处 timeblock 记录成“某王元年四月”，也应优先按题面本地叙事续接到“某王二年四月”。
- 示例3：A 材料写“十月入关”，B 材料写“十一月率军西进欲入关”。如果题面没有歧义，先判为 A 早于 B；不要仅因都在入关阶段就判同一事件。
""".strip()
    user = f"""
{llm.CONCEPT_CONTEXT}

{task}

{examples}

题目信息：
form_id: {question['form_id']}
block: {question['block']}
question_id: {question['question_id']}
agent: {question['agent']}
source_id: {question['source_id']}
target_id: {question.get('target_id') or ''}

题面：
{llm.strip_latex(question.get('question_section', ''))}

"""
    if evidence_mode == "agent_internal" and agent_context is not None:
        user += f"""
AIHAgent 上游结构化结果：
{json.dumps(agent_context, ensure_ascii=False, indent=2)}
""".strip()
        user += "\n\n"

    user += """
只返回 JSON：
{"choice":"A/B/C/D","confidence":0.0到1.0,"reason":"一句话说明使用了哪些题面证据"}
不要输出 markdown 或 JSON 之外的文字。
""".strip()
    return [{"role": "system", "content": role}, {"role": "user", "content": user}]


def build_block_a_review_prompt(question: dict[str, Any], initial_raw: dict[str, Any]) -> list[dict[str, str]]:
    role = "你是历史文本标注复核 Agent，专门复核时间信息原文、下沉句和插叙判定。"
    task = """
请复核下面这道 Block A 题的初始答案，并在必要时纠正。

复核规则：
1. 任何能回答“什么时候 / 在什么阶段 / 在什么事件背景下”的原文短语，都可以算时间信息原文，不要求是绝对年月。
2. 像“某人失败时”“某事结束后”“开始起事时”“等到某人显贵时”“过去到某地服役时”这类事件相对时间短语，通常都应保留为时间信息原文，而不是写“无”。
3. 如果目标句本身陈述了一个具体动作、任命、受伤、逃亡、加封、投降、安葬、显贵、劝说、带兵等事件，即使句子兼有背景说明，通常也不应判为下沉句。
4. 如果目标句通过“过去……时”“当初……时”“某朝灭亡后”“项氏失败时”之类表达，短暂回到另一叙事时间层来解释当前叙事原因或背景，通常应判为插叙/倒叙。
5. 只有在句子确实没有可提取的时间短语时，才填“无”。

泛化示例：
- “某国灭亡后，他隐居山中” -> 时间信息原文应是“某国灭亡后”。
- “攻打某侯国时，被流矢射中” -> 时间信息原文应是“攻打某侯国时”，且通常不是下沉句。
- “后来又加封此人，这是因为早年在边郡服役时有功” -> 若目标句借早年经历解释当前加封，常可判作插叙。
""".strip()
    user = f"""
{llm.CONCEPT_CONTEXT}

{task}

题目信息：
form_id: {question['form_id']}
block: {question['block']}
question_id: {question['question_id']}
agent: {question['agent']}
source_id: {question['source_id']}

题面：
{llm.strip_latex(question.get('question_section', ''))}

初始答案：
{json.dumps(initial_raw, ensure_ascii=False, indent=2)}

请输出复核后的最终 JSON：
{{"time_span":"原文短语或无","sink_yes_no":"是/否/不确定","interlude_yes_no":"是/否/不确定","confidence":0.0到1.0,"reason":"一句话说明修正依据"}}
不要输出 markdown 或 JSON 之外的文字。
""".strip()
    return [{"role": "system", "content": role}, {"role": "user", "content": user}]


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


def consensus_rows(run_rows_by_question: dict[str, list[dict[str, Any]]], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        q_rows = run_rows_by_question[question["question_id"]]
        base = {field: q_rows[0].get(field, "") for field in CSV_FIELDS}
        if question["block"] == "A":
            base["answer_time_span"] = majority([row["answer_time_span"] for row in q_rows], llm.norm_text)
            base["answer_sink_yes_no"] = majority([row["answer_sink_yes_no"] for row in q_rows], llm.norm_yes_no)
            base["answer_interlude_yes_no"] = majority([row["answer_interlude_yes_no"] for row in q_rows], llm.norm_yes_no)
        else:
            base["answer_choice"] = majority([row["answer_choice"] for row in q_rows], norm_choice)
        confidences = []
        for row in q_rows:
            try:
                confidences.append(float(row.get("confidence") or 0))
            except ValueError:
                pass
        base["confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else ""
        base["notes"] = " | ".join(f"run{i+1}:{row.get('notes','')}" for i, row in enumerate(q_rows))
        rows.append(base)
    return rows


def main() -> None:
    llm.load_env_defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--agent-output-root", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=os.getenv("AIH_CHAT_MODEL", ""))
    parser.add_argument("--forms", default="T1,T2,T3")
    parser.add_argument("--blocks", default="A,B,C")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evidence-mode", choices=["visible_only", "agent_internal"], default="visible_only")
    parser.add_argument("--disable-block-a-review", action="store_true")
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("AIH_CHAT_MODEL or --model is required.")
    client, base_url = llm.make_client()
    agent_root = Path(args.agent_output_root) if args.agent_output_root else Path()
    selected_forms = {x.strip() for x in args.forms.split(",") if x.strip()}
    selected_blocks = {x.strip() for x in args.blocks.split(",") if x.strip()}
    questions = [q for q in llm.load_questions() if q["form_id"] in selected_forms and q["block"] in selected_blocks]
    if args.limit:
        questions = questions[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "structured_llm" if args.evidence_mode == "visible_only" else "structured_llm_internal"
    output_dir = Path(args.output_dir) if args.output_dir else PACKAGE_DIR / "outputs" / "current" / f"{prefix}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_index = {}
    timeblock_by_sentence = {}
    crossdoc_evidence = []
    if args.evidence_mode == "agent_internal":
        raise RuntimeError("Final Experiment 2 submission supports visible_only only; agent_internal is excluded from the submitted comparison.")
        crossdoc_evidence = load_crossdoc_evidence(agent_root)

    run_rows_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timing = []
    raw_all = []
    print(
        f"Experiment2 AIHAgent-style API | questions={len(questions)} runs={args.runs} "
        f"model={args.model} base_url={base_url} evidence_mode={args.evidence_mode} "
        f"block_a_review={'off' if args.disable_block_a_review else 'on'}"
    )
    for run_index in range(1, args.runs + 1):
        run_start = time.time()
        run_rows = []
        print(f"[run {run_index}/{args.runs}] start")
        for idx, question in enumerate(questions, start=1):
            t0 = time.time()
            if args.evidence_mode == "agent_internal" and question["block"] != "A":
                agent_context = build_agent_context(question, sentence_index, timeblock_by_sentence, crossdoc_evidence)
            else:
                agent_context = None
            messages = build_prompt(question, args.evidence_mode, agent_context)
            raw = call_llm(client, args.model, messages)
            review_used = False
            if question["block"] == "A" and not args.disable_block_a_review:
                review_messages = build_block_a_review_prompt(question, raw)
                raw = call_llm(client, args.model, review_messages)
                review_used = True
            row = llm.raw_to_row(question, raw)
            elapsed = time.time() - t0
            if review_used:
                row["notes"] = f"[reviewed] {row.get('notes', '')}".strip()
            run_rows.append(row)
            run_rows_by_question[question["question_id"]].append(row)
            raw_all.append({"run": run_index, "question": question, "raw_response": raw, "row": row, "elapsed_seconds": elapsed})
            print(f"[run {run_index}] {idx}/{len(questions)} {question['question_id']} {elapsed:.1f}s")
        run_seconds = time.time() - run_start
        timing.append({"run": run_index, "seconds": run_seconds})
        run_dir = output_dir / f"run_{run_index:02d}"
        write_csv(run_dir / "experiment2_aihagent_api_response.csv", run_rows)

    consensus = consensus_rows(run_rows_by_question, questions)
    score = llm.score_rows(consensus)
    write_csv(output_dir / "experiment-2-aih-agent-api-consensus-response.csv", consensus)
    write_csv(
        output_dir / "experiment-2-aih-agent-api-score-detail.csv",
        score["detail"],
        ["form_id", "block", "question_id", "field", "answer", "correct", "is_correct"],
    )
    (output_dir / "raw-responses.json").write_text(json.dumps(raw_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "experiment-2-aih-agent-api-score.json").write_text(
        json.dumps(
            {
                "label": f"{prefix}_consensus",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "runs": args.runs,
                "questions": len(questions),
                "agent_output_root": str(agent_root),
                "evidence_mode": args.evidence_mode,
                "timing": timing,
                "total_seconds": sum(x["seconds"] for x in timing),
                "method": (
                    "All blocks answered from human-visible question text only, using AIHAgent-style prompts; Block A uses a second-pass review prompt; majority vote across runs."
                    if args.evidence_mode == "visible_only"
                    else "Block A/B/C answered with AIHAgent-style prompts plus upstream sentence/timeblock/crossdoc evidence; Block A uses a second-pass review prompt; majority vote across runs."
                ),
                "block_a_review": not args.disable_block_a_review,
                "consensus_method": "per-question majority vote; ties keep earliest run value; no gold labels used in voting",
                **score,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    latest = PACKAGE_DIR / "outputs" / "latest_structured_llm_output_dir.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(output_dir), encoding="utf-8")
    print(json.dumps({k: score[k] for k in ["component_accuracy", "component_correct", "component_total", "row_strict_accuracy", "row_strict_correct", "row_strict_total"]}, ensure_ascii=False, indent=2))
    print(f"OUT={output_dir}")


if __name__ == "__main__":
    main()
