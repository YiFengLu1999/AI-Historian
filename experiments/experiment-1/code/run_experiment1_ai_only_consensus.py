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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPO_ROOT = PACKAGE_DIR.parents[1]
GENERATOR = SCRIPT_DIR / "generate_experiment1_ai_assisted.py"
FINAL_AGENT_DIR = (
    REPO_ROOT / "src" / "ai_historian" / "profiles" / "evaluation" / "stages"
)
SOURCE_DIR = REPO_ROOT / "src"
SCORER = PACKAGE_DIR / "evaluation" / "score_ai_prefill_variant.js"
LOSS_DIAG = PACKAGE_DIR / "evaluation" / "diagnose_microiou_loss_by_row.js"

DEFAULT_CASES = ["H-C1", "H-C2", "H-C3", "H-C4", "H-C5", "H-C6"]
SINGLE_DOC_CASES = {"H-C1", "H-C2", "H-C3", "H-C4"}

CONSENSUS_FIELDS = [
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
    "ai_sink",
    "ai_sink_reason",
    "ai_interlude",
    "ai_interlude_reason",
    "ai_agent_note",
]


def read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and value:
            env[key] = value
    return env


def load_env(env_file: str, chat_provider: str, model: str) -> dict[str, str]:
    env = os.environ.copy()
    if env_file:
        env.update(read_env_file(Path(env_file).expanduser()))
    env["AIH_CHAT_PROVIDER"] = chat_provider
    env["AIH_CHAT_MODEL"] = model
    env.setdefault("AIH_PIPELINE_CONCURRENCY", "4")
    env.setdefault("AIH_AGENT_CONCURRENCY", "4")
    env.setdefault("AIH_AGENT_MAX_WORKERS", "4")
    env.setdefault("AIH_DISABLE_EMBEDDING", "1")
    from ai_historian.model_config import resolve_chat_config

    resolve_chat_config(env)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(SOURCE_DIR),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


def run_command(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started_at={dt.datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"# command={' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
        log.write(f"\n# finished_at={dt.datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"# returncode={rc}\n")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("case_id", ""),
        row.get("part_id", ""),
        row.get("item_no", ""),
        row.get("sentence_id", ""),
    )


def prediction_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in CONSENSUS_FIELDS)


def month_index(value: str) -> int | None:
    text = str(value or "").strip()
    if text in {"", "-infinity", "+infinity"}:
        return None
    match = re.match(r"^([+-]?\d{4,})-(\d{2})(?:-\d{2})?$", text)
    if not match:
        return None
    return int(match.group(1)) * 12 + int(match.group(2)) - 1


def row_interval(row: dict[str, str]) -> tuple[int | None, int | None]:
    return month_index(row.get("ai_start_ym", "")), month_index(row.get("ai_end_ym", ""))


def interval_distance(left: dict[str, str], right: dict[str, str]) -> int:
    left_start, left_end = row_interval(left)
    right_start, right_end = row_interval(right)
    penalty = 1200
    start_dist = penalty if left_start is None or right_start is None else abs(left_start - right_start)
    end_dist = penalty if left_end is None or right_end is None else abs(left_end - right_end)
    state_dist = 0 if (
        left.get("ai_unknown", ""),
        left.get("ai_sink", ""),
        left.get("ai_interlude", ""),
    ) == (
        right.get("ai_unknown", ""),
        right.get("ai_sink", ""),
        right.get("ai_interlude", ""),
    ) else penalty
    return start_dist + end_dist + state_dist


def structural_quality(row: dict[str, str]) -> tuple[int, int, int]:
    has_range = int(bool(row.get("ai_start_ym") and row.get("ai_end_ym")))
    is_regular = int(not row.get("ai_unknown") and not row.get("ai_sink") and not row.get("ai_interlude"))
    has_iso_range = int(bool(row.get("ai_iso_range")))
    return has_range, is_regular, has_iso_range


def choose_consensus(rows: list[dict[str, str]]) -> dict[str, str]:
    if len(rows) == 1:
        selected = dict(rows[0])
        selected["ai_agent_note"] = f"{selected.get('ai_agent_note', '')} | consensus:single_run".strip()
        return selected

    counts = Counter(prediction_signature(row) for row in rows)
    signature, count = counts.most_common(1)[0]
    if count > len(rows) / 2:
        selected = dict(next(row for row in rows if prediction_signature(row) == signature))
        selected["ai_agent_note"] = f"{selected.get('ai_agent_note', '')} | consensus:majority={count}/{len(rows)}".strip()
        return selected

    best_row = None
    best_score = None
    for row in rows:
        distance = sum(interval_distance(row, other) for other in rows)
        score = (distance, tuple(-x for x in structural_quality(row)), prediction_signature(row))
        if best_score is None or score < best_score:
            best_row = row
            best_score = score
    assert best_row is not None
    selected = dict(best_row)
    selected["ai_agent_note"] = f"{selected.get('ai_agent_note', '')} | consensus:medoid_runs={len(rows)}".strip()
    return selected


def run_case(
    case_id: str,
    repeat_index: int,
    run_dir: Path,
    env: dict[str, str],
    log_dir: Path,
    attempt: int = 1,
) -> Path:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--case-ids",
        case_id,
        "--output-dir",
        str(run_dir),
        "--agent-script-override-dir",
        str(FINAL_AGENT_DIR),
        "--no-pdf",
        "--end-step",
        "11",
        "--timeblock-output-step",
        "11",
        "--microiou-boundary-source",
        "iso_range",
    ]
    if case_id in SINGLE_DOC_CASES:
        cmd.extend(["--skip-crossdoc-presteps"])
    suffix = "" if attempt == 1 else f"_attempt_{attempt:02d}"
    run_command(cmd, env, log_dir / f"{case_id}_run_{repeat_index:02d}{suffix}.log")
    csv_path = run_dir / "tables" / "all_cases_ai_assisted_prefill.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected run CSV not found: {csv_path}")
    return csv_path


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 AI-only with repeated API runs and prediction consensus.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of API runs per case.")
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASES), help="Comma-separated case ids.")
    parser.add_argument("--output-dir", default="", help="Output directory.")
    parser.add_argument("--label", default="experiment1_ai_only_final_api_consensus")
    parser.add_argument("--chat-provider", default=os.getenv("AIH_CHAT_PROVIDER", ""))
    parser.add_argument("--model", default=os.getenv("AIH_CHAT_MODEL", ""))
    parser.add_argument("--env-file", default="", help="Optional env file containing API keys. Not copied into outputs.")
    parser.add_argument("--skip-existing-runs", action="store_true", help="Reuse existing per-case run CSVs under output-dir/runs.")
    parser.add_argument(
        "--parallel-cases",
        type=int,
        default=1,
        help="Run this many isolated H-C cases concurrently; repeats within each case remain sequential.",
    )
    parser.add_argument(
        "--case-attempts",
        type=int,
        default=3,
        help="Maximum attempts for each unfinished case repetition (for transient API failures).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=20.0,
        help="Base delay in seconds before retrying a failed case repetition.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.parallel_cases < 1:
        raise SystemExit("--parallel-cases must be >= 1")
    if args.case_attempts < 1:
        raise SystemExit("--case-attempts must be >= 1")

    cases = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    unknown = set(cases) - set(DEFAULT_CASES)
    if unknown:
        raise SystemExit(f"Unknown case id(s): {', '.join(sorted(unknown))}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or (PACKAGE_DIR / "outputs" / "current" / f"generated_results_api_consensus_{timestamp}")).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    runs_dir = out_dir / "runs"
    logs_dir = out_dir / "orchestration_logs"
    tables_dir = out_dir / "tables"
    for path in [runs_dir, logs_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    env = load_env(args.env_file, args.chat_provider, args.model)
    case_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    fieldnames: list[str] = []
    run_index: dict[str, list[str]] = {}

    def run_case_series(case_id: str) -> tuple[str, list[str]]:
        paths: list[str] = []
        for repeat in range(1, args.repeats + 1):
            run_dir = runs_dir / case_id / f"run_{repeat:02d}"
            csv_path = run_dir / "tables" / "all_cases_ai_assisted_prefill.csv"
            if not args.skip_existing_runs or not csv_path.exists():
                for attempt in range(1, args.case_attempts + 1):
                    try:
                        csv_path = run_case(case_id, repeat, run_dir, env, logs_dir, attempt)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                        if attempt >= args.case_attempts:
                            raise
                        delay = args.retry_delay * attempt
                        print(
                            f"[Experiment1] {case_id} run {repeat:02d} failed "
                            f"(attempt {attempt}/{args.case_attempts}: {exc}); retrying in {delay:.0f}s",
                            flush=True,
                        )
                        time.sleep(delay)
            paths.append(str(csv_path))
        return case_id, paths

    if args.parallel_cases == 1:
        for case_id in cases:
            finished_case, paths = run_case_series(case_id)
            run_index[finished_case] = paths
    else:
        with ThreadPoolExecutor(max_workers=min(args.parallel_cases, len(cases))) as executor:
            futures = {executor.submit(run_case_series, case_id): case_id for case_id in cases}
            for future in as_completed(futures):
                finished_case, paths = future.result()
                run_index[finished_case] = paths
                print(f"[Experiment1] completed case series {finished_case}", flush=True)

    for case_id in cases:
        for csv_name in run_index[case_id]:
            csv_path = Path(csv_name)
            current_fields, rows = read_csv(csv_path)
            if not fieldnames:
                fieldnames = current_fields
            case_rows[case_id].extend(rows)

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    row_order: list[tuple[str, str, str, str]] = []
    for rows in case_rows.values():
        for row in rows:
            key = row_key(row)
            if key not in grouped:
                row_order.append(key)
            grouped[key].append(row)

    if "ai_agent_note" not in fieldnames:
        fieldnames.append("ai_agent_note")

    consensus_rows = [choose_consensus(grouped[key]) for key in row_order]
    combined_csv = tables_dir / "all_cases_ai_only_consensus_prefill.csv"
    write_csv(combined_csv, fieldnames, consensus_rows)
    shutil.copy2(combined_csv, tables_dir / "all_cases_ai_assisted_prefill.csv")

    label = safe_label(args.label)
    run_command(["node", str(SCORER), str(combined_csv), label], env, logs_dir / "score_consensus.log")
    run_command(["node", str(LOSS_DIAG), str(combined_csv), label], env, logs_dir / "loss_consensus.log")

    summary = {
        "label": args.label,
        "output_dir": str(out_dir),
        "combined_csv": str(combined_csv),
        "score_json": str(PACKAGE_DIR / "outputs" / f"ai_variant_score_{label}.json"),
        "loss_csv": str(PACKAGE_DIR / "outputs" / f"microiou_loss_rows_{label}.csv"),
        "repeats": args.repeats,
        "cases": cases,
        "run_csvs": run_index,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "consensus_method": "row majority vote; ties use interval medoid without gold labels",
    }
    summary_path = out_dir / "consensus_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
