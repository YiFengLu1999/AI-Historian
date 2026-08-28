#!/usr/bin/env python3
"""One-command orchestration for frozen checks and full paper regeneration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def run(
    command: list[str],
    *,
    env: dict[str, str],
    dry_run: bool,
    cwd: Path = ROOT,
) -> None:
    print("+", shlex.join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute frozen metrics or regenerate both paper experiments, "
            "including three-run consensus and final scoring."
        )
    )
    parser.add_argument(
        "mode",
        choices=["frozen", "full"],
        help="frozen performs deterministic scoring; full performs new model runs.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/paper-reproduction"))
    parser.add_argument("--provider", default=os.getenv("AIH_CHAT_PROVIDER", ""))
    parser.add_argument("--model", default=os.getenv("AIH_CHAT_MODEL", ""))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel-cases", type=int, default=1)
    parser.add_argument(
        "--skip-experiment1-direct",
        action="store_true",
        help="Run the Agent conditions while retaining the frozen Experiment 1 Direct LLM baseline.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the complete command plan.")
    return parser.parse_args()


def frozen(env: dict[str, str], dry_run: bool) -> None:
    recompute_env = {
        **env,
        "AIH_RECOMPUTED_ROOT": str(ROOT / "experiments" / "experiment-2" / "recomputed"),
    }
    commands = [
        ["node", "experiments/experiment-1/evaluation/score_ai_prefill_variant.js"],
        ["node", "experiments/experiment-2/code/build_human_accuracy.js"],
        ["node", "experiments/experiment-2/code/build_strict_total_html.js"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    ]
    for command in commands:
        run(command, env=recompute_env, dry_run=dry_run)


def full(args: argparse.Namespace, env: dict[str, str]) -> None:
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / stamp

    experiment1_evaluation = run_root / "experiment-1" / "evaluation_profile"
    experiment1_direct = run_root / "experiment-1" / "direct_llm"
    experiment2_direct = run_root / "experiment-2" / "direct_llm"
    experiment2_structured = run_root / "experiment-2" / "structured_llm"
    comparison_env = {
        **env,
        "AIH_RECOMPUTED_ROOT": str(run_root / "experiment-2" / "recomputed"),
    }

    run(
        [
            sys.executable,
            "experiments/experiment-1/code/run_experiment1_ai_only_consensus.py",
            "--repeats",
            str(args.repeats),
            "--output-dir",
            str(experiment1_evaluation),
            "--label",
            f"evaluation_profile_{stamp}",
            "--chat-provider",
            args.provider,
            "--model",
            args.model,
            "--env-file",
            str(args.env_file),
            "--parallel-cases",
            str(args.parallel_cases),
        ],
        env=env,
        dry_run=args.dry_run,
    )

    if not args.skip_experiment1_direct:
        run(
            [
                sys.executable,
                "experiments/experiment-1/direct-llm/run_direct_llm_baseline.py",
                "--runs",
                "1",
                "--output-dir",
                str(experiment1_direct),
                "--label",
                f"direct_llm_{stamp}",
            ],
            env=env,
            dry_run=args.dry_run,
        )
        run(
            [
                "node",
                "experiments/experiment-1/direct-llm/evaluation-scripts/score_direct_llm_variant.js",
                str(experiment1_direct / "tables" / "all_cases_direct_llm_prefill.csv"),
                f"direct_llm_{stamp}",
            ],
            env=env,
            dry_run=args.dry_run,
        )

    run(
        [
            sys.executable,
            "experiments/experiment-2/code/run_direct_llm.py",
            "--runs",
            str(args.repeats),
            "--model",
            args.model,
            "--output-dir",
            str(experiment2_direct),
        ],
        env=env,
        dry_run=args.dry_run,
    )
    run(
        [
            sys.executable,
            "experiments/experiment-2/code/run_structured_llm.py",
            "--runs",
            str(args.repeats),
            "--model",
            args.model,
            "--evidence-mode",
            "visible_only",
            "--output-dir",
            str(experiment2_structured),
        ],
        env=env,
        dry_run=args.dry_run,
    )
    run(
        ["node", "experiments/experiment-2/code/build_human_accuracy.js"],
        env=comparison_env,
        dry_run=args.dry_run,
    )
    run(
        ["node", "experiments/experiment-2/code/build_strict_total_html.js"],
        env=comparison_env,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        manifest = {
            "schema": "aih_full_reproduction_v1",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "provider": args.provider,
            "model": args.model,
            "repeats": args.repeats,
            "experiment_1_evaluation_profile": str(experiment1_evaluation),
            "experiment_1_direct": str(experiment1_direct) if not args.skip_experiment1_direct else "",
            "experiment_2_direct": str(experiment2_direct),
            "experiment_2_structured": str(experiment2_structured),
        }
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "reproduction_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Reproduction complete: {run_root}")


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    env = os.environ.copy()
    env_path = args.env_file.expanduser()
    if env_path.exists():
        env.update(read_env_file(env_path))
    if args.provider:
        env["AIH_CHAT_PROVIDER"] = args.provider
    if args.model:
        env["AIH_CHAT_MODEL"] = args.model
    env["ENV_FILE"] = str(env_path.resolve())

    if args.mode == "frozen":
        frozen(env, args.dry_run)
    else:
        from ai_historian.model_config import resolve_chat_config

        try:
            resolve_chat_config(env, require_api_key=not args.dry_run)
        except RuntimeError as exc:
            raise SystemExit(f"Full regeneration model configuration is invalid: {exc}") from exc
        full(args, env)


if __name__ == "__main__":
    main()
