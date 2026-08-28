#!/usr/bin/env python3
"""A8: Cross-text temporal propagation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from ai_historian.pipeline.paths import PROJECT_ROOT, resolve_run_root

INTERNAL_STAGES = [
    ("10A", "ai_historian.profiles.scalable_fulltext.agent_stages.step_10_tm_generation"),
    ("10B", "ai_historian.profiles.scalable_fulltext.agent_stages.step_10b_cross_document_prealign"),
    ("10C", "ai_historian.profiles.scalable_fulltext.agent_stages.step_10c_single_document_stabilize"),
    ("10D", "ai_historian.profiles.scalable_fulltext.agent_stages.step_10d_crossdoc_temporal_graph"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="A8: Cross-text temporal propagation.")
    parser.add_argument("run_root", help="流水线运行根目录。")
    parser.add_argument(
        "--skip-crossdoc",
        action="store_true",
        help="单文档模式：跳过跨文档内部阶段 10B/10D。",
    )
    parser.add_argument(
        "--legacy-crossdoc",
        action="store_true",
        help="使用旧的全文 10B 跨文档入口；仅用于调试，正式全量默认使用 runtime scope 高召回入口。",
    )
    args = parser.parse_args()

    run_root = resolve_run_root(args.run_root).resolve()
    stages = [
        item
        for item in INTERNAL_STAGES
        if not args.skip_crossdoc or item[0] not in {"10B", "10D"}
    ]

    print("A8 | Cross-text temporal propagation")
    for index, (label, module_name) in enumerate(stages, 1):
        print(f"A8 | {index}/{len(stages)} | START | {label}")
        env = os.environ.copy()
        if label == "10B":
            if args.legacy_crossdoc:
                env["AIH_CROSSDOC_SCOPE_STRATEGY"] = "legacy"
            else:
                env.setdefault("AIH_CROSSDOC_SCOPE_STRATEGY", "runtime_episode_packet")
                env.setdefault("AIH_CROSSDOC_SCOPE_MODE", "episode_packet")
                env.setdefault("AIH_CROSSDOC_SCOPE_SELECTOR", "lexical")
                env.setdefault("AIH_CROSSDOC_SCOPE_CONTEXT_PAD", "1")
                env.setdefault("AIH_CROSSDOC_SCOPE_ANCHOR_SEARCH", "16")
                env.setdefault("AIH_CROSSDOC_SCOPE_TOP_K_PER_PAIR", "30")
                env.setdefault("AIH_CROSSDOC_SCOPE_MAX_CASES", "60")
                env.setdefault("AIH_CROSSDOC_SCOPE_MIN_SCORE", "0.00")
                env.setdefault("AIH_CROSSDOC_CLEAR_EXISTING", "1")
                env.setdefault("AIH_CROSSDOC_PREALIGN_MIN_EPISODE_CONF", "0.55")
                env.setdefault("AIH_CROSSDOC_RECALL_ACCEPT_WEAK_CONTEXT", "1")
                env.setdefault("AIH_CROSSDOC_WEAK_CONTEXT_MIN_CONF", "0.35")
        subprocess.run(
            [sys.executable, "-m", module_name, str(run_root)],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )
        print(f"A8 | {index}/{len(stages)} | DONE | {label}")

    if not (run_root / "timeblock" / "step10output").is_dir():
        raise RuntimeError("A8 完成但未生成 timeblock/step10output。")


if __name__ == "__main__":
    main()
