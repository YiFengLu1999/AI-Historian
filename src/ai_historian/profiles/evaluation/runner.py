"""Orchestrator for the paper-evaluation profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_historian.pipeline.input_manifest import INPUT_MANIFEST_NAME
from ai_historian.pipeline.runner import RunRecorder, Stage, repository_root, run_stage

PREFIX = "ai_historian.profiles.evaluation.stages"
STAGES = [
    Stage("01", f"{PREFIX}.step_01_text_preprocess"),
    Stage("02", f"{PREFIX}.step_02_character_detection"),
    Stage("03", f"{PREFIX}.step_03_time_info_extraction"),
    Stage("04", f"{PREFIX}.step_04_description_detection"),
    Stage("05", f"{PREFIX}.step_05_interlude_detection"),
    Stage("06", f"{PREFIX}.step_06_timeblock_generation"),
    Stage("07", f"{PREFIX}.step_07_timeblock_conversion"),
    Stage("08", f"{PREFIX}.step_08_sequence_sorting"),
    Stage("09", f"{PREFIX}.step_09_granularity_classification"),
    Stage("10A", f"{PREFIX}.step_10_tm_generation"),
    Stage("10C", f"{PREFIX}.step_10c_single_document_stabilize"),
    Stage("11", f"{PREFIX}.step_11_iso_normalization"),
]
CROSS_DOCUMENT_STAGES = {
    "10B": Stage("10B", f"{PREFIX}.step_10b_cross_document_prealign"),
    "10D": Stage("10D", f"{PREFIX}.step_10d_crossdoc_temporal_graph"),
}
SUMMARY_STAGE = Stage("14", f"{PREFIX}.step_14_apply_summary")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paper-evaluation AI Historian profile."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing UTF-8 .txt files.")
    parser.add_argument("--output", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--through-step", type=int, choices=range(1, 12), default=11)
    parser.add_argument(
        "--cross-document",
        action="store_true",
        help="Enable evidence-verified cross-document alignment before normalization.",
    )
    parser.add_argument(
        "--summaries",
        action="store_true",
        help="Generate readable summaries after Step 11.",
    )
    return parser


def selected_stages(through_step: int, cross_document: bool, summaries: bool) -> list[Stage]:
    selected: list[Stage] = []
    for stage in STAGES:
        numeric = int(stage.label[:2])
        if numeric > through_step:
            continue
        if stage.label == "10C" and cross_document:
            selected.append(CROSS_DOCUMENT_STAGES["10B"])
        selected.append(stage)
        if stage.label == "10C" and cross_document:
            selected.append(CROSS_DOCUMENT_STAGES["10D"])
    if summaries:
        selected.append(SUMMARY_STAGE)
    return selected


def run(args: argparse.Namespace, argv: list[str] | None = None) -> Path:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not list(input_dir.glob("*.txt")):
        raise SystemExit(f"No .txt files found in: {input_dir}")
    if not (input_dir / INPUT_MANIFEST_NAME).is_file():
        raise SystemExit(f"Missing {INPUT_MANIFEST_NAME} in: {input_dir}")
    if args.summaries and args.through_step < 11:
        raise SystemExit("--summaries requires --through-step 11")

    stages = selected_stages(args.through_step, args.cross_document, args.summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = repository_root()
    recorder = RunRecorder(
        profile="evaluation",
        input_dir=input_dir,
        output_dir=output_dir,
        argv=argv or sys.argv,
        stages=stages,
        root=root,
    )
    try:
        for stage in stages:
            if stage.label == "01":
                stage = Stage(stage.label, stage.module, (str(output_dir),), stage.env)
                run_stage(stage, run_root=input_dir, root=root)
            else:
                run_stage(stage, run_root=output_dir, root=root)
            recorder.stage_completed(stage.label)
    except Exception as exc:
        recorder.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    recorder.finish("completed")
    print(f"AIH evaluation run completed: {output_dir}")
    return output_dir
