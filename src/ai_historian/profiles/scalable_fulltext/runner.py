"""Orchestrator for the scalable full-text profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_historian.pipeline.input_manifest import INPUT_MANIFEST_NAME
from ai_historian.pipeline.runner import RunRecorder, Stage, repository_root, run_stage

PROFILE = "ai_historian.profiles.scalable_fulltext"
PUBLIC = f"{PROFILE}.stages"
AGENT = f"{PROFILE}.agent_stages"
PIPELINE = [
    Stage("01", f"{PUBLIC}.step_01_text_preprocess"),
    Stage("02", f"{AGENT}.step_02_character_detection"),
    Stage("03", f"{AGENT}.step_03_time_info_extraction"),
    Stage("04", f"{AGENT}.step_04_description_detection"),
    Stage("05", f"{AGENT}.step_05_interlude_detection"),
    Stage("06", f"{PUBLIC}.step_06_timeblock_generation"),
    Stage("07", f"{AGENT}.step_07_timeblock_conversion"),
    Stage("08", f"{AGENT}.step_08_sequence_sorting"),
    Stage("09", f"{AGENT}.step_09_granularity_classification"),
    Stage("10", f"{PUBLIC}.agent_08_cross_text_temporal_propagation"),
    Stage(
        "11",
        f"{AGENT}.step_11_iso_normalization",
        env={
            "AIH_ISO_INPUT_STEP": "10",
            "AIH_ISO_OUTPUT_STEP": "11",
            "AIH_ISO_STEP_LABEL": "A9",
        },
    ),
    Stage("12", f"{PUBLIC}.step_12_cross_document_alignment"),
    Stage("13", f"{PUBLIC}.step_13_timeblock_iso_update"),
    Stage("14", f"{PUBLIC}.step_14_apply_summary"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AI Historian on full texts with runtime-scoped cross-document retrieval."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing UTF-8 .txt files.")
    parser.add_argument("--output", type=Path, required=True, help="Run output directory.")
    parser.add_argument(
        "--resume-from",
        choices=[stage.label for stage in PIPELINE],
        default="01",
        help="Resume at a numbered pipeline stage.",
    )
    parser.add_argument(
        "--through-step",
        choices=[stage.label for stage in PIPELINE],
        default="14",
        help="Stop after a numbered pipeline stage (useful for local validation).",
    )
    parser.add_argument(
        "--skip-crossdoc",
        action="store_true",
        help="Run the within-document path without A8 cross-document propagation.",
    )
    return parser


def run(args: argparse.Namespace, argv: list[str] | None = None) -> Path:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not list(input_dir.glob("*.txt")):
        raise SystemExit(f"No .txt files found in: {input_dir}")
    if not (input_dir / INPUT_MANIFEST_NAME).is_file():
        raise SystemExit(f"Missing {INPUT_MANIFEST_NAME} in: {input_dir}")

    labels = [stage.label for stage in PIPELINE]
    start = labels.index(args.resume_from)
    end = labels.index(args.through_step)
    if start > end:
        raise SystemExit("--resume-from must not come after --through-step")
    stages = PIPELINE[start : end + 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    root = repository_root()
    recorder = RunRecorder(
        profile="scalable_fulltext",
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
            elif stage.label == "10" and args.skip_crossdoc:
                stage = Stage(stage.label, stage.module, ("--skip-crossdoc",), stage.env)
                run_stage(stage, run_root=output_dir, root=root)
            else:
                run_stage(stage, run_root=output_dir, root=root)
            recorder.stage_completed(stage.label)
    except Exception as exc:
        recorder.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    recorder.finish("completed")
    if args.through_step == "14" and not (
        output_dir / "timeblock" / "step14output"
    ).is_dir():
        raise RuntimeError("Full-text pipeline completed without timeblock/step14output")
    print(f"AIH full-text run completed: {output_dir}")
    return output_dir
