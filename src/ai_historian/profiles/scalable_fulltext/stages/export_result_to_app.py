import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from ai_historian.pipeline.paths import resolve_run_root, sentence_step_dir, timeblock_step_dir

APP_EXPORT_SKIP_NAME_SUBSTRINGS = ("step12",)


def list_json_files(directory: Path, skip_name_substrings: Iterable[str] = ()) -> List[Path]:
    skip_tokens = tuple(str(token) for token in skip_name_substrings if str(token))
    return sorted(
        [
            path
            for path in Path(directory).glob("*.json")
            if path.is_file() and not any(token in path.name for token in skip_tokens)
        ],
        key=lambda path: path.name,
    )


def derive_app_export_dir_name(run_root: Path) -> str:
    run_name = Path(run_root).name.strip() or "result"
    if run_name.startswith("result_") and len(run_name) > len("result_"):
        return f"app_base_input_{run_name[len('result_'):]}"
    return f"app_base_input_{run_name}"


def export_result_to_app(run_root: Path) -> Tuple[Path, int, int]:
    sentence_source_dir = sentence_step_dir(run_root, 12)
    timeblock_source_dir = timeblock_step_dir(run_root, 14)

    if not sentence_source_dir.is_dir():
        raise FileNotFoundError(f"缺少 sentence 导出源目录: {sentence_source_dir}")
    if not timeblock_source_dir.is_dir():
        raise FileNotFoundError(f"缺少 timeblock 导出源目录: {timeblock_source_dir}")

    sentence_files = list_json_files(
        sentence_source_dir,
        skip_name_substrings=APP_EXPORT_SKIP_NAME_SUBSTRINGS,
    )
    timeblock_files = list_json_files(timeblock_source_dir)

    if not sentence_files:
        raise FileNotFoundError(f"在 {sentence_source_dir} 下未找到可导出的 JSON 文件。")
    if not timeblock_files:
        raise FileNotFoundError(f"在 {timeblock_source_dir} 下未找到可导出的 JSON 文件。")

    export_root = run_root / "export"
    target_root = export_root / derive_app_export_dir_name(run_root)
    sentence_target_dir = target_root / "sentence"
    timeblock_target_dir = target_root / "timeblock"

    if target_root.exists():
        shutil.rmtree(target_root)

    sentence_target_dir.mkdir(parents=True, exist_ok=True)
    timeblock_target_dir.mkdir(parents=True, exist_ok=True)

    for source_file in sentence_files:
        shutil.copy2(source_file, sentence_target_dir / source_file.name)

    for source_file in timeblock_files:
        shutil.copy2(source_file, timeblock_target_dir / source_file.name)

    return target_root, len(sentence_files), len(timeblock_files)


def main() -> None:
    run_root = resolve_run_root(sys.argv[1] if len(sys.argv) > 1 else None)
    target_root, sentence_count, timeblock_count = export_result_to_app(run_root)
    print(
        f"已导出到 {target_root} | sentence={sentence_count} | timeblock={timeblock_count}"
    )


if __name__ == "__main__":
    main()
