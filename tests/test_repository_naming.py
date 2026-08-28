from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE_PREFIXES = (
    "src/",
    "tests/",
    "scripts/",
    "experiments/experiment-1/code/",
    "experiments/experiment-1/evaluation/",
    "experiments/experiment-1/direct-llm/evaluation-scripts/",
    "experiments/experiment-2/code/",
)
CODE_FILES = {
    "experiments/experiment-1/direct-llm/direct_llm_agent_postprocess.py",
    "experiments/experiment-1/direct-llm/run_direct_llm_baseline.py",
    "experiments/experiment-1/direct-llm/run_direct_llm_baseline.sh",
    "experiments/experiment-1/results/multimodel/analysis/experiment1_multimodel_analysis.mjs",
}
FROZEN_OUTPUT_PREFIXES = (
    "experiments/experiment-1/direct-llm-results/generated_results_",
    "experiments/experiment-1/results/generated_results_",
    "experiments/experiment-2/results/direct-llm/run_",
    "experiments/experiment-2/results/structured-llm/run_",
)


def tracked_files() -> list[str]:
    if not (ROOT / ".git").exists():
        raise unittest.SkipTest("repository naming requires a Git checkout")
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    return [line for line in output.splitlines() if line]


def is_public_path(path: str) -> bool:
    return not (
        path.startswith(CODE_PREFIXES)
        or path in CODE_FILES
        or path.startswith(FROZEN_OUTPUT_PREFIXES)
    )


class RepositoryNamingTests(unittest.TestCase):
    def test_reader_facing_paths_use_kebab_case(self):
        violations = []
        for path in tracked_files():
            if not is_public_path(path):
                continue
            parts = Path(path).parts
            directory_parts = parts[:-1]
            filename = parts[-1]
            if any("_" in part for part in directory_parts):
                violations.append(path)
                continue
            if "_" in filename or re.search(r"experiment\d", filename, re.IGNORECASE):
                violations.append(path)
        self.assertEqual(violations, [], "non-kebab public paths:\n" + "\n".join(violations))

    def test_model_configuration_has_one_template_and_no_removed_variables(self):
        templates = sorted(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*.env.example")
            if ".git" not in path.parts
        )
        self.assertEqual(templates, [".env.example"])

        checked_paths = [
            ROOT / ".env.example",
            ROOT / "README.md",
            ROOT / "README.zh-CN.md",
            ROOT / "docs",
            ROOT / "scripts",
            ROOT / "src",
            ROOT / "experiments" / "experiment-1" / "code",
            ROOT / "experiments" / "experiment-1" / "direct-llm",
            ROOT / "experiments" / "experiment-2" / "code",
        ]
        sources = []
        for path in checked_paths:
            files = path.rglob("*") if path.is_dir() else [path]
            for file_path in files:
                if file_path.is_file() and file_path.suffix in {"", ".md", ".py", ".sh", ".example"}:
                    sources.append(file_path.read_text(encoding="utf-8"))
        combined = "\n".join(sources)
        for removed in ("AIH_API_KEY", "AIH_BASE_URL", "AIH_AGENT_MODEL"):
            self.assertNotIn(removed, combined)


if __name__ == "__main__":
    unittest.main()
