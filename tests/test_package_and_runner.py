from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ai_historian
from ai_historian.pipeline.input_manifest import (
    load_input_manifest,
    source_metadata,
    stable_collection_uuid,
)
from ai_historian.pipeline.runner import SAFE_CONFIG_KEYS, RunRecorder, Stage
from ai_historian.profiles.evaluation.runner import (
    CROSS_DOCUMENT_STAGES,
    SUMMARY_STAGE,
    selected_stages,
)
from ai_historian.profiles.evaluation.runner import (
    STAGES as EVALUATION_STAGES,
)
from ai_historian.profiles.scalable_fulltext.runner import (
    PIPELINE as FULLTEXT_STAGES,
)
from ai_historian.profiles.scalable_fulltext.runner import (
    build_parser,
    run,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "ai_historian"


class PackageBoundaryTests(unittest.TestCase):
    def test_version_metadata_is_consistent(self):
        expected = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertEqual(ai_historian.__version__, expected)
        self.assertIn(expected, citation)
        self.assertIn(
            "repository-code: \"https://github.com/YiFengLu1999/AI-Historian\"",
            citation,
        )
        self.assertIn("preferred-citation:", citation)
        self.assertIn(expected, (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn(expected, (ROOT / "README.zh-CN.md").read_text(encoding="utf-8"))

    def test_ci_uses_repository_reproduction_script_and_supported_versions(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn('python-version: ["3.10", "3.13"]', workflow)
        self.assertIn("uv run python scripts/reproduce_paper.py frozen", workflow)
        self.assertNotIn("uv run aih-reproduce", workflow)

    def test_runtime_package_has_no_legacy_path_injection(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in PACKAGE_ROOT.rglob("*.py")
        )
        self.assertNotIn("sys.path", source)
        self.assertNotIn("AIHAgent_\u539f\u59cb", source)
        self.assertNotIn("aih_agent_fulltext", source)

    def test_fulltext_public_stages_use_shared_chat_completion_wrapper(self):
        for relative_path in (
            "profiles/scalable_fulltext/stages/step_12_cross_document_alignment.py",
            "profiles/scalable_fulltext/stages/step_14_apply_summary.py",
        ):
            source = (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("create_chat_completion(", source)
            self.assertNotIn(".chat.completions.create(", source)

    def test_repository_reproduction_is_not_a_wheel_command(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("aih-reproduce", pyproject)
        self.assertFalse((PACKAGE_ROOT / "cli" / "reproduce.py").exists())

    def test_evaluation_cross_document_stage_order(self):
        labels = [stage.label for stage in selected_stages(11, True, False)]
        self.assertEqual(labels[-5:], ["10A", "10B", "10C", "10D", "11"])

    def test_stage_modules_import_without_clients_or_filesystem_writes(self):
        evaluation = [*EVALUATION_STAGES, *CROSS_DOCUMENT_STAGES.values(), SUMMARY_STAGE]
        module_names = sorted({stage.module for stage in [*evaluation, *FULLTEXT_STAGES]})
        empty_credentials = {
            "DEEPSEEK_API_KEY": "",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "DASHSCOPE_API_KEY": "",
            "AIH_COMPATIBLE_API_KEY": "",
        }

        with (
            mock.patch.dict(os.environ, empty_credentials, clear=False),
            mock.patch.object(
                Path,
                "mkdir",
                side_effect=AssertionError("stage import attempted a filesystem write"),
            ),
        ):
            for module_name in module_names:
                importlib.import_module(module_name)

    def test_fulltext_rejects_reverse_stage_range(self):
        parser = build_parser()
        args = parser.parse_args(
            ["examples/input", "--output", "runs/test", "--resume-from", "02", "--through-step", "01"]
        )
        with self.assertRaisesRegex(SystemExit, "must not come after"):
            run(args, argv=["aih-fulltext", "--resume-from", "02", "--through-step", "01"])


class InputManifestTests(unittest.TestCase):
    def test_collection_uuid_is_location_independent_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            filename = "source.txt"
            (first / filename).write_text("same text\n", encoding="utf-8")
            (second / filename).write_text("same text\n", encoding="utf-8")
            documents = {filename: {"person": "刘邦", "title": "Synthetic Biography"}}

            first_uuid = stable_collection_uuid([first / filename], documents)
            second_uuid = stable_collection_uuid([second / filename], documents)
            self.assertEqual(first_uuid, second_uuid)

            (second / filename).write_text("changed text\n", encoding="utf-8")
            self.assertNotEqual(
                first_uuid,
                stable_collection_uuid([second / filename], documents),
            )

    def test_manifest_decouples_metadata_from_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            text_path = input_dir / "liu-bang-synthetic-biography.txt"
            text_path.write_text("汉元年，刘邦入关。\n", encoding="utf-8")
            (input_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "ai_historian_input_v1",
                        "documents": [
                            {
                                "file": text_path.name,
                                "person": "刘邦",
                                "title": "Synthetic Biography",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            documents = load_input_manifest(input_dir, [text_path])
            resolved = source_metadata(text_path, documents)
            self.assertEqual(resolved["source_person"], "刘邦")
            self.assertEqual(resolved["source_title"], "Synthetic Biography")

    def test_manifest_requires_exact_text_file_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            text_path = input_dir / "source.txt"
            text_path.write_text("text\n", encoding="utf-8")
            (input_dir / "manifest.json").write_text(
                '{"schema":"ai_historian_input_v1","documents":'
                '[{"file":"other.txt","person":"P","title":"T"}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing metadata"):
                load_input_manifest(input_dir, [text_path])


class RunManifestTests(unittest.TestCase):
    def test_manifest_covers_every_non_secret_root_configuration_key(self):
        configured_keys = {
            line.split("=", 1)[0].strip()
            for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "=" in line
        }
        non_secret_keys = {key for key in configured_keys if not key.endswith("API_KEY")}
        self.assertEqual(non_secret_keys - set(SAFE_CONFIG_KEYS), set())

    def test_manifest_hashes_inputs_without_recording_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            output = root / "output"
            inputs.mkdir()
            (inputs / "sample.txt").write_text("sample\n", encoding="utf-8")
            (inputs / "manifest.json").write_text("{}\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "AIH_CHAT_PROVIDER": "deepseek",
                    "AIH_CHAT_MODEL": "test-model",
                    "AIH_CROSSDOC_SCOPE_MAX_CASES": "24",
                    "DEEPSEEK_API_KEY": "secret-value",
                },
                clear=True,
            ):
                recorder = RunRecorder(
                    profile="evaluation",
                    input_dir=inputs,
                    output_dir=output,
                    argv=["aih", "inputs"],
                    stages=[Stage("01", "example.stage")],
                    root=ROOT,
                )
                recorder.finish("completed")

            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["configuration"],
                {
                    "AIH_CHAT_PROVIDER": "deepseek",
                    "AIH_CHAT_MODEL": "test-model",
                    "AIH_CROSSDOC_SCOPE_MAX_CASES": "24",
                },
            )
            self.assertEqual(manifest["stage_configuration"], {})
            self.assertEqual(
                [item["path"] for item in manifest["inputs"]],
                ["manifest.json", "sample.txt"],
            )
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["inputs"]))
            self.assertNotIn("secret-value", json.dumps(manifest))

    def test_manifest_records_non_secret_stage_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            output = root / "output"
            inputs.mkdir()
            (inputs / "sample.txt").write_text("sample\n", encoding="utf-8")

            recorder = RunRecorder(
                profile="fulltext",
                input_dir=inputs,
                output_dir=output,
                argv=["aih-fulltext", "inputs"],
                stages=[
                    Stage(
                        "11",
                        "example.stage",
                        env={
                            "AIH_ISO_INPUT_STEP": "10",
                            "AIH_ISO_OUTPUT_STEP": "11",
                        },
                    )
                ],
                root=ROOT,
            )
            recorder.finish("completed")

            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["stage_configuration"],
                {"11": {"AIH_ISO_INPUT_STEP": "10", "AIH_ISO_OUTPUT_STEP": "11"}},
            )


if __name__ == "__main__":
    unittest.main()
