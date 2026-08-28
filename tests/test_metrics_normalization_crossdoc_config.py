from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "experiments" / "experiment-1" / "evaluation" / "score_ai_prefill_variant.js"


def reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def run_microiou(gold: dict[str, str], prediction: dict[str, str]) -> dict[str, object]:
    script = r"""
const scorer = require(process.argv[1]);
const gold = JSON.parse(process.argv[2]);
const prediction = JSON.parse(process.argv[3]);
const key = `${gold.case_id}::${gold.part_id}::${gold.item_no}::${gold.sentence_id}`;
const windows = scorer.buildGoldWindows([gold]);
const result = scorer.scoreRows([prediction], new Map([[key, gold]]), windows);
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(SCORER), json.dumps(gold), json.dumps(prediction)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class MicroIoUTests(unittest.TestCase):
    def setUp(self):
        self.gold = {
            "case_id": "T",
            "part_id": "T",
            "item_no": "1",
            "sentence_id": "1.1.1",
            "state": "time_range",
            "iso_start": "-0200-01",
            "iso_end": "-0200-06",
            "iso_range": "-0200-01to-0200-06",
        }

    def test_exact_range_has_unit_microiou(self):
        result = run_microiou(self.gold, dict(self.gold))
        self.assertEqual(result["intersectionMonths"], 6)
        self.assertEqual(result["unionMonths"], 6)
        self.assertEqual(result["microIoU"], 1)

    def test_partial_range_uses_month_level_intersection_and_union(self):
        prediction = dict(self.gold, iso_start="-0200-03", iso_end="-0200-08")
        result = run_microiou(self.gold, prediction)
        self.assertEqual(result["intersectionMonths"], 4)
        self.assertEqual(result["unionMonths"], 6)
        self.assertAlmostEqual(result["microIoU"], 2 / 3)


class TemporalNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonicalizer = reload_module(
            "ai_historian.pipeline.time_canonicalizer"
        )

    def test_bare_month_inherits_previous_regnal_year(self):
        self.assertEqual(
            self.canonicalizer.normalize_experiment1_tm("四月", "汉高祖二年冬"),
            "汉高祖二年四月",
        )

    def test_bce_alias_is_normalized_to_evaluation_regnal_time(self):
        self.assertEqual(
            self.canonicalizer.normalize_experiment1_tm("公元前206年十二月"),
            "汉高祖元年十二月",
        )


class CrossDocumentConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(sys, "argv", ["test"]):
            cls.crossdoc = reload_module(
                "ai_historian.profiles.evaluation.stages.step_10d_crossdoc_temporal_graph"
            )
        cls.lookup = {
            "汉高祖元年": "-0206-10",
            "汉高祖二年": "-0205-10",
            "汉高祖三年": "-0204-10",
        }

    def test_supported_interval_is_accepted(self):
        accepted, reason, normalized = self.crossdoc.validate_interval_evidence(
            {
                "relation": "contained_in_source_interval",
                "start_tm": "汉高祖二年",
                "end_tm": "汉高祖三年",
            },
            {"previous_anchor_iso": "-0206-10", "next_anchor_iso": ""},
            self.lookup,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")
        self.assertEqual(normalized["start_iso"], "-0205-10")

    def test_reversed_interval_is_rejected(self):
        accepted, reason, _ = self.crossdoc.validate_interval_evidence(
            {
                "relation": "contained_in_source_interval",
                "start_tm": "汉高祖三年",
                "end_tm": "汉高祖二年",
            },
            {"previous_anchor_iso": "", "next_anchor_iso": ""},
            self.lookup,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "interval_start_after_end")


class ConfigurationLoadingTests(unittest.TestCase):
    def test_all_supported_providers_use_their_own_credential_group(self):
        config = reload_module("ai_historian.model_config")
        cases = {
            "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
            "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
            "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
            "gemini": ("GEMINI_API_KEY", "GEMINI_BASE_URL"),
            "dashscope": ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"),
            "compatible": ("AIH_COMPATIBLE_API_KEY", "AIH_COMPATIBLE_BASE_URL"),
        }
        for provider, (key_env, base_env) in cases.items():
            with self.subTest(provider=provider):
                env = {
                    "AIH_CHAT_PROVIDER": provider,
                    "AIH_CHAT_MODEL": f"{provider}-test-model",
                    key_env: f"{provider}-test-key",
                    base_env: f"https://{provider}.example/v1",
                }
                if provider != "openai":
                    env["OPENAI_API_KEY"] = "unrelated-openai-key"
                resolved = config.resolve_chat_config(env)
                self.assertEqual(resolved.api_key_env, key_env)
                self.assertEqual(resolved.api_key, f"{provider}-test-key")
                self.assertEqual(resolved.base_url_env, base_env)
                self.assertEqual(resolved.base_url, f"https://{provider}.example/v1")

    def test_provider_native_configuration_sets_model_and_endpoint(self):
        env = {
            "AIH_CHAT_PROVIDER": "anthropic",
            "AIH_CHAT_MODEL": "claude-test-model",
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "ANTHROPIC_BASE_URL": "https://anthropic.example/v1/",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = reload_module("ai_historian.model_config")
            resolved = config.resolve_chat_config()
        self.assertEqual(resolved.provider, "anthropic")
        self.assertEqual(resolved.model, "claude-test-model")
        self.assertEqual(resolved.api_key_env, "ANTHROPIC_API_KEY")
        self.assertEqual(resolved.base_url, "https://anthropic.example/v1/")

    def test_selected_provider_does_not_fall_back_to_another_key(self):
        env = {
            "AIH_CHAT_PROVIDER": "deepseek",
            "AIH_CHAT_MODEL": "deepseek-test-model",
            "OPENAI_API_KEY": "wrong-provider-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = reload_module("ai_historian.model_config")
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                config.resolve_chat_config()

    def test_model_is_required_and_not_inferred_from_provider(self):
        env = {
            "AIH_CHAT_PROVIDER": "openai",
            "OPENAI_API_KEY": "openai-test-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = reload_module("ai_historian.model_config")
            with self.assertRaisesRegex(RuntimeError, "AIH_CHAT_MODEL"):
                config.resolve_chat_config()

    def test_provider_aliases_are_rejected(self):
        env = {
            "AIH_CHAT_PROVIDER": "qwen",
            "AIH_CHAT_MODEL": "qwen-test-model",
            "DASHSCOPE_API_KEY": "dashscope-test-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = reload_module("ai_historian.model_config")
            with self.assertRaisesRegex(RuntimeError, "Unsupported AIH_CHAT_PROVIDER"):
                config.resolve_chat_config()

    def test_anthropic_uses_prompt_directed_json(self):
        env = {
            "AIH_CHAT_PROVIDER": "anthropic",
            "AIH_CHAT_MODEL": "claude-test-model",
            "ANTHROPIC_API_KEY": "anthropic-test-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = reload_module("ai_historian.model_config")
            kwargs = config.prepare_chat_request_kwargs(
                model="claude-test-model",
                response_format={"type": "json_object"},
            )
        self.assertNotIn("response_format", kwargs)

    def test_compatible_endpoint_retries_without_response_format(self):
        env = {
            "AIH_CHAT_PROVIDER": "compatible",
            "AIH_CHAT_MODEL": "compatible-test-model",
            "AIH_COMPATIBLE_API_KEY": "compatible-test-key",
            "AIH_COMPATIBLE_BASE_URL": "https://compatible.example/v1",
        }
        completion = object()
        client = mock.Mock()
        client.chat.completions.create.side_effect = [
            RuntimeError("response_format is not supported"),
            completion,
        ]
        with mock.patch.dict(os.environ, env, clear=True):
            config = reload_module("ai_historian.model_config")
            result = config.create_chat_completion(
                client,
                model="compatible-test-model",
                messages=[{"role": "user", "content": "Return JSON."}],
                response_format={"type": "json_object"},
            )

        self.assertIs(result, completion)
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        second_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        self.assertIn("response_format", first_kwargs)
        self.assertNotIn("response_format", second_kwargs)

    def test_embedding_credentials_do_not_fall_back_to_chat_key(self):
        env = {
            "OPENAI_API_KEY": "chat-only-key",
            "AIH_EMBED_BASE_URL": "https://embedding.example/v1",
            "AIH_EMBED_MODEL": "embedding-test-model",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = reload_module("ai_historian.model_config")
            with self.assertRaisesRegex(RuntimeError, "AIH_EMBED_API_KEY"):
                config.resolve_embedding_config(required=True)


if __name__ == "__main__":
    unittest.main()
