from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_historian.profiles.evaluation.stages import (
    step_01_text_preprocess as preprocess,
)
from ai_historian.profiles.evaluation.stages import (
    step_06_timeblock_generation as timeblocks,
)

ROOT = Path(__file__).resolve().parents[1]


class DeterministicStageTests(unittest.TestCase):
    def test_sentence_split_preserves_quoted_punctuation(self):
        text = "高祖说：“继续前进。”随后军队出发。第二日抵达。"
        self.assertEqual(
            preprocess.split_sentences(text),
            ["高祖说：“继续前进。”", "随后军队出发。", "第二日抵达。"],
        )

    def test_timeblock_starts_at_explicit_temporal_anchor(self):
        records = []
        for index, (sentence, time_text) in enumerate(
            [("汉元年，军队出发。", "汉元年"), ("随后抵达。", ""), ("汉二年，返回。", "汉二年")],
            start=1,
        ):
            records.append(
                {
                    "number": f"00000000-0000-0000-0000-000000000000.1.1.{index}",
                    "sentence": sentence,
                    "Original_time_information": {"exist": bool(time_text), "OTI": time_text},
                    "sink": {"Is_it_sinking": False},
                    "Interlude": False,
                }
            )
        blocks = timeblocks.extract_timeblocks(records)["TMB"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["Conversion information"]["time_information_original"], "汉元年")
        self.assertTrue(blocks[0]["timeblock_range"].endswith(".1.1.2"))
        self.assertEqual(blocks[1]["Conversion information"]["time_information_original"], "汉二年")

    def test_timeblock_json_round_trip(self):
        payload = {"TMB": [{"ID": "x", "iso_range": "-0201-10-01to-0200-09-01"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.json"
            timeblocks.save_json(payload, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
