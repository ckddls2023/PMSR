from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.process_infoseek_human_subset import (
    DEFAULT_PARQUET_URL,
    extract_answers,
    extract_question,
    process_rows,
    resolve_parquet_path,
)


class ProcessInfoSeekHumanSubsetTest(unittest.TestCase):
    def test_extract_question_reads_user_prompt_content(self) -> None:
        prompt = [{"role": "user", "content": "What is this landmark called?"}]

        self.assertEqual(extract_question(prompt), "What is this landmark called?")

    def test_extract_answers_uses_candidates_and_ground_truth(self) -> None:
        reward_model = {
            "candidate_answers": '["Nezu Shrine", "Nezu-jinja"]',
            "ground_truth": "Nezu Shrine",
        }

        gold_answer, answers = extract_answers(reward_model)

        self.assertEqual(gold_answer, "Nezu Shrine")
        self.assertEqual(answers, ["Nezu Shrine", "Nezu-jinja"])

    def test_process_rows_writes_images_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_image = root / "source.jpg"
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(source_image)
            rows = [
                {
                    "prompt": [{"role": "user", "content": "What is the architectural style of this building?"}],
                    "images": [{"path": str(source_image), "bytes": None}],
                    "reward_model": {
                        "candidate_answers": '["Ishi-no-ma-zukuri", "gongen-zukuri"]',
                        "ground_truth": "Ishi-no-ma-zukuri",
                    },
                    "data_id": "infoseek_human_0",
                    "image_id": "oven_04958159",
                    "data_source": "mmsearch_r1_infoseek_sub_2k",
                    "category": "infoseek",
                }
            ]

            count = process_rows(
                rows,
                output_jsonl=root / "InfoSeek_human_2k.jsonl",
                image_root=root / "images",
            )

            self.assertEqual(count, 1)
            output = json.loads((root / "InfoSeek_human_2k.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(output["question_id"], "infoseek_human_0")
            self.assertEqual(output["image_id"], "oven_04958159")
            self.assertEqual(output["question"], "What is the architectural style of this building?")
            self.assertEqual(output["gold_answer"], "Ishi-no-ma-zukuri")
            self.assertEqual(output["answer"], ["Ishi-no-ma-zukuri", "gongen-zukuri"])
            self.assertEqual(output["answer_eval"], ["Ishi-no-ma-zukuri", "gongen-zukuri"])
            self.assertEqual(output["data_source"], "mmsearch_r1_infoseek_sub_2k")
            self.assertTrue(Path(output["image_path"]).exists())

    def test_resolve_parquet_path_downloads_default_raw_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "subset.parquet"

            def fake_urlretrieve(url, filename):
                self.assertEqual(url, DEFAULT_PARQUET_URL)
                Path(filename).write_bytes(b"parquet")
                return filename, None

            with patch("scripts.process_infoseek_human_subset.urlretrieve", side_effect=fake_urlretrieve):
                parquet_path = resolve_parquet_path(None, DEFAULT_PARQUET_URL, destination)

            self.assertEqual(parquet_path, destination)
            self.assertEqual(destination.read_bytes(), b"parquet")

    def test_cli_processes_local_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_image = root / "source.jpg"
            Image.new("RGB", (3, 3), color=(0, 255, 0)).save(source_image)
            parquet_path = root / "mmsearch_r1_infoseek_sub_2k.parquet"
            pq.write_table(
                pa.table(
                    {
                        "prompt": [[{"role": "user", "content": "Who designed this building?"}]],
                        "images": [[{"path": str(source_image), "bytes": None}]],
                        "reward_model": [
                            {"candidate_answers": '["Frank Lloyd Wright"]', "ground_truth": "Frank Lloyd Wright"}
                        ],
                        "data_id": ["infoseek_human_1"],
                        "image_id": ["oven_00000001"],
                        "data_source": ["mmsearch_r1_infoseek_sub_2k"],
                        "category": ["infoseek"],
                    }
                ),
                parquet_path,
            )
            output_jsonl = root / "InfoSeek_human_2k.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/process_infoseek_human_subset.py",
                    "--parquet-path",
                    str(parquet_path),
                    "--output-jsonl",
                    str(output_jsonl),
                    "--image-root",
                    str(root / "images"),
                    "--cache-dir",
                    str(root / "cache"),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Wrote 1 InfoSeek Human rows", result.stdout)
            output = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(output["question_id"], "infoseek_human_1")
            self.assertEqual(output["answer_eval"], ["Frank Lloyd Wright"])
            self.assertTrue(Path(output["image_path"]).exists())


if __name__ == "__main__":
    unittest.main()
