from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.process_mmsearch_end2end import (
    answer_candidates,
    has_query_image,
    process_rows,
    snapshot_dataset,
)


class ProcessMMSearchEnd2EndTest(unittest.TestCase):
    def test_has_query_image_rejects_empty_values(self) -> None:
        self.assertFalse(has_query_image(None))
        self.assertFalse(has_query_image(""))
        self.assertFalse(has_query_image({}))
        self.assertFalse(has_query_image([]))
        self.assertTrue(has_query_image({"bytes": b"abc"}))

    def test_answer_candidates_uses_gt_and_alternatives(self) -> None:
        answers = answer_candidates("grey", ["gray", "grey", ""])

        self.assertEqual(answers, ["grey", "gray"])

    def test_process_rows_filters_empty_query_images_and_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_image = root / "conference.png"
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(source_image)
            rows = [
                {
                    "sample_id": "technology_0",
                    "query": "Which days does this conference run from and to?",
                    "query_image": {"path": str(source_image), "bytes": None},
                    "area": "news",
                    "subfield": "technology",
                    "timestamp": "2024-09-29",
                    "gt_requery": "European Conference on Computer Vision 2024 exact date",
                    "gt_answer": "2024-09-29 to 2024-10-04",
                    "alternative_gt_answers": [],
                },
                {
                    "sample_id": "technology_1",
                    "query": "What color shirt did Sundar Pichai wear?",
                    "query_image": None,
                    "gt_answer": "grey",
                    "alternative_gt_answers": ["gray"],
                },
            ]

            count = process_rows(
                rows,
                output_jsonl=root / "MMSearch_end2end.jsonl",
                image_root=root / "images",
            )

            self.assertEqual(count, 1)
            output = json.loads((root / "MMSearch_end2end.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(output["question_id"], "technology_0")
            self.assertEqual(output["image_id"], "technology_0")
            self.assertEqual(output["question"], "Which days does this conference run from and to?")
            self.assertEqual(output["gold_answer"], "2024-09-29 to 2024-10-04")
            self.assertEqual(output["answer"], ["2024-09-29 to 2024-10-04"])
            self.assertEqual(output["answer_eval"], ["2024-09-29 to 2024-10-04"])
            self.assertEqual(output["area"], "news")
            self.assertEqual(output["subfield"], "technology")
            self.assertEqual(output["gt_requery"], "European Conference on Computer Vision 2024 exact date")
            self.assertTrue(Path(output["image_path"]).exists())

    def test_snapshot_dataset_downloads_end2end_parquet_only(self) -> None:
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return "/tmp/mmsearch_snapshot"

        original_module = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = SimpleNamespace(snapshot_download=fake_snapshot_download)
        try:
            snapshot_path = snapshot_dataset("CaraJ/MMSearch", cache_dir="/tmp/hf")
        finally:
            if original_module is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = original_module

        self.assertEqual(snapshot_path, Path("/tmp/mmsearch_snapshot"))
        self.assertEqual(calls[0]["repo_id"], "CaraJ/MMSearch")
        self.assertEqual(calls[0]["repo_type"], "dataset")
        self.assertEqual(calls[0]["allow_patterns"], ["end2end.parquet", "README.md"])

    def test_cli_processes_local_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_image = root / "finance.jpg"
            Image.new("RGB", (3, 3), color=(0, 255, 0)).save(source_image)
            parquet_path = root / "end2end.parquet"
            pq.write_table(
                pa.table(
                    {
                        "sample_id": ["finance_0"],
                        "query": ["What product launched?"],
                        "query_image": [[{"path": str(source_image), "bytes": None}]],
                        "image_search_result": [None],
                        "area": ["news"],
                        "subfield": ["finance"],
                        "timestamp": ["2024-08-15"],
                        "gt_requery": ["xAI product launch"],
                        "gt_answer": ["Grok-2"],
                        "alternative_gt_answers": [["Grok 2"]],
                    }
                ),
                parquet_path,
            )
            output_jsonl = root / "MMSearch_end2end.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/process_mmsearch_end2end.py",
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

            self.assertIn("Wrote 1 MMSearch end2end rows", result.stdout)
            output = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(output["question_id"], "finance_0")
            self.assertEqual(output["answer_eval"], ["Grok-2", "Grok 2"])
            self.assertTrue(Path(output["image_path"]).exists())


if __name__ == "__main__":
    unittest.main()
