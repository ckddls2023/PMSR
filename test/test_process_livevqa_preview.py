from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.process_livevqa_preview import (
    load_detail_map,
    process_entries,
    resolve_image_path,
    snapshot_dataset,
)


class ProcessLiveVQAPreviewTest(unittest.TestCase):
    def test_load_detail_map_accepts_list_and_merges_by_sample_id(self) -> None:
        details = [
            {"sample_id": "Movies_2_2", "topic": "Oscars host returns", "context": "News context"},
            {"sample_id": "Sports_1_1", "topic": "Match result"},
        ]

        detail_map = load_detail_map(details)

        self.assertEqual(detail_map["Movies_2_2"]["topic"], "Oscars host returns")
        self.assertEqual(detail_map["Movies_2_2"]["context"], "News context")

    def test_resolve_image_path_handles_original_absolute_query_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "image"
            image_dir.mkdir()
            expected = image_dir / "859587b317_Conan.jpg"
            Image.new("RGB", (3, 3)).save(expected)

            image_path = resolve_image_path(root, "/mnt/nvme0/bench/image/859587b317_Conan.jpg")

            self.assertEqual(image_path, expected)

    def test_process_entries_writes_images_and_jsonl_with_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "image"
            image_dir.mkdir()
            source_image = image_dir / "859587b317_Conan.jpg"
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(source_image)
            output_jsonl = root / "LiveVQA_test.jsonl"
            output_images = root / "processed_images"
            entries = [
                {
                    "sample_id": "Movies_2_2",
                    "query": "Based on the provided image, when will this individual be hosting the event again?",
                    "query_image": "/mnt/nvme0/bench/image/859587b317_Conan.jpg",
                    "gt_answer": "March 15, 2026",
                }
            ]
            detail_map = {
                "Movies_2_2": {
                    "sample_id": "Movies_2_2",
                    "topic": "Conan O'Brien to return as Oscars host",
                    "context": "The ceremony is scheduled for March 15, 2026.",
                }
            }

            count = process_entries(
                entries,
                dataset_root=root,
                output_jsonl=output_jsonl,
                image_root=output_images,
                detail_map=detail_map,
            )

            self.assertEqual(count, 1)
            output = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(output["question_id"], "Movies_2_2")
            self.assertEqual(output["image_id"], "859587b317_Conan")
            self.assertEqual(output["question"], entries[0]["query"])
            self.assertEqual(output["gold_answer"], "March 15, 2026")
            self.assertEqual(output["answer"], ["March 15, 2026"])
            self.assertEqual(output["answer_eval"], ["March 15, 2026"])
            self.assertEqual(output["topic"], "Conan O'Brien to return as Oscars host")
            self.assertEqual(output["context"], "The ceremony is scheduled for March 15, 2026.")
            self.assertTrue(Path(output["image_path"]).exists())
            self.assertEqual(Path(output["image_path"]).parent, output_images)

    def test_snapshot_dataset_downloads_livevqa_files_only(self) -> None:
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return "/tmp/livevqa_snapshot"

        original_module = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = SimpleNamespace(snapshot_download=fake_snapshot_download)
        try:
            snapshot_path = snapshot_dataset("ONE-Lab/LiveVQA-Research-Preview", cache_dir="/tmp/hf")
        finally:
            if original_module is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = original_module

        self.assertEqual(snapshot_path, Path("/tmp/livevqa_snapshot"))
        self.assertEqual(calls[0]["repo_id"], "ONE-Lab/LiveVQA-Research-Preview")
        self.assertEqual(calls[0]["repo_type"], "dataset")
        self.assertEqual(calls[0]["cache_dir"], "/tmp/hf")
        self.assertEqual(calls[0]["allow_patterns"], ["qa.json", "qa_detailed.json", "image/**", "README.md"])

    def test_cli_uses_local_qa_json_without_snapshot_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "image"
            image_dir.mkdir()
            Image.new("RGB", (4, 4), color=(0, 255, 0)).save(image_dir / "news.jpg")
            qa_json = root / "qa.json"
            qa_json.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "News_1_1",
                            "query": "What is shown?",
                            "query_image": "/mnt/nvme0/bench/image/news.jpg",
                            "gt_answer": "A news image",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            details_json = root / "qa_detailed.json"
            details_json.write_text(json.dumps([{"sample_id": "News_1_1", "topic": "News"}]), encoding="utf-8")
            output_jsonl = root / "LiveVQA_test.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/process_livevqa_preview.py",
                    "--qa-json",
                    str(qa_json),
                    "--qa-detailed-json",
                    str(details_json),
                    "--output-jsonl",
                    str(output_jsonl),
                    "--image-root",
                    str(root / "processed_images"),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Wrote 1 LiveVQA rows", result.stdout)
            output = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(output["question_id"], "News_1_1")
            self.assertEqual(output["topic"], "News")
            self.assertTrue(Path(output["image_path"]).exists())


if __name__ == "__main__":
    unittest.main()
