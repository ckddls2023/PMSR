from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.process_fvqa_test import (
    extract_answers,
    extract_question,
    process_rows,
    snapshot_dataset,
)


class ProcessFVQATest(unittest.TestCase):
    def test_extract_question_reads_user_prompt_content(self) -> None:
        prompt = [{"content": "What country does this building belong to?", "role": "user"}]

        self.assertEqual(extract_question(prompt), "What country does this building belong to?")

    def test_extract_answers_prefers_candidate_answers_and_keeps_ground_truth(self) -> None:
        reward_model = {
            "candidate_answers": '["Malta", "mt", "Republic of Malta"]',
            "ground_truth": "Malta",
            "style": "rule",
        }

        gold_answer, answers = extract_answers(reward_model)

        self.assertEqual(gold_answer, "Malta")
        self.assertEqual(answers, ["Malta", "mt", "Republic of Malta"])

    def test_process_rows_writes_images_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_image = tmp / "source.jpg"
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(source_image)
            output_jsonl = tmp / "fvqa_test.jsonl"
            image_root = tmp / "images"
            rows = [
                {
                    "prompt": [{"content": "Where is this sign located?", "role": "user"}],
                    "images": [{"path": str(source_image), "bytes": None}],
                    "reward_model": {"candidate_answers": '["vegas"]', "ground_truth": "vegas"},
                    "data_source": "mmsearch_r1/fvqa_test",
                    "image_urls": None,
                    "data_id": "fvqa_test_0",
                    "category": "search_required",
                }
            ]

            count = process_rows(rows, output_jsonl=output_jsonl, image_root=image_root, split="test")

            self.assertEqual(count, 1)
            output = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(output["question"], "Where is this sign located?")
            self.assertEqual(output["gold_answer"], "vegas")
            self.assertEqual(output["answer"], ["vegas"])
            self.assertEqual(output["answer_eval"], ["vegas"])
            self.assertEqual(output["question_id"], "fvqa_test_0")
            self.assertEqual(output["image_id"], "fvqa_test_0")
            self.assertEqual(output["data_id"], "fvqa_test_0")
            self.assertEqual(output["category"], "search_required")
            self.assertTrue(Path(output["image_path"]).exists())
            self.assertEqual(Path(output["image_path"]).parent, image_root)

    def test_process_rows_accepts_json_encoded_huggingface_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_buffer = BytesIO()
            Image.new("RGB", (3, 3), color=(0, 255, 0)).save(image_buffer, format="PNG")
            output_jsonl = tmp / "fvqa_test.jsonl"
            rows = [
                {
                    "prompt": json.dumps([{"content": "What is the name of the system shown?", "role": "user"}]),
                    "images": json.dumps([{"bytes": list(image_buffer.getvalue())}]),
                    "reward_model": json.dumps(
                        {"candidate_answers": "[]", "ground_truth": "namus", "style": "rule"}
                    ),
                    "data_source": "mmsearch_r1/fvqa_test",
                    "image_urls": "[]",
                    "data_id": "fvqa_test_42",
                    "category": "search_free",
                }
            ]

            count = process_rows(rows, output_jsonl=output_jsonl, image_root=tmp / "images", split="test")

            self.assertEqual(count, 1)
            output = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(output["question"], "What is the name of the system shown?")
            self.assertEqual(output["gold_answer"], "namus")
            self.assertEqual(output["answer_eval"], ["namus"])
            self.assertEqual(output["candidate_answers"], ["namus"])
            self.assertTrue(Path(output["image_path"]).exists())

    def test_snapshot_dataset_downloads_only_requested_split_parquet(self) -> None:
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return "/tmp/fvqa_snapshot"

        original_module = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = SimpleNamespace(snapshot_download=fake_snapshot_download)
        try:
            snapshot_path = snapshot_dataset("lmms-lab/FVQA", split="test", cache_dir="/tmp/hf")
        finally:
            if original_module is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = original_module

        self.assertEqual(snapshot_path, Path("/tmp/fvqa_snapshot"))
        self.assertEqual(calls[0]["repo_id"], "lmms-lab/FVQA")
        self.assertEqual(calls[0]["repo_type"], "dataset")
        self.assertEqual(calls[0]["cache_dir"], "/tmp/hf")
        self.assertEqual(calls[0]["allow_patterns"], ["fvqa_test.parquet", "README.md"])


if __name__ == "__main__":
    unittest.main()
