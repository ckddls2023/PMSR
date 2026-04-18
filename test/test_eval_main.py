from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import Trajectory
from eval.main import _eval_answer, build_config_from_args, prediction_from_trajectory


class EvalMainTest(unittest.TestCase):
    def test_build_config_strips_vllm_model_prefix(self) -> None:
        args = argparse.Namespace(
            model="vllm:Qwen/Qwen2.5-VL-7B-Instruct",
            api_base="http://127.0.0.1:8000/v1/chat/completions",
            api_key="",
            max_tokens=128,
            temperature=0.0,
            timeout=10,
            retry=0,
            text_kb="",
            text_metadata="",
            text_embed_api_base="",
            pmsr_kb="",
            pmsr_metadata="",
            image_embed_api_base="",
            pmsr_text_embed_api_base="",
            pmsr_fusion="concat",
            return_images=True,
            itercount=1,
            topk=2,
            threshold=0.9,
            verbose=False,
        )

        config = build_config_from_args(args)

        self.assertEqual(config.model, "Qwen/Qwen2.5-VL-7B-Instruct")

    def test_prediction_preserves_dataset_identifiers_and_gold_answer(self) -> None:
        item = {
            "question_id": "fvqa_test_0",
            "image_id": "img0",
            "question": "Where is it?",
            "image_path": "/tmp/image.jpg",
            "gold_answer": "Las Vegas",
            "answer": ["Las Vegas"],
            "entity_text": "Las Vegas",
        }
        traj = Trajectory(question=item["question"], image_path=item["image_path"], final_answer="Las Vegas, Nevada")

        pred = prediction_from_trajectory(item, traj)

        self.assertEqual(pred["question_id"], "fvqa_test_0")
        self.assertEqual(pred["image_id"], "img0")
        self.assertEqual(pred["gold_answer"], "Las Vegas")
        self.assertTrue(pred["answer_eval"])

    def test_eval_answer_supports_list_and_range_targets(self) -> None:
        self.assertTrue(_eval_answer("It is used for urinary tract infections.", {"answer": ["urinary tract infections"]}))
        self.assertTrue(_eval_answer("The value is about 42 meters.", {"answer_eval": {"range": [40, 45]}}))
        self.assertFalse(_eval_answer("The value is about 30 meters.", {"answer_eval": {"range": [40, 45]}}))

    def test_singular_schema_module_reexports_record_types(self) -> None:
        from agents import Record, Trajectory as SchemaTrajectory

        self.assertEqual(Record.__name__, "Record")
        self.assertIs(SchemaTrajectory, Trajectory)


if __name__ == "__main__":
    unittest.main()
