from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import Evidence, Record, SearchResult, Trajectory
from eval.main import _eval_answer, build_config_from_args, output_from_trajectory


class EvalMainTest(unittest.TestCase):
    def test_main_imports_metrics_from_metric_eval(self) -> None:
        import eval.main as eval_main
        import eval.metric_eval as metric_eval

        self.assertIs(eval_main.evaluate_accuracy, metric_eval.evaluate_accuracy)
        self.assertIs(eval_main.evaluate_recall, metric_eval.evaluate_recall)

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

    def test_output_saves_direct_trajectory_with_dataset_metadata(self) -> None:
        item = {
            "question_id": "fvqa_test_0",
            "image_id": "img0",
            "question": "Where is it?",
            "image_path": "/tmp/image.jpg",
            "gold_answer": "Las Vegas",
            "answer_eval": ["Las Vegas", "Vegas"],
            "entity_text": "Las Vegas",
        }
        traj = Trajectory(question=item["question"], image_path=item["image_path"], final_answer="Las Vegas, Nevada")
        traj.records.append(
            Record(
                step=0,
                local_query="Question: Where is it?\nA casino building.",
                global_query="Question: Where is it?\nA casino building.",
                text_results=[
                    SearchResult(Evidence(source="text", modality="text", title="Las Vegas", text="City in Nevada."))
                ],
                image_results=[
                    SearchResult(Evidence(source="pmsr", modality="image", image_path="/tmp/ref.jpg", caption="Las Vegas Strip."))
                ],
                reasoning="The image and retrieved evidence point to Las Vegas.",
                elapsed=1.25,
            )
        )

        output = output_from_trajectory(item, traj)

        self.assertEqual(output["question_id"], "fvqa_test_0")
        self.assertEqual(output["image_id"], "img0")
        self.assertEqual(output["gold_answer"], "Las Vegas")
        self.assertEqual(output["answer_eval"], ["Las Vegas", "Vegas"])
        self.assertNotIn("prediction", output)
        self.assertNotIn("knowledge", output)
        self.assertNotIn("total_pred", output)
        self.assertEqual(output["trajectory"]["final_answer"], "Las Vegas, Nevada")
        self.assertEqual(
            output["trajectory"]["records"][0]["reasoning"],
            "The image and retrieved evidence point to Las Vegas.",
        )
        self.assertIn("City in Nevada.", output["trajectory"]["all_knowledge"])
        self.assertIn("Las Vegas Strip.", output["trajectory"]["all_knowledge"])

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
