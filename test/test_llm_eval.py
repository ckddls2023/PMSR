from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import llm_eval


class LLMEvalTest(unittest.TestCase):
    def test_extract_model_response_uses_trajectory_final_answer(self) -> None:
        row = {
            "answer": "dataset answer",
            "trajectory": {
                "final_answer": "model final answer",
                "records": [{"reasoning": "first"}, {"reasoning": "last"}],
            },
        }

        self.assertEqual(llm_eval.extract_model_response(row), "model final answer")

    def test_extract_gold_answer_prefers_candidate_answer_eval(self) -> None:
        row = {
            "gold_answer": "Ishi-no-ma-zukuri",
            "answer_eval": ["Ishi-no-ma-zukuri", "Ishinoma-zukuri", "gongen-zukuri"],
        }

        self.assertEqual(
            llm_eval.extract_gold_answer(row),
            ["Ishi-no-ma-zukuri", "Ishinoma-zukuri", "gongen-zukuri"],
        )

    def test_has_existing_judge_requires_score_and_reason(self) -> None:
        self.assertTrue(llm_eval.has_existing_judge({"judge_score": "Yes", "judge_reason": "matches"}))
        self.assertFalse(llm_eval.has_existing_judge({"judge_score": "Yes", "judge_reason": ""}))
        self.assertFalse(llm_eval.has_existing_judge({"judge_score": "", "judge_reason": "matches"}))

    def test_process_data_copies_existing_judges_without_api_key(self) -> None:
        rows = [
            {
                "question": "Q?",
                "gold_answer": "A",
                "prediction": "A",
                "judge_score": "Yes",
                "judge_reason": "matches",
            },
            {
                "question": "Q2?",
                "gold_answer": "B",
                "prediction": "C",
                "judge_score": "No",
                "judge_reason": "does not match",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            old_key = llm_eval.API_KEY
            llm_eval.API_KEY = ""
            try:
                llm_eval.process_data(str(input_path), str(output_path))
            finally:
                llm_eval.API_KEY = old_key

            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(len(output_rows), 2)
        self.assertEqual([row["judge_score"] for row in output_rows], ["Yes", "No"])
        self.assertEqual([row["original_index"] for row in output_rows], [0, 1])

    def test_parser_rejects_eval_record_option(self) -> None:
        with self.assertRaises(SystemExit):
            llm_eval.build_parser().parse_args(["--jsonl", "predictions.jsonl", "--eval-record"])


if __name__ == "__main__":
    unittest.main()
