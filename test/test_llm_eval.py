from __future__ import annotations

import sys
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

        self.assertEqual(llm_eval.extract_model_response(row, eval_record=False), "model final answer")

    def test_extract_model_response_uses_last_trajectory_record_for_record_eval(self) -> None:
        row = {
            "trajectory": {
                "all_reasoning": "Reasoning Record #1:\nfirst\n\nReasoning Record #2:\nlast",
                "records": [{"reasoning": "first"}, {"reasoning": "last"}],
            }
        }

        self.assertEqual(llm_eval.extract_model_response(row, eval_record=True), "last")

    def test_extract_gold_answer_prefers_candidate_answer_eval(self) -> None:
        row = {
            "gold_answer": "Ishi-no-ma-zukuri",
            "answer_eval": ["Ishi-no-ma-zukuri", "Ishinoma-zukuri", "gongen-zukuri"],
        }

        self.assertEqual(
            llm_eval.extract_gold_answer(row),
            ["Ishi-no-ma-zukuri", "Ishinoma-zukuri", "gongen-zukuri"],
        )


if __name__ == "__main__":
    unittest.main()
