from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import metric_eval


class MetricEvalTest(unittest.TestCase):
    def test_load_predictions_reads_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"
            path.write_text(
                json.dumps({"question": "q1", "prediction": "alpha"}) + "\n"
                + json.dumps({"question": "q2", "prediction": "beta"}) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(jsonl=str(path), csv=None, pkl=None)

            predictions = metric_eval.load_predictions(args)

        self.assertEqual([p["prediction"] for p in predictions], ["alpha", "beta"])

    def test_evaluate_accuracy_uses_cem_for_pmsr_jsonl_schema(self) -> None:
        predictions = [
            {
                "question": "Where?",
                "trajectory": {"final_answer": "It is in Paris, France."},
                "gold_answer": ["France"],
            },
            {
                "question": "Where?",
                "trajectory": {"final_answer": "It is in Italy."},
                "gold_answer": ["France"],
            },
        ]

        accuracy, flags = metric_eval.evaluate_accuracy(predictions)

        self.assertEqual(flags, [True, False])
        self.assertEqual(accuracy, 0.5)

    def test_evaluate_accuracy_prefers_answer_eval_candidates_over_gold_answer(self) -> None:
        predictions = [
            {
                "question": "What is the architectural style?",
                "trajectory": {"final_answer": "The building is gongen-zukuri."},
                "gold_answer": "Ishi-no-ma-zukuri",
                "answer_eval": ["Ishi-no-ma-zukuri", "Ishinoma-zukuri", "gongen-zukuri"],
            }
        ]

        accuracy, flags = metric_eval.evaluate_accuracy(predictions)

        self.assertEqual(flags, [True])
        self.assertEqual(accuracy, 1.0)

    def test_evaluate_accuracy_uses_top_level_answer_when_answer_eval_is_missing(self) -> None:
        predictions = [
            {
                "question": "What kind of medical usage has this plant?",
                "trajectory": {"final_answer": "The medical usage is urinary tract infections, antioxidant."},
                "answer": [
                    "help urinary tract infections and as an antioxidant.",
                    "urinary tract infections, antioxidant",
                ],
            }
        ]

        accuracy, flags = metric_eval.evaluate_accuracy(predictions)

        self.assertEqual(flags, [True])
        self.assertEqual(accuracy, 1.0)

    def test_evaluate_accuracy_falls_back_when_answer_eval_is_empty_list(self) -> None:
        predictions = [
            {
                "question": "What is the length in metre?",
                "trajectory": {"final_answer": "The bridge is 237 metres long."},
                "gold_answer": "237",
                "answer_eval": [],
            }
        ]

        accuracy, flags = metric_eval.evaluate_accuracy(predictions)

        self.assertEqual(flags, [True])
        self.assertEqual(accuracy, 1.0)

    def test_evaluate_accuracy_uses_existing_answer_eval_when_present(self) -> None:
        predictions = [
            {"prediction": "wrong text", "gold_answer": ["France"], "answer_eval": True},
            {"prediction": "France", "gold_answer": ["France"], "answer_eval": False},
        ]

        accuracy, flags = metric_eval.evaluate_accuracy(predictions)

        self.assertEqual(flags, [True, False])
        self.assertEqual(accuracy, 0.5)

    def test_evaluate_accuracy_uses_bem_when_requested(self) -> None:
        predictions = [
            {
                "question": "Where?",
                "trajectory": {"final_answer": "It is in Paris."},
                "gold_answer": "France",
            }
        ]
        args = argparse.Namespace(bem=True, exact_match=False)

        with patch.object(metric_eval, "initialize_bem_model_and_transformers") as init, patch.object(
            metric_eval, "run_bem_evaluation", return_value=True
        ) as run_bem:
            accuracy, flags = metric_eval.evaluate_accuracy(predictions, args)

        init.assert_called_once()
        run_bem.assert_called_once_with("Where?", "France", "It is in Paris.")
        self.assertEqual(flags, [True])
        self.assertEqual(accuracy, 1.0)

    def test_evaluate_recall_uses_entity_text_against_knowledge(self) -> None:
        predictions = [
            {"entity_text": ["Smilax bona-nox"], "trajectory": {"all_knowledge": "The plant is Smilax bona-nox."}},
            {"entity_text": ["Cornus canadensis"], "trajectory": {"all_knowledge": "The plant is Smilax bona-nox."}},
        ]

        recall, flags = metric_eval.evaluate_recall(predictions)

        self.assertEqual(flags, [True, False])
        self.assertEqual(recall, 0.5)

    def test_evaluate_recall_does_not_use_prr_when_entity_text_exists_but_misses(self) -> None:
        predictions = [
            {
                "entity_text": "Nezu Shrine",
                "answer_eval": ["Ishi-no-ma-zukuri", "Ishinoma-zukuri", "gongen-zukuri"],
                "trajectory": {
                    "all_knowledge": "Japanese Buddhist architecture includes the gongen-zukuri style."
                },
            }
        ]

        recall, flags = metric_eval.evaluate_recall(predictions)

        self.assertEqual(flags, [False])
        self.assertEqual(recall, 0.0)

    def test_evaluate_recall_uses_prr_when_entity_text_is_missing(self) -> None:
        predictions = [
            {
                "entity_text": "",
                "answer_eval": "['NT', 'Near Threatened', 'LR/nt']",
                "trajectory": {"all_knowledge": "This species is listed as Near Threatened."},
            }
        ]

        recall, flags = metric_eval.evaluate_recall(predictions)

        self.assertEqual(flags, [True])
        self.assertEqual(recall, 1.0)

    def test_reasoning_helpers_read_nested_trajectory(self) -> None:
        prediction = {
            "trajectory": {
                "all_reasoning": "Reasoning Record #1:\nFirst.\n\nReasoning Record #2:\nSecond.",
                "records": [{"reasoning": "First."}, {"reasoning": "Second."}],
            }
        }

        self.assertEqual(metric_eval.count_reasoning_records_from_prediction(prediction), 2)
        self.assertEqual(metric_eval.extract_last_reasoning_record_from_prediction(prediction), "Second.")

    def test_iterative_recall_breakdown_uses_structured_record_results(self) -> None:
        predictions = [
            {
                "entity_text": "Alpha",
                "trajectory": {
                    "records": [
                        {
                            "text_results": [{"title": "A", "text": "Alpha fact"}],
                            "image_results": [],
                            "reasoning": "r1",
                        },
                        {
                            "text_results": [],
                            "image_results": [{"caption": "Alpha image caption"}],
                            "reasoning": "r2",
                        },
                    ]
                },
            },
            {
                "entity_text": "Beta",
                "trajectory": {
                    "records": [
                        {
                            "text_results": [{"title": "B", "text": "Other"}],
                            "image_results": [],
                            "reasoning": "r1",
                        },
                        {
                            "text_results": [],
                            "image_results": [{"caption": "Beta image caption"}],
                            "reasoning": "r2",
                        },
                    ]
                },
            },
        ]

        summary = metric_eval.iterative_recall_breakdown(predictions)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["text_hits"], 1)
        self.assertEqual(summary[0]["image_hits"], 0)
        self.assertEqual(summary[0]["combined_hits"], 1)
        self.assertEqual(summary[1]["text_hits"], 1)
        self.assertEqual(summary[1]["image_hits"], 2)
        self.assertEqual(summary[1]["combined_hits"], 2)
        self.assertEqual(summary[1]["new_combined_hits"], 1)

    def test_iterative_recall_breakdown_parses_legacy_flat_knowledge_runs(self) -> None:
        predictions = [
            {
                "entity_text": "Alpha",
                "trajectory": {
                    "records": [{"reasoning": "r1"}, {"reasoning": "r2"}],
                    "all_knowledge": (
                        "Passage Title: T1\n"
                        "Passage Text: Alpha text\n"
                        "Passage: image miss\n"
                        "Passage Title: T2\n"
                        "Passage Text: miss\n"
                        "Passage: Alpha image\n"
                    ),
                },
            }
        ]

        summary = metric_eval.iterative_recall_breakdown(predictions)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["text_hits"], 1)
        self.assertEqual(summary[0]["image_hits"], 0)
        self.assertEqual(summary[0]["combined_hits"], 1)
        self.assertEqual(summary[1]["text_hits"], 1)
        self.assertEqual(summary[1]["image_hits"], 1)
        self.assertEqual(summary[1]["combined_hits"], 1)


if __name__ == "__main__":
    unittest.main()
