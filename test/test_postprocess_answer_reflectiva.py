from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import postprocess_answer_reflectiva as reflectiva


class ReflectivaPostprocessTest(unittest.TestCase):
    def test_prompt_wraps_full_trajectory_reasoning_in_paragraph_tags(self) -> None:
        all_reasoning = "Reasoning Record #1:\nInitial reasoning.\n\nReasoning Record #2:\nMore evidence."

        prompt = reflectiva._build_prompt(
            context=all_reasoning,
        )

        expected_reasoning = all_reasoning.replace("Reasoning Record #", "Passage #")
        self.assertEqual(prompt, f"Consider this paragraph: <paragraph> {expected_reasoning} </paragraph>. ")

    def test_prompt_formats_list_context_like_reference(self) -> None:
        prompt = reflectiva._build_prompt(context=["Reasoning Record #1:\nFirst.", "Reasoning Record #2:\nSecond."])

        self.assertEqual(prompt, "Consider this paragraph: <paragraph> 1:\nFirst.2:\nSecond. </paragraph>. ")

    def test_postprocess_rows_uses_total_pred_and_updates_answer_fields(self) -> None:
        row = {
            "question_id": "q1",
            "question": "What is shown?",
            "image_path": "https://example.com/image.jpg",
            "total_pred": "Reasoning Record #1:\nIt is a tower.",
            "answer": "old answer",
            "prediction": "old answer",
        }
        captured: dict[str, object] = {}

        class FakeClient:
            def answer(self, *, image_path: str, question: str, context: str) -> str:
                captured["image_path"] = image_path
                captured["question"] = question
                captured["context"] = context
                return "refined answer"

        processed = reflectiva.postprocess_rows([row], FakeClient())

        self.assertEqual(processed[0]["answer"], "refined answer")
        self.assertEqual(processed[0]["prediction"], "refined answer")
        self.assertEqual(processed[0]["reflectiva_source_answer"], "old answer")
        self.assertEqual(captured["image_path"], "https://example.com/image.jpg")
        self.assertEqual(captured["question"], "What is shown?")
        self.assertEqual(captured["context"], "Reasoning Record #1:\nIt is a tower.")

    def test_use_lastrecord_extracts_only_final_reasoning_record(self) -> None:
        row = {
            "question": "What is shown?",
            "image_path": "https://example.com/image.jpg",
            "total_pred": "Reasoning Record #1:\nOld.\n\nReasoning Record #2:\nLatest.",
        }
        captured: dict[str, object] = {}

        class FakeClient:
            def answer(self, *, image_path: str, question: str, context: str) -> str:
                captured["context"] = context
                return {"content": "answer"}

        reflectiva.postprocess_rows([row], FakeClient(), use_lastrecord=True)

        self.assertEqual(captured["context"], "Latest.")

    def test_reflectiva_inferencer_passes_context_not_prompt_to_llava_call(self) -> None:
        captured: dict[str, object] = {}

        def fake_answer(**kwargs) -> str:
            captured.update(kwargs)
            return "llava answer"

        inferencer = reflectiva.LLaVAReflectivaInferencer(
            tokenizer=object(),
            model=type("FakeModel", (), {"config": object()})(),
            image_processor=object(),
            model_config=object(),
            conv_mode="llama_3_1",
            answer_fn=fake_answer,
        )

        result = inferencer.answer(
            image_path="/tmp/image.jpg",
            question="What is shown?",
            context="Reasoning Record #1:\nIt is a tower.",
        )

        self.assertEqual(result, "llava answer")
        self.assertEqual(captured["image_path"], "/tmp/image.jpg")
        self.assertEqual(captured["question"], "What is shown?")
        self.assertEqual(captured["context"], "Reasoning Record #1:\nIt is a tower.")
        self.assertNotIn("prompt", captured)


if __name__ == "__main__":
    unittest.main()
