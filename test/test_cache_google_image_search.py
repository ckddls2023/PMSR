from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import Evidence, SearchResult
from scripts.cache_google_image_search import (
    attach_google_image_results,
    convert_legacy_google_image_cache,
    default_output_path,
)


class CacheGoogleImageSearchTest(unittest.TestCase):
    def test_default_output_path_uses_pmsr_cache_suffix(self) -> None:
        self.assertEqual(
            default_output_path(Path("data/fvqa_test.jsonl")),
            Path("data/fvqa_test_pmsr_cache.jsonl"),
        )

    def test_convert_legacy_cache_to_searched_results_schema(self) -> None:
        row = {
            "question_id": "fvqa_test_0",
            "question": "Where is this iconic sign located?",
            "image_path": "/tmp/query.jpg",
            "retrieved_image_path": ["https://example.com/thumb1.jpg", "https://example.com/thumb2.jpg"],
            "retrieved_caption": ["Las Vegas sign", "Welcome sign"],
            "lens_result": "The first result discusses the Las Vegas sign.\nThe second result mentions Nevada.",
        }

        converted = convert_legacy_google_image_cache(row, top_k=2)

        self.assertNotIn("lens_result", converted)
        self.assertNotIn("retrieved_image_path", converted)
        self.assertNotIn("retrieved_caption", converted)
        cached = converted["searched_results"]["google_image"]
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached[0]["search_type"], "google_image")
        self.assertEqual(cached[0]["source"], "google_image")
        self.assertEqual(cached[0]["modality"], "image")
        self.assertEqual(cached[0]["query"], "/tmp/query.jpg")
        self.assertEqual(cached[0]["title"], "Las Vegas sign")
        self.assertEqual(cached[0]["caption"], "Las Vegas sign")
        self.assertEqual(cached[0]["image_path"], "https://example.com/thumb1.jpg")
        self.assertEqual(cached[0]["text"], "The first result discusses the Las Vegas sign.")
        self.assertEqual(cached[0]["rank"], 1)
        self.assertTrue(cached[0]["metadata"]["legacy_cache"])

    def test_attach_google_image_results_preserves_other_search_groups(self) -> None:
        row = {"searched_results": {"web": [{"title": "existing"}]}}
        results = [
            SearchResult(
                evidence=Evidence(
                    source="google_image",
                    modality="image",
                    title="Visual result",
                    text="Visual summary",
                    image_path="https://example.com/image.jpg",
                    caption="Visual result",
                    score=1.0,
                    rank=1,
                ),
                query="query-image",
                search_type="google_image",
            )
        ]

        updated = attach_google_image_results(row, results)

        self.assertEqual(updated["searched_results"]["web"], [{"title": "existing"}])
        self.assertEqual(updated["searched_results"]["google_image"][0]["title"], "Visual result")
        self.assertEqual(updated["searched_results"]["google_image"][0]["query"], "query-image")

    def test_jsonl_round_trip_outputs_current_cache_schema(self) -> None:
        from scripts.cache_google_image_search import process_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "question": "What is shown?",
                        "image_path": "/tmp/query.jpg",
                        "retrieved_image_path": ["https://example.com/thumb.jpg"],
                        "retrieved_caption": ["Example caption"],
                        "lens_result": "Example summary",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            processed = process_jsonl(
                input_path=input_path,
                output_path=output_path,
                mode="convert-existing",
                top_k=5,
                overwrite=True,
            )

            self.assertEqual(processed, 1)
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("searched_results", row)
            self.assertEqual(row["searched_results"]["google_image"][0]["caption"], "Example caption")
            self.assertNotIn("lens_result", row)


if __name__ == "__main__":
    unittest.main()
