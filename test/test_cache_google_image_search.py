from __future__ import annotations

import contextlib
import io
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
    build_parser,
    default_output_path,
)


class CacheGoogleImageSearchTest(unittest.TestCase):
    def test_default_output_path_uses_pmsr_cache_suffix(self) -> None:
        self.assertEqual(
            default_output_path(Path("data/fvqa_test.jsonl")),
            Path("data/fvqa_test_pmsr_cache.jsonl"),
        )

    def test_parser_no_longer_accepts_legacy_conversion_mode(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--jsonl", "data/fvqa_test.jsonl", "--mode", "convert-existing"])

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

    def test_jsonl_round_trip_fetches_current_cache_schema(self) -> None:
        from scripts.cache_google_image_search import process_jsonl

        class FakeGoogleImageSearch:
            def search(self, query: dict[str, str], *, top_k: int) -> list[SearchResult]:
                return [
                    SearchResult(
                        evidence=Evidence(
                            source="google_image",
                            modality="image",
                            title="Example caption",
                            text="Example summary",
                            image_path="https://example.com/thumb.jpg",
                            caption="Example caption",
                            score=1.0,
                            rank=1,
                        ),
                        query=query["image_path"],
                        search_type="google_image",
                    )
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "question": "What is shown?",
                        "image_path": "/tmp/query.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            processed = process_jsonl(
                input_path=input_path,
                output_path=output_path,
                top_k=5,
                searcher=FakeGoogleImageSearch(),
                overwrite=True,
            )

            self.assertEqual(processed, 1)
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("searched_results", row)
            self.assertEqual(row["searched_results"]["google_image"][0]["caption"], "Example caption")

    def test_process_row_requires_searcher_for_missing_current_cache(self) -> None:
        from scripts.cache_google_image_search import process_row

        with self.assertRaises(ValueError):
            process_row({"image_path": "/tmp/query.jpg"}, top_k=5)


if __name__ == "__main__":
    unittest.main()
