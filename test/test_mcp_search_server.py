from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class MCPSearchServerTest(unittest.TestCase):
    def test_format_result_keeps_public_retrieval_fields(self) -> None:
        from agents.schemas import Evidence, SearchResult
        from mcp_server import search_server

        result = SearchResult(
            evidence=Evidence(
                source="text_faiss",
                modality="text",
                title="Pavia Cathedral",
                text="Pavia Cathedral is a church in Pavia, Italy.",
                score=0.63,
                rank=1,
            ),
            query="What is Pavia Cathedral?",
            search_type="text",
        )

        self.assertEqual(
            search_server.format_result(result),
            {
                "rank": 1,
                "score": 0.63,
                "source": "text_faiss",
                "modality": "text",
                "title": "Pavia Cathedral",
                "text": "Pavia Cathedral is a church in Pavia, Italy.",
            },
        )

    def test_text_search_tool_clamps_top_k_and_calls_retriever(self) -> None:
        from mcp_server import search_server

        calls = []
        fake_result = SimpleNamespace(
            to_dict=lambda: {
                "rank": 1,
                "score": 0.5,
                "source": "text_faiss",
                "modality": "text",
                "title": "Pavia Cathedral",
                "text": "Text.",
                "image_path": "",
                "caption": "",
            }
        )
        fake_searcher = SimpleNamespace(search=lambda query, top_k: calls.append((query, top_k)) or [fake_result])

        with patch.object(search_server, "get_text_searcher", return_value=fake_searcher):
            results = search_server.text_search("What is Pavia Cathedral?", top_k=999)

        self.assertEqual(calls, [("What is Pavia Cathedral?", 20)])
        self.assertEqual(results[0]["title"], "Pavia Cathedral")

    def test_image_search_tool_passes_image_and_query_to_pmsr_retriever(self) -> None:
        from mcp_server import search_server

        calls = []
        fake_result = SimpleNamespace(
            to_dict=lambda: {
                "rank": 1,
                "score": 0.7,
                "source": "pmsr_faiss",
                "modality": "image",
                "title": "",
                "text": "",
                "image_path": "/tmp/wiki.jpg",
                "caption": "Reference image caption.",
            }
        )
        fake_searcher = SimpleNamespace(search=lambda query, top_k: calls.append((query, top_k)) or [fake_result])

        with patch.object(search_server, "get_image_searcher", return_value=fake_searcher):
            results = search_server.image_search("/tmp/query.jpg", "What is this?", top_k=3)

        self.assertEqual(calls, [({"image_path": "/tmp/query.jpg", "text": "What is this?"}, 3)])
        self.assertEqual(results[0]["image_path"], "/tmp/wiki.jpg")
        self.assertEqual(results[0]["caption"], "Reference image caption.")

    def test_multimodal_search_returns_text_and_image_results(self) -> None:
        from mcp_server import search_server

        text_calls = []
        image_calls = []
        fake_text = SimpleNamespace(
            search=lambda query, top_k: text_calls.append((query, top_k)) or [
                SimpleNamespace(
                    to_dict=lambda: {
                        "rank": 1,
                        "score": 0.5,
                        "source": "text_faiss",
                        "modality": "text",
                        "title": "Text title",
                        "text": "Text passage.",
                        "image_path": "",
                        "caption": "",
                    }
                )
            ]
        )
        fake_image = SimpleNamespace(
            search=lambda query, top_k: image_calls.append((query, top_k)) or [
                SimpleNamespace(
                    to_dict=lambda: {
                        "rank": 1,
                        "score": 0.7,
                        "source": "pmsr_faiss",
                        "modality": "image",
                        "title": "",
                        "text": "",
                        "image_path": "/tmp/wiki.jpg",
                        "caption": "Caption.",
                    }
                )
            ]
        )

        with patch.object(search_server, "get_text_searcher", return_value=fake_text), patch.object(
            search_server, "get_image_searcher", return_value=fake_image
        ):
            results = search_server.pmsr_multimodal_search(
                "/tmp/query.jpg",
                record_level_query="latest visual clue",
                trajectory_level_query="global evidence need",
                top_k=5,
            )

        self.assertEqual(text_calls, [("latest visual clue", 5), ("global evidence need", 5)])
        self.assertEqual(
            image_calls,
            [
                ({"image_path": "/tmp/query.jpg", "text": "latest visual clue"}, 5),
                ({"image_path": "/tmp/query.jpg", "text": "global evidence need"}, 5),
            ],
        )
        self.assertEqual(results["text_results"][0]["title"], "Text title")
        self.assertEqual(results["image_results"][0]["caption"], "Caption.")

    def test_get_image_searcher_prefers_mllm_when_configured(self) -> None:
        from mcp_server import search_server

        with patch.dict(
            os.environ,
            {
                "MLLM_KB": "/tmp/mllm.index",
                "MLLM_METADATA": "/tmp/mllm.csv",
                "MLLM_EMBED_API_BASE": "http://localhost:8013",
                "MLLM_EMBED_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
            },
            clear=False,
        ), patch("search.pmsr_search.PMSRSearch") as mock_search, patch(
            "search.pmsr_search.FaissKnowledgeBase"
        ):
            search_server.get_image_searcher.cache_clear()
            search_server.get_image_searcher()

        config = mock_search.call_args.args[0]
        self.assertEqual(config.fusion, "mllm")
        self.assertEqual(config.mllm_kb, "/tmp/mllm.index")
        self.assertEqual(config.mllm_embed_api_base, "http://localhost:8013")

    def test_load_env_file_keeps_existing_environment_by_default(self) -> None:
        from mcp_server import search_server

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("PMSR_TEST_ENV=from_file\n", encoding="utf-8")
            previous = os.environ.get("PMSR_TEST_ENV")
            os.environ["PMSR_TEST_ENV"] = "existing"
            try:
                search_server.load_env_file(env_path)
                self.assertEqual(os.environ["PMSR_TEST_ENV"], "existing")
                search_server.load_env_file(env_path, override=True)
                self.assertEqual(os.environ["PMSR_TEST_ENV"], "from_file")
            finally:
                if previous is None:
                    os.environ.pop("PMSR_TEST_ENV", None)
                else:
                    os.environ["PMSR_TEST_ENV"] = previous

        self.assertTrue(search_server.TEXT_SEARCH_DESCRIPTION.startswith("Retrieve text passages"))


if __name__ == "__main__":
    unittest.main()
