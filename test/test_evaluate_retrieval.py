from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import Evidence, SearchResult
from eval.evaluate_retrieval import build_parser, build_pmsr_search, compute_recall, result_to_match_texts


class FakeRetriever:
    def search(self, query, top_k: int = 5):
        del query
        rows = [
            SearchResult(
                evidence=Evidence(
                    source="pmsr_faiss",
                    modality="image",
                    image_path="/tmp/image.jpg",
                    caption="Smilax bona-nox is known for ethnobotanical uses.",
                    score=1.0,
                    rank=1,
                )
            ),
            SearchResult(
                evidence=Evidence(
                    source="pmsr_faiss",
                    modality="image",
                    image_path="/tmp/other.jpg",
                    caption="Another plant.",
                    score=0.8,
                    rank=2,
                )
            ),
        ]
        return rows[:top_k]


class EvaluateRetrievalTest(unittest.TestCase):
    def test_result_to_match_texts_uses_caption_for_pmsr_image_pairs(self) -> None:
        result = SearchResult(
            evidence=Evidence(
                source="pmsr_faiss",
                modality="image",
                title="",
                image_path="/tmp/example.jpg",
                caption="A useful retrieved caption.",
            )
        )

        self.assertEqual(result_to_match_texts(result), ["A useful retrieved caption."])

    def test_compute_recall_reports_r_at_k_from_entity_text(self) -> None:
        dataset = [
            {
                "image_path": "/tmp/query.jpg",
                "question": "What kind of medical usage has this plant?",
                "entity_text": ["Smilax bona-nox"],
                "answer": ["urinary tract infections"],
            }
        ]

        scores = compute_recall(dataset, FakeRetriever(), top_ks=[1, 2])

        self.assertEqual(scores["R@1"], 1.0)
        self.assertEqual(scores["R@2"], 1.0)
        self.assertEqual(scores["total"], 1)

    def test_parser_accepts_mllm_fusion(self) -> None:
        args = build_parser().parse_args(["--pmsr-fusion", "mllm"])

        self.assertEqual(args.pmsr_fusion, "mllm")

    def test_build_pmsr_search_uses_mllm_config_for_mllm_fusion(self) -> None:
        args = build_parser().parse_args(
            [
                "--pmsr-fusion",
                "mllm",
                "--mllm-kb",
                "/tmp/mllm.index",
                "--mllm-metadata",
                "/tmp/mllm.csv",
                "--mllm-embed-api-base",
                "http://localhost:8013",
                "--mllm-model",
                "Qwen/Qwen3-VL-Embedding-2B",
            ]
        )

        with patch("eval.evaluate_retrieval.PMSRSearch") as mock_search:
            build_pmsr_search(args)

        config = mock_search.call_args.args[0]
        self.assertEqual(config.fusion, "mllm")
        self.assertEqual(config.mllm_kb, "/tmp/mllm.index")
        self.assertEqual(config.mllm_metadata, "/tmp/mllm.csv")
        self.assertEqual(config.mllm_embed_api_base, "http://localhost:8013")
        self.assertEqual(config.mllm_model, "Qwen/Qwen3-VL-Embedding-2B")


if __name__ == "__main__":
    unittest.main()
