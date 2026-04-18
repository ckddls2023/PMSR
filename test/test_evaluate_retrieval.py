from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import Evidence, SearchResult
from eval.evaluate_retrieval import compute_recall, result_to_match_texts


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


if __name__ == "__main__":
    unittest.main()
