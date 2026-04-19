from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_knowledge_base import (
    build_metadata_row,
    encode_records,
    l2_normalize_matrix,
)
from search.embedding_client import EmbeddingClient, parse_embeddings


class FakeEmbeddingClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        self.calls.append(list(image_paths))
        return self.vectors[: len(image_paths)]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self.vectors[: len(texts)]

    def embed_mllm(self, *, image_path: str, text: str, instruction: str) -> list[float]:
        self.calls.append([image_path, text, instruction])
        return self.vectors[len(self.calls) - 1]


class CreateKnowledgeBaseTest(unittest.TestCase):
    def test_parse_embeddings_reads_openai_batch_response(self) -> None:
        payload = {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0]},
            ]
        }

        self.assertEqual(parse_embeddings(payload), [[1.0, 0.0], [0.0, 1.0]])

    def test_parse_embeddings_reads_vllm_v2_float_response(self) -> None:
        payload = {"embeddings": {"float": [[1.0, 0.0], [0.0, 1.0]]}}

        self.assertEqual(parse_embeddings(payload), [[1.0, 0.0], [0.0, 1.0]])

    def test_embedding_client_sends_batch_text_inputs(self) -> None:
        response = SimpleNamespace()
        response.raise_for_status = lambda: None
        response.json = lambda: {"embeddings": {"float": [[1.0], [2.0]]}}
        client = EmbeddingClient(api_base="http://localhost:8012", model="Qwen/Qwen3-Embedding-0.6B")

        with patch("search.embedding_client.requests.post", return_value=response) as mock_post:
            vectors = client.embed_texts(["first", "second"])

        self.assertEqual(vectors, [[1.0], [2.0]])
        self.assertEqual(mock_post.call_args.kwargs["json"]["input"], ["first", "second"])

    def test_embedding_client_sends_batch_image_messages(self) -> None:
        response = SimpleNamespace()
        response.raise_for_status = lambda: None
        response.json = lambda: {"data": [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}]}
        client = EmbeddingClient(api_base="http://localhost:8013", model="google/siglip2-giant-opt-patch16-384")

        with patch("search.embedding_client.requests.post", return_value=response) as mock_post:
            vectors = client.embed_images(["https://example.com/a.jpg", "https://example.com/b.jpg"])

        self.assertEqual(vectors, [[1.0], [2.0]])
        self.assertEqual(mock_post.call_args.args[0], "http://localhost:8013/v2/embed")
        inputs = mock_post.call_args.kwargs["json"]["inputs"]
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0]["content"][0]["image_url"]["url"], "https://example.com/a.jpg")
        self.assertEqual(inputs[1]["content"][0]["image_url"]["url"], "https://example.com/b.jpg")
        self.assertEqual(mock_post.call_args.kwargs["json"]["embedding_types"], ["float"])

    def test_embedding_client_sends_mllm_image_payload_with_text_for_documents(self) -> None:
        response = SimpleNamespace()
        response.raise_for_status = lambda: None
        response.json = lambda: {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
        client = EmbeddingClient(api_base="http://localhost:8013", model="Qwen/Qwen3-VL-Embedding-2B")

        with patch("search.embedding_client.requests.post", return_value=response) as mock_post:
            vector = client.embed_mllm(
                image_path="https://example.com/wiki.jpg",
                text="Wikipedia summary caption.",
                instruction="Represent the given Wikipedia image with related text information: ",
            )

        self.assertEqual(vector, [1.0, 2.0])
        self.assertEqual(mock_post.call_args.args[0], "http://localhost:8013/v1/embeddings")
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["messages"],
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Represent the given Wikipedia image with related text information: "},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/wiki.jpg"}},
                        {"type": "text", "text": "Wikipedia summary caption."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                },
            ],
        )

    def test_encode_records_batches_image_and_text_embeddings_for_concat_index(self) -> None:
        records = [
            {"image_path": "https://example.com/image-a.jpg", "wikipedia_summary": "caption a", "wikipedia_content": "content a", "url": "drop"},
            {"image_path": "https://example.com/image-b.jpg", "wikipedia_summary": "caption b", "wikipedia_content": "content b", "title": "drop"},
        ]
        image_client = FakeEmbeddingClient([[1.0, 0.0], [0.0, 1.0]])
        text_client = FakeEmbeddingClient([[0.0, 1.0], [1.0, 0.0]])

        vectors, metadata = encode_records(
            records,
            image_client=image_client,
            text_client=text_client,
            batch_size=2,
            image_field="image_path",
            text_field="wikipedia_summary",
            caption_field="wikipedia_summary",
            fusion="concat",
        )

        self.assertEqual(image_client.calls, [["https://example.com/image-a.jpg", "https://example.com/image-b.jpg"]])
        self.assertEqual(text_client.calls, [["caption a", "caption b"]])
        self.assertEqual(vectors.shape, (2, 4))
        self.assertEqual(metadata[0], {"image_path": "https://example.com/image-a.jpg", "caption": "caption a"})
        self.assertEqual(metadata[1], {"image_path": "https://example.com/image-b.jpg", "caption": "caption b"})

    def test_encode_records_uses_mllm_joint_embeddings_for_mllm_index(self) -> None:
        records = [
            {"image_path": "https://example.com/image-a.jpg", "wikipedia_summary": "caption a"},
            {"image_path": "https://example.com/image-b.jpg", "wikipedia_summary": "caption b"},
        ]
        mllm_client = FakeEmbeddingClient([[1.0, 0.0], [0.0, 2.0]])

        vectors, metadata = encode_records(
            records,
            image_client=None,
            text_client=None,
            mllm_client=mllm_client,
            batch_size=2,
            image_field="image_path",
            text_field="wikipedia_summary",
            caption_field="wikipedia_summary",
            fusion="mllm",
            mllm_instruction="Represent the given Wikipedia image with related text information: ",
        )

        self.assertEqual(
            mllm_client.calls,
            [
                [
                    "https://example.com/image-a.jpg",
                    "caption a",
                    "Represent the given Wikipedia image with related text information: ",
                ],
                [
                    "https://example.com/image-b.jpg",
                    "caption b",
                    "Represent the given Wikipedia image with related text information: ",
                ],
            ],
        )
        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(vectors[1].tolist(), [0.0, 1.0])
        self.assertEqual(metadata[0], {"image_path": "https://example.com/image-a.jpg", "caption": "caption a"})

    def test_build_metadata_row_keeps_only_image_path_and_caption_for_image_rows(self) -> None:
        row = build_metadata_row(
            {"image_path": "img.jpg", "summary": "short caption", "content": "long content", "title": "Title"},
            image_field="image_path",
            text_field="summary",
            caption_field="content",
        )

        self.assertEqual(row, {"image_path": "img.jpg", "caption": "long content"})

    def test_build_metadata_row_splits_wikipedia_contents_for_text_rows(self) -> None:
        row = build_metadata_row(
            {"contents": '"Article Title"\nArticle body text.', "id": "drop"},
            image_field="image_path",
            text_field="contents",
            caption_field="wikipedia_summary",
        )

        self.assertEqual(row, {"title": "Article Title", "text": "Article body text."})

    def test_l2_normalize_matrix_normalizes_rows(self) -> None:
        normalized = l2_normalize_matrix([[3.0, 4.0], [0.0, 0.0]])

        self.assertAlmostEqual(float(normalized[0][0]), 0.6)
        self.assertAlmostEqual(float(normalized[0][1]), 0.8)
        self.assertEqual(normalized[1].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
