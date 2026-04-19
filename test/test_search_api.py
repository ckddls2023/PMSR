from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import build_pmsr_user_message
from search.google_image_search import GoogleImageSearch
from search.google_search import GoogleSearch
from search.embedding_client import EmbeddingClient
from search.faiss_search import CsvMetadataStore, FaissKnowledgeBase, load_metadata
from search.pmsr_search import (
    DEFAULT_MLLM_PASSAGE_INSTRUCTION,
    DEFAULT_MLLM_QUERY_INSTRUCTION,
    PMSRSearch,
    PMSRSearchConfig,
    get_detailed_instruct,
)
from search.text_search import TextSearch, TextSearchConfig


DEFAULT_IMAGE = ROOT / "test" / "image.jpg"


class SearchApiUnitTest(unittest.TestCase):
    def test_get_detailed_instruct_formats_qwen_query_instruction(self) -> None:
        self.assertEqual(
            get_detailed_instruct(
                "Given a web search query, retrieve relevant passages that answer the query",
                "What is the capital of China?",
            ),
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query:What is the capital of China?",
        )

    def test_pmsr_concat_wraps_query_text_with_instruction_before_embedding(self) -> None:
        searcher = object.__new__(PMSRSearch)
        searcher.config = PMSRSearchConfig(fusion="concat")
        searcher.image_client = SimpleNamespace(embed_image=lambda image_path: [1.0, 0.0])
        text_calls: list[str] = []
        searcher.text_client = SimpleNamespace(embed_text=lambda text: text_calls.append(text) or [0.0, 1.0])
        searcher.mllm_client = None

        searcher._encode(image_path="https://example.com/query.jpg", text="What is this building?")

        self.assertEqual(
            text_calls,
            [
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                "Query:What is this building?"
            ],
        )

    def test_pmsr_text_wraps_query_text_with_instruction_before_embedding(self) -> None:
        searcher = object.__new__(PMSRSearch)
        searcher.config = PMSRSearchConfig(fusion="text")
        searcher.image_client = None
        text_calls: list[str] = []
        searcher.text_client = SimpleNamespace(embed_text=lambda text: text_calls.append(text) or [0.0, 1.0])
        searcher.mllm_client = None

        searcher._encode(image_path="", text="Explain gravity")

        self.assertEqual(
            text_calls,
            [
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                "Query:Explain gravity"
            ],
        )

    def test_pmsr_mllm_uses_query_instruction_for_joint_embedding(self) -> None:
        searcher = object.__new__(PMSRSearch)
        searcher.config = PMSRSearchConfig(fusion="mllm", instruction=DEFAULT_MLLM_QUERY_INSTRUCTION)
        searcher.image_client = None
        searcher.text_client = None
        calls: list[dict[str, str]] = []
        searcher.mllm_client = SimpleNamespace(
            embed_mllm=lambda **kwargs: calls.append(kwargs) or [1.0, 0.0]
        )

        searcher._encode(image_path="https://example.com/query.jpg", text="What animal is shown?")

        self.assertEqual(
            calls,
            [
                {
                    "image_path": "https://example.com/query.jpg",
                    "text": "What animal is shown?",
                    "instruction": "Find a Wikipedia image that answers this question: ",
                }
            ],
        )

    def test_embedding_client_sends_mllm_text_embedding_payload(self) -> None:
        response = SimpleNamespace()
        response.raise_for_status = lambda: None
        response.json = lambda: {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
        client = EmbeddingClient(api_base="http://localhost:8013", model="Qwen/Qwen3-VL-Embedding-2B")

        with patch("search.embedding_client.requests.post", return_value=response) as mock_post:
            vector = client.embed_mllm_text(
                text="What animal is shown?",
                instruction=DEFAULT_MLLM_QUERY_INSTRUCTION,
            )

        self.assertEqual(vector, [1.0, 2.0])
        self.assertEqual(mock_post.call_args.args[0], "http://localhost:8013/v1/embeddings")
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {
                "model": "Qwen/Qwen3-VL-Embedding-2B",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": DEFAULT_MLLM_QUERY_INSTRUCTION}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What animal is shown?"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": ""}],
                    },
                ],
                "encoding_format": "float",
                "continue_final_message": True,
                "add_special_tokens": True,
            },
        )

    def test_text_search_applies_e5_query_prefix_before_embedding(self) -> None:
        searcher = object.__new__(TextSearch)
        searcher.config = TextSearchConfig(
            text_kb="/tmp/index.faiss",
            text_metadata="/tmp/metadata.jsonl",
            text_embed_api_base="http://localhost:8011",
        )
        text_calls: list[str] = []
        searcher.text_client = SimpleNamespace(embed_text=lambda text: text_calls.append(text) or [1.0, 0.0])
        searcher.kb = SimpleNamespace(search_vector=lambda vector, top_k, query, search_type: [])

        searcher.search("What is the capital of China?", top_k=3)

        self.assertEqual(text_calls, ["query: What is the capital of China?"])

    def test_text_search_does_not_duplicate_existing_e5_query_prefix(self) -> None:
        searcher = object.__new__(TextSearch)
        searcher.config = TextSearchConfig(
            text_kb="/tmp/index.faiss",
            text_metadata="/tmp/metadata.jsonl",
            text_embed_api_base="http://localhost:8011",
        )
        text_calls: list[str] = []
        searcher.text_client = SimpleNamespace(embed_text=lambda text: text_calls.append(text) or [1.0, 0.0])
        searcher.kb = SimpleNamespace(search_vector=lambda vector, top_k, query, search_type: [])

        searcher.search("query: Explain gravity", top_k=3)

        self.assertEqual(text_calls, ["query: Explain gravity"])

    def test_text_search_truncates_e5_query_to_511_chars_before_embedding(self) -> None:
        searcher = object.__new__(TextSearch)
        searcher.config = TextSearchConfig(
            text_kb="/tmp/index.faiss",
            text_metadata="/tmp/metadata.jsonl",
            text_embed_api_base="http://localhost:8011",
            text_model="intfloat/e5-base-v2",
        )
        formatted = searcher._format_query("a" * 600)

        self.assertEqual(formatted, "query: " + ("a" * (511 - len("query: "))))
        self.assertEqual(len(formatted), 511)

    def test_text_search_truncates_qwen3_query_to_32767_chars_before_embedding(self) -> None:
        searcher = object.__new__(TextSearch)
        searcher.config = TextSearchConfig(
            text_kb="/tmp/index.faiss",
            text_metadata="/tmp/metadata.jsonl",
            text_embed_api_base="http://localhost:8012",
            text_model="Qwen/Qwen3-Embedding-0.6B",
            query_prefix="",
        )
        formatted = searcher._format_query("b" * 33000)

        self.assertEqual(formatted, "b" * 32767)
        self.assertEqual(len(formatted), 32767)

    def test_parser_accepts_qwen_text_embed_api_base_for_pmsr(self) -> None:
        args = build_parser().parse_args(
            [
                "--case",
                "pmsr",
                "--qwen_text_embed_api_base",
                "http://127.0.0.1:8012",
            ]
        )

        self.assertEqual(args.qwen_text_embed_api_base, "http://127.0.0.1:8012")

    def test_google_image_search_wraps_public_url_for_google_lens(self) -> None:
        searcher = GoogleImageSearch(api_key="dummy")

        computed_query = searcher.compute_query(
            {
                "image_url": "https://example.com/image.jpg",
                "question": "What is this?",
            }
        )

        self.assertEqual(computed_query["image_url"], "https://example.com/image.jpg")
        self.assertEqual(
            computed_query["lens_url"],
            "https://lens.google.com/uploadbyurl?url=https://example.com/image.jpg",
        )
        self.assertEqual(computed_query["question"], "What is this?")

    def test_google_search_uses_ollama_web_search(self) -> None:
        response = SimpleNamespace(
            results=[
                SimpleNamespace(
                    title="Smilax bona-nox",
                    content="The roots are used for urinary tract infections.",
                    url="https://example.com/smilax",
                )
            ]
        )
        with patch("search.google_search.ollama.web_search", return_value=response):
            searcher = GoogleSearch(api_key="ollama-key", summarize=False)
            results = searcher.search("Smilax bona-nox medical usage", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].evidence.source, "web")
        self.assertEqual(results[0].evidence.title, "Smilax bona-nox")
        self.assertIn("urinary tract infections", results[0].evidence.text)

    def test_google_image_search_parse_results_returns_image_text_evidence(self) -> None:
        searcher = GoogleImageSearch(api_key="dummy", ollama_api_key="", summarize=False)

        with patch.object(searcher, "_is_invalid_thumbnail", return_value=False):
            results = searcher.parse_results(
                {
                    "lens_results": [
                        {
                            "title": "Smilax bona-nox",
                            "link": "https://example.com/smilax",
                            "source": "Example",
                            "thumbnail": "https://example.com/thumb.jpg",
                        }
                    ]
                },
                question="What kind of medical usage has this plant?",
                top_k=1,
                query="https://example.com/query.jpg",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].evidence.source, "google_image")
        self.assertEqual(results[0].evidence.image_path, "https://example.com/thumb.jpg")
        self.assertEqual(results[0].evidence.caption, "Smilax bona-nox")

    def test_google_image_upload_uses_images0707_and_hashed_object_name(self) -> None:
        searcher = GoogleImageSearch(api_key="dummy", ollama_api_key="", summarize=False)

        with patch("boto3.client") as mock_client:
            mock_s3 = mock_client.return_value
            url = searcher.upload_base64_image("aGVsbG8=", original_name="query.jpg")

        self.assertTrue(url.startswith("https://images0707.s3.ap-southeast-2.amazonaws.com/"))
        self.assertTrue(url.endswith(".jpg"))
        put_object_kwargs = mock_s3.put_object.call_args.kwargs
        self.assertEqual(put_object_kwargs["Bucket"], "images0707")
        self.assertRegex(put_object_kwargs["Key"], r"^[a-f0-9]{24}\.jpg$")

    def test_google_image_lens_request_matches_scrapingdog_api_shape(self) -> None:
        searcher = GoogleImageSearch(api_key="dummy", ollama_api_key="", summarize=False)
        response = SimpleNamespace()
        response.raise_for_status = lambda: None
        response.json = lambda: {"lens_results": []}

        with patch("search.google_image_search.requests.get", return_value=response) as mock_get:
            searcher.search_with_google_lens("https://lens.google.com/uploadbyurl?url=https://example.com/image.jpg")

        request_url = mock_get.call_args.args[0]
        self.assertEqual(
            request_url,
            "https://api.scrapingdog.com/google_lens?api_key=dummy&url=https://lens.google.com/uploadbyurl?url=https://example.com/image.jpg",
        )
        self.assertNotIn("params", mock_get.call_args.kwargs)

    def test_faiss_metadata_hydrates_wiki_contents_schema(self) -> None:
        kb = object.__new__(FaissKnowledgeBase)
        kb.source = "text_faiss"

        result = kb._record_to_result(
            {"id": "1", "contents": '"Horatio Hale"\nBiography text about the article.'},
            row_id=1,
            score=0.5,
            rank=1,
            query="who is horatio hale",
            search_type="text",
        )

        self.assertEqual(result.evidence.title, "Horatio Hale")
        self.assertEqual(result.evidence.text, "Biography text about the article.")
        self.assertEqual(result.evidence.score, 0.5)
        self.assertEqual(result.evidence.rank, 1)
        self.assertEqual(result.to_text_passage(), {"title": "Horatio Hale", "text": "Biography text about the article."})

    def test_faiss_metadata_hydrates_image_caption_schema_without_title(self) -> None:
        kb = object.__new__(FaissKnowledgeBase)
        kb.source = "pmsr_faiss"

        result = kb._record_to_result(
            {"image_path": "/tmp/example.jpg", "caption": "A caption for the retrieved image."},
            row_id=3,
            score=1.0,
            rank=1,
            query="image question",
            search_type="pmsr_concat",
        )
        image_pair = result.to_image_pair()

        self.assertEqual(result.evidence.modality, "image")
        self.assertEqual(result.evidence.title, "")
        self.assertEqual(result.evidence.image_path, "/tmp/example.jpg")
        self.assertEqual(result.evidence.caption, "A caption for the retrieved image.")
        self.assertEqual(image_pair, {"image_path": "/tmp/example.jpg", "caption": "A caption for the retrieved image."})

    def test_faiss_metadata_uses_wikipedia_summary_as_image_caption(self) -> None:
        kb = object.__new__(FaissKnowledgeBase)
        kb.source = "pmsr_faiss"

        result = kb._record_to_result(
            {"image_path": "/tmp/example.jpg", "wikipedia_summary": "Wikipedia summary caption.", "title": "Unused"},
            row_id=4,
            score=1.0,
            rank=1,
            query="image question",
            search_type="pmsr_concat",
        )

        self.assertEqual(result.to_image_pair(), {"image_path": "/tmp/example.jpg", "caption": "Wikipedia summary caption."})

    def test_faiss_image_caption_is_truncated_for_large_metadata(self) -> None:
        kb = object.__new__(FaissKnowledgeBase)
        kb.source = "mllm_faiss"
        caption = "x" * 5000

        result = kb._record_to_result(
            {"image_path": "/tmp/example.jpg", "caption": caption},
            row_id=5,
            score=1.0,
            rank=1,
            query="image question",
            search_type="pmsr_mllm",
        )

        self.assertEqual(len(result.evidence.caption), 4096)
        self.assertEqual(result.evidence.caption, "x" * 4096)

    def test_csv_metadata_loads_and_truncates_fields_larger_than_default_csv_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.csv"
            metadata_path.write_text(
                "image_path,caption,url\n/tmp/example.jpg," + ("x" * 140000) + ",https://example.com\n",
                encoding="utf-8",
            )

            metadata = load_metadata(metadata_path)

            self.assertIsInstance(metadata, CsvMetadataStore)
            self.assertEqual(len(metadata.rows), 0)
            self.assertEqual(metadata.get(0), {"image_path": "/tmp/example.jpg", "caption": "x" * 4096})

    def test_jsonl_metadata_loads_with_indexed_row_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.jsonl"
            metadata_path.write_text(
                '{"id": "0", "contents": "first"}\n{"id": "1", "contents": "second"}\n',
                encoding="utf-8",
            )

            metadata = load_metadata(metadata_path)

            self.assertEqual(len(metadata), 2)
            self.assertEqual(metadata.get(1)["contents"], "second")

    def test_jsonl_metadata_truncates_image_rows_on_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps({"image_path": "/tmp/example.jpg", "wikipedia_summary": "y" * 5000, "url": "drop"})
                + "\n",
                encoding="utf-8",
            )

            metadata = load_metadata(metadata_path)

            self.assertEqual(metadata.get(0), {"image_path": "/tmp/example.jpg", "caption": "y" * 4096})

    def test_pmsr_message_orders_image_pairs_text_passages_image_then_prompt(self) -> None:
        message = build_pmsr_user_message(
            image_text_pairs=[
                {
                    "image_path": "https://example.com/ref.jpg",
                    "caption": "Reference image caption.",
                }
            ],
            text_passages=[
                {
                    "title": "Horatio Hale",
                    "text": "Biography text about the article.",
                }
            ],
            image_path="https://example.com/query.jpg",
            prompt="Question: Who is shown?",
        )

        self.assertEqual(
            message["content"],
            [
                {"type": "text", "text": "Here is relevant knowledge of image and their corresponding description.\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/ref.jpg"}},
                {"type": "text", "text": "Reference image caption."},
                {"type": "text", "text": "Knowledge: Passage Title: Horatio Hale\nPassage Text: Biography text about the article.\n\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/query.jpg"}},
                {"type": "text", "text": "Question: Who is shown?"},
            ],
        )

    def test_redact_secrets_hides_scrapingdog_api_key(self) -> None:
        message = "403 for https://api.scrapingdog.com/google?api_key=secret123&query=test"

        self.assertNotIn("secret123", redact_secrets(message))
        self.assertIn("api_key=REDACTED", redact_secrets(message))


def load_env_file(path: str | Path = ".env", *, override: bool = True) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test PMSR search APIs.")
    parser.add_argument("--text_kb")
    parser.add_argument("--text_metadata")
    parser.add_argument("--pmsr_kb")
    parser.add_argument("--pmsr_metadata")
    parser.add_argument("--mllm_kb")
    parser.add_argument("--mllm_metadata")
    parser.add_argument("--image_embed_api_base")
    parser.add_argument("--text_embed_api_base")
    parser.add_argument("--text_model")
    parser.add_argument("--qwen_text_embed_api_base")
    parser.add_argument("--mllm_embed_api_base")
    parser.add_argument("--mllm_model")
    parser.add_argument("--google-search-api-key")
    parser.add_argument("--google-image-api-key")
    parser.add_argument("--google-image-url")
    parser.add_argument("--query", default="What bird is shown in the image?")
    parser.add_argument("--image")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--pmsr-fusion", choices=["concat", "image", "text", "mllm"], default="concat")
    parser.add_argument(
        "--case",
        choices=["text", "pmsr", "mllm", "google", "google_image", "kb", "all"],
        default="kb",
        help="'kb' runs text and PMSR FAISS searches. MLLM is only run for --case mllm or --case all.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _assert_results(name: str, results: list[object]) -> None:
    if not results:
        raise AssertionError(f"{name} returned no results.")
    first = results[0]
    if not hasattr(first, "score") or not hasattr(first, "to_dict"):
        raise AssertionError(f"{name} result does not match SearchResult interface: {first!r}")
    print(f"[ok] case={name} count={len(results)} first_score={first.score}")


def redact_secrets(message: object) -> str:
    text = str(message)
    text = re.sub(r"([?&]api_key=)[^&\s]+", r"\1REDACTED", text)
    text = re.sub(r"([?&]key=)[^&\s]+", r"\1REDACTED", text)
    return text


def run_text(args: argparse.Namespace) -> None:
    text_kb = args.text_kb or os.environ.get("TEXT_KB", "")
    text_metadata = args.text_metadata or os.environ.get("TEXT_METADATA", "")
    text_embed_api_base = args.text_embed_api_base or os.environ.get("TEXT_EMBED_API_BASE", "")
    text_model = (
        args.text_model
        or os.environ.get("TEXT_MODEL")
        or os.environ.get("QWEN_TEXT_EMBED_MODEL")
        or os.environ.get("TEXT_EMBED_MODEL")
        or TextSearchConfig.__dataclass_fields__["text_model"].default
    )
    missing = [
        name
        for name, value in {
            "--text_kb": text_kb,
            "--text_metadata": text_metadata,
            "--text_embed_api_base": text_embed_api_base,
            "--text_model": text_model,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing text search options: {', '.join(missing)}")
    searcher = TextSearch(
        TextSearchConfig(
            text_kb=text_kb,
            text_metadata=text_metadata,
            text_embed_api_base=text_embed_api_base,
            text_model=text_model,
            timeout=args.timeout,
        )
    )
    results = searcher.search(args.query, top_k=args.top_k)
    _assert_results("text", results)


def run_pmsr(args: argparse.Namespace) -> None:
    if args.pmsr_fusion == "mllm":
        raise SystemExit("Use --case mllm for MLLM KB search.")
    pmsr_kb = args.pmsr_kb or os.environ.get("PMSR_KB", "")
    pmsr_metadata = args.pmsr_metadata or os.environ.get("PMSR_METADATA", "")
    image_embed_api_base = args.image_embed_api_base or os.environ.get("IMAGE_EMBED_API_BASE", "")
    qwen_text_embed_api_base = args.qwen_text_embed_api_base or os.environ.get("QWEN_TEXT_EMBED_API_BASE", "")
    image_path = args.image or os.environ.get("IMAGE_PATH", str(DEFAULT_IMAGE))
    missing = [
        name
        for name, value in {
            "--pmsr_kb": pmsr_kb,
            "--pmsr_metadata": pmsr_metadata,
            "--image_embed_api_base": image_embed_api_base,
            "--qwen_text_embed_api_base": qwen_text_embed_api_base,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing PMSR search options: {', '.join(missing)}")
    searcher = PMSRSearch(
        PMSRSearchConfig(
            pmsr_kb=pmsr_kb,
            pmsr_metadata=pmsr_metadata,
            image_embed_api_base=image_embed_api_base,
            text_embed_api_base=qwen_text_embed_api_base,
            fusion=args.pmsr_fusion,
            timeout=args.timeout,
        )
    )
    results = searcher.search({"image_path": image_path, "text": args.query}, top_k=args.top_k)
    _assert_results("pmsr", results)


def run_mllm(args: argparse.Namespace) -> None:
    mllm_kb = args.mllm_kb or os.environ.get("MLLM_KB", "")
    mllm_metadata = args.mllm_metadata or os.environ.get("MLLM_METADATA", "")
    mllm_embed_api_base = args.mllm_embed_api_base or os.environ.get("MLLM_EMBED_API_BASE", "")
    mllm_model = args.mllm_model or os.environ.get("MLLM_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
    image_path = args.image or os.environ.get("IMAGE_PATH", str(DEFAULT_IMAGE))
    missing = [
        name
        for name, value in {
            "--mllm_kb": mllm_kb,
            "--mllm_metadata": mllm_metadata,
            "--mllm_embed_api_base": mllm_embed_api_base,
            "--mllm_model": mllm_model,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing MLLM search options: {', '.join(missing)}")
    searcher = PMSRSearch(
        PMSRSearchConfig(
            mllm_kb=mllm_kb,
            mllm_metadata=mllm_metadata,
            mllm_embed_api_base=mllm_embed_api_base,
            mllm_model=mllm_model,
            fusion="mllm",
            instruction=DEFAULT_MLLM_QUERY_INSTRUCTION,
            timeout=args.timeout,
        )
    )
    results = searcher.search({"image_path": image_path, "text": args.query}, top_k=args.top_k)
    _assert_results("mllm", results)


def run_google(args: argparse.Namespace) -> None:
    google_api_key = args.google_search_api_key or os.environ.get("OLLAMA_API_KEY", "")
    if not google_api_key:
        raise SystemExit("Missing Google web search option: --google-search-api-key or OLLAMA_API_KEY")
    searcher = GoogleSearch(api_key=google_api_key, timeout=args.timeout)
    results = searcher.search(args.query, top_k=args.top_k)
    _assert_results("google", results)


def run_google_image(args: argparse.Namespace) -> None:
    google_image_api_key = args.google_image_api_key or os.environ.get("GOOGLE_IMAGE_API_KEY", "") or os.environ.get(
        "SCRAPINGDOG_API_KEY", ""
    )
    google_image = args.google_image_url or os.environ.get("GOOGLE_IMAGE_URL", "") or args.image or os.environ.get(
        "IMAGE_PATH", str(DEFAULT_IMAGE)
    )
    if not google_image_api_key:
        raise SystemExit("Missing Google image search option: --google-image-api-key or SCRAPINGDOG_API_KEY")
    searcher = GoogleImageSearch(api_key=google_image_api_key, timeout=args.timeout)
    results = searcher.search({"image_path": google_image, "question": args.query}, top_k=args.top_k)
    _assert_results("google_image", results)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    runners = []
    if args.case in {"text", "kb", "all"}:
        runners.append(("text", run_text))
    if args.case in {"pmsr", "kb", "all"}:
        runners.append(("pmsr", run_pmsr))
    if args.case in {"mllm", "all"}:
        runners.append(("mllm", run_mllm))
    if args.case in {"google", "all"}:
        runners.append(("google", run_google))
    if args.case in {"google_image", "all"}:
        runners.append(("google_image", run_google_image))

    failures = 0
    for name, runner in runners:
        try:
            print(f"[test] case={name}")
            runner(args)
        except (AssertionError, OSError, SystemExit) as exc:
            failures += 1
            print(f"[fail] case={name}: {redact_secrets(exc)}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
