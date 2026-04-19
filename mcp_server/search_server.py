#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MAX_TOP_K = 20
DEFAULT_TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_IMAGE_MODEL = "google/siglip2-giant-opt-patch16-384"

TEXT_SEARCH_DESCRIPTION = (
    "Retrieve text passages from the textual Wikipedia knowledge base using dense text-text "
    "semantic similarity. In PMSR, textual retrieval supplies complementary passages from a "
    "heterogeneous KB to support knowledge-intensive visual question answering."
)
IMAGE_SEARCH_DESCRIPTION = (
    "Retrieve image-text pairs from the multimodal Wikipedia knowledge base for an input image "
    "and query. If MLLM_KB and MLLM_EMBED_API_BASE are configured, PMSR uses a joint MLLM "
    "image-text embedding to retrieve images; otherwise it falls back to concat fusion with "
    "separate image and text embeddings."
)
PMSR_MULTIMODAL_SEARCH_DESCRIPTION = (
    "Run PMSR dual-scope retrieval over heterogeneous KBs. The caller should provide a "
    "record_level_query generated from the latest reasoning record and a trajectory_level_query "
    "generated from the full reasoning trajectory; PMSR searches both scopes over text passages "
    "and multimodal image-text pairs, then merges the evidence."
)


mcp = FastMCP(
    "PMSR Search",
    instructions=(
        "Read-only PMSR retrieval tools for knowledge-intensive visual question answering. "
        "Use text_search for textual Wikipedia evidence, image_search for multimodal image-text "
        "evidence, and pmsr_multimodal_search when both evidence types are useful."
    ),
    json_response=True,
)


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> None:
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


def redact_secrets(message: object) -> str:
    text = str(message)
    text = re.sub(r"([?&]api_key=)[^&\s]+", r"\1REDACTED", text)
    text = re.sub(r"([?&]key=)[^&\s]+", r"\1REDACTED", text)
    text = re.sub(r"(Authorization:\s*Bearer\s+)[^\s]+", r"\1REDACTED", text)
    return text


def clamp_top_k(top_k: int) -> int:
    return max(1, min(int(top_k), MAX_TOP_K))


def _required_env(names: list[str]) -> dict[str, str]:
    values = {name: os.environ.get(name, "") for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required PMSR MCP environment variables: {', '.join(missing)}")
    return values


def _text_model() -> str:
    return (
        os.environ.get("TEXT_MODEL")
        or os.environ.get("QWEN_TEXT_EMBED_MODEL")
        or os.environ.get("TEXT_EMBED_MODEL")
        or DEFAULT_TEXT_MODEL
    )


def _image_model() -> str:
    return os.environ.get("IMAGE_EMBED_MODEL") or DEFAULT_IMAGE_MODEL


def _mllm_model() -> str:
    return os.environ.get("MLLM_EMBED_MODEL") or "Qwen/Qwen3-VL-Embedding-2B"


def _api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""


@lru_cache(maxsize=1)
def get_text_searcher():
    from search.text_search import TextSearch, TextSearchConfig

    env = _required_env(["TEXT_KB", "TEXT_METADATA", "TEXT_EMBED_API_BASE"])
    return TextSearch(
        TextSearchConfig(
            text_kb=env["TEXT_KB"],
            text_metadata=env["TEXT_METADATA"],
            text_embed_api_base=env["TEXT_EMBED_API_BASE"],
            text_model=_text_model(),
            timeout=int(os.environ.get("MCP_SEARCH_TIMEOUT", "120")),
            api_key=_api_key(),
        )
    )


@lru_cache(maxsize=1)
def get_image_searcher():
    from search.pmsr_search import PMSRSearch, PMSRSearchConfig

    if os.environ.get("MLLM_KB") and os.environ.get("MLLM_EMBED_API_BASE"):
        env = _required_env(["MLLM_KB", "MLLM_METADATA", "MLLM_EMBED_API_BASE"])
        return PMSRSearch(
            PMSRSearchConfig(
                mllm_kb=env["MLLM_KB"],
                mllm_metadata=env["MLLM_METADATA"],
                mllm_embed_api_base=env["MLLM_EMBED_API_BASE"],
                mllm_model=_mllm_model(),
                fusion="mllm",
                timeout=int(os.environ.get("MCP_SEARCH_TIMEOUT", "120")),
                api_key=_api_key(),
            )
        )

    env = _required_env(["PMSR_KB", "PMSR_METADATA", "IMAGE_EMBED_API_BASE", "QWEN_TEXT_EMBED_API_BASE"])
    return PMSRSearch(
        PMSRSearchConfig(
            pmsr_kb=env["PMSR_KB"],
            pmsr_metadata=env["PMSR_METADATA"],
            image_embed_api_base=env["IMAGE_EMBED_API_BASE"],
            text_embed_api_base=env["QWEN_TEXT_EMBED_API_BASE"],
            image_model=_image_model(),
            text_model=os.environ.get("PMSR_TEXT_MODEL") or os.environ.get("QWEN_TEXT_EMBED_MODEL") or DEFAULT_TEXT_MODEL,
            fusion=os.environ.get("PMSR_FUSION", "concat"),  # type: ignore[arg-type]
            timeout=int(os.environ.get("MCP_SEARCH_TIMEOUT", "120")),
            api_key=_api_key(),
        )
    )


def format_result(result: Any) -> dict[str, Any]:
    raw = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    formatted: dict[str, Any] = {
        "rank": raw.get("rank", 0),
        "score": raw.get("score", 0.0),
        "source": raw.get("source", ""),
        "modality": raw.get("modality", ""),
    }
    for key in ("title", "text", "image_path", "caption", "url"):
        value = raw.get(key)
        if value:
            formatted[key] = value
    return formatted


def _run_text_search(query: str, top_k: int) -> list[dict[str, Any]]:
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("query must be non-empty.")
    results = get_text_searcher().search(query_text, top_k=clamp_top_k(top_k))
    return [format_result(result) for result in results]


def _run_image_search(image: str, query: str, top_k: int) -> list[dict[str, Any]]:
    image_path = str(image or "").strip()
    query_text = str(query or "").strip()
    if not image_path:
        raise ValueError("image must be a local path, HTTP(S) URL, or data URL.")
    if not query_text:
        raise ValueError("query must be non-empty.")
    results = get_image_searcher().search(
        {"image_path": image_path, "text": query_text},
        top_k=clamp_top_k(top_k),
    )
    return [format_result(result) for result in results]


def _merge_formatted_results(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for result in group:
            key = "|".join(
                str(result.get(field) or "")
                for field in ("source", "title", "text", "image_path", "caption", "url")
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)
    return merged


@mcp.tool(description=TEXT_SEARCH_DESCRIPTION)
def text_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve text passages from PMSR's textual Wikipedia KB using dense text-text similarity."""
    try:
        return _run_text_search(query, top_k)
    except Exception as exc:
        raise RuntimeError(redact_secrets(exc)) from exc


@mcp.tool(description=IMAGE_SEARCH_DESCRIPTION)
def image_search(image: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve image-text pairs from PMSR's multimodal KB using image and query evidence."""
    try:
        return _run_image_search(image, query, top_k)
    except Exception as exc:
        raise RuntimeError(redact_secrets(exc)) from exc


@mcp.tool(description=PMSR_MULTIMODAL_SEARCH_DESCRIPTION)
def pmsr_multimodal_search(
    image: str,
    record_level_query: str,
    trajectory_level_query: str,
    top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve text and image evidence with PMSR record-level and trajectory-level queries."""
    try:
        k = clamp_top_k(top_k)
        return {
            "text_results": _merge_formatted_results(
                _run_text_search(record_level_query, k),
                _run_text_search(trajectory_level_query, k),
            ),
            "image_results": _merge_formatted_results(
                _run_image_search(image, record_level_query, k),
                _run_image_search(image, trajectory_level_query, k),
            ),
        }
    except Exception as exc:
        raise RuntimeError(redact_secrets(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PMSR MCP search server.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--override-env", action="store_true")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8765")))
    return parser


def configure_http_server(host: str, port: int) -> None:
    mcp.settings.host = host
    mcp.settings.port = port


def main() -> None:
    args = build_parser().parse_args()
    load_env_file(args.env_file, override=args.override_env)
    if args.transport != "stdio":
        configure_http_server(args.host, args.port)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
