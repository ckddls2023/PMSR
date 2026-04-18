from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.schemas import SearchResult
from search.base_search import BaseSearch
from search.embedding_client import EmbeddingClient
from search.faiss_search import FaissKnowledgeBase, l2_normalize


DEFAULT_TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


@dataclass(frozen=True, slots=True)
class TextSearchConfig:
    text_kb: str | Path
    text_metadata: str | Path
    text_embed_api_base: str
    text_model: str = DEFAULT_TEXT_MODEL
    query_prefix: str = "query: "
    timeout: int = 60
    api_key: str = ""


class TextSearch(BaseSearch):
    """Text FAISS KB search using a text embedding API."""

    def __init__(self, config: TextSearchConfig) -> None:
        self.config = config
        self.text_client = EmbeddingClient(
            api_base=config.text_embed_api_base,
            model=config.text_model,
            timeout=config.timeout,
            api_key=config.api_key,
        )
        self.kb = FaissKnowledgeBase(
            index_path=config.text_kb,
            metadata_path=config.text_metadata,
            source="text_faiss",
        )

    def search(self, query: Any, top_k: int = 5) -> list[SearchResult]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        embedding_input = self._format_query(query_text)
        query_vector = l2_normalize(self.text_client.embed_text(embedding_input))
        return self.kb.search_vector(query_vector, top_k=top_k, query=query_text, search_type="text")

    def _format_query(self, query: str) -> str:
        if not self.config.query_prefix:
            return query
        if query.startswith(self.config.query_prefix):
            return query
        return f"{self.config.query_prefix}{query}"
