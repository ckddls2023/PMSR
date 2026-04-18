from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agents.schemas import SearchResult
from search.base_search import BaseSearch
from search.embedding_client import EmbeddingClient
from search.faiss_search import FaissKnowledgeBase, l2_normalize


DEFAULT_IMAGE_MODEL = "google/siglip2-giant-opt-patch16-384"
DEFAULT_TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_MLLM_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
FusionMode = Literal["concat", "image", "text", "mllm"]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


@dataclass(frozen=True, slots=True)
class PMSRSearchConfig:
    pmsr_kb: str | Path | None = None
    pmsr_metadata: str | Path | None = None
    mllm_kb: str | Path | None = None
    mllm_metadata: str | Path | None = None
    image_embed_api_base: str = ""
    text_embed_api_base: str = ""
    mllm_embed_api_base: str = ""
    image_model: str = DEFAULT_IMAGE_MODEL
    text_model: str = DEFAULT_TEXT_MODEL
    mllm_model: str = DEFAULT_MLLM_MODEL
    fusion: FusionMode = "concat"
    instruction: str = DEFAULT_QUERY_INSTRUCTION
    image_weight: float = 1.0
    text_weight: float = 1.0
    timeout: int = 60
    api_key: str = ""


class PMSRSearch(BaseSearch):
    """PMSR FAISS search over image/text/MLLM embeddings."""

    def __init__(self, config: PMSRSearchConfig) -> None:
        self.config = config
        self.image_client = self._build_client(config.image_embed_api_base, config.image_model)
        self.text_client = self._build_client(config.text_embed_api_base, config.text_model)
        self.mllm_client = self._build_client(config.mllm_embed_api_base, config.mllm_model)
        self.kb = self._build_kb(config)

    def search(self, query: Any, top_k: int = 5) -> list[SearchResult]:
        image_path, text = self._coerce_query(query)
        vector = self._encode(image_path=image_path, text=text)
        return self.kb.search_vector(vector, top_k=top_k, query=str(query), search_type=f"pmsr_{self.config.fusion}")

    def _encode(self, *, image_path: str, text: str) -> list[float]:
        if self.config.fusion == "image":
            if not image_path:
                raise ValueError("PMSR image search requires image_path.")
            return l2_normalize(self._require_client("image").embed_image(image_path))
        if self.config.fusion == "text":
            if not text:
                raise ValueError("PMSR text search requires text.")
            return l2_normalize(self._require_client("text").embed_text(self._format_text_query(text)))
        if self.config.fusion == "mllm":
            if not image_path:
                raise ValueError("PMSR MLLM search requires image_path.")
            return l2_normalize(
                self._require_client("mllm").embed_mllm(
                    image_path=image_path,
                    text=text,
                    instruction=self.config.instruction,
                )
            )
        if not image_path:
            raise ValueError("PMSR concat search requires image_path.")
        if not text:
            raise ValueError("PMSR concat search requires text.")
        image_vector = [
            self.config.image_weight * value
            for value in l2_normalize(self._require_client("image").embed_image(image_path))
        ]
        text_vector = [
            self.config.text_weight * value
            for value in l2_normalize(self._require_client("text").embed_text(self._format_text_query(text)))
        ]
        return l2_normalize([*image_vector, *text_vector])

    def _format_text_query(self, text: str) -> str:
        return get_detailed_instruct(self.config.instruction, text)

    @staticmethod
    def _coerce_query(query: Any) -> tuple[str, str]:
        if isinstance(query, dict):
            image_path = query.get("image_path") or query.get("image") or ""
            if isinstance(image_path, list):
                image_path = image_path[0] if image_path else ""
            text = query.get("text") or query.get("question") or query.get("caption") or ""
            return str(image_path or ""), str(text or "")
        query_text = str(query or "").strip()
        if Path(query_text).exists() or query_text.startswith(("http://", "https://", "data:")):
            return query_text, ""
        return "", query_text

    @staticmethod
    def _build_kb(config: PMSRSearchConfig) -> FaissKnowledgeBase:
        if config.fusion == "mllm":
            if not config.mllm_kb or not config.mllm_metadata:
                raise ValueError("PMSRSearchConfig with fusion='mllm' requires mllm_kb and mllm_metadata.")
            return FaissKnowledgeBase(
                index_path=config.mllm_kb,
                metadata_path=config.mllm_metadata,
                source="mllm_faiss",
            )
        if not config.pmsr_kb or not config.pmsr_metadata:
            raise ValueError("PMSRSearchConfig requires pmsr_kb and pmsr_metadata for non-MLLM fusion.")
        return FaissKnowledgeBase(
            index_path=config.pmsr_kb,
            metadata_path=config.pmsr_metadata,
            source="pmsr_faiss",
        )

    def _build_client(self, api_base: str, model: str) -> EmbeddingClient | None:
        if not api_base:
            return None
        return EmbeddingClient(
            api_base=api_base,
            model=model,
            timeout=self.config.timeout,
            api_key=self.config.api_key,
        )

    def _require_client(self, client_name: Literal["image", "text", "mllm"]) -> EmbeddingClient:
        client = getattr(self, f"{client_name}_client")
        if client is None:
            raise ValueError(f"PMSR {client_name} embedding requires {client_name}_embed_api_base.")
        return client
