"""Base agent definitions for PMSR."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentConfig:
    """Flat configuration dataclass for PMSRAgent and related agents."""

    # VLM settings
    model: str
    api_base: str = ""
    api_key: str = ""
    max_tokens: int = 32768
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    timeout: int = 300
    retry: int = 3

    # Text retrieval — FAISS path, or HTTP(S) URL for web search (GoogleSearch)
    text_kb: str = ""
    text_metadata: str = ""
    text_embed_api_base: str = ""
    text_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # PMSR image-document KB — always concat (image + text) fusion
    pmsr_kb: str = ""
    pmsr_metadata: str = ""
    image_embed_api_base: str = ""
    pmsr_text_embed_api_base: str = ""
    pmsr_fusion: str = "concat"
    image_model: str = "google/siglip2-giant-opt-patch16-384"
    pmsr_text_model: str = "Qwen/Qwen3-Embedding-0.6B"
    mllm_kb: str = ""
    mllm_metadata: str = ""
    mllm_embed_api_base: str = ""
    mllm_model: str = "Qwen/Qwen3-VL-Embedding-2B"

    # Adaptive stopping similarity embedding. This is intentionally separate
    # from text retrieval so TEXT_EMBED_API_BASE is not reused for query
    # similarity when a better Qwen/MLLM embedding endpoint is available.
    similarity_embed_api_base: str = ""
    similarity_model: str = "Qwen/Qwen3-Embedding-0.6B"
    similarity_embed_mode: str = "text"

    # Optional backends behind the single ReACT pmsr_search tool.
    web_search: bool = False
    google_lens_search: bool = False

    # Whether to include image content in VLM passages from pmsr/cached results
    # True  → to_image_pair()  (VLM loads actual images)
    # False → to_text_passage() (captions/titles only)
    return_images: bool = True

    # Iteration
    max_iter: int = 3
    topk: int = 10
    use_traj_query: bool = True

    # Adaptive stopping (per paper §3.3): always on.
    # Stops when current queries are similar to previous queries (threshold τ).
    threshold: float = 0.9

    verbose: bool = False


class BaseAgent(ABC):
    """Abstract base class for PMSR-style agents."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self, item: dict[str, Any]) -> Any:
        """Run the agent on a single data item."""
