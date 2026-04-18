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
    timeout: int = 300
    retry: int = 3

    # Text retrieval — FAISS path, or HTTP(S) URL for web search (GoogleSearch)
    text_kb: str = ""
    text_metadata: str = ""
    text_embed_api_base: str = ""
    text_model: str = "intfloat/e5-base-v2"

    # PMSR image-document KB — always concat (image + text) fusion
    pmsr_kb: str = ""
    pmsr_metadata: str = ""
    image_embed_api_base: str = ""
    pmsr_text_embed_api_base: str = ""
    pmsr_fusion: str = "concat"
    image_model: str = "google/siglip2-giant-opt-patch16-384"
    pmsr_text_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Whether to include image content in VLM passages from pmsr/cached results
    # True  → to_image_pair()  (VLM loads actual images)
    # False → to_text_passage() (captions/titles only)
    return_images: bool = True

    # Iteration
    max_iter: int = 3
    topk: int = 10

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
