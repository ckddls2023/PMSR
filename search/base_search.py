from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agents.schemas import SearchResult


class BaseSearch(ABC):
    @abstractmethod
    def search(self, query: Any, top_k: int = 5) -> list[SearchResult]:
        """Return normalized ranked search results."""


def clamp_top_k(top_k: int, available: int) -> int:
    if top_k <= 0 or available <= 0:
        return 0
    return min(int(top_k), int(available))
