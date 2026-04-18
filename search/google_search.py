from __future__ import annotations

import os
from typing import Any

import ollama
from ollama import Client

from agents.schemas import Evidence, SearchResult
from search.base_search import BaseSearch


class GoogleSearch(BaseSearch):
    """Ollama web-search wrapper returning normalized PMSR results."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-oss:120b",
        summarize: bool = True,
        host: str = "https://ollama.com",
        timeout: int = 60,
    ) -> None:
        del timeout
        self.model = model
        self.summarize = summarize
        self.host = host
        self.api_key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY", "")
        if self.api_key:
            os.environ.setdefault("OLLAMA_API_KEY", self.api_key)
        self.client: Client | None = None
        if self.summarize and self.api_key:
            self.client = Client(
                host=self.host,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        elif self.summarize:
            self.summarize = False

    def search(self, query: Any, top_k: int = 5) -> list[SearchResult]:
        query_text = str(query or "").strip()
        if not query_text or top_k <= 0:
            return []
        condensed_query = self._condense_query(query_text)
        response = ollama.web_search(condensed_query)
        raw_results = getattr(response, "results", []) or []
        results: list[SearchResult] = []
        for rank, item in enumerate(raw_results[:top_k], start=1):
            title = str(getattr(item, "title", "") or "")
            content = str(getattr(item, "content", "") or "")
            url = str(getattr(item, "url", "") or "")
            text = self._summarize_result(condensed_query, title, content) if content else title
            evidence = Evidence(
                source="web",
                modality="web",
                title=title,
                text=text,
                url=url,
                score=1.0 / rank,
                rank=rank,
                metadata={"raw_content": content[:2000]},
            )
            results.append(SearchResult(evidence=evidence, query=condensed_query, search_type="web"))
        return results

    def search_legacy(self, prompt: str, num: int = 15, return_score: bool = False):
        results = self.search(prompt, top_k=num)
        passages = [result.to_text_passage() for result in results]
        scores = [result.score for result in results]
        if return_score:
            return passages, scores
        return passages

    def _condense_query(self, prompt: str) -> str:
        if len(prompt) <= 400 or self.client is None:
            return prompt
        try:
            response = self.client.chat(
                self.model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Rewrite this query into a concise search-engine query "
                            f"with the main entity and essential point only.\nQuery: {prompt}"
                        ),
                    }
                ],
                stream=False,
            )
            return response["message"]["content"].strip() or prompt
        except Exception:
            return prompt

    def _summarize_result(self, query: str, title: str, content: str) -> str:
        if not self.summarize or self.client is None:
            return content[:1000]
        try:
            response = self.client.chat(
                self.model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Query: {query}\n\nSummarize the following web content, focusing on information relevant "
                            f"to the query. Provide a concise single paragraph.\n\nTitle: {title}\n\nContent: {content[:120000]}"
                        ),
                    }
                ],
                stream=False,
            )
            return response["message"]["content"]
        except Exception:
            return content[:1000]
