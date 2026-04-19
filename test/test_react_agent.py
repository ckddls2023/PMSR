from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base_agent import AgentConfig
from agents.react_agent import ReACTAgent
from agents.schemas import Evidence, SearchResult


def make_agent() -> ReACTAgent:
    agent = object.__new__(ReACTAgent)
    agent.config = AgentConfig(model="dummy", topk=4)
    agent._google_image_cache = {}
    return agent


class ReACTAgentToolTest(unittest.TestCase):
    def test_text_search_uses_web_search_backend(self) -> None:
        agent = make_agent()
        calls: list[tuple[str, int]] = []
        result = SearchResult(Evidence(source="web", modality="web", title="Title", text="Body"))

        class FakeWebSearch:
            def search(self, query: str, top_k: int) -> list[SearchResult]:
                calls.append((query, top_k))
                return [result]

        agent._web_search = FakeWebSearch()
        agent._google_image_search = None
        agent._pmsr_search = None

        text_results, image_results = agent._execute_retrieval(
            "text_search",
            {"query": "Nezu Shrine architecture"},
            "/tmp/query.jpg",
            latest_reasoning="Japanese shrine",
        )

        self.assertEqual(calls, [("Nezu Shrine architecture\nJapanese shrine", 4)])
        self.assertEqual(text_results, [result])
        self.assertEqual(image_results, [])

    def test_image_search_calls_google_lens_once_per_image(self) -> None:
        agent = make_agent()
        calls: list[dict[str, str]] = []
        result = SearchResult(
            Evidence(source="google_image", modality="image", image_path="https://example.com/ref.jpg", caption="Ref")
        )

        class FakeGoogleImageSearch:
            def search(self, query: dict[str, str], top_k: int) -> list[SearchResult]:
                calls.append(query)
                return [result]

        agent._web_search = None
        agent._google_image_search = FakeGoogleImageSearch()
        agent._pmsr_search = None

        first = agent._execute_retrieval(
            "image_search",
            {"query": "What is this building?"},
            "/tmp/query.jpg",
        )
        second = agent._execute_retrieval(
            "image_search",
            {"query": "Different query should not refetch"},
            "/tmp/query.jpg",
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["image_path"], "/tmp/query.jpg")
        self.assertEqual(calls[0]["question"], "What is this building?")
        self.assertEqual(first, ([], [result]))
        self.assertEqual(second, ([], [result]))

    def test_pmsr_search_uses_dual_scope_pmsr_backend_only(self) -> None:
        agent = make_agent()
        calls: list[tuple[str, int]] = []
        local_result = SearchResult(
            Evidence(source="pmsr_faiss", modality="image", image_path="/tmp/local.jpg", caption="Local")
        )
        global_result = SearchResult(
            Evidence(source="pmsr_faiss", modality="image", image_path="/tmp/global.jpg", caption="Global")
        )

        class FakePMSRSearch:
            def search(self, query: dict[str, str], top_k: int) -> list[SearchResult]:
                calls.append((query["text"], top_k))
                return [local_result] if query["text"] == "record query" else [global_result]

        def fail_web(self, query: str, top_k: int) -> list[SearchResult]:
            raise AssertionError("pmsr_search must not call web search")

        agent._web_search = type("FailWeb", (), {"search": fail_web})()
        agent._google_image_search = None
        agent._pmsr_search = FakePMSRSearch()

        text_results, image_results = agent._execute_retrieval(
            "pmsr_search",
            {
                "record_level_query": "record query",
                "trajectory_level_query": "trajectory query",
            },
            "/tmp/query.jpg",
        )

        self.assertEqual(text_results, [])
        self.assertEqual(image_results, [local_result, global_result])
        self.assertEqual(calls, [("record query", 2), ("trajectory query", 2)])

    def test_enabled_tools_reflect_three_distinct_backends(self) -> None:
        agent = make_agent()
        agent._web_search = object()
        agent._google_image_search = object()
        agent._pmsr_search = object()

        names = [tool["function"]["name"] for tool in agent._enabled_tools()]

        self.assertEqual(names, ["text_search", "image_search", "pmsr_search"])


if __name__ == "__main__":
    unittest.main()
