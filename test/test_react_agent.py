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
    return agent


class ReACTAgentToolTest(unittest.TestCase):
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

        agent._pmsr_search = FakePMSRSearch()
        agent._web_search = None
        agent._google_image_search = None
        agent._google_image_cache = {}

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

    def test_pmsr_search_uses_web_search_for_text_when_enabled(self) -> None:
        agent = make_agent()
        agent.config.web_search = True
        calls: list[tuple[str, int]] = []
        web_result = SearchResult(Evidence(source="web", modality="web", title="Web", text="Web evidence"))

        class FakeWebSearch:
            def search(self, query: str, top_k: int) -> list[SearchResult]:
                calls.append((query, top_k))
                return [web_result]

        agent._pmsr_search = None
        agent._web_search = FakeWebSearch()
        agent._google_image_search = None
        agent._google_image_cache = {}

        text_results, image_results = agent._execute_retrieval(
            "pmsr_search",
            {
                "record_level_query": "record query",
                "trajectory_level_query": "trajectory query",
            },
            "/tmp/query.jpg",
        )

        self.assertEqual(text_results, [web_result])
        self.assertEqual(image_results, [])
        self.assertEqual(calls, [("record query", 2), ("trajectory query", 2)])

    def test_pmsr_search_uses_google_lens_for_image_when_enabled(self) -> None:
        agent = make_agent()
        agent.config.google_lens_search = True
        calls: list[dict[str, str]] = []
        lens_result = SearchResult(
            Evidence(source="google_image", modality="image", image_path="https://example.com/ref.jpg", caption="Lens")
        )

        class FakeGoogleImageSearch:
            def search(self, query: dict[str, str], top_k: int) -> list[SearchResult]:
                calls.append(query)
                return [lens_result]

        agent._pmsr_search = object()
        agent._web_search = None
        agent._google_image_search = FakeGoogleImageSearch()
        agent._google_image_cache = {}

        first = agent._execute_retrieval(
            "pmsr_search",
            {
                "record_level_query": "record query",
                "trajectory_level_query": "trajectory query",
            },
            "/tmp/query.jpg",
        )
        second = agent._execute_retrieval(
            "pmsr_search",
            {
                "record_level_query": "second record query",
                "trajectory_level_query": "second trajectory query",
            },
            "/tmp/query.jpg",
        )

        self.assertEqual(first, ([], [lens_result]))
        self.assertEqual(second, ([], [lens_result]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["image_path"], "/tmp/query.jpg")
        self.assertEqual(calls[0]["question"], "record query\ntrajectory query")

    def test_only_pmsr_search_tool_is_enabled(self) -> None:
        agent = make_agent()
        agent._pmsr_search = object()
        agent._web_search = object()
        agent._google_image_search = object()

        names = [tool["function"]["name"] for tool in agent._enabled_tools()]

        self.assertEqual(names, ["pmsr_search"])

    def test_unknown_tool_does_not_retrieve(self) -> None:
        agent = make_agent()
        agent._pmsr_search = object()
        agent._web_search = object()
        agent._google_image_search = object()

        text_results, image_results = agent._execute_retrieval("text_search", {"query": "ignored"}, "/tmp/query.jpg")

        self.assertEqual(text_results, [])
        self.assertEqual(image_results, [])


if __name__ == "__main__":
    unittest.main()
