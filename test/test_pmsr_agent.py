from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import MethodType


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base_agent import AgentConfig
from agents.pmsr_agent import PMSRAgent
from agents.schemas import Evidence, Record, SearchResult, Trajectory


def make_agent() -> PMSRAgent:
    agent = object.__new__(PMSRAgent)
    agent.config = AgentConfig(model="dummy", topk=2, max_iter=1)
    return agent


class PMSRAgentQueryTest(unittest.TestCase):
    def test_generate_sends_user_message_without_system_prompt(self) -> None:
        agent = make_agent()
        captured: dict[str, list[dict]] = {}

        class FakeVLM:
            def chat(self, messages: list[dict]) -> dict[str, str]:
                captured["messages"] = messages
                return {"content": "ok"}

        agent._vlm = FakeVLM()

        result = agent._generate(str(ROOT / "test" / "image.jpg"), "Question: example")

        self.assertEqual(result, "ok")
        self.assertEqual(len(captured["messages"]), 1)
        self.assertEqual(captured["messages"][0]["role"], "user")

    def test_generate_orders_image_pairs_text_passages_query_image_then_prompt(self) -> None:
        agent = make_agent()
        captured: dict[str, list[dict]] = {}

        class FakeVLM:
            def chat(self, messages: list[dict]) -> dict[str, str]:
                captured["content"] = messages[0]["content"]
                return {"content": "ok"}

        agent._vlm = FakeVLM()

        agent._generate(
            "https://example.com/query.jpg",
            "Question: example",
            image_text_pairs=[{"image_path": "https://example.com/ref.jpg", "caption": "ref caption"}],
            text_passages=[{"title": "Title", "text": "Body"}],
        )

        self.assertEqual(
            captured["content"],
            [
                {"type": "text", "text": "Here is relevant knowledge of image and their corresponding description.\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/ref.jpg"}},
                {"type": "text", "text": "ref caption"},
                {"type": "text", "text": "Knowledge: Passage Title: Title\nPassage Text: Body\n\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/query.jpg"}},
                {"type": "text", "text": "Question: example"},
            ],
        )

    def test_description_prompt_matches_reference(self) -> None:
        agent = make_agent()
        captured: dict[str, str] = {}

        def fake_generate(self: PMSRAgent, image_path: str, prompt: str, image_text_pairs=None, text_passages=None) -> str:
            captured["prompt"] = prompt
            return "description"

        agent._generate = MethodType(fake_generate, agent)

        agent._describe_image("/tmp/image.jpg", "What is this?")

        self.assertEqual(
            captured["prompt"],
            "Question: What is this?\nConcisely describe image which is relevant to question.\n",
        )

    def test_step0_reasoning_prompt_matches_reference(self) -> None:
        agent = make_agent()
        captured: dict[str, object] = {}
        text_result = SearchResult(
            evidence=Evidence(source="text", modality="text", title="Plant", text="Plant knowledge."),
            query="q",
            search_type="text",
        )
        image_result = SearchResult(
            evidence=Evidence(source="pmsr", modality="image", image_path="/tmp/ref.jpg", caption="reference caption"),
            query="q",
            search_type="pmsr",
        )

        def fake_generate(
            self: PMSRAgent,
            image_path: str,
            prompt: str,
            image_text_pairs=None,
            text_passages=None,
        ) -> str:
            captured["prompt"] = prompt
            captured["image_text_pairs"] = image_text_pairs
            captured["text_passages"] = text_passages
            return "reasoning"

        agent._generate = MethodType(fake_generate, agent)

        agent._synthesize_reasoning(
            "/tmp/image.jpg",
            "What kind of medical usage has this plant?",
            [text_result],
            [image_result],
            description="A green plant.",
        )

        self.assertEqual(
            captured["prompt"],
            "Question: What kind of medical usage has this plant?\n"
            "Description: A green plant.\n"
            "Based on image, description and knowledge, summarize correct and relevant information with image and question.\n",
        )
        self.assertEqual(captured["image_text_pairs"], [{"image_path": "/tmp/ref.jpg", "caption": "reference caption"}])
        self.assertEqual(captured["text_passages"], [{"title": "Plant", "text": "Plant knowledge."}])

    def test_iterative_reasoning_prompt_matches_reference(self) -> None:
        agent = make_agent()
        captured: dict[str, object] = {}

        def fake_generate(self: PMSRAgent, image_path: str, prompt: str, image_text_pairs=None, text_passages=None) -> str:
            captured["prompt"] = prompt
            captured["image_text_pairs"] = image_text_pairs
            captured["text_passages"] = text_passages
            return "reasoning"

        agent._generate = MethodType(fake_generate, agent)

        agent._synthesize_reasoning("/tmp/image.jpg", "What is this?", [], [])

        self.assertEqual(
            captured["prompt"],
            "Question: What is this?\n"
            "Based on image and knowledge, summarize correct and relevant information with image and question.\n",
        )
        self.assertEqual(captured["image_text_pairs"], [])
        self.assertEqual(captured["text_passages"], [])

    def test_final_answer_prompt_matches_reference(self) -> None:
        agent = make_agent()
        captured: dict[str, str] = {}

        def fake_generate(self: PMSRAgent, image_path: str, prompt: str, image_text_pairs=None, text_passages=None) -> str:
            captured["prompt"] = prompt
            return "answer"

        agent._generate = MethodType(fake_generate, agent)
        traj = Trajectory(question="What is this?", image_path="/tmp/image.jpg")
        traj.records.append(Record(step=0, local_query="q", global_query="q", reasoning="reasoning"))

        agent._final_answer(traj)

        self.assertEqual(
            captured["prompt"],
            "Please answer the following question using the provided information and image.\n\n"
            "Question: What is this?\n"
            "Relevant Knowledge: Reasoning Record #1:\nreasoning\n\n",
        )

    def test_record_level_query_uses_latest_reasoning_record_only(self) -> None:
        agent = make_agent()
        traj = Trajectory(question="What kind of medical usage has this plant?", image_path="/tmp/image.jpg")
        traj.records.append(Record(step=0, local_query="q0", global_query="q0", reasoning="old reasoning"))
        traj.records.append(Record(step=1, local_query="q1", global_query="g1", reasoning="latest reasoning"))

        query = agent._build_record_level_query(traj)

        self.assertIn("What kind of medical usage has this plant?", query)
        self.assertIn("latest reasoning", query)
        self.assertNotIn("old reasoning", query)

    def test_trajectory_level_query_transforms_accumulated_reasoning_with_vlm(self) -> None:
        agent = make_agent()
        captured: dict[str, str] = {}

        def fake_generate(self: PMSRAgent, image_path: str, prompt: str, image_text_pairs=None, text_passages=None) -> str:
            captured["image_path"] = image_path
            captured["prompt"] = prompt
            return "## Analysis\nNeed plant usage.\n## Output\nQuestion: Smilax bona-nox medical uses"

        agent._generate = MethodType(fake_generate, agent)
        traj = Trajectory(question="What kind of medical usage has this plant?", image_path="/tmp/image.jpg")
        traj.records.append(Record(step=0, local_query="q0", global_query="q0", reasoning="The plant appears to be Smilax bona-nox."))

        query = agent._build_trajectory_level_query(traj)

        self.assertEqual(query, "Smilax bona-nox medical uses")
        self.assertEqual(captured["image_path"], "/tmp/image.jpg")
        self.assertIn("**Query**: What kind of medical usage has this plant?", captured["prompt"])
        self.assertIn("**Knowledge**: Reasoning Record #1", captured["prompt"])
        self.assertIn("Generate more accurate question", captured["prompt"])

    def test_iterative_step_retrieves_with_record_and_trajectory_level_queries(self) -> None:
        agent = make_agent()
        calls: list[tuple[str, str]] = []

        def fake_build_record_level_query(self: PMSRAgent, traj: Trajectory) -> str:
            return "local query"

        def fake_build_trajectory_level_query(self: PMSRAgent, traj: Trajectory) -> str:
            return "transformed global query"

        def fake_retrieve_text(self: PMSRAgent, query: str, top_k: int):
            calls.append(("text", query))
            return []

        def fake_retrieve_image(self: PMSRAgent, image_path: str, query: str, top_k: int):
            calls.append(("image", query))
            return []

        agent._build_record_level_query = MethodType(fake_build_record_level_query, agent)
        agent._build_trajectory_level_query = MethodType(fake_build_trajectory_level_query, agent)
        agent._retrieve_text = MethodType(fake_retrieve_text, agent)
        agent._retrieve_image = MethodType(fake_retrieve_image, agent)
        agent._synthesize_reasoning = MethodType(lambda self, image_path, question, text_results, image_results: "reasoning", agent)

        traj = Trajectory(question="Question?", image_path="/tmp/image.jpg")
        traj.records.append(Record(step=0, local_query="q0", global_query="q0", reasoning="previous reasoning"))

        record = agent._iterative_step(traj, step=1)

        self.assertEqual(record.local_query, "local query")
        self.assertEqual(record.global_query, "transformed global query")
        self.assertEqual(
            calls,
            [
                ("text", "local query"),
                ("text", "transformed global query"),
                ("image", "local query"),
                ("image", "transformed global query"),
            ],
        )

    def test_run_builds_candidate_record_before_adaptive_stop(self) -> None:
        agent = make_agent()
        agent.config.max_iter = 2
        calls: list[str] = []

        initial = Record(step=0, local_query="initial", global_query="initial", reasoning="initial reasoning")
        candidate = Record(step=1, local_query="candidate local", global_query="candidate global", reasoning="candidate reasoning")

        agent._step0 = MethodType(lambda self, traj, item: initial, agent)
        agent._final_answer = MethodType(lambda self, traj: "final", agent)

        def fake_iterative_step(self: PMSRAgent, traj: Trajectory, step: int) -> Record:
            calls.append(f"iter:{step}:{len(traj.records)}")
            return candidate

        def fake_should_stop(self: PMSRAgent, traj: Trajectory, record: Record) -> bool:
            calls.append(f"stop:{record.local_query}:{len(traj.records)}")
            return True

        agent._iterative_step = MethodType(fake_iterative_step, agent)
        agent._should_stop = MethodType(fake_should_stop, agent)
        agent._build_record_level_query = MethodType(lambda self, traj: (_ for _ in ()).throw(AssertionError("run should not prebuild record-level query")), agent)
        agent._build_trajectory_level_query = MethodType(lambda self, traj: (_ for _ in ()).throw(AssertionError("run should not prebuild trajectory-level query")), agent)

        traj = agent.run({"question": "Question?", "image_path": "/tmp/image.jpg"})

        self.assertEqual(calls, ["iter:1:1", "stop:candidate local:1"])
        self.assertEqual(traj.records, [initial])
        self.assertEqual(traj.final_answer, "final")

    def test_should_stop_strips_repeated_question_prefix_before_similarity(self) -> None:
        agent = make_agent()
        captured: dict[str, list[str]] = {}

        def fake_check_similarity(self: PMSRAgent, query_texts: list[str], candidate_texts: list[str]) -> float:
            captured["query_texts"] = query_texts
            captured["candidate_texts"] = candidate_texts
            return 0.0

        agent._check_similarity = MethodType(fake_check_similarity, agent)
        traj = Trajectory(question="What kind of medical usage has this plant?", image_path="/tmp/image.jpg")
        traj.records.append(
            Record(
                step=0,
                local_query="Question: What kind of medical usage has this plant?\nThe image contains a green plant.",
                global_query="Smilax bona-nox medical uses",
                reasoning="The image contains a green plant.",
            )
        )
        record = Record(
            step=1,
            local_query="Question: What kind of medical usage has this plant?\nIt may be used for urinary tract infections.",
            global_query="Smilax bona-nox urinary tract infections",
            reasoning="It may be used for urinary tract infections.",
        )

        agent._should_stop(traj, record)

        self.assertEqual(
            captured["query_texts"],
            ["It may be used for urinary tract infections.", "Smilax bona-nox urinary tract infections"],
        )
        self.assertEqual(
            captured["candidate_texts"],
            ["The image contains a green plant.", "Smilax bona-nox medical uses"],
        )

    def test_similarity_formats_e5_inputs_before_embedding(self) -> None:
        agent = make_agent()
        agent.config.similarity_model = "intfloat/e5-base-v2"
        calls: list[str] = []

        class FakeEmbedClient:
            def embed_text(self, text: str) -> list[float]:
                calls.append(text)
                return [1.0, 0.0]

        agent._embed_client = FakeEmbedClient()

        score = agent._check_similarity(["a" * 600], ["b" * 600])

        self.assertEqual(score, 1.0)
        self.assertEqual(calls[0], "query: " + ("a" * (511 - len("query: "))))
        self.assertEqual(calls[1], "query: " + ("b" * (511 - len("query: "))))
        self.assertEqual(len(calls[0]), 511)

    def test_similarity_formats_qwen3_inputs_before_embedding(self) -> None:
        agent = make_agent()
        agent.config.similarity_model = "Qwen/Qwen3-Embedding-0.6B"
        calls: list[str] = []

        class FakeEmbedClient:
            def embed_text(self, text: str) -> list[float]:
                calls.append(text)
                return [1.0, 0.0]

        agent._embed_client = FakeEmbedClient()

        score = agent._check_similarity(["c" * 33000], ["d" * 33000])

        self.assertEqual(score, 1.0)
        self.assertEqual(calls[0], "c" * 32767)
        self.assertEqual(calls[1], "d" * 32767)

    def test_similarity_uses_mllm_text_embedding_when_configured(self) -> None:
        agent = make_agent()
        agent.config.similarity_embed_mode = "mllm"
        agent.config.similarity_model = "Qwen/Qwen3-VL-Embedding-2B"
        calls: list[tuple[str, str]] = []

        class FakeEmbedClient:
            def embed_mllm_text(self, *, text: str, instruction: str) -> list[float]:
                calls.append((text, instruction))
                return [1.0, 0.0]

        agent._embed_client = FakeEmbedClient()

        score = agent._check_similarity(["new query"], ["old query"])

        self.assertEqual(score, 1.0)
        self.assertEqual(calls[0][0], "new query")
        self.assertEqual(calls[1][0], "old query")
        self.assertIn("retrieve relevant passages", calls[0][1])


if __name__ == "__main__":
    unittest.main()
