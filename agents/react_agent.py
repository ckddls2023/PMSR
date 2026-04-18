"""ReACT agent — reasoning-augmented tool calling with text/image/pmsr search."""

from __future__ import annotations

import json
import time
from typing import Any

from agents.base_agent import AgentConfig, BaseAgent
from agents.schemas import Record, SearchResult, Trajectory
from api.openai import OpenAICompatibleClient, build_pmsr_user_message
from agents.pmsr_agent import _build_query


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "text_search",
            "description": (
                "Search the text knowledge base for relevant passages using a reasoning-augmented query. "
                "Use when you need factual or encyclopaedic text about entities in the question or image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query derived from current reasoning and question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": (
                "Search the image-document knowledge base using the query image combined with a text query. "
                "Use when you need visually similar reference documents or image-caption pairs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text query for image-document retrieval.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pmsr_search",
            "description": (
                "Dual-scope search: runs a record-level (local) query from the latest reasoning and a "
                "trajectory-level (global) query from the full reasoning history, then merges results. "
                "Use when you need both fine-grained and broad retrieval simultaneously."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "local_query": {
                        "type": "string",
                        "description": "Focused query from the latest reasoning record.",
                    },
                    "global_query": {
                        "type": "string",
                        "description": "Broader query from the accumulated reasoning trajectory.",
                    },
                },
                "required": ["local_query", "global_query"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a visual question answering expert. "
    "Follow a ReACT loop: think about what information is missing, call a search tool to retrieve it, "
    "observe the results, and repeat until you have enough evidence to answer. "
    "When you have sufficient evidence, provide the final answer directly without calling any more tools."
)

_INITIAL_USER_TEMPLATE = (
    "Question: {question}\n"
    "Use the available tools to retrieve relevant knowledge, then answer the question."
)

_FINAL_ANSWER_PROMPT = (
    "Please answer the following question using the provided information and image.\n\n"
    "Question: {question}\n"
    "Relevant Knowledge: {all_reasoning}\n\n"
)


# ---------------------------------------------------------------------------
# ReACTAgent
# ---------------------------------------------------------------------------


class ReACTAgent(BaseAgent):
    """ReACT tool-calling agent with text_search, image_search, and pmsr_search tools."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._vlm = OpenAICompatibleClient(
            model=config.model,
            api_base=config.api_base or None,
            api_key=config.api_key or None,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            retry=config.retry,
        )
        self._text_search = self._build_text_retriever()
        self._image_search = self._build_image_retriever()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, item: dict[str, Any]) -> Trajectory:
        question = str(item.get("question") or "")
        image_path = str(item.get("image_path") or "")
        traj = Trajectory(question=question, image_path=image_path)

        enabled_tools = self._enabled_tools()
        chat_history: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            build_pmsr_user_message(
                image_path=image_path or None,
                prompt=_INITIAL_USER_TEMPLATE.format(question=question),
            ),
        ]

        for step in range(self.config.max_iter):
            t0 = time.time()
            response = self._vlm.chat(
                chat_history,
                extra_body={
                    "tools": enabled_tools,
                    "tool_choice": "auto",
                } if enabled_tools else {},
            )
            tool_calls_raw: list[dict[str, Any]] = response.get("tool_calls") or []

            if not tool_calls_raw:
                # VLM stopped calling tools → final answer
                traj.final_answer = response["content"]
                break

            chat_history.append(response["assistant_message"])

            global_query = ""
            latest_reasoning = ""

            for tc in tool_calls_raw:
                call_id = tc.get("id") or ""
                fn = tc.get("function") or {}
                tool_name = str(fn.get("name") or "")
                try:
                    args: dict[str, Any] = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}

                if not global_query:
                    global_query = _extract_query_from_args(tool_name, args)

                t_results, i_results = self._execute_retrieval(
                    tool_name, args, image_path, latest_reasoning
                )

                # Consolidate this tool call's results into reasoning; only this is kept
                latest_reasoning = self._synthesize_reasoning(
                    image_path, question, t_results, i_results
                )

                chat_history.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": latest_reasoning,
                })

                if self.config.verbose:
                    print(
                        f"[ReACT step={step}] {tool_name}({args}) "
                        f"→ {len(t_results)}t + {len(i_results)}i results"
                    )

            local_query = _build_query(question, latest_reasoning)

            traj.records.append(Record(
                step=step,
                local_query=local_query,
                global_query=global_query or local_query,
                reasoning=latest_reasoning,
                elapsed=round(time.time() - t0, 2),
            ))

        if not traj.final_answer:
            traj.final_answer = self._final_answer(traj)

        return traj

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_retrieval(
        self, tool_name: str, args: dict[str, Any], image_path: str, latest_reasoning: str = ""
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """Execute one tool call and return (text_results, image_results)."""
        if tool_name == "text_search":
            query = _augment_query(str(args.get("query") or ""), latest_reasoning)
            return self._retrieve_text(query, self.config.topk), []

        if tool_name == "image_search":
            query = _augment_query(str(args.get("query") or ""), latest_reasoning)
            return [], self._retrieve_image(image_path, query, max(self.config.topk // 2, 1))

        if tool_name == "pmsr_search":
            local_query = str(args.get("local_query") or "")
            global_query = str(args.get("global_query") or "")
            half_k = max(self.config.topk // 2, 1)
            text_results = _merge_results(
                self._retrieve_text(local_query, half_k),
                self._retrieve_text(global_query, half_k),
            )
            image_results = _merge_results(
                self._retrieve_image(image_path, local_query, half_k),
                self._retrieve_image(image_path, global_query, half_k),
            )
            return text_results, image_results

        return [], []

    # ------------------------------------------------------------------
    # VLM synthesis and final answer
    # ------------------------------------------------------------------

    def _synthesize_reasoning(
        self,
        image_path: str,
        question: str,
        text_results: list[SearchResult],
        image_results: list[SearchResult],
    ) -> str:
        text_passages = [r.to_text_passage() for r in text_results]
        image_text_pairs = [r.to_image_pair() for r in image_results]
        prompt = (
            f"Question: {question}\n"
            "Based on image and knowledge, summarize correct and relevant information "
            "with image and question.\n"
        )
        user_msg = build_pmsr_user_message(
            image_path=image_path or None,
            prompt=prompt,
            image_text_pairs=image_text_pairs or None,
            text_passages=text_passages or None,
        )
        return self._vlm.chat([user_msg])["content"]

    def _final_answer(self, traj: Trajectory) -> str:
        prompt = _FINAL_ANSWER_PROMPT.format(
            question=traj.question,
            all_reasoning=traj.all_reasoning(),
        )
        user_msg = build_pmsr_user_message(
            image_path=traj.image_path or None,
            prompt=prompt,
        )
        response = self._vlm.chat([{"role": "user", "content": user_msg["content"]}])
        return response["content"]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve_text(self, query: str, top_k: int) -> list[SearchResult]:
        if self._text_search is None or not query:
            return []
        try:
            return self._text_search.search(query, top_k=top_k)
        except Exception as exc:
            if self.config.verbose:
                print(f"[ReACT] text retrieval error: {exc}")
            return []

    def _retrieve_image(self, image_path: str, caption: str, top_k: int) -> list[SearchResult]:
        if self._image_search is None or not caption:
            return []
        try:
            return self._image_search.search(
                {"image_path": image_path, "text": caption}, top_k=top_k
            )
        except Exception as exc:
            if self.config.verbose:
                print(f"[ReACT] pmsr retrieval error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Builder helpers  (mirrors PMSRAgent)
    # ------------------------------------------------------------------

    def _build_text_retriever(self):  # type: ignore[return]
        cfg = self.config
        if not cfg.text_kb:
            return None
        if cfg.text_kb.startswith(("http://", "https://")):
            from search.google_search import GoogleSearch
            return GoogleSearch()
        if not cfg.text_metadata or not cfg.text_embed_api_base:
            return None
        from search.text_search import TextSearch, TextSearchConfig
        return TextSearch(TextSearchConfig(
            text_kb=cfg.text_kb,
            text_metadata=cfg.text_metadata,
            text_embed_api_base=cfg.text_embed_api_base,
            text_model=cfg.text_model,
            api_key=cfg.api_key,
        ))

    def _build_image_retriever(self):  # type: ignore[return]
        cfg = self.config
        if not cfg.pmsr_kb or not cfg.pmsr_metadata:
            return None
        if not cfg.image_embed_api_base or not cfg.pmsr_text_embed_api_base:
            return None
        from search.pmsr_search import PMSRSearch, PMSRSearchConfig
        return PMSRSearch(PMSRSearchConfig(
            pmsr_kb=cfg.pmsr_kb,
            pmsr_metadata=cfg.pmsr_metadata,
            image_embed_api_base=cfg.image_embed_api_base,
            text_embed_api_base=cfg.pmsr_text_embed_api_base,
            image_model=cfg.image_model,
            text_model=cfg.pmsr_text_model,
            fusion=cfg.pmsr_fusion,  # type: ignore[arg-type]
            api_key=cfg.api_key,
        ))

    def _enabled_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if self._text_search is not None:
            tools.append(_TOOLS[0])  # text_search
        if self._image_search is not None:
            tools.append(_TOOLS[1])  # image_search
        if self._text_search is not None or self._image_search is not None:
            tools.append(_TOOLS[2])  # pmsr_search
        return tools


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _augment_query(query: str, reasoning: str) -> str:
    """Append latest reasoning to the tool-provided query for richer retrieval."""
    if reasoning:
        return f"{query}\n{reasoning}" if query else reasoning
    return query


def _extract_query_from_args(tool_name: str, args: dict[str, Any]) -> str:
    """Extract the primary query string from tool call arguments."""
    if tool_name in ("text_search", "image_search"):
        return str(args.get("query") or "")
    if tool_name == "pmsr_search":
        return str(args.get("global_query") or args.get("local_query") or "")
    return ""


def _merge_results(*lists: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for results in lists:
        for r in results:
            key = r.evidence.image_path or (
                (r.evidence.title or "") + "|" + (r.evidence.text or r.evidence.caption or "")[:40]
            )
            if key not in seen:
                seen.add(key)
                merged.append(r)
    return merged
