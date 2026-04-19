"""ReACT-style PMSR agent with model-planned dual-scope search."""

from __future__ import annotations

import json
import time
from typing import Any

from agents.base_agent import AgentConfig, BaseAgent
from agents.schemas import Record, SearchResult, Trajectory
from api.openai import OpenAICompatibleClient, build_pmsr_user_message


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "pmsr_search",
            "description": (
                "Dual-scope PMSR knowledge-base search. Runs a record-level query from the latest reasoning "
                "and a trajectory-level query from the full reasoning history, then merges retrieved evidence. "
                "Depending on runtime options, this single tool may use PMSR KB retrieval, web text search, "
                "or Google Lens image search internally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_level_query": {
                        "type": "string",
                        "description": "Focused query from the latest reasoning record.",
                    },
                    "trajectory_level_query": {
                        "type": "string",
                        "description": "Broader query from the accumulated reasoning trajectory.",
                    },
                },
                "required": ["record_level_query", "trajectory_level_query"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a visual question answering expert. "
    "Follow a ReACT loop using only the pmsr_search tool. "
    "Decompose the question into focused retrieval steps, plan record-level and trajectory-level queries, "
    "observe the PMSR image-text evidence, and repeat until you have enough evidence to answer. "
    "The record-level query should target the latest missing detail. "
    "The trajectory-level query should reflect the full reasoning state and broader entity context. "
    "When you have sufficient evidence, provide the final answer directly without calling any more tools."
)

_INITIAL_USER_TEMPLATE = (
    "Question: {question}\n"
    "Use pmsr_search to retrieve PMSR multimodal evidence step by step, then answer the question."
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
    """ReACT-style agent that lets the model plan PMSR dual-scope search calls."""

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
        self._pmsr_search = self._build_pmsr_retriever()
        self._web_search = self._build_web_retriever()
        self._google_image_search = self._build_google_image_retriever()
        self._google_image_cache: dict[str, list[SearchResult]] = {}

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
            record_query = ""
            latest_reasoning = ""
            latest_t_results: list[SearchResult] = []
            latest_i_results: list[SearchResult] = []

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
                if not record_query:
                    record_query = _extract_record_query_from_args(tool_name, args)

                t_results, i_results = self._execute_retrieval(
                    tool_name, args, image_path, latest_reasoning
                )

                # Consolidate this tool call's results; VLM sees only the synthesis
                latest_reasoning = self._synthesize_reasoning(
                    image_path, question, t_results, i_results
                )
                latest_t_results = t_results
                latest_i_results = i_results

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

            traj.records.append(Record(
                step=step,
                local_query=record_query,
                global_query=global_query or record_query,
                text_results=latest_t_results,
                image_results=latest_i_results,
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
        if tool_name == "pmsr_search":
            record_level_query = str(args.get("record_level_query") or args.get("local_query") or "")
            trajectory_level_query = str(args.get("trajectory_level_query") or args.get("global_query") or "")
            text_results = self._retrieve_web_dual_scope(
                record_level_query,
                trajectory_level_query,
                self.config.topk,
            ) if self.config.web_search else []
            if self.config.google_lens_search:
                image_results = self._retrieve_google_image_once(
                    image_path,
                    _join_queries(record_level_query, trajectory_level_query),
                    self.config.topk,
                )
            else:
                image_results = self._retrieve_pmsr_dual_scope(
                    image_path,
                    record_level_query,
                    trajectory_level_query,
                    self.config.topk,
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

    def _retrieve_web_dual_scope(
        self,
        record_level_query: str,
        trajectory_level_query: str,
        top_k: int,
    ) -> list[SearchResult]:
        if self._web_search is None:
            return []
        half_k = max(top_k // 2, 1)
        try:
            return _merge_results(
                self._web_search.search(record_level_query, top_k=half_k) if record_level_query else [],
                self._web_search.search(trajectory_level_query, top_k=half_k) if trajectory_level_query else [],
            )
        except Exception as exc:
            if self.config.verbose:
                print(f"[ReACT] web retrieval error: {exc}")
            return []

    def _retrieve_google_image_once(self, image_path: str, question: str, top_k: int) -> list[SearchResult]:
        if self._google_image_search is None or not image_path:
            return []
        cache_key = image_path
        if cache_key in self._google_image_cache:
            return self._google_image_cache[cache_key]
        try:
            results = self._google_image_search.search(
                {"image_path": image_path, "question": question}, top_k=top_k
            )
            self._google_image_cache[cache_key] = results
            return results
        except Exception as exc:
            if self.config.verbose:
                print(f"[ReACT] google image retrieval error: {exc}")
            return []

    def _retrieve_pmsr_dual_scope(
        self,
        image_path: str,
        record_level_query: str,
        trajectory_level_query: str,
        top_k: int,
    ) -> list[SearchResult]:
        if self._pmsr_search is None or not image_path:
            return []
        half_k = max(top_k // 2, 1)
        try:
            return _merge_results(
                self._pmsr_search.search(
                    {"image_path": image_path, "text": record_level_query}, top_k=half_k
                ) if record_level_query else [],
                self._pmsr_search.search(
                    {"image_path": image_path, "text": trajectory_level_query}, top_k=half_k
                ) if trajectory_level_query else [],
            )
        except Exception as exc:
            if self.config.verbose:
                print(f"[ReACT] pmsr retrieval error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Builder helpers  (mirrors PMSRAgent)
    # ------------------------------------------------------------------

    def _build_web_retriever(self):  # type: ignore[return]
        if not self.config.web_search:
            return None
        from search.google_search import GoogleSearch
        return GoogleSearch()

    def _build_google_image_retriever(self):  # type: ignore[return]
        if not self.config.google_lens_search:
            return None
        import os
        if not os.environ.get("SCRAPINGDOG_API_KEY"):
            return None
        from search.google_image_search import GoogleImageSearch
        return GoogleImageSearch()

    def _build_pmsr_retriever(self):  # type: ignore[return]
        cfg = self.config
        from search.pmsr_search import PMSRSearch, PMSRSearchConfig
        if cfg.pmsr_fusion == "mllm":
            if not cfg.mllm_kb or not cfg.mllm_metadata or not cfg.mllm_embed_api_base:
                return None
            return PMSRSearch(PMSRSearchConfig(
                mllm_kb=cfg.mllm_kb,
                mllm_metadata=cfg.mllm_metadata,
                mllm_embed_api_base=cfg.mllm_embed_api_base,
                mllm_model=cfg.mllm_model,
                fusion="mllm",
                api_key=cfg.api_key,
            ))

        if not cfg.pmsr_kb or not cfg.pmsr_metadata:
            return None
        if not cfg.image_embed_api_base or not cfg.pmsr_text_embed_api_base:
            return None
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
        if self._pmsr_search is not None or self._web_search is not None or self._google_image_search is not None:
            tools.append(_TOOLS[0])  # pmsr_search
        return tools


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_query_from_args(tool_name: str, args: dict[str, Any]) -> str:
    """Extract the primary query string from tool call arguments."""
    if tool_name == "pmsr_search":
        return str(
            args.get("trajectory_level_query")
            or args.get("global_query")
            or args.get("record_level_query")
            or args.get("local_query")
            or ""
        )
    return ""


def _extract_record_query_from_args(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name != "pmsr_search":
        return ""
    return str(args.get("record_level_query") or args.get("local_query") or "")


def _join_queries(record_level_query: str, trajectory_level_query: str) -> str:
    return "\n".join(query for query in (record_level_query, trajectory_level_query) if query)


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
