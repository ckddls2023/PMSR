"""PMSR Agent — progressive multi-step reasoning with dual-scope retrieval."""

from __future__ import annotations

import time
import re
from typing import Any

from agents.base_agent import AgentConfig, BaseAgent
from agents.schemas import Evidence, Record, SearchResult, Trajectory
from api.openai import OpenAICompatibleClient, build_pmsr_user_message


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_GLOBAL_QUERY_PROMPT = (
    "**Query**: {question}\n"
    "**Knowledge**: {knowledge}\n\n"
    "Please first analyze all the information in a section named Analysis (## Analysis). "
    "Generate more accurate question based on the Knowledge to search more information helpful "
    "to addressing Query.\n"
    "Your response should be in the following format:\n"
    "## Analysis\n"
    "Analysis query and correct knowledge to search more accurately.\n"
    "## Output\n"
    "Question: context-specific new question\n"
)


# ---------------------------------------------------------------------------
# PMSRAgent
# ---------------------------------------------------------------------------

class PMSRAgent(BaseAgent):
    """Progressive Multi-Step Reasoning agent with dual-scope retrieval."""

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
        self._embed_client = self._build_embed_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, item: dict[str, Any]) -> Trajectory:
        question = str(item.get("question") or "")
        image_path = str(item.get("image_path") or "")
        traj = Trajectory(question=question, image_path=image_path)

        traj.records.append(self._step0(traj, item))

        for step in range(1, self.config.max_iter + 1):
            record = self._iterative_step(traj, step)
            if self._should_stop(traj, record):
                if self.config.verbose:
                    print(f"[PMSRAgent] adaptive stop before step {step}")
                break
            traj.records.append(record)

        traj.final_answer = self._final_answer(traj)
        return traj

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _step0(self, traj: Trajectory, item: dict[str, Any]) -> Record:
        t0 = time.time()
        question = traj.question
        image_path = traj.image_path

        description = self._describe_image(image_path, question)
        query = f"Question: {question}\n{description}"

        text_results = self._retrieve_text(query, self.config.topk * 2)
        cached = self._load_cached_image_results(item)
        pmsr_results: list[SearchResult] = []
        if not cached:
            pmsr_results = self._retrieve_image(image_path, query, self.config.topk)
        image_results = self._merge_results(cached, pmsr_results)

        reasoning = self._synthesize_reasoning(
            image_path,
            question,
            text_results,
            image_results,
            description=description,
        )

        return Record(
            step=0,
            local_query=query,
            global_query=query,
            text_results=text_results,
            image_results=image_results,
            reasoning=reasoning,
            elapsed=time.time() - t0,
        )

    def _iterative_step(self, traj: Trajectory, step: int) -> Record:
        t0 = time.time()
        question = traj.question
        image_path = traj.image_path

        local_query = self._build_record_level_query(traj)
        global_query = self._build_trajectory_level_query(traj)

        # Dual-scope retrieval
        text_results = self._merge_results(
            self._retrieve_text(local_query, self.config.topk),
            self._retrieve_text(global_query, self.config.topk),
        )
        image_results = self._merge_results(
            self._retrieve_image(image_path, local_query, self.config.topk),
            self._retrieve_image(image_path, global_query, self.config.topk),
        )

        reasoning = self._synthesize_reasoning(image_path, question, text_results, image_results)

        return Record(
            step=step,
            local_query=local_query,
            global_query=global_query,
            text_results=text_results,
            image_results=image_results,
            reasoning=reasoning,
            elapsed=time.time() - t0,
        )

    def _build_record_level_query(self, traj: Trajectory) -> str:
        """Build the record-level retrieval query from the latest reasoning record."""
        return _build_query(traj.question, traj.latest_reasoning())

    def _build_trajectory_level_query(self, traj: Trajectory) -> str:
        """Transform accumulated reasoning into a focused trajectory-level query."""
        knowledge = traj.all_reasoning() or traj.latest_reasoning()
        prompt = _GLOBAL_QUERY_PROMPT.format(question=traj.question, knowledge=knowledge)
        try:
            raw_query = self._generate(traj.image_path, prompt)
        except Exception as exc:
            if self.config.verbose:
                print(f"[PMSRAgent] global query transformation failed: {exc}")
            return _build_query(traj.question, knowledge)
        parsed_query = _extract_generated_question(raw_query)
        return parsed_query or _build_query(traj.question, knowledge)

    # ------------------------------------------------------------------
    # VLM calls
    # ------------------------------------------------------------------

    def _generate(
        self,
        image_path: str,
        prompt: str,
        image_text_pairs: list[Any] | None = None,
        text_passages: list[Any] | None = None,
    ) -> str:
        user_msg = build_pmsr_user_message(
            image_path=image_path or None,
            prompt=prompt,
            image_text_pairs=image_text_pairs,
            text_passages=text_passages,
        )
        return self._vlm.chat([user_msg])["content"]

    def _describe_image(self, image_path: str, question: str) -> str:
        return self._generate(
            image_path,
            f"Question: {question}\nConcisely describe image which is relevant to question.\n",
        )

    def _synthesize_reasoning(
        self,
        image_path: str,
        question: str,
        text_results: list[SearchResult],
        image_results: list[SearchResult],
        description: str = "",
    ) -> str:
        text_passages = self._to_text_passages(text_results)
        image_text_pairs = self._to_image_text_pairs(image_results)
        if description:
            prompt = (
                f"Question: {question}\n"
                f"Description: {description}\n"
            )
            prompt += (
                "Based on image, description and knowledge, summarize correct and relevant "
                "information with image and question.\n"
            )
        else:
            prompt = f"Question: {question}\n"
            prompt += "Based on image and knowledge, summarize correct and relevant information with image and question.\n"
        return self._generate(
            image_path,
            prompt,
            image_text_pairs=image_text_pairs,
            text_passages=text_passages,
        )

    def _final_answer(self, traj: Trajectory) -> str:
        prompt = (
            "Please answer the following question using the provided information and image.\n\n"
            f"Question: {traj.question}\n"
            f"Relevant Knowledge: {traj.all_reasoning()}\n\n"
        )
        return self._generate(traj.image_path, prompt)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve_text(self, query: str, top_k: int) -> list[SearchResult]:
        if self._text_search is None:
            return []
        try:
            return self._text_search.search(query, top_k=top_k)
        except Exception as exc:
            if self.config.verbose:
                print(f"[PMSRAgent] text retrieval error: {exc}")
            return []

    def _retrieve_image(self, image_path: str, caption: str, top_k: int) -> list[SearchResult]:
        """Search pmsr_kb with concat(image + text) query."""
        if self._image_search is None:
            return []
        try:
            return self._image_search.search(
                {"image_path": image_path, "text": caption}, top_k=top_k
            )
        except Exception as exc:
            if self.config.verbose:
                print(f"[PMSRAgent] pmsr retrieval error: {exc}")
            return []

    def _load_cached_image_results(self, item: dict[str, Any]) -> list[SearchResult]:
        """Load Google Image Search results cached by scripts/cache_google_image_search.py."""
        raw_list = (item.get("searched_results") or {}).get("google_image") or []
        results: list[SearchResult] = []
        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            evidence = Evidence(
                source="google_image",
                modality="image",
                image_path=raw.get("image_path") or "",
                caption=raw.get("caption") or "",
            )
            results.append(SearchResult(
                evidence=evidence,
                query=item.get("question") or "",
                search_type="google_image",
            ))
        return results

    # ------------------------------------------------------------------
    # Adaptive stopping
    # ------------------------------------------------------------------

    def _should_stop(self, traj: Trajectory, record: Record) -> bool:
        """Paper §3.3: stop when δquery(t) ≥ τ.

        δquery(t) = max sim between the candidate record queries and all
        previous record queries.
        """
        if len(traj.records) < 1:
            return False

        # All queries used in previous records
        previous: list[str] = []
        for rec in traj.records:
            previous.append(_strip_question_prefix(rec.local_query))
            if rec.global_query != rec.local_query:
                previous.append(_strip_question_prefix(rec.global_query))

        candidate = [_strip_question_prefix(record.local_query)]
        if record.global_query != record.local_query:
            candidate.append(_strip_question_prefix(record.global_query))

        delta = self._check_similarity(candidate, previous)
        if self.config.verbose:
            print(f"[Adaptive] δ={delta:.3f}  τ={self.config.threshold}")
        return delta >= self.config.threshold

    def _check_similarity(self, query_texts: list[str], candidate_texts: list[str]) -> float:
        """Max cosine similarity between query_texts and candidate_texts.

        Embeds all texts via text_embed_api_base (EmbeddingClient),
        then computes cosine similarity as dot product of L2-normalised vectors.
        """
        if self._embed_client is None or not query_texts or not candidate_texts:
            return 0.0
        try:
            from search.faiss_search import l2_normalize
            q_vecs = [l2_normalize(self._embed_client.embed_text(t)) for t in query_texts]
            c_vecs = [l2_normalize(self._embed_client.embed_text(t)) for t in candidate_texts]
            return max(
                float(sum(q * c for q, c in zip(qv, cv)))
                for qv in q_vecs
                for cv in c_vecs
            )
        except Exception as exc:
            if self.config.verbose:
                print(f"[Adaptive] similarity computation failed: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Merging / formatting
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _to_text_passages(text_results: list[SearchResult]) -> list[dict[str, Any]]:
        return [result.to_text_passage() for result in text_results]

    def _to_image_text_pairs(self, image_results: list[SearchResult]) -> list[dict[str, Any]]:
        return [result.to_image_pair() for result in image_results]

    # ------------------------------------------------------------------
    # Builder helpers
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

    def _build_embed_client(self):  # type: ignore[return]
        """EmbeddingClient used solely for adaptive stopping similarity computation."""
        cfg = self.config
        if not cfg.text_embed_api_base:
            return None
        from search.embedding_client import EmbeddingClient
        return EmbeddingClient(
            api_base=cfg.text_embed_api_base,
            model=cfg.text_model,
            api_key=cfg.api_key,
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _build_query(question: str, context: str) -> str:
    if context:
        return f"Question: {question}\n{context}"
    return f"Question: {question}"


def _strip_question_prefix(query: str) -> str:
    return re.sub(r"(?is)^Question\s*:\s*.+?\n", "", str(query or "").strip(), count=1).strip()


def _extract_generated_question(text: str) -> str:
    output_match = re.search(r"(?is)##\s*Output.*?Question\s*:\s*(.+?)(?:\n|$)", text.strip())
    if output_match:
        return output_match.group(1).strip()
    fallback_match = re.search(r"(?i)Question\s*:\s*(.+?)(?:\n|$)", text.strip())
    if fallback_match:
        return fallback_match.group(1).strip()
    return ""
