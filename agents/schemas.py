from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EvidenceModality = Literal["text", "image", "multimodal", "web"]


@dataclass(frozen=True, slots=True)
class Evidence:
    source: str
    modality: EvidenceModality
    title: str = ""
    text: str = ""
    url: str = ""
    image_path: str = ""
    caption: str = ""
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text_passage(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text or self.caption,
            "url": self.url,
            "source": self.source,
            "score": self.score,
            **self.metadata,
        }

    def to_image_pair(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path or self.url,
            "caption": self.caption or self.title or self.text,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "score": self.score,
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    evidence: Evidence
    query: str = ""
    search_type: str = ""

    @property
    def score(self) -> float:
        return self.evidence.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "search_type": self.search_type,
            "source": self.evidence.source,
            "modality": self.evidence.modality,
            "title": self.evidence.title,
            "text": self.evidence.text,
            "url": self.evidence.url,
            "image_path": self.evidence.image_path,
            "caption": self.evidence.caption,
            "score": self.evidence.score,
            "rank": self.evidence.rank,
            "metadata": dict(self.evidence.metadata),
        }

    def to_text_passage(self) -> dict[str, Any]:
        return self.evidence.to_text_passage()

    def to_image_pair(self) -> dict[str, Any]:
        return self.evidence.to_image_pair()


# ---------------------------------------------------------------------------
# PMSR trajectory types
# ---------------------------------------------------------------------------


@dataclass
class Record:
    """One step in the PMSR trajectory."""

    step: int
    local_query: str
    global_query: str
    text_results: list[SearchResult] = field(default_factory=list)
    image_results: list[SearchResult] = field(default_factory=list)
    reasoning: str = ""
    follow_up_question: str = ""
    elapsed: float = 0.0


@dataclass
class Trajectory:
    """Accumulates the full progressive reasoning state for one item."""

    question: str
    image_path: str
    records: list[Record] = field(default_factory=list)
    final_answer: str = ""

    def latest_reasoning(self) -> str:
        return self.records[-1].reasoning if self.records else ""

    def all_reasoning(self) -> str:
        parts: list[str] = []
        for rec in self.records:
            if rec.reasoning:
                parts.append(f"Reasoning Record #{rec.step + 1}:\n{rec.reasoning}")
        return "\n\n".join(parts)

    def history_questions(self) -> list[str]:
        questions: list[str] = [self.question]
        for rec in self.records:
            if rec.follow_up_question:
                questions.append(rec.follow_up_question)
        return questions

    def all_knowledge(self) -> str:
        parts: list[str] = []
        for rec in self.records:
            for r in rec.text_results:
                p = r.to_text_passage()
                title = str(p.get("title") or "").strip()
                text = str(p.get("text") or "").strip()
                entry = f"{title}\n{text}" if title and text and title != text else text or title
                if entry:
                    parts.append(entry)
            for r in rec.image_results:
                p = r.to_image_pair()
                caption = str(p.get("caption") or p.get("title") or "").strip()
                if caption:
                    parts.append(caption)
        return "\n\n".join(parts)
