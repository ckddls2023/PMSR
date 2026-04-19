from __future__ import annotations

import csv
import json
import sys
from array import array
from pathlib import Path
from typing import Any, Sequence

from agents.schemas import Evidence, SearchResult
from search.base_search import clamp_top_k

MAX_IMAGE_TEXT_CHARS = 4096


def l2_normalize(values: Sequence[float]) -> list[float]:
    import math

    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class MetadataStore:
    def __len__(self) -> int:
        raise NotImplementedError

    def get(self, row_id: int) -> dict[str, Any]:
        raise NotImplementedError


class ListMetadataStore(MetadataStore):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = [normalize_metadata_row(row) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, row_id: int) -> dict[str, Any]:
        if row_id < 0 or row_id >= len(self.rows):
            return {}
        return self.rows[row_id]


class JsonlMetadataStore(MetadataStore):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offsets = array("Q")
        self._next_offset = 0
        self._eof = False

    def __len__(self) -> int:
        self._ensure_offset(None)
        return len(self.offsets)

    def get(self, row_id: int) -> dict[str, Any]:
        if row_id < 0:
            return {}
        self._ensure_offset(row_id)
        if row_id >= len(self.offsets):
            return {}
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[row_id])
            line = handle.readline().decode("utf-8")
        return normalize_metadata_row(json.loads(line)) if line.strip() else {}

    def _ensure_offset(self, row_id: int | None) -> None:
        if self._eof:
            return
        if row_id is not None and row_id < len(self.offsets):
            return
        with self.path.open("rb") as handle:
            handle.seek(self._next_offset)
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    self._eof = True
                    break
                if line.strip():
                    self.offsets.append(offset)
                    if row_id is not None and row_id < len(self.offsets):
                        break
            self._next_offset = handle.tell()


def load_metadata(path: str | Path) -> MetadataStore:
    metadata_path = Path(path)
    if metadata_path.suffix == ".jsonl":
        return JsonlMetadataStore(metadata_path)
    if metadata_path.suffix == ".json":
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return ListMetadataStore([row for row in payload if isinstance(row, dict)])
        raise ValueError(f"JSON metadata must be a list of objects: {metadata_path}")
    set_csv_field_size_limit()
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        return ListMetadataStore(list(csv.DictReader(handle)))


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


def truncate_image_text(text: Any) -> str:
    return str(text or "").strip()[:MAX_IMAGE_TEXT_CHARS]


def normalize_metadata_row(row: dict[str, Any]) -> dict[str, Any]:
    image_path = str(row.get("image_path") or "").strip()
    if not image_path:
        return row
    caption = (
        row.get("caption")
        or row.get("wikipedia_summary")
        or row.get("text")
        or row.get("wikipedia_content")
        or row.get("contents")
        or ""
    )
    return {"image_path": image_path, "caption": truncate_image_text(caption)}


def split_wiki_contents(contents: str) -> tuple[str, str]:
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if not lines:
        return "", ""
    title = lines[0].strip().strip('"')
    body = "\n".join(lines[1:]).strip()
    return title, body or title


class FaissKnowledgeBase:
    def __init__(self, *, index_path: str | Path, metadata_path: str | Path, source: str = "pmsr_faiss") -> None:
        import faiss

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = load_metadata(self.metadata_path)
        self.source = source

    def search_vector(self, vector: Sequence[float], *, top_k: int, query: str, search_type: str) -> list[SearchResult]:
        import numpy as np
        import faiss

        query_vector = np.asarray([l2_normalize(vector)], dtype="float32")
        if query_vector.shape[1] != self.index.d:
            raise ValueError(f"Query dimension {query_vector.shape[1]} does not match FAISS index dimension {self.index.d}.")
        k = clamp_top_k(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)
        results: list[SearchResult] = []
        for rank, (row_id, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if row_id < 0:
                continue
            record = self.metadata.get(int(row_id))
            results.append(self._record_to_result(record, int(row_id), float(score), rank, query, search_type))
        return results

    def _record_to_result(
        self,
        record: dict[str, Any],
        row_id: int,
        score: float,
        rank: int,
        query: str,
        search_type: str,
    ) -> SearchResult:
        contents_title, contents_text = split_wiki_contents(str(record.get("contents") or ""))
        image_path = str(record.get("image_path") or "").strip()
        if image_path:
            evidence = Evidence(
                source=self.source,
                modality="image",
                image_path=image_path,
                caption=truncate_image_text(record.get("caption") or record.get("wikipedia_summary") or ""),
                score=score,
                rank=rank,
            )
        else:
            evidence = Evidence(
                source=self.source,
                modality="text",
                title=str(record.get("title") or contents_title or "").strip(),
                text=str(record.get("text") or contents_text or "").strip(),
                score=score,
                rank=rank,
            )
        return SearchResult(evidence=evidence, query=query, search_type=search_type)
