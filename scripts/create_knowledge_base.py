#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search.embedding_client import EmbeddingClient
from search.pmsr_search import DEFAULT_MLLM_PASSAGE_INSTRUCTION


FusionMode = Literal["concat", "image", "text", "mllm"]


class BatchEmbeddingClient(Protocol):
    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_mllm(self, *, image_path: str, text: str, instruction: str) -> list[float]:
        ...


def load_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def l2_normalize_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def split_wiki_contents(contents: str) -> tuple[str, str]:
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if not lines:
        return "", ""
    title = lines[0].strip().strip('"')
    body = "\n".join(lines[1:]).strip()
    return title, body or title


def build_metadata_row(
    record: dict[str, Any],
    *,
    image_field: str,
    text_field: str,
    caption_field: str,
) -> dict[str, Any]:
    image_path = str(record.get(image_field) or "").strip()
    if image_path:
        return {
            "image_path": image_path,
            "caption": str(record.get(caption_field) or record.get(text_field) or "").strip(),
        }

    title = str(record.get("title") or "").strip()
    text = str(record.get("text") or "").strip()
    if not text:
        title_from_contents, text_from_contents = split_wiki_contents(str(record.get(text_field) or ""))
        title = title or title_from_contents
        text = text_from_contents
    return {"title": title, "text": text}


def _valid_image_value(value: str) -> bool:
    if value.startswith(("http://", "https://", "data:")):
        return True
    return Path(value).exists()


def _fuse_embeddings(
    image_vectors: list[list[float]],
    text_vectors: list[list[float]],
    *,
    fusion: FusionMode,
) -> np.ndarray:
    if fusion == "image":
        return l2_normalize_matrix(image_vectors)
    if fusion == "text":
        return l2_normalize_matrix(text_vectors)
    image_matrix = l2_normalize_matrix(image_vectors)
    text_matrix = l2_normalize_matrix(text_vectors)
    if image_matrix.shape[0] != text_matrix.shape[0]:
        raise ValueError(f"Image/text batch size mismatch: {image_matrix.shape[0]} != {text_matrix.shape[0]}")
    return l2_normalize_matrix(np.concatenate([image_matrix, text_matrix], axis=1))


def encode_records(
    records: list[dict[str, Any]],
    *,
    image_client: BatchEmbeddingClient | None,
    text_client: BatchEmbeddingClient | None,
    batch_size: int,
    image_field: str,
    text_field: str,
    caption_field: str,
    fusion: FusionMode = "concat",
    mllm_client: BatchEmbeddingClient | None = None,
    mllm_instruction: str = DEFAULT_MLLM_PASSAGE_INSTRUCTION,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if fusion in {"concat", "image"} and image_client is None:
        raise ValueError(f"fusion={fusion!r} requires an image embedding client.")
    if fusion in {"concat", "text"} and text_client is None:
        raise ValueError(f"fusion={fusion!r} requires a text embedding client.")
    if fusion == "mllm" and mllm_client is None:
        raise ValueError("fusion='mllm' requires an MLLM embedding client.")

    all_vectors: list[np.ndarray] = []
    all_metadata: list[dict[str, Any]] = []
    for batch in tqdm(list(iter_batches(records, batch_size)), desc="Embedding batches"):
        valid_batch: list[dict[str, Any]] = []
        for record in batch:
            image_value = str(record.get(image_field) or "")
            text_value = str(record.get(text_field) or "")
            if fusion in {"concat", "image", "mllm"} and (not image_value or not _valid_image_value(image_value)):
                continue
            if fusion in {"concat", "text", "mllm"} and not text_value:
                continue
            valid_batch.append(record)
        if not valid_batch:
            continue

        image_vectors: list[list[float]] = []
        text_vectors: list[list[float]] = []
        if fusion in {"concat", "image"}:
            image_vectors = image_client.embed_images([str(record.get(image_field) or "") for record in valid_batch])  # type: ignore[union-attr]
        if fusion in {"concat", "text"}:
            text_vectors = text_client.embed_texts([str(record.get(text_field) or "") for record in valid_batch])  # type: ignore[union-attr]

        if fusion == "mllm":
            vectors = l2_normalize_matrix(
                [
                    mllm_client.embed_mllm(  # type: ignore[union-attr]
                        image_path=str(record.get(image_field) or ""),
                        text=str(record.get(text_field) or ""),
                        instruction=mllm_instruction,
                    )
                    for record in valid_batch
                ]
            )
        else:
            vectors = _fuse_embeddings(image_vectors, text_vectors, fusion=fusion)
        all_vectors.append(vectors)
        all_metadata.extend(
            build_metadata_row(
                record,
                image_field=image_field,
                text_field=text_field,
                caption_field=caption_field,
            )
            for record in valid_batch
        )

    if not all_vectors:
        return np.empty((0, 0), dtype="float32"), []
    return np.vstack(all_vectors).astype("float32"), all_metadata


def write_metadata_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_faiss_index(path: str | Path, vectors: np.ndarray) -> None:
    if vectors.size == 0:
        raise ValueError("No vectors to index.")
    import faiss

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype="float32"))
    faiss.write_index(index, str(output_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create PMSR FAISS KB using embedding API servers.")
    parser.add_argument("--input-jsonl", "--jsonl", dest="input_jsonl", required=True)
    parser.add_argument("--index-output", required=True, help="Output FAISS index path.")
    parser.add_argument("--metadata-output", required=True, help="Output metadata CSV path.")
    parser.add_argument("--image-embed-api-base", default=os.environ.get("IMAGE_EMBED_API_BASE", ""))
    parser.add_argument(
        "--text-embed-api-base",
        default=os.environ.get("QWEN_TEXT_EMBED_API_BASE", os.environ.get("TEXT_EMBED_API_BASE", "")),
    )
    parser.add_argument("--image-model", default=os.environ.get("IMAGE_EMBED_MODEL", "google/siglip2-giant-opt-patch16-384"))
    parser.add_argument("--text-model", default=os.environ.get("QWEN_TEXT_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"))
    parser.add_argument("--mllm-embed-api-base", default=os.environ.get("MLLM_EMBED_API_BASE", ""))
    parser.add_argument("--mllm-model", default=os.environ.get("MLLM_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B"))
    parser.add_argument("--mllm-instruction", default=os.environ.get("MLLM_PASSAGE_INSTRUCTION", DEFAULT_MLLM_PASSAGE_INSTRUCTION))
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--text-field", default="wikipedia_summary")
    parser.add_argument("--caption-field", default="wikipedia_summary")
    parser.add_argument("--fusion", choices=["concat", "image", "text", "mllm"], default="concat")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--api-key", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_client = None
    text_client = None
    mllm_client = None
    if args.fusion in {"concat", "image"}:
        if not args.image_embed_api_base:
            raise SystemExit("--image-embed-api-base is required for image or concat fusion.")
        image_client = EmbeddingClient(
            api_base=args.image_embed_api_base,
            model=args.image_model,
            timeout=args.timeout,
            api_key=args.api_key,
        )
    if args.fusion in {"concat", "text"}:
        if not args.text_embed_api_base:
            raise SystemExit("--text-embed-api-base is required for text or concat fusion.")
        text_client = EmbeddingClient(
            api_base=args.text_embed_api_base,
            model=args.text_model,
            timeout=args.timeout,
            api_key=args.api_key,
        )
    if args.fusion == "mllm":
        if not args.mllm_embed_api_base:
            raise SystemExit("--mllm-embed-api-base is required for mllm fusion.")
        mllm_client = EmbeddingClient(
            api_base=args.mllm_embed_api_base,
            model=args.mllm_model,
            timeout=args.timeout,
            api_key=args.api_key,
        )

    records = load_jsonl(args.input_jsonl, limit=args.limit)
    vectors, metadata = encode_records(
        records,
        image_client=image_client,
        text_client=text_client,
        mllm_client=mllm_client,
        batch_size=args.batch_size,
        image_field=args.image_field,
        text_field=args.text_field,
        caption_field=args.caption_field,
        fusion=args.fusion,
        mllm_instruction=args.mllm_instruction,
    )
    write_faiss_index(args.index_output, vectors)
    write_metadata_csv(args.metadata_output, metadata)
    print(f"Wrote {vectors.shape[0]} vectors with dim={vectors.shape[1]} to {args.index_output}")
    print(f"Wrote metadata to {args.metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
