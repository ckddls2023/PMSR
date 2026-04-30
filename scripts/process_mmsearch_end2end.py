#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm


DEFAULT_DATASET = "CaraJ/MMSearch"
DEFAULT_OUTPUT_JSONL = Path("data/MMSearch_end2end.jsonl")
DEFAULT_IMAGE_ROOT = Path("data/processed/mmsearch/end2end/images")
DEFAULT_CACHE_DIR = Path("data/processed/mmsearch/end2end/hf_cache")
DEFAULT_INSTRUCTION = "Using the provided image, answer the question: "


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def has_query_image(value: Any) -> bool:
    value = maybe_json(value)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray, Image.Image)):
        return True
    if isinstance(value, dict):
        return bool(value.get("bytes") is not None or value.get("path"))
    if isinstance(value, list):
        return any(has_query_image(item) for item in value)
    return True


def first_image(value: Any) -> Any:
    value = maybe_json(value)
    if isinstance(value, list):
        for item in value:
            if has_query_image(item):
                return item
        return None
    return value


def image_bytes_from_entry(entry: Any) -> bytes | None:
    if isinstance(entry, Image.Image):
        buffer = BytesIO()
        entry.save(buffer, format="PNG")
        return buffer.getvalue()
    if isinstance(entry, (bytes, bytearray)):
        return bytes(entry)
    if isinstance(entry, list) and all(isinstance(item, int) for item in entry):
        return bytes(entry)
    if isinstance(entry, dict):
        raw_bytes = entry.get("bytes")
        if raw_bytes is not None:
            return image_bytes_from_entry(raw_bytes)
        path = entry.get("path")
        if path:
            return Path(path).expanduser().read_bytes()
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            return None
        if text.startswith("data:") and "," in text:
            text = text.split(",", 1)[1]
        try:
            return base64.b64decode(text, validate=True)
        except Exception:
            path = Path(text).expanduser()
            if path.exists():
                return path.read_bytes()
    return None


def save_image(image_entry: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = image_bytes_from_entry(image_entry)
    if image_bytes is None:
        raise ValueError("Could not extract image bytes from MMSearch query_image.")
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(output_path, format="JPEG", quality=95)


def answer_candidates(gt_answer: Any, alternatives: Any) -> list[Any]:
    alternatives = maybe_json(alternatives)
    candidates: list[Any] = []
    for answer in [gt_answer]:
        if answer is not None and str(answer).strip() and answer not in candidates:
            candidates.append(answer)
    if isinstance(alternatives, list):
        for answer in alternatives:
            if answer is not None and str(answer).strip() and answer not in candidates:
                candidates.append(answer)
    elif alternatives is not None and str(alternatives).strip() and alternatives not in candidates:
        candidates.append(alternatives)
    return candidates


def process_rows(
    rows: Iterable[dict[str, Any]],
    *,
    output_jsonl: str | Path,
    image_root: str | Path,
    split: str = "end2end",
    limit: int | None = None,
    overwrite: bool = True,
) -> int:
    output_path = Path(output_jsonl)
    image_root_path = Path(image_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_root_path.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output JSONL already exists: {output_path}")

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        iterator = tqdm(rows, desc="Processing MMSearch end2end rows", disable=not sys.stderr.isatty())
        for index, row in enumerate(iterator):
            if limit is not None and count >= limit:
                break
            query_image = row.get("query_image")
            if not has_query_image(query_image):
                continue

            sample_id = str(row.get("sample_id") or f"mmsearch_{split}_{index}")
            image_path = image_root_path / f"{sample_id}.jpg"
            save_image(first_image(query_image), image_path)

            gold_answer = row.get("gt_answer")
            answers = answer_candidates(gold_answer, row.get("alternative_gt_answers"))
            record = {
                "question_id": sample_id,
                "image_id": sample_id,
                "question": str(row.get("query") or "").strip(),
                "answer_eval": answers,
                "data_split": split,
                "wikidata_value": None,
                "wikidata_range": None,
                "entity_id": None,
                "entity_text": None,
                "image_path": str(image_path),
                "gold_answer": str(gold_answer or "").strip(),
                "instruction": DEFAULT_INSTRUCTION,
                "answer": answers,
                "sample_id": sample_id,
                "area": row.get("area"),
                "subfield": row.get("subfield"),
                "timestamp": row.get("timestamp"),
                "gt_requery": row.get("gt_requery"),
                "alternative_gt_answers": maybe_json(row.get("alternative_gt_answers")) or [],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def snapshot_dataset(
    dataset: str = DEFAULT_DATASET,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=dataset,
            repo_type="dataset",
            cache_dir=cache_dir,
            revision=revision,
            allow_patterns=["end2end.parquet", "README.md"],
        )
    )


def load_rows(parquet_path: str | Path, *, cache_dir: str | Path | None = None) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and process MMSearch end2end rows into PMSR JSONL.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--parquet-path", help="Use an existing local end2end.parquet instead of downloading.")
    parser.add_argument("--snapshot-dir", help="Use an existing local MMSearch snapshot directory instead of downloading.")
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--revision")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.parquet_path:
        parquet_path = Path(args.parquet_path)
    else:
        snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else snapshot_dataset(
            args.dataset,
            cache_dir=args.cache_dir,
            revision=args.revision,
        )
        parquet_path = snapshot_dir / "end2end.parquet"

    if Path(args.image_root).exists() and args.overwrite:
        shutil.rmtree(args.image_root)
    count = process_rows(
        load_rows(parquet_path, cache_dir=args.cache_dir),
        output_jsonl=args.output_jsonl,
        image_root=args.image_root,
        split="end2end",
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Wrote {count} MMSearch end2end rows to {args.output_jsonl}")
    print(f"Saved images under {args.image_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
