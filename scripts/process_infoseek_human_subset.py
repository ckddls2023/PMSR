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
from urllib.request import urlretrieve

from PIL import Image
from tqdm import tqdm


DEFAULT_PARQUET_URL = (
    "https://raw.githubusercontent.com/EvolvingLMMs-Lab/multimodal-search-r1/"
    "main/mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet"
)
DEFAULT_PARQUET_PATH = Path("data/processed/infoseek_human/mmsearch_r1_infoseek_sub_2k.parquet")
DEFAULT_OUTPUT_JSONL = Path("data/InfoSeek_human_2k.jsonl")
DEFAULT_IMAGE_ROOT = Path("data/processed/infoseek_human/images")
DEFAULT_CACHE_DIR = Path("data/processed/infoseek_human/hf_cache")
DEFAULT_INSTRUCTION = "Using the provided image, obtain documents that address the subsequent question: "


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return " ".join(part.strip() for part in parts if part.strip()).strip()
    return str(content or "").strip()


def extract_question(prompt: Any) -> str:
    prompt = maybe_json(prompt)
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user" and message.get("content") is not None:
                return _content_to_text(message.get("content"))
        for message in prompt:
            if isinstance(message, dict) and message.get("content") is not None:
                return _content_to_text(message.get("content"))
    if isinstance(prompt, dict):
        return _content_to_text(prompt.get("content") or prompt.get("text") or prompt)
    return str(prompt or "").strip()


def extract_answers(reward_model: Any) -> tuple[str, list[Any]]:
    reward_model = maybe_json(reward_model)
    if not isinstance(reward_model, dict):
        reward_model = {}
    gold_answer = str(reward_model.get("ground_truth") or "").strip()
    candidates = maybe_json(reward_model.get("candidate_answers") or [])
    if isinstance(candidates, list):
        answers = [answer for answer in candidates if str(answer).strip()]
    elif candidates:
        answers = [candidates]
    else:
        answers = []
    if not answers and gold_answer:
        answers = [gold_answer]
    return gold_answer, answers


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


def first_image(images: Any) -> Any:
    images = maybe_json(images)
    if isinstance(images, list):
        if not images:
            raise ValueError("InfoSeek Human row has no images.")
        return images[0]
    return images


def save_image(image_entry: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = image_bytes_from_entry(image_entry)
    if image_bytes is None:
        raise ValueError("Could not extract image bytes from InfoSeek Human row.")
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(output_path, format="JPEG", quality=95)


def process_rows(
    rows: Iterable[dict[str, Any]],
    *,
    output_jsonl: str | Path,
    image_root: str | Path,
    split: str = "infoseek_human",
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
        iterator = tqdm(rows, desc="Processing InfoSeek Human rows", disable=not sys.stderr.isatty())
        for index, row in enumerate(iterator):
            if limit is not None and count >= limit:
                break
            data_id = str(row.get("data_id") or row.get("sample_id") or f"infoseek_human_{index}")
            image_id = str(row.get("image_id") or data_id)
            image_path = image_root_path / f"{image_id}.jpg"
            save_image(first_image(row.get("images") or row.get("query_image")), image_path)

            gold_answer, answers = extract_answers(row.get("reward_model"))
            record = {
                "question_id": data_id,
                "image_id": image_id,
                "question": extract_question(row.get("prompt")),
                "answer_eval": answers,
                "data_split": split,
                "wikidata_value": None,
                "wikidata_range": None,
                "entity_id": None,
                "entity_text": None,
                "image_path": str(image_path),
                "gold_answer": gold_answer,
                "instruction": DEFAULT_INSTRUCTION,
                "answer": answers,
                "data_id": data_id,
                "sample_id": row.get("sample_id"),
                "data_source": row.get("data_source"),
                "category": row.get("category"),
                "candidate_answers": answers,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def resolve_parquet_path(parquet_path: str | Path | None, parquet_url: str, destination: str | Path) -> Path:
    if parquet_path:
        return Path(parquet_path).expanduser()
    destination_path = Path(destination).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not destination_path.exists():
        urlretrieve(parquet_url, destination_path)
    return destination_path


def load_rows(parquet_path: str | Path, *, cache_dir: str | Path | None = None) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and process InfoSeek Human 2K subset into PMSR JSONL.")
    parser.add_argument("--parquet-url", default=DEFAULT_PARQUET_URL)
    parser.add_argument("--parquet-path", help="Use an existing local parquet instead of downloading from GitHub.")
    parser.add_argument("--download-path", default=str(DEFAULT_PARQUET_PATH))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    parquet_path = resolve_parquet_path(args.parquet_path, args.parquet_url, args.download_path)

    if Path(args.image_root).exists() and args.overwrite:
        shutil.rmtree(args.image_root)
    count = process_rows(
        load_rows(parquet_path, cache_dir=args.cache_dir),
        output_jsonl=args.output_jsonl,
        image_root=args.image_root,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Wrote {count} InfoSeek Human rows to {args.output_jsonl}")
    print(f"Saved images under {args.image_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
