#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


DEFAULT_DATASET = "ONE-Lab/LiveVQA-Research-Preview"
DEFAULT_OUTPUT_JSONL = Path("data/LiveVQA_test.jsonl")
DEFAULT_IMAGE_ROOT = Path("data/processed/livevqa/images")
DEFAULT_CACHE_DIR = Path("data/processed/livevqa/hf_cache")
DEFAULT_INSTRUCTION = "Using the provided image, answer the question: "


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_entries(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        for key in ("data", "samples", "qa"):
            value = data.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        return [data]
    raise ValueError(f"Unsupported LiveVQA qa.json structure: {type(data).__name__}")


def load_detail_map(details: Any) -> dict[str, dict[str, Any]]:
    if details is None:
        return {}
    if isinstance(details, (str, Path)):
        details = load_json(details)
    if isinstance(details, dict):
        if all(isinstance(value, dict) for value in details.values()):
            return {str(key): value for key, value in details.items()}
        for key in ("data", "samples", "qa"):
            value = details.get(key)
            if isinstance(value, list):
                details = value
                break
        else:
            sample_id = details.get("sample_id")
            return {str(sample_id): details} if sample_id else {}
    if isinstance(details, list):
        output: dict[str, dict[str, Any]] = {}
        for entry in details:
            if not isinstance(entry, dict):
                continue
            sample_id = entry.get("sample_id")
            if sample_id:
                output[str(sample_id)] = entry
        return output
    return {}


def resolve_image_path(dataset_root: str | Path, query_image: str) -> Path:
    root = Path(dataset_root)
    raw_path = Path(str(query_image))
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    candidates.extend(
        [
            root / str(query_image).lstrip("/"),
            root / raw_path.name,
            root / "image" / raw_path.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve LiveVQA image path {query_image!r} under {root}")


def copy_image(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def answer_list(answer: Any) -> list[Any]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [item for item in answer if str(item).strip()]
    text = str(answer).strip()
    return [text] if text else []


def process_entries(
    entries: Iterable[dict[str, Any]],
    *,
    dataset_root: str | Path,
    output_jsonl: str | Path,
    image_root: str | Path,
    detail_map: dict[str, dict[str, Any]] | None = None,
    split: str = "test",
    limit: int | None = None,
    overwrite: bool = True,
) -> int:
    output_path = Path(output_jsonl)
    image_root_path = Path(image_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_root_path.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output JSONL already exists: {output_path}")

    details = detail_map or {}
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        iterator = tqdm(entries, desc="Processing LiveVQA rows", disable=not sys.stderr.isatty())
        for index, entry in enumerate(iterator):
            if limit is not None and count >= limit:
                break
            sample_id = str(entry.get("sample_id") or f"livevqa_{split}_{index}")
            question = str(entry.get("query") or "").strip()
            query_image = str(entry.get("query_image") or "").strip()
            if not query_image:
                raise ValueError(f"LiveVQA row {sample_id} has no query_image.")

            source_image = resolve_image_path(dataset_root, query_image)
            image_id = source_image.stem
            output_image_path = image_root_path / source_image.name
            copy_image(source_image, output_image_path)

            gold_answer = entry.get("gt_answer")
            answers = answer_list(gold_answer)
            detail = details.get(sample_id, {})
            record = {
                "question_id": sample_id,
                "image_id": image_id,
                "question": question,
                "answer_eval": answers,
                "data_split": split,
                "wikidata_value": None,
                "wikidata_range": None,
                "entity_id": None,
                "entity_text": None,
                "image_path": str(output_image_path),
                "gold_answer": str(gold_answer or "").strip(),
                "instruction": DEFAULT_INSTRUCTION,
                "answer": answers,
                "sample_id": sample_id,
                "query_image": query_image,
                "topic": detail.get("topic"),
                "context": detail.get("context"),
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
            allow_patterns=["qa.json", "qa_detailed.json", "image/**", "README.md"],
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and process LiveVQA Research Preview into PMSR JSONL.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--revision")
    parser.add_argument("--snapshot-dir", help="Use an existing local snapshot directory instead of downloading.")
    parser.add_argument("--qa-json", help="Use an explicit local qa.json path.")
    parser.add_argument("--qa-detailed-json", help="Use an explicit local qa_detailed.json path.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
    elif args.qa_json:
        snapshot_dir = Path(args.qa_json).parent
    else:
        snapshot_dir = snapshot_dataset(
            args.dataset,
            cache_dir=args.cache_dir,
            revision=args.revision,
        )
    qa_json = Path(args.qa_json) if args.qa_json else snapshot_dir / "qa.json"
    qa_detailed_json = Path(args.qa_detailed_json) if args.qa_detailed_json else snapshot_dir / "qa_detailed.json"
    detail_map = load_detail_map(qa_detailed_json) if qa_detailed_json.exists() else {}

    if Path(args.image_root).exists() and args.overwrite:
        shutil.rmtree(args.image_root)
    count = process_entries(
        load_entries(qa_json),
        dataset_root=snapshot_dir,
        output_jsonl=args.output_jsonl,
        image_root=args.image_root,
        detail_map=detail_map,
        split=args.split,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Wrote {count} LiveVQA rows to {args.output_jsonl}")
    print(f"Saved images under {args.image_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
