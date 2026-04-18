#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import Evidence, SearchResult
from search.google_image_search import GoogleImageSearch


LEGACY_CACHE_FIELDS = ("lens_result", "retrieved_image_path", "retrieved_caption")
SEARCH_GROUP = "google_image"


def load_env_file(path: str | Path = ".env", *, override: bool = True) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_pmsr_cache{input_path.suffix}")


def _clean_row(row: dict[str, Any], *, drop_legacy_fields: bool) -> dict[str, Any]:
    cleaned = dict(row)
    if drop_legacy_fields:
        for field in LEGACY_CACHE_FIELDS:
            cleaned.pop(field, None)
    searched_results = cleaned.get("searched_results")
    if not isinstance(searched_results, dict):
        searched_results = {}
    cleaned["searched_results"] = dict(searched_results)
    return cleaned


def attach_google_image_results(
    row: dict[str, Any],
    results: Iterable[SearchResult | dict[str, Any]],
    *,
    drop_legacy_fields: bool = True,
) -> dict[str, Any]:
    cleaned = _clean_row(row, drop_legacy_fields=drop_legacy_fields)
    normalized: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, SearchResult):
            normalized.append(result.to_dict())
        elif isinstance(result, dict):
            normalized.append(dict(result))
        else:
            raise TypeError(f"Unsupported search result type: {type(result).__name__}")
    cleaned["searched_results"][SEARCH_GROUP] = normalized
    return cleaned


def has_current_google_image_cache(row: dict[str, Any]) -> bool:
    searched_results = row.get("searched_results")
    if not isinstance(searched_results, dict):
        return False
    results = searched_results.get(SEARCH_GROUP)
    return isinstance(results, list) and len(results) > 0


def has_legacy_google_image_cache(row: dict[str, Any]) -> bool:
    return any(field in row for field in LEGACY_CACHE_FIELDS)


def _legacy_summaries(row: dict[str, Any]) -> list[str]:
    lens_result = row.get("lens_result")
    if isinstance(lens_result, list):
        return [str(item).strip() for item in lens_result if str(item).strip()]
    if isinstance(lens_result, str):
        return [line.strip().removeprefix("Passage:").strip() for line in lens_result.splitlines() if line.strip()]
    return []


def _legacy_list(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def convert_legacy_google_image_cache(
    row: dict[str, Any],
    *,
    top_k: int = 5,
    drop_legacy_fields: bool = True,
) -> dict[str, Any]:
    image_paths = _legacy_list(row, "retrieved_image_path")
    captions = _legacy_list(row, "retrieved_caption")
    summaries = _legacy_summaries(row)
    result_count = min(top_k, max(len(image_paths), len(captions), len(summaries)))

    results: list[SearchResult] = []
    query = str(row.get("image_path") or row.get("image_url") or "")
    question = str(row.get("question") or "")
    for index in range(result_count):
        image_path = image_paths[index] if index < len(image_paths) else ""
        caption = captions[index] if index < len(captions) else ""
        summary = summaries[index] if index < len(summaries) else caption
        title = caption
        rank = index + 1
        evidence = Evidence(
            source=SEARCH_GROUP,
            modality="image",
            title=title,
            text=summary,
            image_path=image_path,
            caption=caption,
            score=1.0 / rank,
            rank=rank,
            metadata={
                "legacy_cache": True,
                "question": question,
            },
        )
        results.append(SearchResult(evidence=evidence, query=query, search_type=SEARCH_GROUP))
    return attach_google_image_results(row, results, drop_legacy_fields=drop_legacy_fields)


def fetch_google_image_cache(
    row: dict[str, Any],
    searcher: GoogleImageSearch,
    *,
    top_k: int,
    drop_legacy_fields: bool = True,
) -> dict[str, Any]:
    image_value = row.get("image_url") or row.get("image_path")
    if not image_value:
        raise ValueError("Row does not contain image_path or image_url.")
    query = {
        "image_path": str(image_value),
        "question": str(row.get("question") or ""),
    }
    results = searcher.search(query, top_k=top_k)
    return attach_google_image_results(row, results, drop_legacy_fields=drop_legacy_fields)


def process_row(
    row: dict[str, Any],
    *,
    mode: str,
    top_k: int,
    searcher: GoogleImageSearch | None = None,
    refresh: bool = False,
    drop_legacy_fields: bool = True,
) -> dict[str, Any]:
    if not refresh and has_current_google_image_cache(row):
        return _clean_row(row, drop_legacy_fields=drop_legacy_fields)
    if mode in {"auto", "convert-existing"} and not refresh and has_legacy_google_image_cache(row):
        return convert_legacy_google_image_cache(row, top_k=top_k, drop_legacy_fields=drop_legacy_fields)
    if mode == "convert-existing":
        return _clean_row(row, drop_legacy_fields=drop_legacy_fields)
    if searcher is None:
        raise ValueError("A GoogleImageSearch instance is required when fetching missing cache rows.")
    return fetch_google_image_cache(row, searcher, top_k=top_k, drop_legacy_fields=drop_legacy_fields)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def process_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    mode: str,
    top_k: int,
    searcher: GoogleImageSearch | None = None,
    refresh: bool = False,
    drop_legacy_fields: bool = True,
    limit: int | None = None,
    overwrite: bool = False,
    resume: bool = True,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to read and write the same JSONL path. Use a separate --output path.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    start_index = _count_lines(output_path) if resume and output_path.exists() else 0
    processed = 0
    mode_open = "a" if start_index else "w"
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(mode_open, encoding="utf-8") as outfile:
        for line_index, line in enumerate(infile):
            if line_index < start_index:
                continue
            if limit is not None and processed >= limit:
                break
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            updated = process_row(
                row,
                mode=mode,
                top_k=top_k,
                searcher=searcher,
                refresh=refresh,
                drop_legacy_fields=drop_legacy_fields,
            )
            outfile.write(json.dumps(updated, ensure_ascii=False) + "\n")
            outfile.flush()
            processed += 1
    return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache Google image search results into PMSR searched_results schema.")
    parser.add_argument("--jsonl", required=True, help="Input JSONL file.")
    parser.add_argument("--output", help="Output JSONL file. Defaults to *_pmsr_cache.jsonl.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of image search results to cache per row.")
    parser.add_argument("--mode", choices=["auto", "convert-existing", "fetch"], default="auto")
    parser.add_argument("--refresh", action="store_true", help="Fetch again even when cached results already exist.")
    parser.add_argument("--limit", type=int, help="Optional smoke-test row limit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file instead of resuming.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from an existing output file.")
    parser.add_argument("--keep-legacy-fields", action="store_true", help="Keep lens_result and retrieved_* fields.")
    parser.add_argument("--env-file", default=".env", help="Environment file to load before reading API keys.")
    parser.add_argument(
        "--lens-api-key",
        "--scrapingdog-api-key",
        dest="lens_api_key",
        default=None,
        help="ScrapingDog API key. Defaults to SCRAPINGDOG_API_KEY from the environment.",
    )
    parser.add_argument(
        "--ollama-api-key",
        default=None,
        help="Ollama API key for optional result summarization. Defaults to OLLAMA_API_KEY.",
    )
    parser.add_argument("--ollama-model", default="gpt-oss:120b")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--no-summarize", action="store_true", help="Disable Ollama summarization during live fetches.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(args.env_file, override=True)

    input_path = Path(args.jsonl)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    searcher = None
    if args.mode == "fetch" or args.refresh:
        api_key = args.lens_api_key or os.environ.get("SCRAPINGDOG_API_KEY", "")
        if not api_key:
            raise SystemExit("Missing ScrapingDog API key. Set SCRAPINGDOG_API_KEY or pass --lens-api-key.")
        searcher = GoogleImageSearch(
            api_key=api_key,
            ollama_api_key=args.ollama_api_key or os.environ.get("OLLAMA_API_KEY", ""),
            ollama_model=args.ollama_model,
            summarize=not args.no_summarize,
            timeout=args.timeout,
        )

    processed = process_jsonl(
        input_path=input_path,
        output_path=output_path,
        mode=args.mode,
        top_k=args.top_k,
        searcher=searcher,
        refresh=args.refresh,
        drop_legacy_fields=not args.keep_legacy_fields,
        limit=args.limit,
        overwrite=args.overwrite,
        resume=not args.no_resume,
    )
    print(f"Processed {processed} rows.")
    print(f"Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
