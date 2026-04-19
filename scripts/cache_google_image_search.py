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

from agents.schemas import SearchResult
from search.google_image_search import GoogleImageSearch


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


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    searched_results = cleaned.get("searched_results")
    if not isinstance(searched_results, dict):
        searched_results = {}
    cleaned["searched_results"] = dict(searched_results)
    return cleaned


def attach_google_image_results(
    row: dict[str, Any],
    results: Iterable[SearchResult | dict[str, Any]],
) -> dict[str, Any]:
    cleaned = _clean_row(row)
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


def fetch_google_image_cache(
    row: dict[str, Any],
    searcher: GoogleImageSearch,
    *,
    top_k: int,
) -> dict[str, Any]:
    image_value = row.get("image_url") or row.get("image_path")
    if not image_value:
        raise ValueError("Row does not contain image_path or image_url.")
    query = {
        "image_path": str(image_value),
        "question": str(row.get("question") or ""),
    }
    results = searcher.search(query, top_k=top_k)
    return attach_google_image_results(row, results)


def process_row(
    row: dict[str, Any],
    *,
    top_k: int,
    searcher: GoogleImageSearch | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    if not refresh and has_current_google_image_cache(row):
        return _clean_row(row)
    if searcher is None:
        raise ValueError("A GoogleImageSearch instance is required when fetching missing cache rows.")
    return fetch_google_image_cache(row, searcher, top_k=top_k)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def process_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    top_k: int,
    searcher: GoogleImageSearch | None = None,
    refresh: bool = False,
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
                top_k=top_k,
                searcher=searcher,
                refresh=refresh,
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
    parser.add_argument("--refresh", action="store_true", help="Fetch again even when cached results already exist.")
    parser.add_argument("--limit", type=int, help="Optional smoke-test row limit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file instead of resuming.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from an existing output file.")
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
    api_key = args.lens_api_key or os.environ.get("SCRAPINGDOG_API_KEY", "")
    if api_key:
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
        top_k=args.top_k,
        searcher=searcher,
        refresh=args.refresh,
        limit=args.limit,
        overwrite=args.overwrite,
        resume=not args.no_resume,
    )
    print(f"Processed {processed} rows.")
    print(f"Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
