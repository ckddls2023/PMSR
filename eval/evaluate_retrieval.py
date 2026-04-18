from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import SearchResult
from search.pmsr_search import PMSRSearch, PMSRSearchConfig


DEFAULT_DATA = ROOT / "data" / "EVQA_test.jsonl"


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def result_to_match_texts(result: SearchResult | dict[str, Any]) -> list[str]:
    if isinstance(result, SearchResult):
        payload = result.to_dict()
    else:
        payload = result

    texts: list[str] = []
    for field in ("caption", "title", "text", "url"):
        value = payload.get(field)
        if value:
            texts.append(str(value))
    return texts


def item_targets(item: dict[str, Any]) -> list[str]:
    targets = as_list(item.get("entity_text"))
    if targets:
        return targets
    targets = as_list(item.get("wikipedia_title"))
    if targets:
        return targets
    return as_list(item.get("answer"))


def target_matches(target: str, docs: Sequence[str]) -> bool:
    normalized_docs = [normalize_text(doc) for doc in docs]
    target_parts = [part.strip() for part in str(target).split("|") if part.strip()]
    if not target_parts:
        return False
    return all(
        any(normalize_text(part) in doc for doc in normalized_docs)
        for part in target_parts
    )


def has_recall_hit(docs: Sequence[str], targets: Sequence[str]) -> bool:
    return any(target_matches(target, docs) for target in targets)


def compute_recall(
    dataset: Iterable[dict[str, Any]],
    retriever: PMSRSearch,
    *,
    top_ks: Sequence[int] = (5, 10, 20),
    use_question: bool = True,
) -> dict[str, float | int]:
    sorted_top_ks = sorted(set(int(k) for k in top_ks))
    max_k = max(sorted_top_ks)
    hits = {k: 0 for k in sorted_top_ks}
    total = 0
    errors = 0

    for item in tqdm(dataset, desc="Evaluating PMSR KB recall"):
        image_path = item.get("image_path", "")
        question = item.get("question", "")
        targets = item_targets(item)
        if not image_path or not targets:
            continue

        query = {"image_path": image_path}
        if use_question:
            query["text"] = question

        try:
            results = retriever.search(query, top_k=max_k)
        except Exception as exc:
            errors += 1
            item_id = item.get("question_id") or item.get("dataset_image_ids") or total
            print(f"[warn] retrieval failed item={item_id}: {exc}", file=sys.stderr)
            continue

        total += 1
        match_texts_by_rank = [result_to_match_texts(result) for result in results]
        for k in sorted_top_ks:
            docs = [
                text
                for rank_texts in match_texts_by_rank[:k]
                for text in rank_texts
            ]
            if has_recall_hit(docs, targets):
                hits[k] += 1

    scores: dict[str, float | int] = {
        f"R@{k}": hits[k] / total if total else 0.0
        for k in sorted_top_ks
    }
    scores["total"] = total
    scores["errors"] = errors
    return scores


def parse_top_ks(values: list[str] | None) -> list[int]:
    if not values:
        return [5, 10, 20]
    top_ks: list[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                top_ks.append(int(item))
    return top_ks or [5, 10, 20]


def build_pmsr_search(args: argparse.Namespace) -> PMSRSearch:
    pmsr_kb = args.pmsr_kb or os.environ.get("PMSR_KB", "")
    pmsr_metadata = args.pmsr_metadata or os.environ.get("PMSR_METADATA", "")
    image_embed_api_base = args.image_embed_api_base or os.environ.get("IMAGE_EMBED_API_BASE", "")
    qwen_text_embed_api_base = args.qwen_text_embed_api_base or os.environ.get("QWEN_TEXT_EMBED_API_BASE", "")
    missing = [
        name
        for name, value in {
            "--pmsr_kb": pmsr_kb,
            "--pmsr_metadata": pmsr_metadata,
            "--image_embed_api_base": image_embed_api_base,
            "--qwen_text_embed_api_base": qwen_text_embed_api_base,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing PMSR retrieval options: {', '.join(missing)}")

    return PMSRSearch(
        PMSRSearchConfig(
            pmsr_kb=pmsr_kb,
            pmsr_metadata=pmsr_metadata,
            image_embed_api_base=image_embed_api_base,
            text_embed_api_base=qwen_text_embed_api_base,
            fusion=args.pmsr_fusion,
            timeout=args.timeout,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PMSR KB retrieval recall on EVQA.")
    parser.add_argument("--data", "--jsonl_path", dest="data", default=str(DEFAULT_DATA))
    parser.add_argument("--pmsr_kb")
    parser.add_argument("--pmsr_metadata")
    parser.add_argument("--image_embed_api_base")
    parser.add_argument("--qwen_text_embed_api_base")
    parser.add_argument("--pmsr-fusion", choices=["concat", "image", "text"], default="concat")
    parser.add_argument("--top-k", "--topk", dest="top_ks", action="append", help="Comma-separated or repeated K values.")
    parser.add_argument("--limit", "--max_samples", dest="limit", type=int)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--no-question", action="store_true", help="Do not include the question text in PMSR query encoding.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    top_ks = parse_top_ks(args.top_ks)
    dataset = load_jsonl(args.data, limit=args.limit)
    retriever = build_pmsr_search(args)
    scores = compute_recall(dataset, retriever, top_ks=top_ks, use_question=not args.no_question)

    print("\n==== PMSR KB Recall ====")
    print(f"data: {args.data}")
    print(f"total: {scores['total']}")
    print(f"errors: {scores['errors']}")
    for k in sorted(top_ks):
        print(f"R@{k}: {scores[f'R@{k}']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
