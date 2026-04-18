"""PMSR evaluation entry point."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base_agent import AgentConfig
from agents.pmsr_agent import PMSRAgent
from agents.schemas import Trajectory
from eval.metric_eval import evaluate_accuracy, evaluate_recall


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------


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


def save_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config_from_args(args: argparse.Namespace) -> AgentConfig:
    model = str(args.model or "")
    if model.startswith("vllm:"):
        model = model.split(":", 1)[1]
    api_base = (
        getattr(args, "api_base", None)
        or os.getenv("VLLM_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    )
    api_key = getattr(args, "api_key", None) or os.getenv("OPENAI_API_KEY") or ""

    text_kb = getattr(args, "text_kb", None) or os.getenv("TEXT_KB") or ""
    text_metadata = getattr(args, "text_metadata", None) or os.getenv("TEXT_METADATA") or ""
    text_embed_api_base = (
        getattr(args, "text_embed_api_base", None) or os.getenv("TEXT_EMBED_API_BASE") or ""
    )

    pmsr_kb = getattr(args, "pmsr_kb", None) or os.getenv("PMSR_KB") or ""
    pmsr_metadata = getattr(args, "pmsr_metadata", None) or os.getenv("PMSR_METADATA") or ""
    image_embed_api_base = (
        getattr(args, "image_embed_api_base", None) or os.getenv("IMAGE_EMBED_API_BASE") or ""
    )
    pmsr_text_embed_api_base = (
        getattr(args, "pmsr_text_embed_api_base", None)
        or os.getenv("PMSR_TEXT_EMBED_API_BASE")
        or os.getenv("QWEN_TEXT_EMBED_API_BASE")
        or ""
    )

    return AgentConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        retry=args.retry,
        text_kb=text_kb,
        text_metadata=text_metadata,
        text_embed_api_base=text_embed_api_base,
        pmsr_kb=pmsr_kb,
        pmsr_metadata=pmsr_metadata,
        image_embed_api_base=image_embed_api_base,
        pmsr_text_embed_api_base=pmsr_text_embed_api_base,
        pmsr_fusion=args.pmsr_fusion,
        return_images=args.return_images,
        max_iter=args.itercount,
        topk=args.topk,
        threshold=args.threshold,
        verbose=args.verbose,
    )


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------


def _clean_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model.split("/")[-1])


def build_output_path(args: argparse.Namespace, config: AgentConfig) -> Path:
    output_dir = Path(args.output_dir or ROOT / "outputs")
    stem = _clean_model_name(config.model)
    stem += f"_iter{config.max_iter}_topk{config.topk}"

    if config.text_kb:
        if config.text_kb.startswith(("http://", "https://")):
            stem += "_web"
        else:
            stem += "_text"
    if config.pmsr_kb:
        stem += "_pmsr"
    if not config.return_images:
        stem += "_noimg"

    return output_dir / f"{stem}.jsonl"


# ---------------------------------------------------------------------------
# Prediction / evaluation
# ---------------------------------------------------------------------------


def prediction_from_trajectory(item: dict[str, Any], traj: Trajectory) -> dict[str, Any]:
    answer = traj.final_answer
    return {
        "question_id": item.get("question_id", item.get("dataset_image_ids", "")),
        "image_id": item.get("image_id", item.get("dataset_image_ids", "")),
        "dataset_name": item.get("dataset_name", ""),
        "question": item.get("question", ""),
        "image_path": item.get("image_path", ""),
        "choices": item.get("choices", []),
        "label": item.get("label", ""),
        "gold_answer": item.get("gold_answer", item.get("answer", "")),
        "entity_text": item.get("entity_text", ""),
        "answer": answer,
        "answer_eval": _eval_answer(answer, item),
        "knowledge": traj.all_knowledge(),
        "history_questions": traj.history_questions(),
        "total_pred": traj.all_reasoning(),
        "prediction": answer,
    }


def _eval_answer(answer: str, item: dict[str, Any]) -> bool:
    answer_norm = answer.lower().strip()

    answer_eval = item.get("answer_eval")
    if isinstance(answer_eval, dict) and isinstance(answer_eval.get("range"), list) and len(answer_eval["range"]) == 2:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_norm)
        lower, upper = answer_eval["range"]
        for value in numbers:
            try:
                number = float(value)
            except ValueError:
                continue
            if lower <= number <= upper:
                return True

    raw_targets: list[Any] = []
    for field in ("entity_text", "gold_answer", "answer", "answer_eval"):
        value = item.get(field)
        if isinstance(value, list):
            raw_targets.extend(value)
        elif isinstance(value, str):
            raw_targets.append(value)
    targets = [str(t).lower().strip() for t in raw_targets if str(t).strip()]
    if not targets:
        return False
    for target in targets:
        if not target:
            continue
        parts = [p.strip() for p in target.split("|") if p.strip()]
        if all(part in answer_norm for part in parts):
            return True
    return False


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PMSR evaluation")

    # Data
    parser.add_argument("--data", default=str(ROOT / "data" / "EVQA_test.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--env-file", dest="env_file", default=".env")

    # Model / VLM
    parser.add_argument("--model", default=os.getenv("MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--api-base", "--api_base", dest="api_base", default=None)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retry", type=int, default=3)

    # Text KB: file path OR web-search URL (e.g. https://ollama.com/api/web_search)
    parser.add_argument("--text-kb", dest="text_kb", default=None,
                        help="FAISS index path or web-search URL for text retrieval")
    parser.add_argument("--text-metadata", dest="text_metadata", default=None)
    parser.add_argument("--text-embed-api-base", dest="text_embed_api_base", default=None)

    # PMSR image-document KB (concat image+text fusion)
    parser.add_argument("--pmsr-kb", dest="pmsr_kb", default=None)
    parser.add_argument("--pmsr-metadata", dest="pmsr_metadata", default=None)
    parser.add_argument("--pmsr-image-embed-api-base", dest="image_embed_api_base", default=None,
                        help="API base for image embedding model (env: IMAGE_EMBED_API_BASE)")
    parser.add_argument("--pmsr-text-embed-api-base", dest="pmsr_text_embed_api_base", default=None)
    parser.add_argument("--pmsr-fusion", dest="pmsr_fusion", default="concat",
                        choices=["text", "concat", "image", "mllm"])

    # Retrieval / iteration
    parser.add_argument("--itercount", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Adaptive stopping similarity threshold τ (default: 0.9)")
    parser.add_argument("--no-return-images", dest="return_images", action="store_false",
                        help="Use text passages only from image KB (no base64 image loading)")

    parser.add_argument("--bem", action="store_true",
                        help="Measure accuracy with BEM answer-equivalence model")
    parser.add_argument("--verbose", action="store_true")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    data = load_jsonl(args.data, limit=args.limit)
    if args.verbose:
        print(f"Loaded {len(data)} items from {args.data}")

    config = build_config_from_args(args)
    agent = PMSRAgent(config)
    output_path = build_output_path(args, config)

    if args.verbose:
        print(f"Model:   {config.model}")
        print(f"Text KB: {config.text_kb or '(none)'}")
        print(f"PMSR KB: {config.pmsr_kb or '(none)'}")
        print(f"Output:  {output_path}")

    # Always resume from existing output
    predictions: list[dict[str, Any]] = []
    existing_ids: set[Any] = set()
    if output_path.exists():
        predictions = load_jsonl(output_path)
        existing_ids = {p.get("question_id") or p.get("image_path") for p in predictions} - {None}
        if args.verbose:
            print(f"Resuming from {len(predictions)} existing predictions")

    for i, item in enumerate(data):
        qid = item.get("question_id") or item.get("image_path")
        if qid in existing_ids:
            continue

        t_start = time.time()
        try:
            traj = agent.run(item)
        except Exception as exc:
            print(f"[warn] item {i} failed: {exc}", file=sys.stderr)
            continue

        pred = prediction_from_trajectory(item, traj)
        pred["elapsed"] = round(time.time() - t_start, 2)
        predictions.append(pred)

        if args.verbose:
            print(
                f"[{i + 1}/{len(data)}] {item.get('question', '')[:60]!r}"
                f" → {pred['answer'][:50]!r}"
                f" | {'✓' if pred['answer_eval'] else '✗'}"
                f" | {pred['elapsed']:.1f}s"
            )

        if len(predictions) % 5 == 0:
            save_jsonl(output_path, predictions)

    save_jsonl(output_path, predictions)

    acc, _ = evaluate_accuracy(predictions, args)
    acc_label = "Accuracy (BEM)" if args.bem else "Accuracy (CEM)"
    ret, _ = evaluate_recall(predictions, args)

    print("\n==== Results ====")
    print(f"Data:     {args.data}")
    print(f"Model:    {config.model}")
    print(f"Items:    {len(predictions)}")
    print(f"{acc_label}: {acc:.4f}  ({int(acc * len(predictions))}/{len(predictions)})")
    print(f"Retrieval recall: {ret:.4f}")
    print(f"Output:   {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
