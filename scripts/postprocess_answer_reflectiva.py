#!/usr/bin/env python3
"""Postprocess PMSR answers with local LLaVA ReflectiVA inference.

The input is the JSONL output produced by eval/main.py. Each row already
contains the PMSR reasoning trajectory in `total_pred`; this script sends the
query image, question, and trajectory reasoning to a local ReflectiVA model,
then writes a new JSONL file with refined `answer` and `prediction` fields.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL = "aimagelab/ReflectiVA"
DEFAULT_MODEL_NAME = "llava_llama_3.1"
DEFAULT_CONV_MODE = "llama_3_1"
DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN = f"{DEFAULT_IMAGE_TOKEN}\n\n"


class ReflectivaClient(Protocol):
    def answer(self, *, image_path: str, question: str, context: str) -> str:
        ...


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


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_last_reasoning_record(total_pred: str) -> str:
    """Extract the final reasoning record from Trajectory.all_reasoning()."""
    if not total_pred:
        return ""
    records = re.split(r"Reasoning Record #\d+:\n?", str(total_pred))
    for record in reversed(records):
        stripped = record.strip()
        if stripped:
            return stripped
    return ""


def resolve_all_reasoning(row: dict[str, Any], *, use_lastrecord: bool = False) -> str:
    all_reasoning = (
        row.get("total_pred")
        or row.get("all_reasoning")
        or row.get("trajectory_all_reasoning")
        or row.get("reasoning")
        or ""
    )
    all_reasoning = str(all_reasoning).strip()
    if use_lastrecord:
        return extract_last_reasoning_record(all_reasoning)
    return all_reasoning


def _build_prompt(*, context: str | list[str]) -> str:
    """Build the ReflectiVA paragraph prompt from PMSR trajectory reasoning."""
    paragraphs = "<paragraph> "
    if isinstance(context, list):
        for item in context:
            paragraphs += f"{item}".replace("Reasoning Record #", "")
    else:
        paragraphs += f"{context}".replace("Reasoning Record #", "Passage #")
    paragraphs += " </paragraph>"
    return f"Consider this paragraph: {paragraphs}. "


def build_reflectiva_prompt(*, question: str, all_reasoning: str) -> str:
    return _build_prompt(context=all_reasoning)


def _load_llava_modules() -> dict[str, Any]:
    """Import LLaVA only for actual local ReflectiVA inference."""
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN as LLAVA_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    return {
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "DEFAULT_IMAGE_TOKEN": LLAVA_IMAGE_TOKEN,
        "conv_templates": conv_templates,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
    }


def get_llava_vlm_answer(
    *,
    image_path: str,
    question: str,
    context: str,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    conv_mode: str,
    model_config: Any,
    temperature: float = 0.2,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 128,
) -> str:
    """Run one ReflectiVA/LLaVA inference pass with PMSR trajectory context."""
    modules = _load_llava_modules()
    image_token = f"{modules['DEFAULT_IMAGE_TOKEN']}\n\n"
    device = model.device
    conv = modules["conv_templates"][conv_mode].copy()
    conv.append_message(conv.roles[0], image_token + f"{question}")
    conv.append_message(conv.roles[1], "[Retrieval]")
    prompt = _build_prompt(context=context)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_for_tokenizer = conv.get_prompt()

    input_ids = (
        modules["tokenizer_image_token"](
            prompt_for_tokenizer,
            tokenizer,
            modules["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(device)
    )
    pil_img = Image.open(image_path).convert("RGB")
    image_tensor = modules["process_images"]([pil_img], image_processor, model_config)[0]
    image_tensor = image_tensor.to(dtype=torch.float16, device=device, non_blocking=True).unsqueeze(0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=pil_img.size,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return answer.strip()


class LLaVAReflectivaInferencer:
    """Small adapter around local ReflectiVA/LLaVA inference."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        image_processor: Any,
        model_config: Any,
        conv_mode: str,
        answer_fn=get_llava_vlm_answer,
        temperature: float = 0.2,
        top_p: float | None = None,
        num_beams: int = 1,
        max_new_tokens: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.answer_fn = answer_fn
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

    def answer(self, *, image_path: str, question: str, context: str) -> str:
        return self.answer_fn(
            image_path=image_path,
            question=question,
            context=context,
            tokenizer=self.tokenizer,
            model=self.model,
            image_processor=self.image_processor,
            conv_mode=self.conv_mode,
            model_config=self.model_config,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
        )


def build_reflectiva_client(
    *,
    model_path: str = DEFAULT_MODEL,
    model_name: str = DEFAULT_MODEL_NAME,
    conv_mode: str = DEFAULT_CONV_MODE,
    temperature: float = 0.2,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 128,
) -> LLaVAReflectivaInferencer:
    modules = _load_llava_modules()
    model_path = os.path.expanduser(model_path)
    tokenizer, model, image_processor, _ = modules["load_pretrained_model"](model_path, None, model_name)
    return LLaVAReflectivaInferencer(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        model_config=model.config,
        conv_mode=conv_mode,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )


def postprocess_row(
    row: dict[str, Any],
    client: ReflectivaClient,
    *,
    use_lastrecord: bool = False,
    original_index: int | None = None,
) -> dict[str, Any]:
    question = str(row.get("question") or "")
    image_path = str(row.get("image_path") or "")
    all_reasoning = resolve_all_reasoning(row, use_lastrecord=use_lastrecord)
    answer = client.answer(
        image_path=image_path,
        question=question,
        context=all_reasoning,
    )

    processed = dict(row)
    if "reflectiva_source_answer" not in processed:
        processed["reflectiva_source_answer"] = processed.get("prediction", processed.get("answer", ""))
    processed["answer"] = str(answer).strip()
    processed["prediction"] = str(answer).strip()
    processed["reflectiva_reasoning_source"] = "last_record" if use_lastrecord else "all_reasoning"
    if original_index is not None and "original_index" not in processed:
        processed["original_index"] = original_index
    return processed


def postprocess_rows(
    rows: list[dict[str, Any]],
    client: ReflectivaClient,
    *,
    use_lastrecord: bool = False,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    return [
        postprocess_row(
            row,
            client,
            use_lastrecord=use_lastrecord,
            original_index=start_index + offset,
        )
        for offset, row in enumerate(rows)
    ]


def default_output_path(input_path: str | Path, *, use_lastrecord: bool = False) -> Path:
    path = Path(input_path)
    suffix = "_reflectiva_lastrr" if use_lastrecord else "_reflectiva"
    return path.with_name(f"{path.stem}{suffix}.jsonl")


def _processed_indices(rows: list[dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
    for row in rows:
        if "original_index" not in row:
            continue
        try:
            indices.add(int(row["original_index"]))
        except (TypeError, ValueError):
            continue
    return indices


def process_jsonl_resumable(
    *,
    input_path: str | Path,
    output_path: str | Path,
    client: ReflectivaClient,
    use_lastrecord: bool = False,
    limit: int | None = None,
    save_every: int = 10,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    input_rows = load_jsonl(input_path, limit=limit)
    output = Path(output_path)
    processed_rows = load_jsonl(output) if output.exists() else []
    processed_indices = _processed_indices(processed_rows)

    for index, row in enumerate(input_rows):
        if index in processed_indices:
            continue
        if verbose:
            print(f"[reflectiva] row {index + 1}/{len(input_rows)} image={row.get('image_path', '')}")
        processed_rows.append(
            postprocess_row(
                row,
                client,
                use_lastrecord=use_lastrecord,
                original_index=index,
            )
        )
        if save_every > 0 and len(processed_rows) % save_every == 0:
            save_jsonl(output, processed_rows)

    save_jsonl(output, processed_rows)
    return processed_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postprocess PMSR JSONL answers with local LLaVA ReflectiVA.")
    parser.add_argument("--input", "--jsonl", "--csv", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", default="")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model-path", "--model", dest="model_path", default=os.getenv("REFLECTIVA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--model-name", default=os.getenv("REFLECTIVA_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--conv-mode", default=os.getenv("REFLECTIVA_CONV_MODE", DEFAULT_CONV_MODE))
    parser.add_argument("--max-new-tokens", "--max-tokens", dest="max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--use-lastrecord",
        action="store_true",
        help="Use only the last PMSR reasoning record instead of full trajectory all_reasoning.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(args.env_file)

    output_path = args.output_path or default_output_path(args.input_path, use_lastrecord=args.use_lastrecord)
    client = build_reflectiva_client(
        model_path=args.model_path,
        model_name=args.model_name,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    rows = process_jsonl_resumable(
        input_path=args.input_path,
        output_path=output_path,
        client=client,
        use_lastrecord=args.use_lastrecord,
        limit=args.limit,
        save_every=args.save_every,
        verbose=args.verbose,
    )
    print(f"Finished {len(rows)} rows. Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
