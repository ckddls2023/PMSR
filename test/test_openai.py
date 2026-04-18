from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import build_multimodal_user_message, build_text_message, chat_completion
from api.openai import ProviderError, normalize_chat_completions_url


DEFAULT_PROMPT = "Answer with exactly one word: ok"
DEFAULT_IMAGE_PROMPT = "Describe this image in one short sentence."
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_IMAGE_PATH = ROOT / "test" / "image.jpg"


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


def split_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                items.append(item)
    return items


def normalize_user_api_base(api_base: str) -> str:
    # Tolerate the common typo `http://147,47...` in shell commands.
    return api_base.replace(",", ".").rstrip("/")


def resolve_api_bases(args: argparse.Namespace) -> list[str]:
    bases = [base.strip() for base in args.api_base or [] if base.strip()]
    if not bases:
        env_base = os.environ.get("API_BASE", "").strip()
        if env_base:
            bases = [env_base]
    if not bases:
        raise SystemExit("Missing API base. Pass --api-base/--api_base or set API_BASE in .env/environment.")
    return [normalize_user_api_base(base) for base in bases]


def resolve_models(args: argparse.Namespace) -> list[str]:
    models = split_values(args.model)
    if models:
        return models
    env_model = os.environ.get("MODEL", "").strip() or os.environ.get("OPENAI_MODEL", "").strip()
    return [env_model or DEFAULT_MODEL]


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    for key_name in args.api_key_env:
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test OpenAI-compatible chat endpoints.")
    parser.add_argument(
        "--api-base",
        "--api_base",
        dest="api_base",
        action="append",
        help="OpenAI-compatible endpoint base. Accepts http://host:port, /v1, or /v1/chat/completions. Can be repeated. Falls back to API_BASE.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model name to test. Can be repeated or comma-separated. Falls back to MODEL, OPENAI_MODEL, then Qwen/Qwen2.5-VL-7B-Instruct.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-prompt", default=DEFAULT_IMAGE_PROMPT)
    parser.add_argument(
        "--case",
        choices=["text", "image", "all"],
        default="text",
        help="Which model-call case to run.",
    )
    parser.add_argument(
        "--image",
        "--image-path",
        dest="image_path",
        default=os.environ.get("IMAGE_PATH", str(DEFAULT_IMAGE_PATH)),
        help="Image path or URL for --case image/all. Defaults to IMAGE_PATH or test/image.jpg.",
    )
    parser.add_argument("--api-key", help="Explicit API key. Usually unnecessary for local vLLM endpoints.")
    parser.add_argument(
        "--api-key-env",
        action="append",
        default=["API_KEY", "OPENAI_API_KEY", "VLLM_API_KEY", "OPENROUTER_API_KEY"],
        help="Environment variable to check for API key. Can be repeated.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def run_case(
    *,
    case_name: str,
    api_base: str,
    model: str,
    messages: list[dict],
    api_key: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> None:
    endpoint = normalize_chat_completions_url(api_base)
    print(f"[test] case={case_name} endpoint={endpoint}")
    print(f"[test] case={case_name} model={model}")
    result = chat_completion(
        None,
        model,
        messages,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = result["content"].strip()
    if not content:
        raise AssertionError(f"{case_name} endpoint returned empty content.")
    print(f"[ok] case={case_name} {content[:500]}")


def build_cases(args: argparse.Namespace) -> list[tuple[str, list[dict]]]:
    cases: list[tuple[str, list[dict]]] = []
    if args.case in {"text", "all"}:
        cases.append(("text", [build_text_message("user", args.prompt)]))
    if args.case in {"image", "all"}:
        image_path = str(args.image_path).strip()
        if not image_path:
            raise SystemExit("Missing image path for image test case.")
        if not image_path.startswith(("http://", "https://", "data:")) and not Path(image_path).exists():
            raise SystemExit(f"Image path does not exist for image test case: {image_path}")
        cases.append(("image", [build_multimodal_user_message(args.image_prompt, [image_path])]))
    return cases


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    api_bases = resolve_api_bases(args)
    models = resolve_models(args)
    api_key = resolve_api_key(args)
    cases = build_cases(args)

    failures = 0
    for api_base in api_bases:
        for model in models:
            for case_name, messages in cases:
                try:
                    run_case(
                        case_name=case_name,
                        api_base=api_base,
                        model=model,
                        messages=messages,
                        api_key=api_key,
                        timeout=args.timeout,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                except (ProviderError, AssertionError, OSError) as exc:
                    failures += 1
                    print(f"[fail] case={case_name} endpoint={api_base} model={model}: {exc}", file=sys.stderr)
                    if not args.continue_on_error:
                        return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
