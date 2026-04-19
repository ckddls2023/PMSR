from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_MODEL = "google/siglip2-giant-opt-patch16-384"
DEFAULT_TEXT_MODEL = "intfloat/e5-base-v2"
DEFAULT_MLLM_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_TEXT = "A small bird on a branch."
DEFAULT_IMAGE = ROOT / "test" / "image.jpg"


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


def normalize_api_base(api_base: str) -> str:
    value = api_base.replace(",", ".").strip().rstrip("/")
    if value.endswith("/v1/embeddings"):
        return value
    if value.endswith("/embeddings"):
        return value
    if value.endswith("/v1"):
        return f"{value}/embeddings"
    return f"{value}/v1/embeddings"


def normalize_v2_embed_url(api_base: str) -> str:
    value = api_base.replace(",", ".").strip().rstrip("/")
    if value.endswith("/v2/embed"):
        return value
    if value.endswith("/v2"):
        return f"{value}/embed"
    return f"{value}/v2/embed"


def image_path_to_data_url(image_path: str | Path) -> str:
    path_text = str(image_path)
    if path_text.startswith(("http://", "https://", "data:")):
        return path_text
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def parse_embedding(payload: dict[str, Any]) -> list[float]:
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return first["embedding"]
    embedding = payload.get("embedding")
    if isinstance(embedding, list):
        return embedding
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        return embeddings[0]
    raise AssertionError(f"Embedding response has no embedding vector: {payload}")


def assert_embedding_vector(name: str, embedding: list[float]) -> None:
    if not embedding:
        raise AssertionError(f"{name} embedding is empty.")
    if not all(isinstance(value, (float, int)) for value in embedding[:16]):
        raise AssertionError(f"{name} embedding contains non-numeric values in the first 16 entries.")


def post_embedding(
    *,
    name: str,
    api_base: str,
    model: str,
    payload: dict[str, Any],
    timeout: int,
    api_key: str,
) -> list[float]:
    endpoint = normalize_api_base(api_base)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    print(f"[test] case={name} endpoint={endpoint}")
    print(f"[test] case={name} model={model}")
    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    if not 200 <= response.status_code < 300:
        raise AssertionError(f"{name} endpoint returned HTTP {response.status_code}: {response.text[:1000]}")
    embedding = parse_embedding(response.json())
    assert_embedding_vector(name, embedding)
    print(f"[ok] case={name} dim={len(embedding)}")
    return embedding


def post_v2_embed(
    *,
    name: str,
    api_base: str,
    model: str,
    payload: dict[str, Any],
    timeout: int,
    api_key: str,
) -> list[float]:
    endpoint = normalize_v2_embed_url(api_base)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    print(f"[test] case={name} endpoint={endpoint}")
    print(f"[test] case={name} model={model}")
    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    if not 200 <= response.status_code < 300:
        raise AssertionError(f"{name} endpoint returned HTTP {response.status_code}: {response.text[:1000]}")
    payload_json = response.json()
    embeddings = payload_json.get("embeddings")
    if isinstance(embeddings, dict):
        float_embeddings = embeddings.get("float")
        if isinstance(float_embeddings, list) and float_embeddings and isinstance(float_embeddings[0], list):
            embedding = float_embeddings[0]
        else:
            raise AssertionError(f"{name} response did not include embeddings.float: {payload_json}")
    elif isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        embedding = embeddings[0]
    else:
        raise AssertionError(f"{name} response did not include a Cohere-compatible embedding: {payload_json}")
    assert_embedding_vector(name, embedding)
    print(f"[ok] case={name} dim={len(embedding)}")
    return embedding


def build_image_payload(model: str, image_path: str | Path) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path_to_data_url(image_path)},
                    }
                ],
            }
        ],
        "encoding_format": "float",
    }


def build_v2_image_payload(model: str, image_path: str | Path, text: str, mode: str) -> dict[str, Any]:
    data_url = image_path_to_data_url(image_path)
    if mode in {"images", "images_no_input_type"}:
        payload = {
            "model": model,
            "images": [data_url],
            "embedding_types": ["float"],
        }
        if mode == "images":
            payload["input_type"] = "image"
        return payload
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": data_url},
        },
    ]
    if mode == "image_text":
        content.insert(0, {"type": "text", "text": text})
    return {
        "model": model,
        "inputs": [
            {
                "content": content,
            }
        ],
        "embedding_types": ["float"],
    }


def build_text_payload(model: str, text: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": text,
        "encoding_format": "float",
    }


def build_mllm_payload(model: str, image_path: str | Path, text: str, instruction: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
            },
        ],
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
    }


def resolve_base(cli_value: str | None, env_name: str, case_name: str) -> str:
    value = (cli_value or os.environ.get(env_name, "")).strip()
    if not value:
        raise SystemExit(f"Missing {case_name} API base. Pass --{case_name}_api_base or set {env_name}.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test PMSR embedding API servers.")
    parser.add_argument("--image_embed_api_base", help="SigLIP2 image embedding endpoint base.")
    parser.add_argument("--text_embed_api_base", help="Text embedding endpoint base.")
    parser.add_argument("--mllm_embed_api_base", help="Qwen3-VL multimodal embedding endpoint base.")
    parser.add_argument("--image-model")
    parser.add_argument("--text-model")
    parser.add_argument("--mllm-model")
    parser.add_argument("--image")
    parser.add_argument("--text")
    parser.add_argument("--instruction")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--case", choices=["image", "image_v2", "text", "mllm", "all"], default="all")
    parser.add_argument(
        "--image-v2-mode",
        choices=["image", "image_text", "images", "images_no_input_type"],
        default="image",
        help="Cohere-compatible /v2/embed image payload mode. SigLIP usually requires image or images.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    image_model = args.image_model or os.environ.get("IMAGE_EMBED_MODEL", DEFAULT_IMAGE_MODEL)
    text_model = args.text_model or os.environ.get("TEXT_EMBED_MODEL", DEFAULT_TEXT_MODEL)
    mllm_model = args.mllm_model or os.environ.get("MLLM_EMBED_MODEL", DEFAULT_MLLM_MODEL)
    image_path = args.image or os.environ.get("IMAGE_PATH", str(DEFAULT_IMAGE))
    text = args.text or os.environ.get("EMBED_TEXT", DEFAULT_TEXT)
    instruction = args.instruction or os.environ.get("MLLM_EMBED_INSTRUCTION", "Represent the user's input.")
    api_key = args.api_key if args.api_key is not None else os.environ.get("API_KEY", "")

    cases: list[tuple[str, str, str, dict[str, Any]]] = []
    if args.case in {"image", "all"}:
        api_base = resolve_base(args.image_embed_api_base, "IMAGE_EMBED_API_BASE", "image_embed")
        cases.append(("image", api_base, image_model, build_image_payload(image_model, image_path)))
    if args.case in {"image_v2", "all"}:
        api_base = resolve_base(args.image_embed_api_base, "IMAGE_EMBED_API_BASE", "image_embed")
        cases.append(
            (
                "image_v2",
                api_base,
                image_model,
                build_v2_image_payload(image_model, image_path, text, args.image_v2_mode),
            )
        )
    if args.case in {"text", "all"}:
        api_base = resolve_base(args.text_embed_api_base, "TEXT_EMBED_API_BASE", "text_embed")
        cases.append(("text", api_base, text_model, build_text_payload(text_model, text)))
    if args.case in {"mllm", "all"}:
        api_base = resolve_base(args.mllm_embed_api_base, "MLLM_EMBED_API_BASE", "mllm_embed")
        cases.append(
            (
                "mllm",
                api_base,
                mllm_model,
                build_mllm_payload(mllm_model, image_path, text, instruction),
            )
        )

    failures = 0
    for name, api_base, model, payload in cases:
        try:
            if name == "image_v2":
                post_v2_embed(
                    name=name,
                    api_base=api_base,
                    model=model,
                    payload=payload,
                    timeout=args.timeout,
                    api_key=api_key,
                )
            else:
                post_embedding(
                    name=name,
                    api_base=api_base,
                    model=model,
                    payload=payload,
                    timeout=args.timeout,
                    api_key=api_key,
                )
        except (AssertionError, OSError, requests.RequestException) as exc:
            failures += 1
            print(f"[fail] case={name}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
