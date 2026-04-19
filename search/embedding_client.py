from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import requests


def normalize_embedding_url(api_base: str) -> str:
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
    if value.endswith("/v1/embeddings"):
        return value[: -len("/v1/embeddings")] + "/v2/embed"
    if value.endswith("/v1"):
        return value[: -len("/v1")] + "/v2/embed"
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
    return parse_embeddings(payload)[0]


def parse_embeddings(payload: dict[str, Any]) -> list[list[float]]:
    data = payload.get("data")
    if isinstance(data, list) and data:
        vectors = [
            row["embedding"]
            for row in sorted(data, key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0)
            if isinstance(row, dict) and isinstance(row.get("embedding"), list)
        ]
        if vectors:
            return vectors
    embedding = payload.get("embedding")
    if isinstance(embedding, list):
        return [embedding]
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list):
        if embeddings and isinstance(embeddings[0], list):
            return embeddings
    if isinstance(embeddings, dict):
        float_embeddings = embeddings.get("float")
        if isinstance(float_embeddings, list) and (
            not float_embeddings or isinstance(float_embeddings[0], list)
        ):
            return float_embeddings
    raise ValueError("Embedding response did not include a vector.")


class EmbeddingClient:
    def __init__(self, *, api_base: str, model: str, timeout: int = 60, api_key: str = "") -> None:
        self.endpoint = normalize_embedding_url(api_base)
        self.v2_embed_endpoint = normalize_v2_embed_url(api_base)
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def _post_embeddings(self, payload: dict[str, Any]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return parse_embeddings(response.json())

    def _post(self, payload: dict[str, Any]) -> list[float]:
        return self._post_embeddings(payload)[0]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._post_embeddings(
            {
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            }
        )

    def embed_image(self, image_path: str | Path) -> list[float]:
        return self.embed_images([image_path])[0]

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        if not image_paths:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "inputs": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_path_to_data_url(image_path)},
                        }
                    ]
                }
                for image_path in image_paths
            ],
            "embedding_types": ["float"],
        }
        response = requests.post(self.v2_embed_endpoint, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return parse_embeddings(response.json())

    def embed_mllm(self, *, image_path: str | Path, text: str, instruction: str) -> list[float]:
        return self._post(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": instruction}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_path_to_data_url(image_path)},
                            },
                            {"type": "text", "text": text},
                        ],
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
        )

    def embed_mllm_text(self, *, text: str, instruction: str) -> list[float]:
        return self._post(
            {
                "model": self.model,
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
        )
