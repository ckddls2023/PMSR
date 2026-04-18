"""OpenAI-compatible chat client for local and hosted model endpoints.

This module is the API foundation for the PMSR refactor. It keeps model calls
behind a small interface that works with OpenAI, vLLM, OpenRouter, and other
OpenAI-compatible chat/completions servers.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Iterable

import requests


DEFAULT_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


class ProviderError(RuntimeError):
    """Base class for API client failures."""


class ConfigurationError(ProviderError):
    """Raised when endpoint or credential configuration is invalid."""


class RequestError(ProviderError):
    """Raised when an HTTP request fails."""


class ResponseError(ProviderError):
    """Raised when an endpoint response has an unexpected shape."""


APIError = ProviderError
APIConfigurationError = ConfigurationError
APIRequestError = RequestError
APIResponseError = ResponseError


def normalize_chat_completions_url(api_base: str | None) -> str:
    """Return a concrete `/chat/completions` URL.

    The codebase historically passes full URLs such as
    `http://host:8003/v1/chat/completions`. Newer configs may pass a base URL
    such as `http://host:8003/v1`. Both forms are accepted.
    """

    value = (api_base or os.getenv("OPENAI_BASE_URL") or DEFAULT_CHAT_COMPLETIONS_URL).strip()
    if not value:
        raise ConfigurationError("Missing OpenAI-compatible API base URL.")
    value = value.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def image_path_to_data_url(image_path: str | Path) -> str:
    """Convert a local image path to a base64 data URL.

    HTTP(S) URLs are already valid OpenAI `image_url` values and are returned as
    strings unchanged.
    """

    path_text = str(image_path)
    if path_text.startswith(("http://", "https://", "data:")):
        return path_text

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": text}


def build_multimodal_user_message(
    text: str,
    image_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for image_path in image_paths or ():
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path_to_data_url(image_path)},
            }
        )
    if text:
        content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def build_pmsr_user_message(
    *,
    image_path: str | list[str] | None = None,
    prompt: str | None = None,
    image_text_pairs: list[Any] | None = None,
    text_passages: list[Any] | None = None,
) -> dict[str, Any]:
    """Build the multimodal message shape used by the PMSR loop.

    Ordering follows the MedAgent/Qwen wrapper pattern:
    image-text-pair prompt, image-text pairs, text knowledge, query image,
    then the final task prompt.
    """

    content: list[dict[str, Any]] = []

    def append_image_pair(passage: Any) -> None:
        if isinstance(passage, dict):
            passage_image = passage.get("image_path")
            if passage_image:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path_to_data_url(passage_image)},
                    }
                )
            if passage_image:
                passage_text = passage.get("caption") or passage.get("text")
            else:
                passage_title = str(passage.get("title") or "").strip()
                passage_body = str(passage.get("text") or passage.get("caption") or "").strip()
                passage_text = (
                    f"{passage_title}\n{passage_body}"
                    if passage_title and passage_body and passage_title != passage_body
                    else passage_body or passage_title
                )
            if passage_text:
                content.append({"type": "text", "text": str(passage_text)})
        elif isinstance(passage, tuple) and len(passage) == 2:
            passage_image, passage_text = passage
            if passage_image:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path_to_data_url(passage_image)},
                    }
                )
            if passage_text:
                content.append({"type": "text", "text": f"Passage: {passage_text}"})
        elif isinstance(passage, str):
            content.append({"type": "text", "text": f"Passage: {passage}"})

    if image_text_pairs:
        content.append(
            {
                "type": "text",
                "text": "Here is relevant knowledge of image and their corresponding description.\n",
            }
        )
        for passage in image_text_pairs:
            append_image_pair(passage)

    if text_passages:
        content.append({"type": "text", "text": f"Knowledge: {_format_text_passages(text_passages)}"})

    image_paths: list[str] = []
    if isinstance(image_path, list):
        image_paths = [str(path) for path in image_path if path]
    elif image_path:
        image_paths = [str(image_path)]
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path_to_data_url(path)},
            }
        )

    if prompt:
        content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def _format_text_passages(text_passages: list[Any]) -> str:
    knowledge = ""
    for passage in text_passages:
        if isinstance(passage, dict):
            title = str(passage.get("title") or "").strip()
            text = str(passage.get("text") or passage.get("caption") or "").strip()
            knowledge += f"Passage Title: {title}\n"
            knowledge += f"Passage Text: {text}\n\n"
        elif isinstance(passage, str):
            knowledge += f"{passage}\n\n"
    return knowledge


class OpenAICompatibleClient:
    """Small OpenAI-compatible chat/completions client."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: int = 300,
        retry: int = 3,
        wait: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.api_url = normalize_chat_completions_url(api_base)
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        output_tokens = self.max_tokens if max_tokens is None else max_tokens
        if output_tokens is not None:
            payload["max_tokens"] = output_tokens
        merged_extra_body = dict(self.extra_body)
        merged_extra_body.update(extra_body or {})
        payload.update(merged_extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retry + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                if 200 <= response.status_code < 300:
                    return self._parse_response(response)
                last_error = RequestError(
                    f"HTTP {response.status_code} from {self.api_url}: {response.text[:1000]}"
                )
            except requests.RequestException as exc:
                last_error = RequestError(f"Request failed for {self.api_url}: {exc}")

            if attempt < self.retry:
                time.sleep(self.wait * (attempt + 1))

        raise last_error or RequestError(f"Request failed for {self.api_url}")

    def _parse_response(self, response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ResponseError(f"Response was not JSON: {response.text[:1000]}") from exc

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ResponseError("Response did not include non-empty `choices`.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ResponseError("Response choice did not include a message object.")
        content = message.get("content")
        reasoning = message.get("reasoning")
        tool_calls = message.get("tool_calls")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if (not isinstance(content, str) or not content.strip()) and isinstance(reasoning, str):
            content = reasoning
        if content is None:
            content = ""
        if not isinstance(content, str):
            if tool_calls:
                content = ""
            else:
                raise ResponseError("Response message content was not a string.")
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return {
            "provider": "openai",
            "content": content.strip(),
            "reasoning": reasoning if isinstance(reasoning, str) else "",
            "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
            "assistant_message": assistant_message,
            "message": message,
            "raw": payload,
            "usage": payload.get("usage", {}),
            "model": payload.get("model", self.model),
        }


def chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = 300,
    retry: int = 3,
    wait: float = 2.0,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one OpenAI-compatible chat completion."""

    client = OpenAICompatibleClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
        retry=retry,
        wait=wait,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return client.chat(messages)


def is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


__all__ = [
    "APIConfigurationError",
    "APIError",
    "APIRequestError",
    "APIResponseError",
    "ConfigurationError",
    "DEFAULT_CHAT_COMPLETIONS_URL",
    "OpenAICompatibleClient",
    "ProviderError",
    "RequestError",
    "ResponseError",
    "build_multimodal_user_message",
    "build_pmsr_user_message",
    "build_text_message",
    "chat",
    "image_path_to_data_url",
    "is_rate_limit_error",
    "normalize_chat_completions_url",
]
