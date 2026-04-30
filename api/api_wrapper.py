"""OpenAI-compatible API entrypoints for PMSR."""

from __future__ import annotations

from typing import Any

from .openai import (
    ConfigurationError,
    ProviderError,
    RequestError,
    ResponseError,
    build_multimodal_user_message,
    build_pmsr_user_message,
    build_text_message,
    chat,
    is_rate_limit_error,
)


def chat_completion(
    provider: str | None,
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run an OpenAI-compatible chat completion.

    `provider` is accepted for parity with VideoAgent. Endpoint selection is
    controlled by `api_base` and the model string in this repository.
    """

    del provider
    return chat(model=model, messages=messages, **kwargs)


__all__ = [
    "ConfigurationError",
    "ProviderError",
    "RequestError",
    "ResponseError",
    "build_multimodal_user_message",
    "build_pmsr_user_message",
    "build_text_message",
    "chat_completion",
    "is_rate_limit_error",
]
