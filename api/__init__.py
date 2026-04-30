"""OpenAI-compatible API utilities for PMSR."""

from .api_wrapper import (
    ConfigurationError,
    ProviderError,
    RequestError,
    ResponseError,
    build_multimodal_user_message,
    build_pmsr_user_message,
    build_text_message,
    chat_completion,
    is_rate_limit_error,
)
from .openai import (
    APIConfigurationError,
    APIError,
    APIRequestError,
    APIResponseError,
    OpenAICompatibleClient,
    image_path_to_data_url,
    normalize_chat_completions_url,
)

__all__ = [
    "APIConfigurationError",
    "APIError",
    "APIRequestError",
    "APIResponseError",
    "ConfigurationError",
    "OpenAICompatibleClient",
    "ProviderError",
    "RequestError",
    "ResponseError",
    "build_multimodal_user_message",
    "build_pmsr_user_message",
    "build_text_message",
    "chat_completion",
    "image_path_to_data_url",
    "is_rate_limit_error",
    "normalize_chat_completions_url",
]
