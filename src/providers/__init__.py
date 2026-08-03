"""Provider-neutral inference transports used across ROBIN phases."""

from .openai_compatible import (
    ChatCompletion,
    OpenAICompatibleChatClient,
    checkpoint_protocol_hash,
    deployment_id_from_url,
    is_local_url,
    normalize_openai_base_url,
    resolve_credentials,
    retry_after_seconds,
)

__all__ = [
    "ChatCompletion",
    "OpenAICompatibleChatClient",
    "checkpoint_protocol_hash",
    "deployment_id_from_url",
    "is_local_url",
    "normalize_openai_base_url",
    "resolve_credentials",
    "retry_after_seconds",
]
