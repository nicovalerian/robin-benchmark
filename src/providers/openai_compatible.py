"""Shared OpenAI-compatible chat transport and endpoint utilities."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
LOCAL_PLACEHOLDER_KEY = "local"


def normalize_openai_base_url(url: str) -> str:
    """Return an OpenAI-compatible base URL ending in exactly one ``/v1``."""
    if not url or not url.strip():
        raise ValueError("An explicit OpenAI-compatible base_url is required")
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"Invalid OpenAI-compatible base_url: {url!r}")
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def is_local_url(url: str) -> bool:
    """Return whether ``url`` points at a loopback inference server."""
    return (urlsplit(url).hostname or "").lower() in _LOCAL_HOSTS


def deployment_id_from_url(url: str) -> str:
    """Extract the deployment path segment immediately before ``/v1``."""
    parts = [p for p in urlsplit(normalize_openai_base_url(url)).path.split("/") if p]
    return parts[-2] if len(parts) >= 2 else ""


def resolve_credentials(
    model_cfg: dict[str, Any],
    default_api_key: str = "",
    *,
    default_api_key_env: str = "ELICE_API_KEY",
) -> tuple[str, str, str]:
    """Resolve an explicit endpoint and its credential without cross-host fallback."""
    base_url = normalize_openai_base_url(str(model_cfg.get("base_url") or ""))
    local = is_local_url(base_url)
    api_key_env = model_cfg.get("api_key_env")

    if local and not api_key_env:
        return base_url, LOCAL_PLACEHOLDER_KEY, "local (no auth)"
    if not api_key_env:
        raise ValueError(
            f"Remote base_url '{base_url}' must declare api_key_env; "
            "refusing to guess or reuse credentials across hosts"
        )

    key = os.getenv(str(api_key_env), "")
    if not key and api_key_env == default_api_key_env:
        key = default_api_key
    if local and not key:
        return base_url, LOCAL_PLACEHOLDER_KEY, f"{api_key_env} (unset; local, no auth)"
    return base_url, key, str(api_key_env)


def checkpoint_protocol_hash(protocol: dict[str, Any]) -> str:
    """Build a stable short identity for parameters that affect model output."""
    payload = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def retry_after_seconds(exc: Exception) -> float | None:
    """Read Retry-After from an OpenAI SDK exception when available."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    return {}


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    finish_reason: str | None
    usage: dict[str, Any]
    response_model: str | None
    effective_parameters: dict[str, Any]


class OpenAICompatibleChatClient:
    """Low-level async client for one explicitly configured deployment."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        base_url: str,
        api_key: str,
        temperature: float | None,
        max_output_tokens: int,
        token_parameter: str = "max_completion_tokens",
        extra_body: dict[str, Any] | None = None,
        timeout_seconds: float = 240.0,
        discover_model: bool = True,
    ) -> None:
        if token_parameter not in {"max_completion_tokens", "max_tokens", "auto"}:
            raise ValueError(f"Unsupported token_parameter: {token_parameter}")
        self.model_name = model_name
        self.requested_model_id = model_id
        self.resolved_model_id: str | None = None
        self.base_url = normalize_openai_base_url(base_url)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.token_parameter = token_parameter
        self.extra_body = dict(extra_body or {})
        self.discover_model = discover_model
        self._model_lock = asyncio.Lock()
        self._client = AsyncOpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout_seconds)

    async def resolve_model_id(self) -> str:
        if self.resolved_model_id:
            return self.resolved_model_id
        async with self._model_lock:
            if self.resolved_model_id:
                return self.resolved_model_id
            if not self.discover_model and self.requested_model_id != "auto":
                self.resolved_model_id = self.requested_model_id
                return self.resolved_model_id

            page = await self._client.models.list()
            ids = [item.id for item in page.data if getattr(item, "id", None)]
            requested = self.requested_model_id
            if requested != "auto":
                if requested not in ids:
                    raise RuntimeError(
                        f"Configured model_id {requested!r} is not advertised by {self.base_url}; "
                        f"available IDs: {ids}"
                    )
                self.resolved_model_id = requested
                return requested
            if len(ids) == 1:
                self.resolved_model_id = ids[0]
                return ids[0]

            needle = re.sub(r"[^a-z0-9]+", "", self.model_name.lower())
            matches = [
                mid for mid in ids
                if needle in re.sub(r"[^a-z0-9]+", "", mid.lower())
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Could not resolve one model ID for {self.model_name!r}; available IDs: {ids}"
                )
            self.resolved_model_id = matches[0]
            return matches[0]

    def protocol(self) -> dict[str, Any]:
        token_parameter = "max_completion_tokens" if self.token_parameter == "auto" else self.token_parameter
        return {
            "api": "chat.completions",
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "token_parameter": token_parameter,
            "extra_body": self.extra_body,
        }

    async def complete(self, messages: list[dict[str, str]]) -> ChatCompletion:
        model_id = await self.resolve_model_id()
        token_parameter = "max_completion_tokens" if self.token_parameter == "auto" else self.token_parameter
        kwargs: dict[str, Any] = {"model": model_id, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        kwargs[token_parameter] = self.max_output_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc)
            if self.token_parameter != "auto" or "max_completion_tokens" not in message:
                raise
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = self.max_output_tokens
            token_parameter = "max_tokens"
            response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        effective = self.protocol()
        effective["token_parameter"] = token_parameter
        return ChatCompletion(
            content=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason,
            usage=_model_dump(response.usage),
            response_model=getattr(response, "model", None),
            effective_parameters=effective,
        )

    async def aclose(self) -> None:
        await self._client.close()
