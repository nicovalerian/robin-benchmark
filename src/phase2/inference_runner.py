"""ROBIN Phase 2 provider-neutral OpenAI-compatible inference runner."""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asyncio_throttle import Throttler
from tqdm.asyncio import tqdm as atqdm

from providers import (
    OpenAICompatibleChatClient,
    checkpoint_protocol_hash,
    deployment_id_from_url,
    is_local_url,
    resolve_credentials,
    retry_after_seconds,
)


def checkpoint_key(
    provider: str,
    deployment_id: str,
    resolved_model_id: str,
    sample_id: str,
    level: int,
    protocol_hash: str,
) -> str:
    """Return the resume identity for one model/protocol/sample/level call."""
    return "|".join(
        (provider, deployment_id, resolved_model_id, sample_id, str(level), protocol_hash)
    )


def select_models(
    models_config: list[dict],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    add: list[str] | None = None,
) -> list[dict]:
    """Filter or extend configured models, matching names and model IDs."""
    resolved = [dict(model) for model in models_config]
    for spec in add or []:
        spec = spec.strip()
        if not spec:
            continue
        name, separator, model_id = spec.partition("=")
        if not separator:
            model_id = name
        resolved.append(
            {"name": name, "provider": "openai-compatible", "model_id": model_id}
        )

    def matches(model: dict, token: str) -> bool:
        return token in (model.get("name"), model.get("model_id"))

    if include:
        unknown = [token for token in include if not any(matches(m, token) for m in resolved)]
        if unknown:
            raise ValueError(f"--models matched nothing: {unknown}")
        resolved = [m for m in resolved if any(matches(m, token) for token in include)]
    if exclude:
        unknown = [token for token in exclude if not any(matches(m, token) for m in resolved)]
        if unknown:
            raise ValueError(f"--exclude-models matched nothing: {unknown}")
        resolved = [m for m in resolved if not any(matches(m, token) for token in exclude)]
    if not resolved:
        raise ValueError("No models left after applying model filters")
    return resolved


def _model_setting(model_cfg: dict, key: str, default: Any) -> Any:
    return model_cfg[key] if key in model_cfg else default


async def prepare_model_config(
    model_cfg: dict,
    default_api_key: str,
    *,
    temperature: float | None,
    max_output_tokens: int,
    timeout_seconds: float,
) -> dict:
    """Resolve credentials/model ID and freeze the model's protocol identity."""
    base_url, api_key, key_source = resolve_credentials(model_cfg, default_api_key)
    if not api_key:
        raise RuntimeError(
            f"No API key for model '{model_cfg.get('name', '?')}': set env var {key_source}"
        )

    client = OpenAICompatibleChatClient(
        model_name=model_cfg["name"],
        model_id=model_cfg.get("model_id", "auto"),
        base_url=base_url,
        api_key=api_key,
        temperature=_model_setting(model_cfg, "temperature", temperature),
        max_output_tokens=_model_setting(
            model_cfg, "max_output_tokens", max_output_tokens
        ),
        token_parameter=model_cfg.get("token_parameter", "max_completion_tokens"),
        extra_body=model_cfg.get("extra_body"),
        timeout_seconds=_model_setting(model_cfg, "timeout_seconds", timeout_seconds),
        discover_model=model_cfg.get("discover_model_id", True),
    )
    try:
        resolved_model_id = await client.resolve_model_id()
        protocol = client.protocol()
        if model_cfg.get("protocol_deviation"):
            protocol["protocol_deviation"] = model_cfg["protocol_deviation"]
    finally:
        await client.aclose()

    prepared = dict(model_cfg)
    prepared["base_url"] = base_url
    prepared["_resolved_model_id"] = resolved_model_id
    prepared["_deployment_id"] = deployment_id_from_url(base_url)
    prepared["_effective_parameters"] = protocol
    prepared["_protocol_hash"] = checkpoint_protocol_hash(protocol)
    return prepared


@dataclass
class InferenceResult:
    model_name: str
    prompt: str
    response: str
    tokens_used: int
    latency_ms: float
    success: bool
    error_message: str | None = None
    error_type: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    response_model: str | None = None
    effective_parameters: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleInferenceClient:
    """Single-model retry, throttle, and concurrency wrapper."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        api_key: str,
        base_url: str,
        temperature: float | None = 0.0,
        max_output_tokens: int = 1024,
        token_parameter: str = "max_completion_tokens",
        extra_body: dict[str, Any] | None = None,
        timeout_seconds: float = 240.0,
        max_concurrency: int = 4,
        rate_limit_per_minute: int = 30,
        max_retries: int = 6,
        seed: int = 42,
        discover_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._throttler = Throttler(rate_limit=rate_limit_per_minute, period=60)
        self._rng = random.Random(seed)
        self._transport = OpenAICompatibleChatClient(
            model_name=model_name,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            token_parameter=token_parameter,
            extra_body=extra_body,
            timeout_seconds=timeout_seconds,
            discover_model=discover_model,
        )

    @property
    def resolved_model_id(self) -> str:
        return self._transport.resolved_model_id or self._transport.requested_model_id

    @property
    def protocol(self) -> dict[str, Any]:
        return self._transport.protocol()

    async def prepare(self) -> str:
        return await self._transport.resolve_model_id()

    async def generate(self, prompt: str) -> InferenceResult:
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._throttler, self._semaphore:
                    completion = await asyncio.wait_for(
                        self._transport.complete(
                            [{"role": "user", "content": prompt}]
                        ),
                        timeout=self.timeout_seconds,
                    )
                if completion.content:
                    total_tokens = completion.usage.get("total_tokens", 0) or 0
                    return InferenceResult(
                        model_name=self.resolved_model_id,
                        prompt=prompt,
                        response=completion.content,
                        tokens_used=int(total_tokens),
                        latency_ms=(time.perf_counter() - started) * 1000,
                        success=True,
                        finish_reason=completion.finish_reason,
                        usage=completion.usage,
                        response_model=completion.response_model,
                        effective_parameters=completion.effective_parameters,
                    )
                last_error = RuntimeError(
                    "empty completion"
                    + (
                        f" (finish_reason={completion.finish_reason}; increase max_output_tokens)"
                        if completion.finish_reason
                        else ""
                    )
                )
                backoff = 2.0
            except asyncio.TimeoutError:
                last_error = RuntimeError(
                    f"timed out after {self.timeout_seconds:.0f}s"
                )
                backoff = 10.0 * attempt + self._rng.uniform(0, 5)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                retry_after = retry_after_seconds(exc)
                is_capacity = any(
                    marker in str(exc).lower()
                    for marker in ("429", "503", "rate_limit", "overloaded")
                )
                backoff = (
                    retry_after
                    if retry_after is not None
                    else (5 * 2 ** (attempt - 1) if is_capacity else 2.0 * attempt)
                )
                backoff += self._rng.uniform(0, 2)
            if attempt < self.max_retries:
                await asyncio.sleep(backoff)

        error = str(last_error)
        lowered = error.lower()
        error_type = (
            "timeout"
            if "timed out" in lowered
            else "empty_completion"
            if "empty completion" in lowered
            else "rate_limit"
            if any(marker in lowered for marker in ("429", "503", "rate_limit", "overloaded"))
            else "api_error"
        )
        return InferenceResult(
            model_name=self.resolved_model_id,
            prompt=prompt,
            response="",
            tokens_used=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            success=False,
            error_message=error,
            error_type=error_type,
            effective_parameters=self.protocol,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


class InferenceRunner:
    """Run one deployment at a time and checkpoint every response."""

    def __init__(
        self,
        models_config: list[dict],
        api_key: str,
        temperature: float | None = 0.0,
        max_output_tokens: int = 1024,
        timeout_seconds: float = 240.0,
        max_concurrency: int = 4,
        rate_limit_per_minute: int = 30,
        max_retries: int = 6,
    ) -> None:
        self.models_config = models_config
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_retries = max_retries

    async def run_model(
        self,
        model_cfg: dict,
        prompts: list[dict],
        output_path: Path,
        completed: set[str],
        lock: asyncio.Lock,
        progress_cb=None,
    ) -> None:
        model_name = model_cfg["name"]
        configured_model_id = model_cfg.get("model_id", "auto")
        resolved_model_id = model_cfg.get("_resolved_model_id", configured_model_id)
        base_url, api_key, key_source = resolve_credentials(model_cfg, self.api_key)
        if not api_key:
            raise RuntimeError(f"No API key for model '{model_name}': set env var {key_source}")

        client = OpenAICompatibleInferenceClient(
            model_name=model_name,
            model_id=resolved_model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=_model_setting(model_cfg, "temperature", self.temperature),
            max_output_tokens=_model_setting(
                model_cfg, "max_output_tokens", self.max_output_tokens
            ),
            token_parameter=model_cfg.get("token_parameter", "max_completion_tokens"),
            extra_body=model_cfg.get("extra_body"),
            timeout_seconds=_model_setting(
                model_cfg, "timeout_seconds", self.timeout_seconds
            ),
            max_concurrency=_model_setting(
                model_cfg, "max_concurrency", self.max_concurrency
            ),
            rate_limit_per_minute=_model_setting(
                model_cfg, "rate_limit_per_minute", self.rate_limit_per_minute
            ),
            max_retries=_model_setting(model_cfg, "max_retries", self.max_retries),
            discover_model=False,
        )
        resolved_model_id = await client.prepare()
        provider = model_cfg.get("provider", "openai-compatible")
        deployment_id = model_cfg.get("_deployment_id") or deployment_id_from_url(base_url)
        protocol_hash = model_cfg.get("_protocol_hash") or checkpoint_protocol_hash(client.protocol)

        output_handle = open(output_path, "a", encoding="utf-8")

        async def run_one(prompt: dict) -> None:
            key = checkpoint_key(
                provider,
                deployment_id,
                resolved_model_id,
                prompt["id"],
                prompt["level"],
                protocol_hash,
            )
            if key in completed:
                if progress_cb:
                    progress_cb()
                return

            result = await client.generate(prompt["text"])
            record = {
                "provider": provider,
                "deployment_id": deployment_id,
                "model_name": model_name,
                "configured_model_id": configured_model_id,
                "model_id": resolved_model_id,
                "protocol_hash": protocol_hash,
                "sample_id": prompt["id"],
                "level": prompt["level"],
                "category": prompt.get("category", ""),
                "prompt": prompt["text"],
                "response": result.response,
                "success": result.success,
                "tokens_used": result.tokens_used,
                "usage": result.usage,
                "finish_reason": result.finish_reason,
                "response_model": result.response_model,
                "effective_parameters": model_cfg.get(
                    "_effective_parameters", result.effective_parameters
                ),
                "latency_ms": round(result.latency_ms),
                "error": result.error_message,
                "error_type": result.error_type,
            }
            async with lock:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed.add(key)
            if progress_cb:
                progress_cb()

        try:
            await atqdm.gather(
                *(run_one(prompt) for prompt in prompts),
                desc=f"  {model_name}",
                unit="resp",
                leave=False,
            )
        finally:
            output_handle.close()
            await client.aclose()
