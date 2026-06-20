"""
ROBIN Phase 2 — Inference runner.

All models are accessed via DigitalOcean Serverless Inference using the
OpenAI-compatible Chat Completions endpoint. The runner uses the same
AsyncOpenAI + Semaphore + Throttler stack as Phase 1's perturbation engine.

Loop structure: model-outer, all prompts-inner. One model is saturated to
its rate limit at a time, results checkpointed per-response, then the next
model starts. This avoids N-model burst patterns that thrash rate limits.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from asyncio_throttle import Throttler
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm


DO_BASE_URL = "https://inference.do-ai.run/v1/"


def resolve_credentials(model_cfg: dict, default_api_key: str) -> tuple[str, str, str]:
    """Resolve (base_url, api_key, source) for a model — BYOK support.

    A model entry may bring its own provider/key:
      - ``base_url``    : OpenAI-compatible endpoint (default: DigitalOcean).
      - ``api_key_env`` : name of the env var holding that provider's key.
    When neither is given, the model uses the DigitalOcean default key/endpoint,
    so existing configs keep working unchanged.

    Security: the default DigitalOcean key is only ever sent to the default
    DigitalOcean endpoint. A model that points at a *different* ``base_url``
    must supply its own ``api_key_env`` — otherwise we would leak the DO
    credential to a foreign (possibly attacker-chosen) host. Refuse rather
    than fall back to the default key for a non-default endpoint.
    """
    base_url = model_cfg.get("base_url") or DO_BASE_URL
    api_key_env = model_cfg.get("api_key_env")
    if api_key_env:
        return base_url, (os.getenv(api_key_env) or ""), api_key_env
    if base_url.rstrip("/") != DO_BASE_URL.rstrip("/"):
        raise ValueError(
            f"Model with custom base_url '{base_url}' must also set 'api_key_env'; "
            "refusing to send the default DigitalOcean key to a non-default endpoint."
        )
    return base_url, default_api_key, "DIGITALOCEAN_INFERENCE_KEY"


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


class DOInferenceClient:
    """
    Single-model async client wrapping AsyncOpenAI for DO Serverless Inference.
    Mirrors Phase 1's _call_one: Semaphore + Throttler + exponential backoff.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str = DO_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_concurrency: int = 20,
        rate_limit_per_minute: int = 100,
        max_retries: int = 6,
        seed: int = 42,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._throttler = Throttler(rate_limit=rate_limit_per_minute, period=60)
        self._rng = random.Random(seed)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=90.0,
        )

    async def generate(self, prompt: str) -> InferenceResult:
        t0 = time.perf_counter()
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._throttler, self._semaphore:
                    resp = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self.model_id,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            messages=[{"role": "user", "content": prompt}],
                        ),
                        timeout=120.0,
                    )

                # Reasoning models put final answer in content; it is null only
                # when max_tokens was exhausted inside the thinking chain.
                text = (resp.choices[0].message.content or "").strip()
                tokens = resp.usage.total_tokens if resp.usage else 0

                if text:
                    return InferenceResult(
                        model_name=self.model_id,
                        prompt=prompt,
                        response=text,
                        tokens_used=tokens,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        success=True,
                    )
                last_err = RuntimeError("empty completion")
                backoff = 2.0

            except asyncio.TimeoutError:
                last_err = RuntimeError("timed out after 120s")
                backoff = 10.0 * attempt + self._rng.uniform(0, 5)
            except Exception as exc:
                last_err = exc
                is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
                backoff = (5 * 2 ** (attempt - 1)) if is_rate_limit else (2.0 * attempt)
                backoff += self._rng.uniform(0, 2)

            await asyncio.sleep(backoff)

        err_str = str(last_err)
        err_type = (
            "timeout" if "timed out" in err_str.lower() else
            "empty_completion" if "empty completion" in err_str.lower() else
            "rate_limit" if "429" in err_str or "rate_limit" in err_str.lower() else
            "api_error"
        )
        return InferenceResult(
            model_name=self.model_id,
            prompt=prompt,
            response="",
            tokens_used=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error_message=err_str,
            error_type=err_type,
        )

    async def aclose(self) -> None:
        await self._client.close()


class InferenceRunner:
    """
    Runs all configured models over the full prompt set.
    One model at a time (model-outer loop), all prompts concurrent (inner).
    Results checkpointed per-response to output_path.
    """

    def __init__(
        self,
        models_config: list[dict],
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_concurrency: int = 20,
        rate_limit_per_minute: int = 100,
        max_retries: int = 6,
    ):
        self.models_config = models_config
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_retries = max_retries

    async def run_model(
        self,
        model_cfg: dict,
        prompts: list[dict],  # each: {id, level, text}
        output_path: Path,
        completed: set[str],  # set of "model_id|sample_id|level"
        lock: asyncio.Lock,
        progress_cb=None,
    ) -> None:
        model_name = model_cfg["name"]
        model_id = model_cfg["model_id"]

        base_url, api_key, key_source = resolve_credentials(model_cfg, self.api_key)
        if not api_key:
            raise RuntimeError(
                f"No API key for model '{model_name}': set env var {key_source}"
            )

        client = DOInferenceClient(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_concurrency=self.max_concurrency,
            rate_limit_per_minute=self.rate_limit_per_minute,
            max_retries=self.max_retries,
        )

        async def _run_one(p: dict) -> None:
            key = f"{model_id}|{p['id']}|{p['level']}"
            if key in completed:
                if progress_cb:
                    progress_cb()
                return

            result = await client.generate(p["text"])

            record = {
                "model_name": model_name,
                "model_id": model_id,
                "sample_id": p["id"],
                "level": p["level"],
                "category": p.get("category", ""),
                "prompt": p["text"],
                "response": result.response,
                "success": result.success,
                "tokens_used": result.tokens_used,
                "latency_ms": round(result.latency_ms),
                "error": result.error_message,
                "error_type": result.error_type,
            }

            async with lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed.add(key)

            if progress_cb:
                progress_cb()

        tasks = [_run_one(p) for p in prompts]
        await atqdm.gather(*tasks, desc=f"  {model_name}", unit="resp", leave=False)
        await client.aclose()
