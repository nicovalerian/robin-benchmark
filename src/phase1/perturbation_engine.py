"""
LLM-based perturbation engine for ROBIN Phase 1.

Given a base Indonesian instruction, generates the four ROBIN perturbation
levels (L0-L3) through an explicit OpenAI-compatible deployment. Post-processing
applies IndoCollex word substitutions and discourse particles deterministically
so the LLM can focus on structural transformations.

    L0  Formal Indonesian          (control)
    L1  Lexical Borrowing          (absorbed EN nouns, no morphological change)
    L2  Morphological Fusion       (ID affixes on EN verb stems
                                    + EN connectors / discourse fillers
                                    + programmatic discourse particles)
    L3  Intra-Sentential Switching (L2 + clause-level switching
                                    + ID-root+EN-suffix
                                    + ID phonological respelling
                                    + IndoCollex word substitutions
                                    + random lowercase sentence starts)

Semantic core and verifiable constraints are preserved across all levels.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from asyncio_throttle import Throttler

from providers import (
    OpenAICompatibleChatClient,
    deployment_id_from_url,
    resolve_credentials,
    retry_after_seconds,
)

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "perturbation_prompts.yaml"
DEFAULT_INDOCOLLEX_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "indocollex.json"
DEFAULT_MODEL = "gpt-5.6-luna"

DISCOURSE_PARTICLES = ["dong", "sih", "deh", "nih", "ya", "loh", "kan", "gitu"]


@dataclass
class PerturbedInstruction:
    original: str
    level_0_clean: str
    level_1_mild: str
    level_2_jaksel: str
    level_3_adversarial: str
    metadata: dict = field(default_factory=dict)


class PerturbationEngine:
    """LLM-based four-level perturbation generator with IndoCollex post-processing."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        api_key_env: str = "ELICE_API_KEY",
        provider: str = "elice",
        gold_model: str | None = None,
        gold_base_url: str | None = None,
        gold_api_key_env: str | None = None,
        prompts_path: str | Path | None = None,
        indocollex_path: str | Path | None = None,
        max_concurrency: int = 8,
        temperature: float | None = 0.4,
        gold_temperature: float | None = 0.0,
        max_retries: int = 6,
        rate_limit_per_minute: int = 15,
        max_tokens: int = 1024,
        token_parameter: str = "max_completion_tokens",
        gold_token_parameter: str | None = None,
        extra_body: dict[str, Any] | None = None,
        gold_extra_body: dict[str, Any] | None = None,
        timeout_seconds: float = 240.0,
        discover_model_id: bool = True,
        protocol_deviation: str | None = None,
        seed: int | None = None,
    ):
        perturbation_cfg = {
            "base_url": base_url,
            "api_key_env": api_key_env,
        }
        normalized_base_url, perturbation_key, key_source = resolve_credentials(
            perturbation_cfg, api_key or ""
        )
        if not perturbation_key:
            raise RuntimeError(f"Missing perturbation API key. Set {key_source}.")

        resolved_gold_model = gold_model or model
        gold_cfg = {
            "base_url": gold_base_url or base_url,
            "api_key_env": gold_api_key_env or api_key_env,
        }
        normalized_gold_base_url, gold_key, gold_key_source = resolve_credentials(
            gold_cfg, api_key or ""
        )
        if not gold_key:
            raise RuntimeError(f"Missing gold-reference API key. Set {gold_key_source}.")

        self.provider = provider
        self.model = model
        self.gold_model = resolved_gold_model
        self.base_url = normalized_base_url
        self.gold_base_url = normalized_gold_base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.protocol_deviation = protocol_deviation
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._throttler = Throttler(rate_limit=rate_limit_per_minute, period=60)
        self._rng = random.Random(seed)
        self._perturbation_client = OpenAICompatibleChatClient(
            model_name="perturbation-model",
            model_id=model,
            base_url=normalized_base_url,
            api_key=perturbation_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            token_parameter=token_parameter,
            extra_body=extra_body,
            timeout_seconds=timeout_seconds,
            discover_model=discover_model_id,
        )
        self._shares_gold_client = (
            resolved_gold_model == model
            and normalized_gold_base_url == normalized_base_url
            and gold_temperature == temperature
            and (gold_token_parameter or token_parameter) == token_parameter
            and (gold_extra_body or {}) == (extra_body or {})
        )
        if self._shares_gold_client:
            self._gold_client = self._perturbation_client
        else:
            self._gold_client = OpenAICompatibleChatClient(
                model_name="gold-reference-model",
                model_id=resolved_gold_model,
                base_url=normalized_gold_base_url,
                api_key=gold_key,
                temperature=gold_temperature,
                max_output_tokens=max_tokens,
                token_parameter=gold_token_parameter or token_parameter,
                extra_body=gold_extra_body,
                timeout_seconds=timeout_seconds,
                discover_model=discover_model_id,
            )
        self._prompts = self._load_prompts(prompts_path or DEFAULT_PROMPTS_PATH)
        self._indocollex = self._load_indocollex(indocollex_path or DEFAULT_INDOCOLLEX_PATH)

    # ------------------------------------------------------------------ loaders

    @staticmethod
    def _load_prompts(path: str | Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key in ("shared", "level_0", "level_1", "level_2", "level_3"):
            if key not in data:
                raise ValueError(f"perturbation_prompts.yaml missing '{key}'")
        return data

    @staticmethod
    def _load_indocollex(path: str | Path) -> dict[str, str]:
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        # Keep only entries where formal ≠ colloquial
        return {k: v for k, v in raw.items() if k != v}

    # ------------------------------------------------------------------ post-processing

    def _post_process_l2(self, text: str) -> str:
        """Add an Indonesian discourse particle if none is present in the L2 output."""
        words_lower = {w.strip(".,!?;:") for w in text.lower().split()}
        has_particle = any(p in words_lower for p in DISCOURSE_PARTICLES)
        if not has_particle and self._rng.random() < 0.65:
            particle = self._rng.choice(DISCOURSE_PARTICLES)
            text = text.rstrip()
            if text and text[-1] in ".!?":
                text = text[:-1] + f" {particle}."
            else:
                text = text + f" {particle}."
        return text

    def _post_process_l3(self, text: str) -> str:
        """Apply IndoCollex substitutions, discourse particles, and random lowercase."""
        # 1. IndoCollex word substitutions (~45% probability per eligible word)
        for formal, colloq in self._indocollex.items():
            if self._rng.random() < 0.45:
                text = re.sub(
                    r"\b" + re.escape(formal) + r"\b",
                    colloq,
                    text,
                    flags=re.IGNORECASE,
                )

        # 2. Discourse particle if none present
        words_lower = {w.strip(".,!?;:") for w in text.lower().split()}
        has_particle = any(p in words_lower for p in DISCOURSE_PARTICLES)
        if not has_particle and self._rng.random() < 0.80:
            particle = self._rng.choice(DISCOURSE_PARTICLES)
            text = text.rstrip()
            if text and text[-1] in ".!?":
                text = text[:-1] + f" {particle}."
            else:
                text = text + f" {particle}."

        # 3. Randomly lowercase the first letter of non-opening sentences (25% chance)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        result = [sentences[0]] if sentences else []
        for sent in sentences[1:]:
            if sent and self._rng.random() < 0.25:
                sent = sent[0].lower() + sent[1:]
            result.append(sent)
        return " ".join(result)

    # ------------------------------------------------------------------ LLM call

    def _style_plan(self, level: int, style_slot: int) -> str:
        """Return the externally selected style plan for a perturbation level."""
        plans = self._prompts[f"level_{level}"].get("style_plans", [])
        if not plans:
            return "No additional style plan applies at this level."
        return str(plans[style_slot % len(plans)]).strip()

    async def _call_one(self, level: int, instruction: str, style_slot: int = 0) -> str:
        prompts = self._prompts
        system_msg = prompts["shared"]["system"].strip()
        output_rule = prompts["shared"]["output_format"].strip()
        user_template = prompts[f"level_{level}"]["user"]
        user_msg = user_template.format(
            instruction=instruction.strip(),
            style_plan=self._style_plan(level, style_slot),
        ) + "\n\n" + output_rule

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._throttler, self._semaphore:
                    # asyncio.wait_for provides a hard cancellation timeout independent
                    # of the httpx-level timeout, preventing indefinite server stalls.
                    completion = await asyncio.wait_for(
                        self._perturbation_client.complete(
                            [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg},
                            ]
                        ),
                        timeout=self.timeout_seconds,
                    )
                text = completion.content
                if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
                    text = text[1:-1].strip()
                if text:
                    return text
                last_err = RuntimeError("empty completion")
                backoff = 2.0
            except asyncio.TimeoutError:
                last_err = RuntimeError(
                    f"call timed out after {self.timeout_seconds:.0f}s"
                )
                backoff = 3.0 * attempt + self._rng.uniform(0, 3)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
                retry_after = retry_after_seconds(exc)
                backoff = retry_after if retry_after is not None else (
                    (5 * 2 ** (attempt - 1)) if is_rate_limit else (2.0 * attempt)
                )
                backoff += self._rng.uniform(0, 2)
            await asyncio.sleep(backoff)

        if level == 0:
            return instruction.strip()
        raise RuntimeError(f"L{level} perturbation failed after {self.max_retries} retries: {last_err}")

    # ------------------------------------------------------------------ gold generation

    async def generate_gold(self, constrained_instruction: str) -> str:
        """Generate a deterministic gold reference response for a constrained instruction.

        Called once per sample in parallel with perturb_async. The configured
        gold protocol is used, including omission of unsupported parameters.
        """
        system_msg = (
            "Anda adalah asisten yang membantu. Jawab instruksi berikut dengan tepat "
            "dan penuhi semua persyaratan yang disebutkan dalam instruksi."
        )
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._throttler, self._semaphore:
                    completion = await asyncio.wait_for(
                        self._gold_client.complete(
                            [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": constrained_instruction.strip()},
                            ]
                        ),
                        timeout=self.timeout_seconds,
                    )
                text = completion.content
                if text:
                    return text
                last_err = RuntimeError("empty gold completion")
                backoff = 2.0
            except asyncio.TimeoutError:
                last_err = RuntimeError(
                    f"gold generation timed out after {self.timeout_seconds:.0f}s"
                )
                backoff = 3.0 * attempt + self._rng.uniform(0, 3)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
                retry_after = retry_after_seconds(exc)
                backoff = retry_after if retry_after is not None else (
                    (5 * 2 ** (attempt - 1)) if is_rate_limit else (2.0 * attempt)
                )
                backoff += self._rng.uniform(0, 2)
            await asyncio.sleep(backoff)
        raise RuntimeError(f"Gold generation failed after {self.max_retries} retries: {last_err}")

    # ------------------------------------------------------------------ async pipeline

    async def perturb_async(
        self, instruction: str, style_slot: int = 0
    ) -> PerturbedInstruction:
        l0 = await self._call_one(0, instruction, style_slot)
        l1, l2, l3 = await asyncio.gather(
            self._call_one(1, l0, style_slot),
            self._call_one(2, l0, style_slot),
            self._call_one(3, l0, style_slot),
        )
        l2 = self._post_process_l2(l2)
        l3 = self._post_process_l3(l3)
        return PerturbedInstruction(
            original=instruction,
            level_0_clean=l0,
            level_1_mild=l1,
            level_2_jaksel=l2,
            level_3_adversarial=l3,
            metadata={
                "engine": "openai-compatible",
                "provider": self.provider,
                "deployment_id": deployment_id_from_url(self.base_url),
                "model": self._perturbation_client.resolved_model_id or self.model,
                "gold_reference_deployment_id": deployment_id_from_url(self.gold_base_url),
                "gold_reference_model": self._gold_client.resolved_model_id or self.gold_model,
                "style_slot": style_slot,
                "style_plans": {
                    "level_2": self._style_plan(2, style_slot),
                    "level_3": self._style_plan(3, style_slot),
                },
                "effective_parameters": {
                    **self._perturbation_client.protocol(),
                    **(
                        {"protocol_deviation": self.protocol_deviation}
                        if self.protocol_deviation
                        else {}
                    ),
                },
            },
        )

    # ------------------------------------------------------------------ sync wrappers

    def perturb(self, instruction: str) -> PerturbedInstruction:
        return asyncio.run(self.perturb_async(instruction))

    async def aclose(self) -> None:
        if self._shares_gold_client:
            await self._perturbation_client.aclose()
        else:
            await asyncio.gather(
                self._perturbation_client.aclose(),
                self._gold_client.aclose(),
            )
