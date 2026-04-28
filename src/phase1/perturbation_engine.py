"""
LLM-based perturbation engine for ROBIN Phase 1.

Given a base Indonesian instruction, generates the four ROBIN perturbation
levels (L0-L3) via DigitalOcean Serverless Inference. Post-processing
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
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from asyncio_throttle import Throttler
from openai import AsyncOpenAI

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "perturbation_prompts.yaml"
DEFAULT_INDOCOLLEX_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "indocollex.json"
DEFAULT_BASE_URL = "https://inference.do-ai.run/v1/"
DEFAULT_MODEL = "minimax-m2.5"

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
        base_url: str = DEFAULT_BASE_URL,
        prompts_path: str | Path | None = None,
        indocollex_path: str | Path | None = None,
        max_concurrency: int = 8,
        temperature: float = 0.4,
        max_retries: int = 6,
        rate_limit_per_minute: int = 15,
        seed: int | None = None,
    ):
        key = api_key or os.getenv("DIGITALOCEAN_INFERENCE_KEY") or os.getenv("MODEL_ACCESS_KEY")
        if not key:
            raise RuntimeError(
                "Missing DigitalOcean inference key. Set DIGITALOCEAN_INFERENCE_KEY "
                "or MODEL_ACCESS_KEY in your .env file."
            )

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._throttler = Throttler(rate_limit=rate_limit_per_minute, period=60)
        self._rng = random.Random(seed)
        self._client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=90.0)
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

    async def _call_one(self, level: int, instruction: str) -> str:
        prompts = self._prompts
        system_msg = prompts["shared"]["system"].strip()
        output_rule = prompts["shared"]["output_format"].strip()
        user_template = prompts[f"level_{level}"]["user"]
        user_msg = user_template.format(instruction=instruction.strip()) + "\n\n" + output_rule

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._throttler, self._semaphore:
                    # asyncio.wait_for provides a hard cancellation timeout independent
                    # of the httpx-level timeout, preventing indefinite server stalls.
                    resp = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self.model,
                            temperature=self.temperature,
                            max_tokens=512,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg},
                            ],
                        ),
                        timeout=120.0,
                    )
                text = (resp.choices[0].message.content or "").strip()
                if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
                    text = text[1:-1].strip()
                if text:
                    return text
                last_err = RuntimeError("empty completion")
                backoff = 2.0
            except asyncio.TimeoutError:
                last_err = RuntimeError("call timed out after 120s")
                backoff = 10.0 * attempt + self._rng.uniform(0, 5)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
                backoff = (5 * 2 ** (attempt - 1)) if is_rate_limit else (2.0 * attempt)
                backoff += self._rng.uniform(0, 2)
            await asyncio.sleep(backoff)

        if level == 0:
            return instruction.strip()
        raise RuntimeError(f"L{level} perturbation failed after {self.max_retries} retries: {last_err}")

    # ------------------------------------------------------------------ async pipeline

    async def perturb_async(self, instruction: str) -> PerturbedInstruction:
        l0 = await self._call_one(0, instruction)
        l1, l2, l3 = await asyncio.gather(
            self._call_one(1, l0),
            self._call_one(2, l0),
            self._call_one(3, l0),
        )
        l2 = self._post_process_l2(l2)
        l3 = self._post_process_l3(l3)
        return PerturbedInstruction(
            original=instruction,
            level_0_clean=l0,
            level_1_mild=l1,
            level_2_jaksel=l2,
            level_3_adversarial=l3,
            metadata={"engine": "do-serverless", "model": self.model},
        )

    # ------------------------------------------------------------------ sync wrappers

    def perturb(self, instruction: str) -> PerturbedInstruction:
        return asyncio.run(self.perturb_async(instruction))

    def get_all_levels(self, instruction: str) -> dict[int, str]:
        p = self.perturb(instruction)
        return {
            0: p.level_0_clean,
            1: p.level_1_mild,
            2: p.level_2_jaksel,
            3: p.level_3_adversarial,
        }

    async def aclose(self) -> None:
        await self._client.close()
