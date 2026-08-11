#!/usr/bin/env python3
"""Preflight OpenRouter model availability, reasoning policy, pricing, and key limit."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

MODELS_URL = "https://openrouter.ai/api/v1/models"
KEY_URL = "https://openrouter.ai/api/v1/key"


def _get_json(url: str, *, api_key: str | None = None) -> dict:
    headers = {"Accept": "application/json", "User-Agent": "robin-benchmark/phase2-preflight"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _money_per_million(per_token: str | float | int | None) -> float:
    return float(per_token or 0.0) * 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ROBIN OpenRouter Phase 2 config")
    parser.add_argument("--config", default="configs/openrouter_phase2.yaml")
    parser.add_argument("--samples", type=int, default=750)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--no-key-check", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    inference = config.get("inference", {})
    models = inference.get("models", [])
    if not models:
        raise SystemExit("No inference.models found")

    try:
        catalog = _get_json(MODELS_URL).get("data", [])
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not fetch OpenRouter model catalog: {exc}") from exc
    by_id = {m.get("id"): m for m in catalog if m.get("id")}

    requests_per_model = args.samples * args.levels
    input_tokens = int(inference.get("estimate_input_tokens_per_request", 350))
    output_tokens = int(inference.get("estimate_output_tokens_per_request", 1024))
    budget = float(inference.get("budget_usd", 20.0))

    print(f"Config: {args.config}")
    print(f"Plan: {args.samples} samples x {args.levels} levels x {len(models)} models")
    print(f"Requests: {requests_per_model * len(models):,}")
    print(f"Conservative estimate: {input_tokens} input + {output_tokens} output tokens/request\n")

    errors: list[str] = []
    estimated_total = 0.0
    for cfg in models:
        model_id = cfg.get("model_id")
        meta = by_id.get(model_id)
        if not meta:
            errors.append(f"{model_id}: not present in GET /api/v1/models")
            continue

        created = datetime.fromtimestamp(int(meta.get("created", 0)), tz=timezone.utc)
        pricing = meta.get("pricing") or {}
        p_in = _money_per_million(pricing.get("prompt"))
        p_out = _money_per_million(pricing.get("completion"))
        cost = requests_per_model * (
            input_tokens * float(pricing.get("prompt") or 0.0)
            + output_tokens * float(pricing.get("completion") or 0.0)
        )
        estimated_total += cost

        reasoning = meta.get("reasoning") or {}
        mandatory = bool(reasoning.get("mandatory", False))
        default_enabled = reasoning.get("default_enabled")
        configured_reasoning = (cfg.get("extra_body") or {}).get("reasoning") or {}
        disabled = configured_reasoning.get("enabled") is False

        supported = set(meta.get("supported_parameters") or [])
        if cfg.get("token_parameter", "max_tokens") not in supported:
            errors.append(
                f"{model_id}: configured token parameter {cfg.get('token_parameter')!r} "
                f"not advertised in supported_parameters"
            )
        if mandatory and disabled:
            errors.append(f"{model_id}: reasoning is mandatory but config tries to disable it")
        if not disabled:
            errors.append(f"{model_id}: reasoning is not explicitly disabled in config")
        if created.year not in {2025, 2026}:
            errors.append(f"{model_id}: release year {created.year} is outside requested 2025-2026")

        print(
            f"{cfg.get('name', model_id):24} {model_id:38} "
            f"{created.date()}  ${p_in:.3f}/${p_out:.3f} per M  "
            f"reasoning mandatory={mandatory} default={default_enabled!s:5}  "
            f"worst-case=${cost:.3f}"
        )

    print(f"\nConservative catalog-price estimate: ${estimated_total:.2f}")
    print(f"Configured budget:                   ${budget:.2f}")
    if estimated_total > budget:
        errors.append(
            f"estimated ${estimated_total:.2f} exceeds configured ${budget:.2f} budget"
        )

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not args.no_key_check:
        if not api_key:
            errors.append("OPENROUTER_API_KEY is missing from environment/.env")
        else:
            try:
                key_info = (_get_json(KEY_URL, api_key=api_key).get("data") or {})
                limit = key_info.get("limit")
                remaining = key_info.get("limit_remaining")
                usage = key_info.get("usage")
                print("\nAPI key check: OK")
                print(f"  key usage       : {usage}")
                print(f"  spending limit  : {limit}")
                print(f"  limit remaining : {remaining}")
                if remaining is not None and float(remaining) + 1e-9 < estimated_total:
                    errors.append(
                        f"key limit remaining ${float(remaining):.2f} is below the "
                        f"conservative estimate ${estimated_total:.2f}"
                    )
                if limit is None:
                    print("  WARNING: this key has no API-key spending limit")
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                errors.append(f"could not validate OPENROUTER_API_KEY via /api/v1/key: {exc}")

    if errors:
        print("\nPREFLIGHT FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(2)

    print("\nPREFLIGHT PASSED")


if __name__ == "__main__":
    main()
