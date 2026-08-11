#!/usr/bin/env python3
"""
ROBIN Phase 2 — Model Inference

Runs all configured models over the 3,000 ROBIN prompts (750 samples × 4
perturbation levels). Output is a flat JSONL where each line is one
(model, sample, level) response.

Loop: model-outer, all prompts-inner. One model is processed at a time so
its rate limit is saturated cleanly; the next model starts after. Results
are checkpointed per-response so any crash is resumable.
"""
import argparse
import asyncio
import hashlib
import io
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from phase2 import (  # noqa: E402
    InferenceRunner,
    checkpoint_key,
    prepare_model_config,
)
from providers import deployment_id_from_url  # noqa: E402
from utils import load_config, load_jsonl, setup_logger  # noqa: E402


LEVEL_KEYS = ["level_0_clean", "level_1_mild", "level_2_jaksel", "level_3_adversarial"]


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [t.strip() for t in value.split(",") if t.strip()]


def _load_completed(output_path: Path) -> set[str]:
    """Return protocol-aware checkpoint keys for successful rows only."""
    if not output_path.exists():
        return set()
    completed = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("success"):
                continue
            required = (
                "provider",
                "deployment_id",
                "model_id",
                "sample_id",
                "level",
                "protocol_hash",
            )
            if all(field in row for field in required):
                completed.add(
                    checkpoint_key(
                        row["provider"],
                        row["deployment_id"],
                        row["model_id"],
                        row["sample_id"],
                        row["level"],
                        row["protocol_hash"],
                    )
                )
    return completed


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_run_manifest(
    path: Path,
    *,
    config_path: str,
    dataset_path: str,
    models: list[dict],
    excluded_perturbation_model: str | None,
    phase1_config: dict,
) -> None:
    perturbation_llm = phase1_config.get("llm", {})
    gold_llm = phase1_config.get("gold_llm", {})
    providers = sorted({model.get("provider", "openai-compatible") for model in models})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": providers[0] if len(providers) == 1 else "mixed",
        "api": "chat.completions",
        "config_path": config_path,
        "config_sha256": _sha256(config_path),
        "dataset_path": dataset_path,
        "dataset_sha256": _sha256(dataset_path),
        "code_commit": _code_commit(),
        "excluded_perturbation_model": excluded_perturbation_model,
        "phase1": {
            "perturbation_model": perturbation_llm.get(
                "model_id", perturbation_llm.get("model")
            ),
            "perturbation_deployment_id": (
                deployment_id_from_url(perturbation_llm["base_url"])
                if perturbation_llm.get("base_url")
                else None
            ),
            "gold_reference_model": gold_llm.get(
                "model_id", gold_llm.get("model")
            ),
            "gold_reference_deployment_id": (
                deployment_id_from_url(gold_llm["base_url"])
                if gold_llm.get("base_url")
                else None
            ),
        },
        "models": [
            {
                "name": model["name"],
                "provider": model.get("provider", "openai-compatible"),
                "deployment_root": model["base_url"].removesuffix("/v1"),
                "deployment_id": model["_deployment_id"],
                "configured_model_id": model.get("model_id", "auto"),
                "resolved_model_id": model["_resolved_model_id"],
                "effective_parameters": model["_effective_parameters"],
                "protocol_hash": model["_protocol_hash"],
                "excluded_from_primary_ranking": bool(
                    excluded_perturbation_model
                    and excluded_perturbation_model
                    in (model["name"], model["_resolved_model_id"], model.get("model_id"))
                ),
            }
            for model in models
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _length_bucket(sample: dict) -> str:
    """Coarse length bucket used only for deterministic smoke-test sampling."""
    for constraint in sample.get("constraints", []):
        if constraint.get("constraint_type") != "length":
            continue
        target = constraint.get("target_value")
        if isinstance(target, list) and len(target) == 2:
            try:
                upper = int(target[1])
            except (TypeError, ValueError):
                break
            if upper <= 50:
                return "short"
            if upper <= 150:
                return "medium"
            return "long"
    return "unknown"


def _select_dataset_subset(
    dataset: list[dict],
    *,
    limit: int | None,
    strategy: str,
    seed: int,
) -> list[dict]:
    if not limit or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError("--limit must be positive")
    if strategy == "head":
        return dataset[:limit]

    # Round-robin over category x length-bucket strata. This makes a small
    # smoke run much more likely to catch protocol bugs than taking the first N
    # rows while remaining deterministic and reproducible.
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in dataset:
        strata[(sample.get("category", "unknown"), _length_bucket(sample))].append(sample)
    for values in strata.values():
        rng.shuffle(values)

    ordered_keys = sorted(strata)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < limit and ordered_keys:
        key = ordered_keys[cursor % len(ordered_keys)]
        values = strata[key]
        if values:
            selected.append(values.pop())
        if not values:
            ordered_keys.remove(key)
            if not ordered_keys:
                break
            cursor %= len(ordered_keys)
        else:
            cursor += 1
    return selected


def _latest_records(output_path: Path) -> dict[str, dict]:
    """Return the latest row for each protocol-aware checkpoint identity."""
    latest: dict[str, dict] = {}
    if not output_path.exists():
        return latest
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            required = (
                "provider",
                "deployment_id",
                "model_id",
                "sample_id",
                "level",
                "protocol_hash",
            )
            if not all(field in row for field in required):
                continue
            key = checkpoint_key(
                row["provider"],
                row["deployment_id"],
                row["model_id"],
                row["sample_id"],
                row["level"],
                row["protocol_hash"],
            )
            latest[key] = row
    return latest


def _build_prompts(dataset: list[dict]) -> list[dict]:
    """Flatten dataset into a list of {id, level, category, text} dicts.

    original_input (the passage/content the instruction refers to) is appended
    to the prompt when non-empty. The perturbation engine's R5 rule strips raw
    input content from L0-L3 texts, so Phase 2 is the authoritative source for
    context that models need to answer information-extraction tasks.
    """
    prompts = []
    for sample in dataset:
        original_input = (sample.get("original_input") or "").strip()
        for level_key in LEVEL_KEYS:
            text = (sample.get("perturbations") or {}).get(level_key, "")
            if not text:
                continue
            if original_input:
                text = f"{text}\n\n{original_input}"
            level_num = int(level_key.split("_")[1])
            prompts.append({
                "id": sample["id"],
                "level": level_num,
                "category": sample.get("category", ""),
                "text": text,
            })
    return prompts


async def run_inference(args):
    logger = setup_logger("robin-phase2", level="INFO")
    logger.info("Starting ROBIN Phase 2: Model Inference")

    config = load_config(args.config)
    inference_cfg = config.get("inference", {})
    models_config = inference_cfg.get("models", [])

    if not models_config:
        logger.error("No models configured in config file")
        return

    # Optional per-run model selection (filter/add) without editing the YAML.
    from phase2 import select_models  # noqa: E402
    explicit_excludes = _split_csv(args.exclude_models) or []
    if args.exclude_perturbation_model:
        matches_phase2 = any(
            args.exclude_perturbation_model in (model.get("name"), model.get("model_id"))
            for model in models_config
        )
        if matches_phase2:
            explicit_excludes.append(args.exclude_perturbation_model)
        else:
            logger.info(
                "Perturbation model is not in the Phase 2 set; no ranking exclusion needed"
            )
    try:
        models_config = select_models(
            models_config,
            include=_split_csv(args.models),
            exclude=explicit_excludes or None,
            add=args.add_model,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Models for this run: {', '.join(m['name'] for m in models_config)}")

    # Every remote deployment declares its endpoint and credential environment.
    from phase2.inference_runner import resolve_credentials, is_local_url  # noqa: E402

    default_key = os.getenv("ELICE_API_KEY", "")
    missing = []
    for mc in models_config:
        name = mc.get("name", mc.get("model_id", "?"))
        try:
            base_url, key, source = resolve_credentials(mc, default_key)
        except ValueError as exc:
            logger.error(f"Model '{name}': {exc}")
            return
        # Local servers use a placeholder key (no auth) — only require non-empty.
        # Cloud keys must look real (length guard catches truncated/blank vars).
        ok = bool(key) if is_local_url(base_url) else (key and len(key) >= 10)
        if not ok:
            missing.append((name, source))
    if missing:
        for name, source in missing:
            logger.error(f"Model '{name}': API key env var '{source}' not set or too short")
        return
    dataset = load_jsonl(args.input)
    try:
        dataset = _select_dataset_subset(
            dataset,
            limit=args.limit,
            strategy=args.limit_strategy,
            seed=args.limit_seed,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Loaded {len(dataset)} samples")
    if args.limit:
        logger.info(
            "Subset strategy: %s (seed=%s)", args.limit_strategy, args.limit_seed
        )

    prompts = _build_prompts(dataset)
    logger.info(f"Built {len(prompts)} prompts ({len(dataset)} samples x {len(LEVEL_KEYS)} levels)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_output_tokens = inference_cfg.get(
        "max_output_tokens", inference_cfg.get("max_tokens", 1024)
    )
    prepared_models = []
    for model_cfg in models_config:
        prepared_models.append(
            await prepare_model_config(
                model_cfg,
                default_key,
                temperature=inference_cfg.get("temperature", 0.0),
                max_output_tokens=max_output_tokens,
                timeout_seconds=inference_cfg.get("timeout_seconds", 240),
            )
        )
    models_config = prepared_models

    _write_run_manifest(
        Path(args.manifest),
        config_path=args.config,
        dataset_path=args.input,
        models=models_config,
        excluded_perturbation_model=args.exclude_perturbation_model,
        phase1_config=config.get("perturbation", {}),
    )
    logger.info(f"Run manifest: {args.manifest}")

    completed = _load_completed(output_path)
    if completed:
        logger.info(f"Resuming: {len(completed)} responses already written")

    runner = InferenceRunner(
        models_config=models_config,
        api_key=default_key,
        temperature=inference_cfg.get("temperature", 0.0),
        max_output_tokens=max_output_tokens,
        timeout_seconds=inference_cfg.get("timeout_seconds", 240),
        max_concurrency=inference_cfg.get("max_concurrent", 4),
        rate_limit_per_minute=inference_cfg.get("rate_limit_per_minute", 30),
        max_retries=inference_cfg.get("max_retries", 6),
    )

    lock = asyncio.Lock()
    total = len(models_config) * len(prompts)
    already_done = sum(
        checkpoint_key(
            model_cfg.get("provider", "openai-compatible"),
            model_cfg["_deployment_id"],
            model_cfg["_resolved_model_id"],
            prompt["id"],
            prompt["level"],
            model_cfg["_protocol_hash"],
        ) in completed
        for model_cfg in models_config
        for prompt in prompts
    )

    print(f"\nTotal requests : {total}")
    print(f"Already done   : {already_done}")
    print(f"Remaining      : {total - already_done}")
    print(f"Models         : {len(models_config)}\n")

    pbar = tqdm(total=total, initial=already_done, unit="resp", desc="Phase 2")

    for model_cfg in models_config:
        model_name = model_cfg["name"]
        remaining = sum(
            1 for p in prompts
            if checkpoint_key(
                model_cfg.get("provider", "openai-compatible"),
                model_cfg["_deployment_id"],
                model_cfg["_resolved_model_id"],
                p["id"],
                p["level"],
                model_cfg["_protocol_hash"],
            ) not in completed
        )
        logger.info(f"Model: {model_name} — {remaining}/{len(prompts)} prompts remaining")

        if remaining == 0:
            pbar.update(len(prompts))
            continue

        await runner.run_model(
            model_cfg=model_cfg,
            prompts=prompts,
            output_path=output_path,
            completed=completed,
            lock=lock,
            progress_cb=lambda: pbar.update(1),
        )

    pbar.close()

    # Summarize only identities belonging to this exact run/protocol. Failed
    # attempts may coexist with a later successful retry in the append-only
    # JSONL, and the same output file may contain an older protocol hash.
    expected_keys = {
        checkpoint_key(
            model_cfg.get("provider", "openai-compatible"),
            model_cfg["_deployment_id"],
            model_cfg["_resolved_model_id"],
            prompt["id"],
            prompt["level"],
            model_cfg["_protocol_hash"],
        )
        for model_cfg in models_config
        for prompt in prompts
    }
    latest = {
        key: row
        for key, row in _latest_records(output_path).items()
        if key in expected_keys
    }
    total_records = len(latest)
    success = 0
    error_types: dict[str, int] = {}
    total_cost = 0.0
    reasoning_violations = 0
    truncated_total = 0
    diagnostics: dict[str, dict] = defaultdict(
        lambda: {
            "total": 0,
            "success": 0,
            "reasoning": 0,
            "truncated": 0,
            "routes": defaultdict(int),
            "first_error": None,
        }
    )
    for r in latest.values():
        model_name = r.get("model_name") or r.get("model_id") or "unknown"
        diag = diagnostics[model_name]
        diag["total"] += 1

        if r.get("success"):
            success += 1
            diag["success"] += 1
        elif r.get("error_type"):
            t = r["error_type"]
            error_types[t] = error_types.get(t, 0) + 1
            if diag["first_error"] is None and r.get("error"):
                diag["first_error"] = str(r["error"])

        if r.get("reasoning_violation"):
            reasoning_violations += 1
            diag["reasoning"] += 1
        if r.get("was_truncated"):
            truncated_total += 1
            diag["truncated"] += 1

        route = r.get("openrouter_selected_provider") or "metadata-unavailable"
        diag["routes"][route] += 1

        try:
            total_cost += float((r.get("usage") or {}).get("cost") or 0.0)
        except (TypeError, ValueError):
            pass
    failed = total_records - success

    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Total responses     : {total_records}")
    print(f"Successful          : {success}")
    print(f"Failed              : {failed}")
    print(f"Reasoning violations: {reasoning_violations}")
    print(f"Truncated (all)     : {truncated_total}")
    if error_types:
        print(f"Error breakdown     : {error_types}")
    if total_cost:
        print(f"Reported cost       : ${total_cost:.4f}")
    print(f"Output              : {output_path}")

    print("\nPer-model routing diagnostics:")
    for model_name in sorted(diagnostics):
        diag = diagnostics[model_name]
        routes = ", ".join(
            f"{name}={count}"
            for name, count in sorted(diag["routes"].items())
        )
        print(
            f"  {model_name:28} "
            f"ok={diag['success']}/{diag['total']}  "
            f"reasoning={diag['reasoning']}  "
            f"truncated={diag['truncated']}  "
            f"routes=[{routes}]"
        )
        if diag["first_error"]:
            # One representative error is enough to diagnose model-level parameter
            # incompatibility without flooding the console with 40 identical lines.
            compact_error = " ".join(diag["first_error"].split())
            print(f"    first_error: {compact_error[:900]}")

    if success != total or failed:
        logger.error(
            "Phase 2 is incomplete: expected %d successful responses, got %d. "
            "Fix/retry before Phase 3.",
            total,
            success,
        )
        raise SystemExit(2)

    logger.info("Phase 2 complete!")


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 2: Model Inference")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--input", type=str, default="data/processed/robin_dataset.jsonl")
    parser.add_argument("--output", type=str, default="data/output/inference_results.jsonl")
    parser.add_argument("--manifest", type=str, default="artifacts/run_manifest.json",
                        help="Protocol and resolved-model manifest written before inference")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for smoke testing)")
    parser.add_argument(
        "--limit-strategy",
        choices=("head", "stratified"),
        default="head",
        help="How --limit chooses samples; use stratified for smoke tests",
    )
    parser.add_argument(
        "--limit-seed",
        type=int,
        default=42,
        help="Deterministic seed for --limit-strategy stratified",
    )
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated names/model_ids to keep (default: all configured)")
    parser.add_argument("--exclude-models", type=str, default=None,
                        help="Comma-separated names/model_ids to drop")
    parser.add_argument("--exclude-perturbation-model", type=str, default=None,
                        help="Exclude the Phase 1 perturbation model when it is in Phase 2")
    parser.add_argument("--add-model", action="append", default=None,
                        help="Ad-hoc model 'name=model_id'; requires endpoint fields in config")
    args = parser.parse_args()

    load_dotenv()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
