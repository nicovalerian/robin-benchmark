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
import io
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from phase2 import InferenceRunner  # noqa: E402
from utils import load_config, load_jsonl, setup_logger  # noqa: E402


LEVEL_KEYS = ["level_0_clean", "level_1_mild", "level_2_jaksel", "level_3_adversarial"]


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [t.strip() for t in value.split(",") if t.strip()]


def _load_completed(output_path: Path) -> set[str]:
    """Return set of 'model_id|sample_id|level' keys already written."""
    if not output_path.exists():
        return set()
    completed = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = f"{row['model_id']}|{row['sample_id']}|{row['level']}"
            completed.add(key)
    return completed


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
    try:
        models_config = select_models(
            models_config,
            include=_split_csv(args.models),
            exclude=_split_csv(args.exclude_models),
            add=args.add_model,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Models for this run: {', '.join(m['name'] for m in models_config)}")

    # Default credential = DigitalOcean. Models may override per-entry (BYOK) via
    # `base_url` + `api_key_env`; validate every model can resolve a key up front.
    from phase2.inference_runner import resolve_credentials, is_local_url  # noqa: E402

    default_key = os.getenv("DIGITALOCEAN_INFERENCE_KEY", "")
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
    api_key = default_key

    dataset = load_jsonl(args.input)
    if args.limit:
        dataset = dataset[:args.limit]
    logger.info(f"Loaded {len(dataset)} samples")

    prompts = _build_prompts(dataset)
    logger.info(f"Built {len(prompts)} prompts ({len(dataset)} samples x {len(LEVEL_KEYS)} levels)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = _load_completed(output_path)
    if completed:
        logger.info(f"Resuming: {len(completed)} responses already written")

    runner = InferenceRunner(
        models_config=models_config,
        api_key=api_key,
        temperature=inference_cfg.get("temperature", 0.0),
        max_tokens=inference_cfg.get("max_tokens", 1024),
        max_concurrency=inference_cfg.get("max_concurrent", 20),
        rate_limit_per_minute=inference_cfg.get("rate_limit_per_minute", 100),
        max_retries=inference_cfg.get("max_retries", 6),
    )

    lock = asyncio.Lock()
    total = len(models_config) * len(prompts)
    already_done = len(completed)

    print(f"\nTotal requests : {total}")
    print(f"Already done   : {already_done}")
    print(f"Remaining      : {total - already_done}")
    print(f"Models         : {len(models_config)}\n")

    pbar = tqdm(total=total, initial=already_done, unit="resp", desc="Phase 2")

    for model_cfg in models_config:
        model_name = model_cfg["name"]
        remaining = sum(
            1 for p in prompts
            if f"{model_cfg['model_id']}|{p['id']}|{p['level']}" not in completed
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

    # Summary — stream the output file line by line instead of loading the whole
    # (potentially 20k+ record) file into memory just to tally success/failure.
    total_records = 0
    success = 0
    error_types: dict[str, int] = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total_records += 1
            if r.get("success"):
                success += 1
            elif r.get("error_type"):
                t = r["error_type"]
                error_types[t] = error_types.get(t, 0) + 1
    failed = total_records - success

    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Total responses : {total_records}")
    print(f"Successful      : {success}")
    print(f"Failed          : {failed}")
    if error_types:
        print(f"Error breakdown : {error_types}")
    print(f"Output          : {output_path}")

    logger.info("Phase 2 complete!")


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 2: Model Inference")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--input", type=str, default="data/processed/robin_dataset.jsonl")
    parser.add_argument("--output", type=str, default="data/output/inference_results.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for smoke testing)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated names/model_ids to keep (default: all configured)")
    parser.add_argument("--exclude-models", type=str, default=None,
                        help="Comma-separated names/model_ids to drop")
    parser.add_argument("--add-model", action="append", default=None,
                        help="Ad-hoc DO model 'name=model_id' (or just model_id); repeatable")
    args = parser.parse_args()

    load_dotenv()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
