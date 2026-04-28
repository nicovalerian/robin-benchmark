#!/usr/bin/env python3
"""
ROBIN Phase 1 — Dataset Construction

Loads the Indonesian Alpaca-cleaned dataset, picks a balanced subset,
injects verifiable constraints, and uses DigitalOcean Serverless
Inference (openai-gpt-5.4-mini) to generate the four perturbation
levels (L0-L3) for each base instruction.

Output: 1 JSONL row per base instruction containing all four levels.
With --samples 750 this yields 750 base × 4 levels = 3000 prompts.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1 import TaskClassifier, ConstraintInjector, PerturbationEngine  # noqa: E402
from utils import load_config, save_jsonl, setup_logger  # noqa: E402


async def _process_one(engine, sample, classifier, injector, idx):
    instruction = (sample.get("instruction") or "").strip()
    input_text = (sample.get("input") or "").strip()
    output_text = (sample.get("output") or "").strip()

    if not instruction or len(instruction) < 20:
        return None

    category = classifier.classify(instruction, input_text)
    full_instruction = f"{instruction}\n{input_text}".strip() if input_text else instruction

    constrained = injector.inject_constraints(
        instruction=full_instruction,
        category=category,
        gold_response=output_text,
    )

    try:
        perturbed = await engine.perturb_async(constrained.constrained_instruction)
    except Exception as exc:
        return {"_error": str(exc), "_id": idx, "_category": category}

    return {
        "id": f"robin_{idx:05d}",
        "category": category,
        "original_instruction": instruction,
        "original_input": input_text,
        "gold_response": output_text,
        "constraints": [
            {
                "constraint_type": c.constraint_type,
                "requirement": c.requirement,
                "verification_regex": c.verification_regex,
                "target_value": c.target_value,
            }
            for c in constrained.constraints
        ],
        "perturbations": {
            "level_0_clean": perturbed.level_0_clean,
            "level_1_mild": perturbed.level_1_mild,
            "level_2_jaksel": perturbed.level_2_jaksel,
            "level_3_adversarial": perturbed.level_3_adversarial,
        },
        "perturbation_meta": perturbed.metadata,
    }


async def _run(args, logger):
    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    pert_config = config.get("perturbation", {})
    llm_cfg = pert_config.get("llm", {})

    random.seed(args.seed)

    source = dataset_config.get("source", "ilhamfadheel/alpaca-cleaned-indonesian")
    logger.info(f"Loading dataset: {source}")
    source_dataset = load_dataset(source, split="train")

    classifier = TaskClassifier(dataset_config.get("categories"))
    injector = ConstraintInjector(
        constraint_types=config.get("constraints", {}).get("types", ["keyword", "length"]),
        seed=args.seed,
    )
    engine = PerturbationEngine(
        model=llm_cfg.get("model", "minimax-m2.5"),
        base_url=llm_cfg.get("base_url", "https://inference.do-ai.run/v1/"),
        prompts_path=pert_config.get("prompts_path"),
        indocollex_path=config.get("paths", {}).get("data_raw", "data/raw") + "/indocollex.json",
        max_concurrency=llm_cfg.get("max_concurrency", 8),
        temperature=llm_cfg.get("temperature", 0.4),
        max_retries=llm_cfg.get("max_retries", 6),
        rate_limit_per_minute=llm_cfg.get("rate_limit_per_minute", 15),
        seed=args.seed,
    )

    target_size = args.samples or dataset_config.get("target_size", 750)
    weights = classifier.get_distribution_weights()
    category_targets = {cat: max(1, int(target_size * w)) for cat, w in weights.items()}
    # Distribute any remainder (from floor rounding) to the largest category.
    remainder = target_size - sum(category_targets.values())
    if remainder > 0:
        largest = max(category_targets, key=category_targets.get)
        category_targets[largest] += remainder
    logger.info(f"Target distribution: {category_targets}")

    indices = list(range(len(source_dataset)))
    random.shuffle(indices)

    # Index source samples by category up front so top-up rounds can pull
    # extra rows without re-classifying.
    by_category: dict[str, list] = {cat: [] for cat in weights}
    for idx in indices:
        sample = source_dataset[idx]
        instruction = (sample.get("instruction") or "").strip()
        input_text = (sample.get("input") or "").strip()
        if not instruction or len(instruction) < 20:
            continue
        cat = classifier.classify(instruction, input_text)
        if cat in by_category:
            by_category[cat].append(sample)

    robin_samples: list = []
    failures: list = []
    category_counts = {cat: 0 for cat in weights}
    cat_cursors = {cat: 0 for cat in weights}

    round_idx = 0
    max_rounds = 4
    while round_idx < max_rounds:
        round_idx += 1
        # Pick fresh batch to fill remaining gaps per category.
        batch = []
        for cat, target in category_targets.items():
            need = target - category_counts[cat]
            if need <= 0:
                continue
            # Oversample by 25% on round 1 only, to absorb residual failures.
            ask = int(need * 1.25) if round_idx == 1 else need
            available = by_category[cat][cat_cursors[cat]:cat_cursors[cat] + ask]
            cat_cursors[cat] += ask
            for s in available:
                batch.append((cat, s))

        if not batch:
            break

        logger.info(
            f"Round {round_idx}: perturbing {len(batch)} candidates "
            f"(remaining gaps: { {c: category_targets[c]-category_counts[c] for c in weights} })"
        )

        start_id = len(robin_samples) + len(failures)
        coros = [
            _process_one(engine, s, classifier, injector, start_id + i)
            for i, (_cat, s) in enumerate(batch)
        ]
        results = await tqdm_asyncio.gather(*coros, desc=f"Round {round_idx}")

        for (cat, _s), r in zip(batch, results):
            if r is None:
                continue
            if "_error" in r:
                failures.append(r)
                continue
            # Stop adding to a category once full so we don't overshoot from oversampling.
            if category_counts[cat] >= category_targets[cat]:
                continue
            robin_samples.append(r)
            category_counts[cat] += 1

        if all(category_counts[c] >= category_targets[c] for c in weights):
            logger.info(f"All category targets met after round {round_idx}.")
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(robin_samples, output_path)

    if failures:
        fail_path = output_path.parent / "phase1_failures.jsonl"
        save_jsonl(failures, fail_path)
        logger.warning(f"{len(failures)} samples failed; logged to {fail_path}")

    stats = {
        "total_samples": len(robin_samples),
        "total_prompts": len(robin_samples) * 4,
        "failures": len(failures),
        "category_distribution": category_counts,
        "config_used": str(args.config),
        "source_dataset": source,
        "perturbation_model": llm_cfg.get("model", "openai-gpt-5.4-mini"),
    }
    with open(output_path.parent / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated {len(robin_samples)} samples ({len(robin_samples) * 4} prompts)")
    logger.info(f"Saved to: {output_path}")
    await engine.aclose()


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 1: Dataset Construction")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--output", type=str, default="data/processed/robin_dataset.jsonl")
    parser.add_argument("--samples", type=int, default=None,
                        help="Override target_size from config (default: use config = 750)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()
    logger = setup_logger("robin-phase1", level="INFO")
    logger.info("Starting ROBIN Phase 1: Dataset Construction")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run(args, logger))
    logger.info("Phase 1 complete!")


if __name__ == "__main__":
    main()
