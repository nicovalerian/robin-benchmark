#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1 import TaskClassifier, ConstraintInjector, PerturbationEngine
from utils import load_config, save_jsonl, setup_logger


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 1: Dataset Construction")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--output", type=str, default="data/processed/robin_dataset.jsonl")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to generate (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    logger = setup_logger("robin-phase1", level="INFO")
    logger.info("Starting ROBIN Phase 1: Dataset Construction")
    
    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    
    random.seed(args.seed)
    
    logger.info(f"Loading dataset: {dataset_config.get('source')}")
    source_dataset = load_dataset(
        dataset_config.get("source", "FreedomIntelligence/alpaca-gpt4-indonesian"),
        split="train",
    )
    
    classifier = TaskClassifier(dataset_config.get("categories"))
    injector = ConstraintInjector(
        constraint_types=config.get("constraints", {}).get("types", ["keyword", "length"]),
        seed=args.seed,
    )
    perturbation_engine = PerturbationEngine(
        indocollex_path=config.get("perturbation", {}).get("levels", {}).get("3", {}).get("indocollex_path"),
        seed=args.seed,
    )
    
    target_size = args.samples if args.samples else dataset_config.get("target_size", 30)
    weights = classifier.get_distribution_weights()
    
    category_counts = {cat: 0 for cat in weights}
    category_targets = {cat: max(1, int(target_size * weight)) for cat, weight in weights.items()}
    
    logger.info(f"Target distribution: {category_targets}")
    
    robin_samples = []
    indices = list(range(len(source_dataset)))
    random.shuffle(indices)
    
    for idx in tqdm(indices, desc="Processing samples"):
        if len(robin_samples) >= target_size:
            break
        
        sample = source_dataset[idx]
        
        conversations = sample.get("conversations", [])
        if len(conversations) >= 2:
            instruction = conversations[0].get("value", "")
            output_text = conversations[1].get("value", "")
        else:
            instruction = sample.get("instruction", "")
            output_text = sample.get("output", "")
        
        input_text = sample.get("input", "")
        
        if not instruction or len(instruction) < 20:
            continue
        
        category = classifier.classify(instruction, input_text)
        
        if category_counts[category] >= category_targets[category]:
            continue
        
        full_instruction = f"{instruction}\n{input_text}".strip() if input_text else instruction
        
        constrained = injector.inject_constraints(
            instruction=full_instruction,
            category=category,
            gold_response=output_text,
        )
        
        perturbed = perturbation_engine.perturb(constrained.constrained_instruction)
        
        robin_sample = {
            "id": f"robin_{len(robin_samples):05d}",
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
        }
        
        robin_samples.append(robin_sample)
        category_counts[category] += 1
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(robin_samples, output_path)
    
    logger.info(f"Generated {len(robin_samples)} samples")
    logger.info(f"Category distribution: {category_counts}")
    logger.info(f"Saved to: {output_path}")
    
    stats = {
        "total_samples": len(robin_samples),
        "category_distribution": category_counts,
        "config_used": str(args.config),
    }
    stats_path = output_path.parent / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Phase 1 complete!")


if __name__ == "__main__":
    main()
