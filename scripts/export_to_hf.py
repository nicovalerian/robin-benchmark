#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_jsonl, setup_logger


def flatten_sample(sample: dict) -> dict:
    flattened = {
        "id": sample["id"],
        "category": sample["category"],
        "original_instruction": sample.get("original_instruction", ""),
        "original_input": sample.get("original_input", ""),
        "gold_response": sample.get("gold_response", ""),
    }
    
    perturbations = sample.get("perturbations", {})
    flattened["level_0_clean"] = perturbations.get("level_0_clean", "")
    flattened["level_1_mild"] = perturbations.get("level_1_mild", "")
    flattened["level_2_jaksel"] = perturbations.get("level_2_jaksel", "")
    flattened["level_3_adversarial"] = perturbations.get("level_3_adversarial", "")
    
    constraints = sample.get("constraints", [])
    flattened["constraints_json"] = json.dumps(constraints, ensure_ascii=False)
    
    return flattened


def main():
    parser = argparse.ArgumentParser(description="Export ROBIN to HuggingFace format")
    parser.add_argument("--input", type=str, default="data/processed/robin_dataset.jsonl")
    parser.add_argument("--output", type=str, default="data/output/robin_hf")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-name", type=str, default="robin-benchmark")
    args = parser.parse_args()
    
    logger = setup_logger("robin-export")
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Loading dataset from {input_path}")
    raw_samples = load_jsonl(input_path)
    
    flattened_samples = [flatten_sample(s) for s in raw_samples]
    logger.info(f"Processed {len(flattened_samples)} samples")
    
    dataset = Dataset.from_list(flattened_samples)
    
    split_idx = int(len(flattened_samples) * 0.9)
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(flattened_samples[:split_idx]),
        "test": Dataset.from_list(flattened_samples[split_idx:]),
    })
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_dict.save_to_disk(str(output_path))
    logger.info(f"Saved HuggingFace dataset to {output_path}")
    
    if args.push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {args.hub_name}")
        dataset_dict.push_to_hub(args.hub_name)
        logger.info("Push complete!")
    
    logger.info("Export complete!")
    logger.info(f"  Train samples: {len(dataset_dict['train'])}")
    logger.info(f"  Test samples: {len(dataset_dict['test'])}")


if __name__ == "__main__":
    main()
