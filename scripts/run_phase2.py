#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase2 import InferenceRunner
from utils import load_config, load_jsonl, save_jsonl, setup_logger, get_env_var


async def run_inference(args):
    logger = setup_logger("robin-phase2", level="INFO")
    logger.info("Starting ROBIN Phase 2: Model Inference")
    
    config = load_config(args.config)
    inference_config = config.get("inference", {})
    
    dataset = load_jsonl(args.input)
    logger.info(f"Loaded {len(dataset)} samples")
    
    providers_config = []
    for model_cfg in inference_config.get("models", []):
        provider = model_cfg.get("provider")
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "together": "TOGETHER_API_KEY",
            "google": "GOOGLE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        api_key = get_env_var(api_key_map.get(provider, ""), required=False)
        
        if api_key:
            providers_config.append({
                "name": model_cfg.get("name"),
                "provider": provider,
                "model_id": model_cfg.get("model_id"),
                "api_key": api_key,
            })
            logger.info(f"Configured provider: {model_cfg.get('name')}")
        else:
            logger.warning(f"Skipping {model_cfg.get('name')}: API key not found")
    
    if not providers_config:
        logger.error("No providers configured. Please set API keys in .env")
        return
    
    runner = InferenceRunner(
        providers_config=providers_config,
        temperature=inference_config.get("temperature", 0.0),
        max_tokens=inference_config.get("max_tokens", 512),
        max_concurrent=inference_config.get("max_concurrent", 5),
        rate_limit_delay=inference_config.get("rate_limit_delay", 0.5),
    )
    
    results = []
    perturbation_levels = ["level_0_clean", "level_1_mild", "level_2_jaksel", "level_3_adversarial"]
    
    for sample in dataset:
        sample_results = {
            "id": sample["id"],
            "category": sample["category"],
            "constraints": sample["constraints"],
            "gold_response": sample["gold_response"],
            "model_responses": {},
        }
        
        for level_key in perturbation_levels:
            prompt = sample["perturbations"].get(level_key, "")
            if not prompt:
                continue
            
            for model_name in runner.providers:
                logger.info(f"Running {model_name} on {sample['id']} - {level_key}")
                result = await runner.run_single(model_name, prompt)
                
                if model_name not in sample_results["model_responses"]:
                    sample_results["model_responses"][model_name] = {}
                
                level_num = int(level_key.split("_")[1])
                sample_results["model_responses"][model_name][level_num] = {
                    "prompt": prompt,
                    "response": result.response,
                    "success": result.success,
                    "tokens_used": result.tokens_used,
                    "latency_ms": result.latency_ms,
                    "error": result.error_message,
                }
        
        results.append(sample_results)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, output_path)
    
    logger.info(f"Saved inference results to: {output_path}")
    logger.info("Phase 2 complete!")


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 2: Model Inference")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--input", type=str, default="data/processed/robin_dataset.jsonl")
    parser.add_argument("--output", type=str, default="data/output/inference_results.jsonl")
    args = parser.parse_args()
    
    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
