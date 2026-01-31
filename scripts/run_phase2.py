#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase2 import InferenceRunner
from utils import load_config, load_jsonl, save_jsonl, setup_logger, get_env_var


API_KEY_MAP = {
    "openai": ("OPENAI_API_KEY", "OpenAI (GPT-4o)"),
    "anthropic": ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
    "together": ("TOGETHER_API_KEY", "Together AI (Llama, Qwen, Mistral)"),
    "google": ("GOOGLE_API_KEY", "Google AI Studio (Gemini, Gemma)"),
    "groq": ("GROQ_API_KEY", "Groq (Llama, Mixtral)"),
    "huggingface": ("HUGGINGFACE_API_KEY", "HuggingFace (Cendol)"),
}


def check_api_keys(models_config: list[dict], interactive: bool = True) -> dict[str, str]:
    available_keys = {}
    missing_providers = []
    
    providers_needed = set(m.get("provider") for m in models_config)
    
    placeholder_patterns = ["...", "sk-...", "sk-ant-...", "gsk_...", "hf_...", "your-", "xxx", "placeholder"]
    
    for provider in providers_needed:
        if provider not in API_KEY_MAP:
            continue
            
        env_var, display_name = API_KEY_MAP[provider]
        api_key = os.getenv(env_var, "")
        
        is_placeholder = any(p in api_key.lower() for p in placeholder_patterns) or len(api_key) < 10
        
        if api_key and not is_placeholder:
            available_keys[provider] = api_key
        else:
            missing_providers.append((provider, env_var, display_name))
    
    if missing_providers and interactive:
        print("\n" + "=" * 60)
        print("API KEY CONFIGURATION")
        print("=" * 60)
        print("\nThe following API keys are missing or not configured:")
        for provider, env_var, display_name in missing_providers:
            print(f"  - {display_name} ({env_var})")
        
        print("\nYou can:")
        print("  1. Enter API keys now (they will be saved to .env)")
        print("  2. Press Enter to skip and only use available providers")
        print()
        
        for provider, env_var, display_name in missing_providers:
            key = input(f"Enter {display_name} API key (or press Enter to skip): ").strip()
            if key:
                available_keys[provider] = key
                save_key_to_env(env_var, key)
                print(f"  [OK] Saved {env_var} to .env")
    
    return available_keys


def save_key_to_env(env_var: str, api_key: str):
    env_path = Path(__file__).parent.parent / ".env"
    
    if env_path.exists():
        with open(env_path, "r") as f:
            lines = f.readlines()
        
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[i] = f"{env_var}={api_key}\n"
                updated = True
                break
        
        if not updated:
            lines.append(f"{env_var}={api_key}\n")
        
        with open(env_path, "w") as f:
            f.writelines(lines)
    else:
        with open(env_path, "w") as f:
            f.write(f"{env_var}={api_key}\n")


def filter_models_by_available_keys(models_config: list[dict], available_keys: dict[str, str]) -> list[dict]:
    filtered = []
    for model in models_config:
        provider = model.get("provider")
        if provider in available_keys:
            filtered.append({
                **model,
                "api_key": available_keys[provider]
            })
    return filtered


async def run_inference(args):
    logger = setup_logger("robin-phase2", level="INFO")
    logger.info("Starting ROBIN Phase 2: Model Inference")
    
    config = load_config(args.config)
    inference_config = config.get("inference", {})
    models_config = inference_config.get("models", [])
    
    if not models_config:
        logger.error("No models configured in config file")
        return
    
    print("\n" + "=" * 60)
    print("ROBIN BENCHMARK - Phase 2: Model Inference")
    print("=" * 60)
    
    available_keys = check_api_keys(models_config, interactive=not args.non_interactive)
    
    if not available_keys:
        logger.error("No API keys available. Please configure at least one provider.")
        print("\nTo configure API keys, either:")
        print("  1. Run this script again without --non-interactive")
        print("  2. Edit .env file directly")
        print("  3. Set environment variables")
        return
    
    providers_config = filter_models_by_available_keys(models_config, available_keys)
    
    if not providers_config:
        logger.error("No models available with current API keys")
        return
    
    print("\n" + "-" * 40)
    print("Models that will be evaluated:")
    for model in providers_config:
        print(f"  [OK] {model['name']} ({model['provider']})")
    print("-" * 40)
    
    skipped_models = [m for m in models_config if m.get("provider") not in available_keys]
    if skipped_models:
        print("\nModels skipped (no API key):")
        for model in skipped_models:
            print(f"  [--] {model['name']} ({model['provider']})")
    
    if not args.yes:
        confirm = input("\nProceed with inference? [Y/n]: ").strip().lower()
        if confirm and confirm != 'y':
            print("Aborted.")
            return
    
    dataset = load_jsonl(args.input)
    logger.info(f"Loaded {len(dataset)} samples")
    
    runner = InferenceRunner(
        providers_config=providers_config,
        temperature=inference_config.get("temperature", 0.0),
        max_tokens=inference_config.get("max_tokens", 512),
        max_concurrent=inference_config.get("max_concurrent", 5),
        rate_limit_delay=inference_config.get("rate_limit_delay", 0.5),
    )
    
    results = []
    perturbation_levels = ["level_0_clean", "level_1_mild", "level_2_jaksel", "level_3_adversarial"]
    
    total_requests = len(dataset) * len(perturbation_levels) * len(runner.providers)
    completed = 0
    
    print(f"\nRunning inference: {total_requests} total requests")
    print(f"  {len(dataset)} samples × {len(perturbation_levels)} levels × {len(runner.providers)} models")
    print()
    
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
                completed += 1
                progress = f"[{completed}/{total_requests}]"
                print(f"{progress} {model_name} | {sample['id']} | {level_key}", end="\r")
                
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
                
                if not result.success:
                    logger.warning(f"Failed: {model_name} on {sample['id']}: {result.error_message}")
        
        results.append(sample_results)
    
    print()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, output_path)
    
    success_count = sum(
        1 for r in results 
        for m in r["model_responses"].values() 
        for l in m.values() 
        if l.get("success")
    )
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total requests: {total_requests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_requests - success_count}")
    print(f"Output saved to: {output_path}")
    
    logger.info("Phase 2 complete!")


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 2: Model Inference")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--input", type=str, default="data/processed/robin_dataset.jsonl")
    parser.add_argument("--output", type=str, default="data/output/inference_results.jsonl")
    parser.add_argument("--non-interactive", action="store_true", help="Don't prompt for missing API keys")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()
    
    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
