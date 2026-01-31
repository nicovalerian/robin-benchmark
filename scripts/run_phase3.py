#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase3 import ConstraintChecker, SemanticScorer, LLMJudge
from utils import load_config, load_jsonl, save_jsonl, setup_logger, get_env_var


async def run_evaluation(args):
    logger = setup_logger("robin-phase3", level="INFO")
    logger.info("Starting ROBIN Phase 3: Hybrid Evaluation")
    
    config = load_config(args.config)
    eval_config = config.get("evaluation", {})
    
    results = load_jsonl(args.input)
    logger.info(f"Loaded {len(results)} inference results")
    
    constraint_checker = ConstraintChecker()
    semantic_scorer = SemanticScorer(
        bert_model=eval_config.get("semantic_scorer", {}).get("bert_model", "indolem/indobert-base-uncased"),
    )
    
    judge_api_key = get_env_var("OPENAI_API_KEY", required=False)
    llm_judge = None
    if judge_api_key and eval_config.get("llm_judge", {}).get("enabled", True):
        llm_judge = LLMJudge(
            api_key=judge_api_key,
            model=eval_config.get("llm_judge", {}).get("model", "gpt-4o"),
        )
        logger.info("LLM Judge configured")
    else:
        logger.warning("LLM Judge disabled (no API key)")
    
    evaluated_results = []
    
    for result in results:
        sample_id = result["id"]
        constraints = result["constraints"]
        gold_response = result["gold_response"]
        
        evaluated_sample = {
            "id": sample_id,
            "category": result["category"],
            "evaluations": {},
        }
        
        for model_name, model_responses in result.get("model_responses", {}).items():
            evaluated_sample["evaluations"][model_name] = {}
            
            for level, response_data in model_responses.items():
                response = response_data.get("response", "")
                
                constraint_results = constraint_checker.check_all_constraints(
                    response, constraints
                )
                constraint_pass_rate = constraint_checker.get_pass_rate(constraint_results)
                
                semantic_score = semantic_scorer.score(response, gold_response)
                
                judge_score = None
                if llm_judge and args.run_judge:
                    try:
                        judge_result = await llm_judge.judge(
                            instruction=response_data.get("prompt", ""),
                            response=response,
                            reference=gold_response,
                        )
                        if judge_result.success:
                            judge_score = judge_result.score
                    except Exception as e:
                        logger.warning(f"Judge failed for {sample_id}: {e}")
                
                combined_score = (
                    constraint_pass_rate * 0.4 +
                    semantic_score.combined_score * 0.3 +
                    (judge_score / 5.0 if judge_score else 0.5) * 0.3
                )
                
                evaluated_sample["evaluations"][model_name][level] = {
                    "constraint_pass_rate": constraint_pass_rate,
                    "constraint_details": [
                        {"type": r.constraint_type, "passed": r.passed, "details": r.details}
                        for r in constraint_results
                    ],
                    "semantic_scores": {
                        "rouge_l_f1": semantic_score.rouge_l_f1,
                        "bert_f1": semantic_score.bert_f1,
                        "combined": semantic_score.combined_score,
                    },
                    "judge_score": judge_score,
                    "combined_score": combined_score,
                }
        
        evaluated_results.append(evaluated_sample)
        logger.info(f"Evaluated: {sample_id}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(evaluated_results, output_path)
    
    logger.info(f"Saved evaluation results to: {output_path}")
    logger.info("Phase 3 complete!")


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 3: Evaluation")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--input", type=str, default="data/output/inference_results.jsonl")
    parser.add_argument("--output", type=str, default="data/output/evaluation_results.jsonl")
    parser.add_argument("--run-judge", action="store_true", help="Run LLM-as-judge evaluation")
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
