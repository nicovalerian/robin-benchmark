#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase4 import PDRCalculator, SkillMapper
from utils import load_config, load_jsonl, setup_logger
from visualization import (
    plot_pdr_bars,
    plot_heatmap,
    plot_perturbation_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 4: Analysis & Reporting")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--input", type=str, default="data/output/evaluation_results.jsonl")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations without prompt")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualizations without prompt")
    args = parser.parse_args()
    
    logger = setup_logger("robin-phase4", level="INFO")
    logger.info("Starting ROBIN Phase 4: Analysis & Reporting")
    
    config = load_config(args.config)
    
    results = load_jsonl(args.input)
    logger.info(f"Loaded {len(results)} evaluation results")
    
    pdr_calculator = PDRCalculator()
    skill_mapper = SkillMapper()
    
    model_level_scores: dict[str, dict[int, dict[str, list[float]]]] = {}
    model_results: dict[str, list[dict]] = {}
    
    for sample in results:
        category = sample["category"]
        
        for model_name, evaluations in sample.get("evaluations", {}).items():
            if model_name not in model_level_scores:
                model_level_scores[model_name] = {}
                model_results[model_name] = []
            
            for level_str, eval_data in evaluations.items():
                level = int(level_str)
                
                if level not in model_level_scores[model_name]:
                    model_level_scores[model_name][level] = {
                        "constraint_pass_rate": [],
                        "semantic_score": [],
                        "combined_score": [],
                    }
                
                model_level_scores[model_name][level]["constraint_pass_rate"].append(
                    eval_data.get("constraint_pass_rate", 0)
                )
                model_level_scores[model_name][level]["semantic_score"].append(
                    eval_data.get("semantic_scores", {}).get("combined", 0)
                )
                model_level_scores[model_name][level]["combined_score"].append(
                    eval_data.get("combined_score", 0)
                )
                
                model_results[model_name].append({
                    "category": category,
                    "perturbation_level": level,
                    "combined_score": eval_data.get("combined_score", 0),
                    "constraints_passed": {
                        d["type"]: d["passed"]
                        for d in eval_data.get("constraint_details", [])
                    },
                })
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_pdr_results = {}
    for model_name, level_scores in model_level_scores.items():
        avg_scores = {
            level: {
                metric: sum(scores) / len(scores) if scores else 0
                for metric, scores in metrics.items()
            }
            for level, metrics in level_scores.items()
        }
        
        pdr_results = pdr_calculator.calculate_for_model(
            model_name=model_name,
            results=avg_scores,
            baseline_level=0,
        )
        
        level_pdrs = pdr_calculator.aggregate_by_level(pdr_results)
        robustness_score = pdr_calculator.get_robustness_score(pdr_results)
        
        all_pdr_results[model_name] = {
            "level_scores": avg_scores,
            "pdr_by_level": level_pdrs,
            "robustness_score": robustness_score,
            "pdr_details": [
                {
                    "metric": r.metric_name,
                    "level": r.perturbation_level,
                    "baseline": r.baseline_score,
                    "perturbed": r.perturbed_score,
                    "pdr": r.pdr_percentage,
                }
                for r in pdr_results
            ],
        }
    
    with open(output_dir / "pdr_analysis.json", "w") as f:
        json.dump(all_pdr_results, f, indent=2)
    logger.info("Saved PDR analysis")
    
    all_profiles = {}
    for model_name, results_list in model_results.items():
        profile = skill_mapper.create_profile(model_name, results_list)
        all_profiles[model_name] = {
            "category_scores": profile.category_scores,
            "constraint_scores": profile.constraint_scores,
            "level_scores": {str(k): v for k, v in profile.level_scores.items()},
            "radar_data": skill_mapper.get_radar_data(profile),
        }
    
    with open(output_dir / "skill_profiles.json", "w") as f:
        json.dump(all_profiles, f, indent=2)
    logger.info("Saved skill profiles")
    
    summary = {
        "total_samples": len(results),
        "models_evaluated": list(model_level_scores.keys()),
        "robustness_ranking": sorted(
            [(m, d["robustness_score"]) for m, d in all_pdr_results.items()],
            key=lambda x: x[1],
            reverse=True,
        ),
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 50)
    logger.info("ROBIN Benchmark Results Summary")
    logger.info("=" * 50)
    logger.info(f"Total samples evaluated: {summary['total_samples']}")
    logger.info(f"Models: {', '.join(summary['models_evaluated'])}")
    logger.info("\nRobustness Ranking (higher = more robust):")
    for rank, (model, score) in enumerate(summary["robustness_ranking"], 1):
        logger.info(f"  {rank}. {model}: {score:.2f}")
    logger.info("=" * 50)
    
    # Handle visualization generation
    should_visualize = False
    if args.visualize:
        should_visualize = True
    elif args.no_visualize:
        should_visualize = False
    else:
        response = input("\nGenerate visualizations? [Y/n]: ").strip().lower()
        should_visualize = response != 'n'
    
    if should_visualize:
        logger.info("\nGenerating visualizations...")
        figures_dir = output_dir / "figures"
        
        # PDR comparison bar chart
        pdr_for_plot = {m: d["pdr_by_level"] for m, d in all_pdr_results.items()}
        path = plot_pdr_bars(pdr_for_plot, str(figures_dir))
        logger.info(f"  Saved: {path}")
        
        # Pass rate heatmap
        heatmap_data = {
            m: {int(k): v["constraint_pass_rate"] for k, v in d["level_scores"].items()}
            for m, d in all_pdr_results.items()
        }
        path = plot_heatmap(heatmap_data, str(figures_dir))
        logger.info(f"  Saved: {path}")
        
        # Perturbation trend
        trend_data = {
            m: {int(k): v["combined_score"] for k, v in d["level_scores"].items()}
            for m, d in all_pdr_results.items()
        }
        path = plot_perturbation_comparison(trend_data, str(figures_dir))
        logger.info(f"  Saved: {path}")
        
        # Perturbation examples (load Phase 1 dataset)
        dataset_path = Path("data/processed/robin_dataset.jsonl")
        if dataset_path.exists():
            from visualization.plots import plot_perturbation_examples
            dataset = load_jsonl(dataset_path)
            path = plot_perturbation_examples(dataset, output_dir=str(figures_dir))
            logger.info(f"  Saved: {path}")
        
        logger.info(f"\nAll visualizations saved to: {figures_dir}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Phase 4 complete!")


if __name__ == "__main__":
    main()
