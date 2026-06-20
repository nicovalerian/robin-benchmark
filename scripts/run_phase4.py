#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase4 import PDRCalculator, SkillMapper
from utils import load_jsonl, setup_logger
from visualization import (
    plot_robustness_comparison,
    plot_constraint_breakdown,
    plot_heatmap,
    plot_perturbation_comparison,
)

CONSTRAINT_TYPES = ("keyword", "length", "format")


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 4: Analysis & Reporting")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--input", type=str, default="data/output/evaluation_results.jsonl")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--dataset", type=str, default="data/processed/robin_dataset.jsonl",
                        help="Phase 1 dataset used for the perturbation-examples figure")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations without prompt")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualizations without prompt")
    args = parser.parse_args()
    
    logger = setup_logger("robin-phase4", level="INFO")
    logger.info("Starting ROBIN Phase 4: Analysis & Reporting")

    results = load_jsonl(args.input)
    logger.info(f"Loaded {len(results)} evaluation results")
    
    pdr_calculator = PDRCalculator()
    skill_mapper = SkillMapper()
    
    # Phase 3 writes a flat JSONL: one record per (model, sample, level).
    model_level_scores: dict[str, dict[int, dict[str, list[float]]]] = {}
    model_results: dict[str, list[dict]] = {}
    # Per-constraint-type pass booleans: model -> level -> type -> [bool,...].
    # Averaging the three into one CPR hides that keyword/format are saturated
    # while length is the only discriminating constraint, so we keep them split.
    model_cpr_type: dict[str, dict[int, dict[str, list[bool]]]] = {}

    for record in results:
        model_name = record["model_name"]
        level = int(record["level"])

        if model_name not in model_level_scores:
            model_level_scores[model_name] = {}
            model_results[model_name] = []
            model_cpr_type[model_name] = {}

        if level not in model_level_scores[model_name]:
            model_level_scores[model_name][level] = {
                "constraint_pass_rate": [],
                "semantic_score": [],
                "combined_score": [],
            }
            model_cpr_type[model_name][level] = {t: [] for t in CONSTRAINT_TYPES}

        model_level_scores[model_name][level]["constraint_pass_rate"].append(record.get("cpr", 0))
        model_level_scores[model_name][level]["semantic_score"].append(record.get("semantic", 0))
        model_level_scores[model_name][level]["combined_score"].append(record.get("composite", 0))

        for cr in record.get("constraint_results", []):
            ctype = cr.get("type")
            if ctype in model_cpr_type[model_name][level]:
                model_cpr_type[model_name][level][ctype].append(bool(cr.get("passed")))

        model_results[model_name].append({
            "category": record.get("category", ""),
            "perturbation_level": level,
            "combined_score": record.get("composite", 0),
            "constraints_passed": {
                cr["type"]: cr["passed"]
                for cr in record.get("constraint_results", [])
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
        wlr = pdr_calculator.get_worst_level_robustness(avg_scores)
        mean_level = pdr_calculator.get_mean_level(avg_scores)

        # Per-constraint-type compliance, per level and overall (mean over levels).
        cpr_type_by_level: dict[int, dict[str, float]] = {}
        for lvl, types in model_cpr_type[model_name].items():
            cpr_type_by_level[lvl] = {
                t: (sum(vals) / len(vals) if vals else 0.0)
                for t, vals in types.items()
            }
        cpr_type_overall = {
            t: (
                round(sum(cpr_type_by_level[lvl][t] for lvl in cpr_type_by_level)
                      / len(cpr_type_by_level), 4)
                if cpr_type_by_level else 0.0
            )
            for t in CONSTRAINT_TYPES
        }

        all_pdr_results[model_name] = {
            "level_scores": avg_scores,
            "pdr_by_level": level_pdrs,
            # PRIMARY metric (capability-aware): min composite across L0-L3.
            "worst_level_robustness": wlr,
            "mean_level_score": mean_level,
            # CPR split by constraint type — length is the real discriminator;
            # keyword/format saturate near 1.0 (see eval-design notes).
            "cpr_by_type_overall": cpr_type_overall,
            "cpr_by_type_by_level": {
                str(lvl): {t: round(v, 4) for t, v in types.items()}
                for lvl, types in cpr_type_by_level.items()
            },
            # SECONDARY (shape only) — 100 - mean PDR; do not rank on this alone.
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
    
    # PRIMARY ranking: Worst-Level Robustness (min composite across L0-L3).
    wlr_ranking = sorted(
        [
            {
                "model": m,
                "worst_level_robustness": d["worst_level_robustness"],
                "mean_level_score": d["mean_level_score"],
                "cpr_by_type": d["cpr_by_type_overall"],  # keyword/length/format split
                "pdr_robustness": d["robustness_score"],  # secondary shape metric
            }
            for m, d in all_pdr_results.items()
        ],
        key=lambda r: r["worst_level_robustness"],
        reverse=True,
    )

    summary = {
        "total_samples": len(results),
        "models_evaluated": list(model_level_scores.keys()),
        "primary_metric": "worst_level_robustness (min composite over L0-L3)",
        "ranking": wlr_ranking,
        # kept for backward compatibility; PDR is a secondary shape diagnostic
        "robustness_ranking_pdr": sorted(
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
    logger.info("\nPRIMARY Ranking - Worst-Level Robustness (higher = more robust):")
    for rank, r in enumerate(wlr_ranking, 1):
        c = r["cpr_by_type"]
        logger.info(
            f"  {rank}. {r['model']}: WLR={r['worst_level_robustness']:.3f} "
            f"(mean={r['mean_level_score']:.3f}, PDR-robustness={r['pdr_robustness']:.1f})"
        )
        logger.info(
            f"       CPR by type -> keyword={c['keyword']:.2f} "
            f"length={c['length']:.2f} format={c['format']:.2f}"
        )
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
        
        # Robustness comparison: WLR + mean-level (primary) + PDR-by-level (secondary)
        wlr_for_plot = {m: d["worst_level_robustness"] for m, d in all_pdr_results.items()}
        mean_for_plot = {m: d["mean_level_score"] for m, d in all_pdr_results.items()}
        pdr_for_plot = {m: d["pdr_by_level"] for m, d in all_pdr_results.items()}
        path = plot_robustness_comparison(
            wlr_for_plot, pdr_for_plot, str(figures_dir), mean_data=mean_for_plot
        )
        logger.info(f"  Saved: {path}")

        # Per-constraint-type compliance breakdown (keyword/length/format)
        cpr_type_for_plot = {m: d["cpr_by_type_overall"] for m, d in all_pdr_results.items()}
        path = plot_constraint_breakdown(cpr_type_for_plot, str(figures_dir))
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
        
        # Perturbation examples (load the Phase 1 dataset used for THIS run)
        dataset_path = Path(args.dataset)
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
