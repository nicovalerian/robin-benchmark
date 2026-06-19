from dataclasses import dataclass


@dataclass
class PDRResult:
    model_name: str
    metric_name: str
    baseline_score: float
    perturbed_score: float
    pdr_percentage: float
    perturbation_level: int


class PDRCalculator:
    def calculate_pdr(
        self,
        baseline_score: float,
        perturbed_score: float,
    ) -> float:
        if baseline_score == 0:
            return 0.0
        pdr = ((baseline_score - perturbed_score) / baseline_score) * 100
        return round(pdr, 2)
    
    def calculate_for_model(
        self,
        model_name: str,
        results: dict[int, dict[str, float]],
        baseline_level: int = 0,
    ) -> list[PDRResult]:
        pdr_results = []
        
        if baseline_level not in results:
            return pdr_results
        
        baseline_scores = results[baseline_level]
        
        for level, level_scores in results.items():
            if level == baseline_level:
                continue
            
            for metric_name, perturbed_score in level_scores.items():
                baseline_score = baseline_scores.get(metric_name, 0)
                pdr = self.calculate_pdr(baseline_score, perturbed_score)
                
                pdr_results.append(PDRResult(
                    model_name=model_name,
                    metric_name=metric_name,
                    baseline_score=baseline_score,
                    perturbed_score=perturbed_score,
                    pdr_percentage=pdr,
                    perturbation_level=level,
                ))
        
        return pdr_results
    
    def aggregate_by_level(
        self,
        results: list[PDRResult],
    ) -> dict[int, float]:
        level_pdrs: dict[int, list[float]] = {}
        
        for result in results:
            level = result.perturbation_level
            if level not in level_pdrs:
                level_pdrs[level] = []
            level_pdrs[level].append(result.pdr_percentage)
        
        return {
            level: sum(pdrs) / len(pdrs) if pdrs else 0.0
            for level, pdrs in level_pdrs.items()
        }
    
    def get_robustness_score(
        self,
        pdr_results: list[PDRResult],
    ) -> float:
        """SECONDARY (shape) metric: 100 - mean PDR. Kept for diagnostics only.

        This rewards a flat score curve regardless of its height, so a model that
        is uniformly low (floor effect) scores as "robust" as one that is uniformly
        high. Do NOT rank on this alone — use get_worst_level_robustness().
        """
        if not pdr_results:
            return 100.0
        avg_pdr = sum(r.pdr_percentage for r in pdr_results) / len(pdr_results)
        return max(0, 100 - avg_pdr)

    def get_worst_level_robustness(
        self,
        avg_scores: dict[int, dict[str, float]],
        metric: str = "combined_score",
    ) -> float:
        """PRIMARY metric: Worst-Level Robustness (WLR) = min score across all levels.

        Capability-aware analogue of adversarial "robust accuracy": a model is only
        as robust as its worst perturbation level, so a model that is low everywhere
        cannot hide behind a flat curve. Reported on the [0, 1] composite scale.
        """
        vals = [lv.get(metric, 0.0) for lv in avg_scores.values()]
        return round(min(vals), 4) if vals else 0.0

    def get_mean_level(
        self,
        avg_scores: dict[int, dict[str, float]],
        metric: str = "combined_score",
    ) -> float:
        """Secondary capability metric: mean score across all levels (area under
        the perturbation curve). Reported alongside WLR."""
        vals = [lv.get(metric, 0.0) for lv in avg_scores.values()]
        return round(sum(vals) / len(vals), 4) if vals else 0.0
