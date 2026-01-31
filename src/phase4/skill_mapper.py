from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillProfile:
    model_name: str
    category_scores: dict[str, float] = field(default_factory=dict)
    constraint_scores: dict[str, float] = field(default_factory=dict)
    level_scores: dict[int, float] = field(default_factory=dict)


class SkillMapper:
    CATEGORIES = [
        "logical_reasoning",
        "mathematical_reasoning",
        "creative_writing",
        "information_extraction",
        "coding",
    ]
    
    CONSTRAINT_TYPES = ["keyword", "length"]
    
    PERTURBATION_LEVELS = [0, 1, 2, 3]
    
    def create_profile(
        self,
        model_name: str,
        results: list[dict[str, Any]],
    ) -> SkillProfile:
        category_results: dict[str, list[float]] = {c: [] for c in self.CATEGORIES}
        constraint_results: dict[str, list[float]] = {c: [] for c in self.CONSTRAINT_TYPES}
        level_results: dict[int, list[float]] = {l: [] for l in self.PERTURBATION_LEVELS}
        
        for result in results:
            category = result.get("category", "")
            level = result.get("perturbation_level", 0)
            score = result.get("combined_score", 0.0)
            constraints_passed = result.get("constraints_passed", {})
            
            if category in category_results:
                category_results[category].append(score)
            
            if level in level_results:
                level_results[level].append(score)
            
            for ctype, passed in constraints_passed.items():
                if ctype in constraint_results:
                    constraint_results[ctype].append(1.0 if passed else 0.0)
        
        return SkillProfile(
            model_name=model_name,
            category_scores={
                cat: sum(scores) / len(scores) if scores else 0.0
                for cat, scores in category_results.items()
            },
            constraint_scores={
                ctype: sum(scores) / len(scores) if scores else 0.0
                for ctype, scores in constraint_results.items()
            },
            level_scores={
                level: sum(scores) / len(scores) if scores else 0.0
                for level, scores in level_results.items()
            },
        )
    
    def compare_profiles(
        self,
        profiles: list[SkillProfile],
    ) -> dict[str, dict[str, float]]:
        comparison = {}
        
        for profile in profiles:
            comparison[profile.model_name] = {
                **{f"cat_{k}": v for k, v in profile.category_scores.items()},
                **{f"const_{k}": v for k, v in profile.constraint_scores.items()},
                **{f"level_{k}": v for k, v in profile.level_scores.items()},
            }
        
        return comparison
    
    def get_radar_data(
        self,
        profile: SkillProfile,
    ) -> dict[str, Any]:
        labels = list(profile.category_scores.keys())
        values = list(profile.category_scores.values())
        
        return {
            "labels": labels,
            "values": values,
            "model": profile.model_name,
        }
