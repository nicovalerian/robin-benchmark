import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class PerturbedInstruction:
    original: str
    level_0_clean: str
    level_1_mild: str
    level_2_jaksel: str
    level_3_adversarial: str
    metadata: dict = field(default_factory=dict)


class PerturbationEngine:
    ENGLISH_REPLACEMENTS = {
        "konsep": "concept",
        "proses": "process",
        "strategi": "strategy",
        "sistem": "system",
        "analisis": "analysis",
        "metode": "method",
        "hasil": "result",
        "masalah": "problem",
        "solusi": "solution",
        "informasi": "information",
        "teknologi": "technology",
        "komunikasi": "communication",
        "organisasi": "organization",
        "implementasi": "implementation",
        "evaluasi": "evaluation",
    }
    
    JAKSEL_PATTERNS = [
        (r"\byang\b", "which is"),
        (r"\bsangat\b", "so"),
        (r"\bbenar-benar\b", "literally"),
        (r"\bseperti\b", "like"),
        (r"\btetapi\b", "but"),
        (r"\bdan\b", "and"),
        (r"\batau\b", "or"),
        (r"\buntuk\b", "for"),
        (r"\bdengan\b", "with"),
    ]
    
    JAKSEL_SUFFIXES = [
        " sih", " dong", " deh", " nih", " gitu", " kan", " lho"
    ]
    
    TYPO_CHARS = {
        "a": ["q", "s", "z", "w"],
        "e": ["w", "r", "d", "3"],
        "i": ["u", "o", "k", "8"],
        "o": ["i", "p", "l", "0"],
        "u": ["y", "i", "j", "7"],
    }
    
    def __init__(
        self,
        indocollex_path: str | Path | None = None,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.indocollex = self._load_indocollex(indocollex_path)
    
    def _load_indocollex(self, path: str | Path | None) -> dict[str, str]:
        if path is None:
            return self._get_default_slang_mappings()
        
        path = Path(path)
        if not path.exists():
            return self._get_default_slang_mappings()
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _get_default_slang_mappings(self) -> dict[str, str]:
        return {
            "sudah": "udah",
            "tidak": "nggak",
            "dengan": "sama",
            "saya": "gue",
            "kamu": "lo",
            "apa": "apaan",
            "bagaimana": "gimana",
            "mengapa": "kenapa",
            "kapan": "kapan",
            "dimana": "dimana",
            "siapa": "siapa",
            "ini": "nih",
            "itu": "tuh",
            "yang": "yg",
            "juga": "jg",
            "kalau": "kalo",
            "sekarang": "skrg",
            "memang": "emang",
            "begitu": "gitu",
            "begini": "gini",
            "seperti": "kaya",
            "tetapi": "tapi",
            "hanya": "cuma",
            "sangat": "banget",
            "banyak": "banyak",
            "sedikit": "dikit",
            "benar": "bener",
            "salah": "salah",
            "baik": "baek",
            "buruk": "jelek",
            "besar": "gede",
            "kecil": "kecil",
        }
    
    def perturb(self, instruction: str) -> PerturbedInstruction:
        level_0 = self._normalize_to_clean(instruction)
        level_1 = self._apply_mild_mixing(level_0)
        level_2 = self._apply_jaksel_mixing(level_0)
        level_3 = self._apply_adversarial_noise(level_0)
        
        return PerturbedInstruction(
            original=instruction,
            level_0_clean=level_0,
            level_1_mild=level_1,
            level_2_jaksel=level_2,
            level_3_adversarial=level_3,
            metadata={
                "engine_version": "1.0.0",
                "seed": self.rng.getstate()[1][0],
            }
        )
    
    def _normalize_to_clean(self, text: str) -> str:
        return text.strip()
    
    def _apply_mild_mixing(self, text: str, replacement_rate: float = 0.10) -> str:
        words = text.split()
        result = []
        
        for word in words:
            word_lower = word.lower().strip(".,!?;:")
            if word_lower in self.ENGLISH_REPLACEMENTS:
                if self.rng.random() < replacement_rate:
                    replacement = self.ENGLISH_REPLACEMENTS[word_lower]
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    result.append(replacement)
                    continue
            result.append(word)
        
        return " ".join(result)
    
    def _apply_jaksel_mixing(self, text: str, replacement_rate: float = 0.30) -> str:
        result = text
        
        for pattern, replacement in self.JAKSEL_PATTERNS:
            if self.rng.random() < replacement_rate:
                result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
        
        sentences = result.split(".")
        new_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and i < len(sentences) - 1:
                if self.rng.random() < 0.3:
                    suffix = self.rng.choice(self.JAKSEL_SUFFIXES)
                    sentence = sentence + suffix
            new_sentences.append(sentence)
        
        return ". ".join(new_sentences)
    
    def _apply_adversarial_noise(
        self, 
        text: str, 
        typo_rate: float = 0.05,
        slang_rate: float = 0.20,
    ) -> str:
        words = text.split()
        result = []
        
        for word in words:
            word_lower = word.lower().strip(".,!?;:")
            punctuation = ""
            if word and word[-1] in ".,!?;:":
                punctuation = word[-1]
                word = word[:-1]
            
            if word_lower in self.indocollex and self.rng.random() < slang_rate:
                replacement = self.indocollex[word_lower]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement + punctuation)
                continue
            
            if self.rng.random() < typo_rate and len(word) > 3:
                word = self._introduce_typo(word)
            
            result.append(word + punctuation)
        
        return " ".join(result)
    
    def _introduce_typo(self, word: str) -> str:
        typo_type = self.rng.choice(["swap", "delete", "replace"])
        
        if len(word) < 3:
            return word
        
        pos = self.rng.randint(1, len(word) - 2)
        
        if typo_type == "swap" and pos < len(word) - 1:
            chars = list(word)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return "".join(chars)
        
        elif typo_type == "delete":
            return word[:pos] + word[pos + 1:]
        
        elif typo_type == "replace":
            char = word[pos].lower()
            if char in self.TYPO_CHARS:
                replacement = self.rng.choice(self.TYPO_CHARS[char])
                return word[:pos] + replacement + word[pos + 1:]
        
        return word
    
    def get_all_levels(self, instruction: str) -> dict[int, str]:
        perturbed = self.perturb(instruction)
        return {
            0: perturbed.level_0_clean,
            1: perturbed.level_1_mild,
            2: perturbed.level_2_jaksel,
            3: perturbed.level_3_adversarial,
        }
