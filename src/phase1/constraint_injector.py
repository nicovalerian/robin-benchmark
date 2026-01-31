import random
import re
from dataclasses import dataclass, field
from typing import Literal, Any


@dataclass
class Constraint:
    constraint_type: Literal["keyword", "length"]
    requirement: str
    verification_regex: str | None = None
    target_value: Any = None


@dataclass
class ConstrainedInstruction:
    original_instruction: str
    constrained_instruction: str
    constraints: list[Constraint] = field(default_factory=list)
    gold_response: str = ""
    category: str = ""


class ConstraintInjector:
    LENGTH_TEMPLATES = {
        "word_count_exact": "Jawab dalam tepat {n} kata.",
        "word_count_range": "Jawab dalam {min}-{max} kata.",
        "sentence_count": "Gunakan maksimal {n} kalimat.",
        "paragraph_count": "Tulis dalam {n} paragraf.",
    }
    
    KEYWORD_TEMPLATES = {
        "must_include": "Pastikan jawaban mengandung kata: {keywords}.",
        "must_include_all": "Jawaban harus mengandung semua kata berikut: {keywords}.",
    }
    
    CATEGORY_KEYWORDS = {
        "logical_reasoning": [
            "kesimpulan", "premis", "argumen", "valid", "logis",
            "sebab", "akibat", "inferensi", "deduksi", "induksi"
        ],
        "mathematical_reasoning": [
            "hasil", "total", "jumlah", "persentase", "rata-rata",
            "rumus", "perhitungan", "nilai", "angka", "solusi"
        ],
        "creative_writing": [
            "karakter", "setting", "plot", "konflik", "klimaks",
            "narasi", "dialog", "deskripsi", "suasana", "emosi"
        ],
        "information_extraction": [
            "poin", "utama", "ringkasan", "fakta", "data",
            "informasi", "daftar", "aspek", "elemen", "komponen"
        ],
        "coding": [
            "fungsi", "variabel", "output", "input", "return",
            "parameter", "algoritma", "error", "debug", "test"
        ],
    }

    def __init__(
        self,
        constraint_types: list[str] | None = None,
        seed: int = 42,
    ):
        self.constraint_types = constraint_types or ["keyword", "length"]
        self.rng = random.Random(seed)
    
    def inject_constraints(
        self,
        instruction: str,
        category: str,
        gold_response: str = "",
    ) -> ConstrainedInstruction:
        constraints = []
        constraint_texts = []
        
        if "keyword" in self.constraint_types:
            keyword_constraint = self._generate_keyword_constraint(
                category, gold_response
            )
            constraints.append(keyword_constraint)
            constraint_texts.append(keyword_constraint.requirement)
        
        if "length" in self.constraint_types:
            length_constraint = self._generate_length_constraint(gold_response)
            constraints.append(length_constraint)
            constraint_texts.append(length_constraint.requirement)
        
        constraint_suffix = " ".join(constraint_texts)
        constrained_instruction = f"{instruction.rstrip('.')}. {constraint_suffix}"
        
        return ConstrainedInstruction(
            original_instruction=instruction,
            constrained_instruction=constrained_instruction,
            constraints=constraints,
            gold_response=gold_response,
            category=category,
        )
    
    def _generate_keyword_constraint(
        self, 
        category: str,
        gold_response: str,
    ) -> Constraint:
        category_words = self.CATEGORY_KEYWORDS.get(category, [])
        
        keywords_to_use = []
        if gold_response and self.rng.random() > 0.5:
            response_words = re.findall(r"\b[a-zA-Z]{4,}\b", gold_response.lower())
            if response_words:
                keywords_to_use = self.rng.sample(
                    response_words, 
                    min(2, len(response_words))
                )
        
        if not keywords_to_use and category_words:
            keywords_to_use = self.rng.sample(
                category_words,
                min(2, len(category_words))
            )
        
        if not keywords_to_use:
            keywords_to_use = ["hasil", "jawaban"]
        
        keywords_str = ", ".join(keywords_to_use)
        requirement = self.KEYWORD_TEMPLATES["must_include"].format(
            keywords=keywords_str
        )
        
        regex_pattern = "(?=.*" + ")(?=.*".join(
            re.escape(kw) for kw in keywords_to_use
        ) + ")"
        
        return Constraint(
            constraint_type="keyword",
            requirement=requirement,
            verification_regex=regex_pattern,
            target_value=keywords_to_use,
        )
    
    def _generate_length_constraint(
        self,
        gold_response: str,
    ) -> Constraint:
        if gold_response:
            word_count = len(gold_response.split())
            target_min = max(20, int(word_count * 0.7))
            target_max = int(word_count * 1.3)
        else:
            target_min = self.rng.choice([30, 50, 75])
            target_max = target_min + self.rng.choice([20, 30, 50])
        
        requirement = self.LENGTH_TEMPLATES["word_count_range"].format(
            min=target_min, max=target_max
        )
        
        return Constraint(
            constraint_type="length",
            requirement=requirement,
            verification_regex=None,
            target_value=[target_min, target_max],
        )
