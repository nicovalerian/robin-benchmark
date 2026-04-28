import random
import re
from dataclasses import dataclass, field
from typing import Literal, Any


@dataclass
class Constraint:
    constraint_type: Literal["keyword", "length", "format"]
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
        "word_count_range": "Jawab dalam {min}-{max} kata.",
    }

    KEYWORD_TEMPLATES = {
        "must_include": "Pastikan jawaban mengandung kata: {keywords}.",
    }

    # Format constraints with regex patterns to verify them.
    # Keyed by format_id: (requirement_text, verification_regex)
    FORMAT_OPTIONS = {
        "json":           ("Jawab dalam format JSON.",
                           r"\{[\s\S]*\}|\[[\s\S]*\]"),
        "numbered_list":  ("Gunakan format daftar bernomor.",
                           r"^\s*\d+[\.\)]"),
        "bullet_list":    ("Gunakan format poin-poin.",
                           r"^\s*[-•*]"),
        "table":          ("Tampilkan jawaban dalam format tabel.",
                           r"\|.+\|"),
    }

    # Preferred format types per category — ordered by fit.
    CATEGORY_FORMAT_PREFS = {
        "coding":                 ["json", "numbered_list"],
        "mathematical_reasoning": ["json", "numbered_list"],
        "information_extraction": ["table", "numbered_list", "bullet_list"],
        "logical_reasoning":      ["numbered_list", "bullet_list"],
        "creative_writing":       ["bullet_list", "numbered_list"],
    }

    # Used only as last-resort fallback when gold response yields nothing usable.
    CATEGORY_KEYWORDS = {
        "logical_reasoning":      ["kesimpulan", "sebab", "akibat", "valid", "logis",
                                   "inferensi", "premis", "argumen"],
        "mathematical_reasoning": ["hasil", "total", "jumlah", "persentase", "rata-rata",
                                   "rumus", "perhitungan", "nilai", "solusi"],
        "creative_writing":       ["karakter", "plot", "konflik", "narasi",
                                   "dialog", "deskripsi", "suasana", "emosi"],
        "information_extraction": ["ringkasan", "fakta", "informasi", "aspek",
                                   "elemen", "komponen"],
        "coding":                 ["fungsi", "variabel", "output", "input",
                                   "parameter", "algoritma", "error"],
    }

    # Common Indonesian function words that carry no constraint signal.
    _STOPWORDS = {
        # Pronouns & determiners
        "yang", "ini", "itu", "kami", "kita", "mereka", "anda", "saya",
        "kamu", "dia", "sebuah", "suatu", "setiap", "para",
        # Conjunctions & discourse
        "dan", "atau", "serta", "pula", "tetapi", "tapi", "namun", "agar",
        "maka", "jika", "bila", "karena", "bahwa", "supaya",
        # Prepositions
        "pada", "dari", "untuk", "dengan", "dalam", "oleh", "ke", "kepada",
        "bagi", "tentang", "antara",
        # Auxiliaries & common verbs
        "adalah", "ada", "akan", "tidak", "juga", "saja", "bisa", "sudah",
        "dapat", "lebih", "harus", "perlu", "boleh", "telah", "sedang",
        # High-frequency nouns that carry no constraint signal
        "berikut", "tersebut", "diberikan", "contoh", "cara", "jenis",
        "tipe", "lain", "sama", "baru", "satu", "dua", "tiga",
    }

    def __init__(
        self,
        constraint_types: list[str] | None = None,
        seed: int = 42,
    ):
        self.constraint_types = constraint_types or ["keyword", "length", "format"]
        self.rng = random.Random(seed)

    def inject_constraints(
        self,
        instruction: str,
        category: str,
        gold_response: str = "",
    ) -> ConstrainedInstruction:
        constraints = []

        if "keyword" in self.constraint_types:
            constraints.append(self._generate_keyword_constraint(category, gold_response))

        if "length" in self.constraint_types:
            constraints.append(self._generate_length_constraint(gold_response))

        if "format" in self.constraint_types:
            constraints.append(self._generate_format_constraint(category))

        # Constraints are evaluation metadata only — NOT embedded in the prompt.
        # Phase 3 checks model responses against these constraints.
        return ConstrainedInstruction(
            original_instruction=instruction,
            constrained_instruction=instruction,
            constraints=constraints,
            gold_response=gold_response,
            category=category,
        )

    def _generate_keyword_constraint(
        self,
        category: str,
        gold_response: str,
    ) -> Constraint:
        keywords_to_use = []

        # Primary source: meaningful words from the gold response.
        # This guarantees the constraint is achievable by a correct answer.
        if gold_response:
            candidates = re.findall(r"\b[a-zA-Z]{4,}\b", gold_response.lower())
            candidates = [w for w in candidates if w not in self._STOPWORDS]
            candidates = list(dict.fromkeys(candidates))  # deduplicate, preserve order
            if len(candidates) >= 2:
                keywords_to_use = self.rng.sample(candidates, 2)
            elif candidates:
                keywords_to_use = candidates

        # Fallback: category vocab, only when gold gave us nothing usable.
        if not keywords_to_use:
            pool = self.CATEGORY_KEYWORDS.get(category, [])
            if pool:
                keywords_to_use = self.rng.sample(pool, min(2, len(pool)))

        if not keywords_to_use:
            keywords_to_use = ["hasil", "jawaban"]

        keywords_str = ", ".join(keywords_to_use)
        requirement = self.KEYWORD_TEMPLATES["must_include"].format(keywords=keywords_str)
        regex_pattern = "(?=.*" + ")(?=.*".join(re.escape(kw) for kw in keywords_to_use) + ")"

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
            if word_count <= 10:
                target_min = max(1, int(word_count * 0.8))
                target_max = max(target_min + 10, word_count * 3)
            else:
                target_min = max(10, int(word_count * 0.7))
                target_max = max(target_min + 10, int(word_count * 1.3))
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

    def _generate_format_constraint(self, category: str) -> Constraint:
        prefs = self.CATEGORY_FORMAT_PREFS.get(category, list(self.FORMAT_OPTIONS.keys()))
        fmt_id = self.rng.choice(prefs)
        requirement, regex = self.FORMAT_OPTIONS[fmt_id]

        return Constraint(
            constraint_type="format",
            requirement=requirement,
            verification_regex=regex,
            target_value=fmt_id,
        )
