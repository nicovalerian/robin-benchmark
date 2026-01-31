import re
from dataclasses import dataclass


@dataclass
class TaskCategory:
    name: str
    weight: float
    keywords: list[str]
    

class TaskClassifier:
    CATEGORIES = {
        "logical_reasoning": TaskCategory(
            name="logical_reasoning",
            weight=0.25,
            keywords=[
                "logika", "deduksi", "silogisme", "kesimpulan", "premis",
                "argumen", "valid", "inferensi", "reasoning", "logic",
                "jika", "maka", "karena", "sebab", "akibat"
            ]
        ),
        "mathematical_reasoning": TaskCategory(
            name="mathematical_reasoning", 
            weight=0.20,
            keywords=[
                "hitung", "matematika", "rumus", "angka", "persamaan",
                "jumlah", "kurang", "kali", "bagi", "persen",
                "berapa", "total", "rata-rata", "statistik"
            ]
        ),
        "creative_writing": TaskCategory(
            name="creative_writing",
            weight=0.20,
            keywords=[
                "cerita", "puisi", "narasi", "kreatif", "tulis",
                "dongeng", "fiksi", "karakter", "plot", "dialog",
                "deskripsi", "imajinatif", "kisah"
            ]
        ),
        "information_extraction": TaskCategory(
            name="information_extraction",
            weight=0.20,
            keywords=[
                "rangkum", "ekstrak", "informasi", "fakta", "data",
                "ringkasan", "poin", "utama", "sebutkan", "daftar",
                "identifikasi", "temukan", "cari"
            ]
        ),
        "coding": TaskCategory(
            name="coding",
            weight=0.15,
            keywords=[
                "kode", "program", "fungsi", "algoritma", "python",
                "javascript", "code", "script", "debug", "error",
                "variabel", "loop", "array", "class"
            ]
        ),
    }
    
    def __init__(self, custom_categories: dict | None = None):
        self.categories = self.CATEGORIES.copy()
        if custom_categories:
            for name, config in custom_categories.items():
                self.categories[name] = TaskCategory(
                    name=name,
                    weight=config.get("weight", 0.2),
                    keywords=config.get("keywords", [])
                )
    
    def classify(self, instruction: str, input_text: str = "") -> str:
        combined_text = f"{instruction} {input_text}".lower()
        scores = {}
        
        for category_name, category in self.categories.items():
            score = sum(
                1 for keyword in category.keywords 
                if re.search(rf"\b{re.escape(keyword)}\b", combined_text)
            )
            scores[category_name] = score
        
        if max(scores.values()) == 0:
            return self._classify_by_heuristics(combined_text)
        
        best_category = "logical_reasoning"
        best_score = 0
        for cat, score in scores.items():
            if score > best_score:
                best_score = score
                best_category = cat
        return best_category
    
    def _classify_by_heuristics(self, text: str) -> str:
        if re.search(r"\d+\s*[\+\-\*\/\=]", text):
            return "mathematical_reasoning"
        if re.search(r"(def |function |class |import |print\()", text):
            return "coding"
        if re.search(r"(ceritakan|tuliskan|buatkan cerita)", text):
            return "creative_writing"
        if re.search(r"(jelaskan|apa itu|mengapa)", text):
            return "information_extraction"
        return "logical_reasoning"
    
    def get_distribution_weights(self) -> dict[str, float]:
        return {name: cat.weight for name, cat in self.categories.items()}
