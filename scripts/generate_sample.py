#!/usr/bin/env python3
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1 import TaskClassifier, ConstraintInjector, PerturbationEngine
from utils import save_jsonl, setup_logger


SAMPLE_INSTRUCTIONS = [
    {
        "instruction": "Jelaskan konsep demokrasi dalam 100 kata.",
        "category": "information_extraction",
        "output": "Demokrasi adalah sistem pemerintahan di mana kekuasaan tertinggi berada di tangan rakyat. Dalam demokrasi, warga negara memiliki hak untuk berpartisipasi dalam pengambilan keputusan politik, baik secara langsung maupun melalui perwakilan yang mereka pilih. Prinsip-prinsip utama demokrasi meliputi kebebasan berpendapat, hak pilih yang setara, perlindungan hak asasi manusia, dan supremasi hukum. Demokrasi memungkinkan pergantian kekuasaan secara damai melalui pemilihan umum yang bebas dan adil."
    },
    {
        "instruction": "Jika semua kucing adalah hewan, dan Mimi adalah kucing, apa kesimpulannya?",
        "category": "logical_reasoning",
        "output": "Berdasarkan silogisme yang diberikan, kita dapat menyimpulkan bahwa Mimi adalah hewan. Ini adalah contoh klasik dari penalaran deduktif, di mana kesimpulan logis mengikuti dari premis-premis yang diberikan. Premis mayor menyatakan bahwa semua kucing termasuk dalam kategori hewan, dan premis minor mengidentifikasi Mimi sebagai anggota kategori kucing."
    },
    {
        "instruction": "Berapakah hasil dari 15 x 8 + 32?",
        "category": "mathematical_reasoning",
        "output": "Untuk menghitung 15 x 8 + 32, kita perlu mengikuti urutan operasi matematika (PEMDAS). Pertama, kita hitung perkalian: 15 x 8 = 120. Kemudian, kita tambahkan 32: 120 + 32 = 152. Jadi, hasil dari 15 x 8 + 32 adalah 152."
    },
    {
        "instruction": "Tulis cerita pendek tentang seorang nelayan yang menemukan harta karun.",
        "category": "creative_writing",
        "output": "Pak Hasan adalah nelayan tua yang setiap hari berlayar di perairan tenang desa pesisir. Suatu pagi, jala nya tersangkut pada sesuatu di dasar laut. Dengan susah payah, ia menarik jala tersebut dan terkejut menemukan peti kayu tua berisi koin emas. Namun, alih-alih menjadi serakah, Pak Hasan membawa harta itu ke kepala desa dan menggunakannya untuk membangun sekolah bagi anak-anak nelayan."
    },
    {
        "instruction": "Buatlah fungsi Python yang menghitung faktorial dari suatu bilangan.",
        "category": "coding",
        "output": "def faktorial(n):\n    if n < 0:\n        raise ValueError('Bilangan harus non-negatif')\n    if n == 0 or n == 1:\n        return 1\n    hasil = 1\n    for i in range(2, n + 1):\n        hasil *= i\n    return hasil\n\n# Contoh penggunaan:\nprint(faktorial(5))  # Output: 120"
    },
    {
        "instruction": "Apa perbedaan antara iklim dan cuaca?",
        "category": "information_extraction",
        "output": "Cuaca adalah kondisi atmosfer pada waktu dan tempat tertentu dalam jangka pendek, seperti hujan, panas, atau berangin hari ini. Iklim adalah pola rata-rata cuaca dalam jangka panjang, biasanya 30 tahun atau lebih, di suatu wilayah. Cuaca dapat berubah setiap jam atau hari, sedangkan iklim relatif stabil selama dekade."
    },
    {
        "instruction": "Sebuah toko memberikan diskon 20%. Jika harga asli baju adalah Rp 250.000, berapa harga setelah diskon?",
        "category": "mathematical_reasoning",
        "output": "Untuk menghitung harga setelah diskon 20% dari Rp 250.000, kita perlu menghitung nilai diskonnya terlebih dahulu. Diskon = 20% x Rp 250.000 = Rp 50.000. Kemudian, harga setelah diskon = Rp 250.000 - Rp 50.000 = Rp 200.000. Jadi, harga baju setelah diskon adalah Rp 200.000."
    },
    {
        "instruction": "Tulis puisi singkat tentang matahari terbit.",
        "category": "creative_writing",
        "output": "Fajar menyingsing di ufuk timur,\nMerah jingga menghiasi cakrawala,\nBurung-burung bernyanyi riang,\nMenyambut hari yang baru.\n\nMatahari bangkit perlahan,\nMenyinari bumi dengan hangatnya,\nHarapan baru terlahir kembali,\nDalam pelukan cahaya pagi."
    },
    {
        "instruction": "Jika hujan, maka jalanan basah. Jalanan kering. Apa yang bisa disimpulkan tentang hujan?",
        "category": "logical_reasoning",
        "output": "Berdasarkan logika modus tollens, jika premis 'jika hujan maka jalanan basah' benar, dan kita mengamati bahwa 'jalanan kering' (tidak basah), maka kita dapat menyimpulkan bahwa 'tidak hujan'. Ini karena jika hujan turun, jalanan seharusnya basah. Karena jalanan kering, berarti kondisi yang menyebabkan basah (hujan) tidak terjadi."
    },
    {
        "instruction": "Buatlah fungsi JavaScript untuk membalik urutan array.",
        "category": "coding",
        "output": "function balikArray(arr) {\n    const hasil = [];\n    for (let i = arr.length - 1; i >= 0; i--) {\n        hasil.push(arr[i]);\n    }\n    return hasil;\n}\n\n// Contoh penggunaan:\nconst angka = [1, 2, 3, 4, 5];\nconsole.log(balikArray(angka)); // Output: [5, 4, 3, 2, 1]"
    },
]


def generate_sample_dataset(target_size: int = 100) -> list[dict]:
    classifier = TaskClassifier()
    injector = ConstraintInjector(seed=42)
    perturbation_engine = PerturbationEngine(
        indocollex_path=Path(__file__).parent.parent / "data" / "raw" / "indocollex.json",
        seed=42,
    )
    
    samples = []
    sample_idx = 0
    
    while len(samples) < target_size:
        base = random.choice(SAMPLE_INSTRUCTIONS)
        
        instruction = base["instruction"]
        category = base["category"]
        gold_response = base["output"]
        
        variation = random.randint(1, 5)
        if variation > 1:
            instruction = f"{instruction} (Variasi {variation})"
        
        constrained = injector.inject_constraints(
            instruction=instruction,
            category=category,
            gold_response=gold_response,
        )
        
        perturbed = perturbation_engine.perturb(constrained.constrained_instruction)
        
        sample = {
            "id": f"robin_{sample_idx:05d}",
            "category": category,
            "original_instruction": instruction,
            "original_input": "",
            "gold_response": gold_response,
            "constraints": [
                {
                    "constraint_type": c.constraint_type,
                    "requirement": c.requirement,
                    "verification_regex": c.verification_regex,
                    "target_value": c.target_value,
                }
                for c in constrained.constraints
            ],
            "perturbations": {
                "level_0_clean": perturbed.level_0_clean,
                "level_1_mild": perturbed.level_1_mild,
                "level_2_jaksel": perturbed.level_2_jaksel,
                "level_3_adversarial": perturbed.level_3_adversarial,
            },
        }
        
        samples.append(sample)
        sample_idx += 1
    
    return samples


def main():
    logger = setup_logger("robin-sample")
    logger.info("Generating sample ROBIN dataset (100 samples)")
    
    random.seed(42)
    samples = generate_sample_dataset(100)
    
    output_path = Path(__file__).parent.parent / "data" / "processed" / "robin_sample_100.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(samples, output_path)
    
    logger.info(f"Generated {len(samples)} samples")
    logger.info(f"Saved to: {output_path}")
    
    category_counts = {}
    for s in samples:
        cat = s["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    logger.info(f"Category distribution: {category_counts}")
    
    logger.info("\n--- Sample Entry ---")
    sample = samples[0]
    logger.info(f"ID: {sample['id']}")
    logger.info(f"Category: {sample['category']}")
    logger.info(f"Level 0 (Clean): {sample['perturbations']['level_0_clean'][:100]}...")
    logger.info(f"Level 2 (Jaksel): {sample['perturbations']['level_2_jaksel'][:100]}...")
    logger.info(f"Level 3 (Adversarial): {sample['perturbations']['level_3_adversarial'][:100]}...")


if __name__ == "__main__":
    main()
