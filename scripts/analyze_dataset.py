#!/usr/bin/env python3
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
"""
Quality analysis for ROBIN Phase 1 output.

Checks:
  - Total sample / prompt counts
  - Category distribution
  - L0==L1 exact-match rate (identity failure)
  - L2 connector/particle presence rate
  - L3 phonological respelling coverage
  - L3 ID-root+EN-suffix coverage
  - L3 clause-switch indicators
  - Constraint type distribution
  - Average instruction lengths per level
"""
import json
import re
from collections import Counter
from pathlib import Path

DATASET_PATH = Path("data/processed/robin_dataset.jsonl")
STATS_PATH = Path("data/processed/dataset_stats.json")

RESPELLED_WORDS = {
    "plis", "rili", "litereli", "meik", "rait", "cek", "greit",
    "nais", "krezi", "apdet", "oke", "wow", "hei",
}

ID_ROOT_EN_SUFFIX_PATTERN = re.compile(
    r"\b[a-z]{3,}(?:ly|ness)\b",
    re.IGNORECASE,
)

EN_CONNECTORS = {
    "which", "that", "so that", "because", "actually", "i mean",
    "basically", "like", "literally", "by the way",
}

ID_PARTICLES = {"dong", "sih", "deh", "nih", "ya", "loh", "kan", "gitu"}

COMMON_ENGLISH_CLAUSE_WORDS = {
    "the", "is", "are", "was", "were", "have", "has", "had",
    "make", "makes", "made", "that", "which", "this", "there",
    "your", "our", "their", "will", "would", "should", "could",
    "please", "ensure", "include", "provide", "write", "give",
    "show", "list", "describe", "explain", "create", "generate",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_respelling(text):
    words = {w.strip(".,!?;:\"'").lower() for w in text.split()}
    return bool(words & RESPELLED_WORDS)


def has_id_root_en_suffix(text):
    matches = ID_ROOT_EN_SUFFIX_PATTERN.findall(text)
    # Filter out pure English words
    pure_english = {
        "really", "literally", "basically", "actually", "finally",
        "generally", "specifically", "clearly", "simply", "exactly",
        "already", "directly", "easily", "quickly", "strongly",
        "properly", "likely", "only", "only", "nearly", "closely",
    }
    filtered = [m for m in matches if m.lower() not in pure_english]
    return bool(filtered), filtered


def has_en_connector(text):
    text_lower = text.lower()
    for conn in EN_CONNECTORS:
        if conn in text_lower:
            return True
    return False


def has_clause_switch(text):
    words = text.lower().split()
    en_count = sum(1 for w in words if w.strip(".,!?;:\"'") in COMMON_ENGLISH_CLAUSE_WORDS)
    return en_count >= 3


def has_particle(text):
    words = {w.strip(".,!?;:\"'").lower() for w in text.split()}
    return bool(words & ID_PARTICLES)


def word_count(text):
    return len(text.split())


def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found. Run Phase 1 first.")
        sys.exit(1)

    samples = load_jsonl(DATASET_PATH)
    n = len(samples)
    print(f"\n{'='*60}")
    print(f"ROBIN Phase 1 Quality Report")
    print(f"{'='*60}")
    print(f"Total samples:   {n}")
    print(f"Total prompts:   {n * 4}")

    # Stats file
    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            stats = json.load(f)
        print(f"Failures:        {stats.get('failures', 'N/A')}")
        print(f"Perturbation model: {stats.get('perturbation_model', 'N/A')}")

    # --- Category distribution ---
    cat_counts = Counter(s["category"] for s in samples)
    print(f"\n{'─'*40}")
    print("Category distribution:")
    targets = {"logical_reasoning": 188, "mathematical_reasoning": 150,
               "creative_writing": 150, "information_extraction": 150, "coding": 112}
    for cat, target in sorted(targets.items(), key=lambda x: -x[1]):
        got = cat_counts.get(cat, 0)
        diff = got - target
        marker = "✓" if got == target else f"Δ{diff:+d}"
        print(f"  {cat:<28} {got:>4} / {target}  {marker}")

    # --- Constraint types ---
    constraint_type_counts = Counter()
    for s in samples:
        for c in s.get("constraints", []):
            constraint_type_counts[c["constraint_type"]] += 1
    print(f"\n{'─'*40}")
    print("Constraints per type (across all samples):")
    for ct, count in sorted(constraint_type_counts.items()):
        print(f"  {ct:<20} {count}")

    # --- Per-level analysis ---
    l0_texts, l1_texts, l2_texts, l3_texts = [], [], [], []
    for s in samples:
        p = s.get("perturbations", {})
        l0_texts.append(p.get("level_0_clean", ""))
        l1_texts.append(p.get("level_1_mild", ""))
        l2_texts.append(p.get("level_2_jaksel", ""))
        l3_texts.append(p.get("level_3_adversarial", ""))

    # Average lengths
    print(f"\n{'─'*40}")
    print("Average instruction length (words):")
    for label, texts in [("L0", l0_texts), ("L1", l1_texts), ("L2", l2_texts), ("L3", l3_texts)]:
        avg = sum(word_count(t) for t in texts) / max(len(texts), 1)
        print(f"  {label}: {avg:.1f}")

    # --- L0 == L1 identity rate ---
    l0_eq_l1 = sum(1 for a, b in zip(l0_texts, l1_texts) if a.strip() == b.strip())
    pct = l0_eq_l1 / n * 100
    marker = "✓" if pct < 10 else ("⚠" if pct < 30 else "✗")
    print(f"\n{'─'*40}")
    print(f"L0 == L1 exact match:  {l0_eq_l1}/{n}  ({pct:.1f}%)  {marker}")
    if pct >= 5:
        print("  (target: < 10% — high means L1 is not injecting borrowings)")

    # --- L2 quality ---
    l2_with_connector = sum(1 for t in l2_texts if has_en_connector(t))
    l2_with_particle = sum(1 for t in l2_texts if has_particle(t))
    print(f"\n{'─'*40}")
    print("L2 Morphological Fusion quality:")
    print(f"  EN connector present:   {l2_with_connector}/{n}  ({l2_with_connector/n*100:.1f}%)")
    print(f"  ID particle present:    {l2_with_particle}/{n}  ({l2_with_particle/n*100:.1f}%)")

    # --- L3 quality ---
    l3_with_respelling = sum(1 for t in l3_texts if has_respelling(t))
    l3_with_suffix = sum(1 for t in l3_texts if has_id_root_en_suffix(t)[0])
    l3_with_clause = sum(1 for t in l3_texts if has_clause_switch(t))
    l3_with_particle = sum(1 for t in l3_texts if has_particle(t))
    print(f"\n{'─'*40}")
    print("L3 Intra-Sentential Switching quality:")
    print(f"  Phonological respelling: {l3_with_respelling}/{n}  ({l3_with_respelling/n*100:.1f}%)")
    print(f"  ID-root+EN-suffix:       {l3_with_suffix}/{n}  ({l3_with_suffix/n*100:.1f}%)")
    print(f"  Clause switch (≥3 EN):   {l3_with_clause}/{n}  ({l3_with_clause/n*100:.1f}%)")
    print(f"  ID particle present:     {l3_with_particle}/{n}  ({l3_with_particle/n*100:.1f}%)")

    all_three = sum(
        1 for t in l3_texts
        if has_respelling(t) and has_id_root_en_suffix(t)[0] and has_clause_switch(t)
    )
    print(f"  All 3 mechanisms:        {all_three}/{n}  ({all_three/n*100:.1f}%)")

    # --- Sample showcase ---
    print(f"\n{'─'*40}")
    print("Sample (first 3 entries):")
    for i, s in enumerate(samples[:3]):
        p = s.get("perturbations", {})
        print(f"\n  [{i}] {s['id']}  ({s['category']})")
        print(f"  L0: {p.get('level_0_clean','')[:120]}")
        print(f"  L1: {p.get('level_1_mild','')[:120]}")
        print(f"  L2: {p.get('level_2_jaksel','')[:120]}")
        print(f"  L3: {p.get('level_3_adversarial','')[:120]}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
