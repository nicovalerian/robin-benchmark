#!/usr/bin/env python3
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
"""
Pre-publication validation / QA for the ROBIN Phase 1 dataset.

Run this on data/processed/robin_dataset.jsonl BEFORE committing the canonical
dataset to GitHub or before kicking off Phase 2 (so a bad row doesn't cost hours
of inference). It is read-only — it never modifies the dataset.

Checks (ERROR = blocks publishing, WARN = inspect but not fatal):
  ERROR  required fields present + correct types
  ERROR  ids unique, contiguous, well-formed (robin_00000..N-1)
  ERROR  original_instruction unique (no duplicate base items)
  ERROR  all four perturbation levels present and non-empty
  ERROR  every constraint requirement appears verbatim in each of L0-L3
  ERROR  each requirement appears exactly once per level (no leaked dup)
  ERROR  length constraint range sane (0 < min < max)
  ERROR  format constraint is one of the known formats
  ERROR  category is one of the known categories
  WARN   keyword constraint word(s) actually present in gold_response
  WARN   L0 == L1 exact-match rate (high => L1 not borrowing)
  WARN   category counts vs target distribution

Exit code 0 if no ERROR-level issues, 1 otherwise.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --input data/processed/robin_dataset.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {
    "id": str,
    "category": str,
    "original_instruction": str,
    "gold_response": str,
    "constraints": list,
    "perturbations": dict,
}

LEVEL_KEYS = ["level_0_clean", "level_1_mild", "level_2_jaksel", "level_3_adversarial"]

KNOWN_CATEGORIES = {
    "logical_reasoning", "mathematical_reasoning", "creative_writing",
    "information_extraction", "coding",
}

TARGET_DISTRIBUTION = {
    "logical_reasoning": 188, "mathematical_reasoning": 150,
    "creative_writing": 150, "information_extraction": 150, "coding": 112,
}

KNOWN_FORMAT_REQUIREMENTS = {
    "Jawab dalam format JSON.",
    "Gunakan format daftar bernomor.",
    "Gunakan format poin-poin.",
    "Tampilkan jawaban dalam format tabel.",
}

ID_RE = re.compile(r"^robin_\d{5}$")
# Pull "kata: a, b." out of the keyword requirement line.
KEYWORD_REQ_RE = re.compile(r"mengandung kata:\s*(.+?)\.?$", re.IGNORECASE)


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warns: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"FATAL: line {lineno} is not valid JSON: {exc}")
    return rows


def validate_row(row: dict, rep: Report) -> None:
    rid = row.get("id", "<no-id>")

    # --- schema ---
    for field, typ in REQUIRED_FIELDS.items():
        if field not in row:
            rep.error(f"{rid}: missing field '{field}'")
        elif not isinstance(row[field], typ):
            rep.error(f"{rid}: field '{field}' should be {typ.__name__}, got {type(row[field]).__name__}")

    if row.get("category") not in KNOWN_CATEGORIES:
        rep.error(f"{rid}: unknown category {row.get('category')!r}")

    perturbations = row.get("perturbations", {})
    constraints = row.get("constraints", [])

    # --- levels present + non-empty ---
    for key in LEVEL_KEYS:
        text = perturbations.get(key, "")
        if not isinstance(text, str) or not text.strip():
            rep.error(f"{rid}: perturbation '{key}' missing or empty")

    # --- constraints embedded verbatim, exactly once, in every level ---
    requirements = [c.get("requirement", "") for c in constraints]
    for key in LEVEL_KEYS:
        text = perturbations.get(key, "") or ""
        for req in requirements:
            if not req:
                continue
            count = text.count(req)
            if count == 0:
                rep.error(f"{rid}/{key}: constraint requirement not found verbatim: {req!r}")
            elif count > 1:
                rep.error(f"{rid}/{key}: constraint requirement appears {count}x (leaked dup): {req!r}")

    # --- per-constraint sanity ---
    seen_types = set()
    for c in constraints:
        ctype = c.get("constraint_type")
        seen_types.add(ctype)
        req = c.get("requirement", "")

        if ctype == "length":
            tv = c.get("target_value")
            lo, hi = _length_range(tv, req)
            if lo is None or hi is None:
                rep.error(f"{rid}: length constraint has no parseable range ({tv!r} / {req!r})")
            elif not (0 < lo < hi):
                rep.error(f"{rid}: length range not sane: {lo}-{hi}")

        elif ctype == "format":
            if req not in KNOWN_FORMAT_REQUIREMENTS:
                rep.error(f"{rid}: unknown format requirement: {req!r}")

        elif ctype == "keyword":
            tv = c.get("target_value")
            kws = tv if isinstance(tv, list) else _keyword_terms(req)
            gold = (row.get("gold_response", "") or "").lower()
            missing = [k for k in kws if k.lower() not in gold]
            if missing:
                rep.warn(f"{rid}: keyword(s) {missing} not in gold_response (model may not satisfy own constraint)")

    for needed in ("keyword", "length", "format"):
        if needed not in seen_types:
            rep.error(f"{rid}: missing '{needed}' constraint (has: {sorted(seen_types)})")


def _length_range(target_value, requirement):
    # target_value may be [min, max] or {"min":..,"max":..}; fall back to the text.
    if isinstance(target_value, (list, tuple)) and len(target_value) == 2:
        return target_value[0], target_value[1]
    if isinstance(target_value, dict) and "min" in target_value and "max" in target_value:
        return target_value["min"], target_value["max"]
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", requirement or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _keyword_terms(requirement):
    m = KEYWORD_REQ_RE.search(requirement or "")
    if not m:
        return []
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def main():
    parser = argparse.ArgumentParser(description="Validate the ROBIN Phase 1 dataset before publishing")
    parser.add_argument("--input", type=str, default="data/processed/robin_dataset.jsonl")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"FATAL: {path} not found. Run Phase 1 first.")

    rows = load_jsonl(path)
    rep = Report()

    print(f"\n{'='*64}")
    print("ROBIN Dataset Validation")
    print(f"{'='*64}")
    print(f"File:    {path}")
    print(f"Rows:    {len(rows)}  ({len(rows) * 4} prompts)")

    # --- global: ids ---
    ids = [r.get("id", "") for r in rows]
    for rid in ids:
        if not ID_RE.match(rid):
            rep.error(f"malformed id: {rid!r}")
    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    if dup_ids:
        rep.error(f"duplicate ids: {dup_ids[:10]}{' ...' if len(dup_ids) > 10 else ''}")
    expected = [f"robin_{i:05d}" for i in range(len(rows))]
    if sorted(ids) != expected:
        rep.error("ids are not contiguous robin_00000..N-1 (run finalize / regenerate)")

    # --- global: unique base instructions ---
    instr = [r.get("original_instruction", "") for r in rows]
    dup_instr = [i for i, c in Counter(instr).items() if c > 1]
    if dup_instr:
        rep.error(f"{len(dup_instr)} duplicate original_instruction value(s)")

    # --- per-row ---
    for row in rows:
        validate_row(row, rep)

    # --- WARN: L0==L1 rate ---
    l0_eq_l1 = sum(
        1 for r in rows
        if (r.get("perturbations", {}).get("level_0_clean", "") or "").strip()
        == (r.get("perturbations", {}).get("level_1_mild", "") or "").strip()
    )
    if rows:
        pct = l0_eq_l1 / len(rows) * 100
        if pct >= 30:
            rep.warn(f"L0==L1 exact-match rate is {pct:.1f}% (>=30%: many items have no borrowable noun)")

    # --- WARN: category distribution ---
    cat_counts = Counter(r.get("category") for r in rows)
    print(f"\n{'-'*40}\nCategory distribution:")
    for cat, target in sorted(TARGET_DISTRIBUTION.items(), key=lambda x: -x[1]):
        got = cat_counts.get(cat, 0)
        mark = "ok" if got == target else f"d{got - target:+d}"
        print(f"  {cat:<26} {got:>4} / {target}  {mark}")

    # --- output ---
    print(f"\n{'-'*40}")
    if rep.warns:
        print(f"WARNINGS ({len(rep.warns)}):")
        for w in rep.warns[:40]:
            print(f"  WARN  {w}")
        if len(rep.warns) > 40:
            print(f"  ... {len(rep.warns) - 40} more")
    if rep.errors:
        print(f"\nERRORS ({len(rep.errors)}):")
        for e in rep.errors[:60]:
            print(f"  ERROR {e}")
        if len(rep.errors) > 60:
            print(f"  ... {len(rep.errors) - 60} more")

    print(f"\n{'='*64}")
    if rep.errors:
        print(f"FAILED: {len(rep.errors)} error(s), {len(rep.warns)} warning(s). Do NOT publish.")
        print(f"{'='*64}\n")
        sys.exit(1)
    print(f"PASSED: 0 errors, {len(rep.warns)} warning(s). Safe to publish.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
