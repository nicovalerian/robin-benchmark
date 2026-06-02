"""
ROBIN Phase 3 — Stream A: Rule-based constraint checker.

Each ROBIN sample carries three constraints stored as a list in the dataset:
  1. keyword  — pre-computed lookahead regex; match is case-insensitive
  2. length   — word count must fall within [min, max] from target_value
  3. format   — pre-computed regex; list types need MULTILINE, JSON tries parse first

All checks produce a binary pass/fail ConstraintResult.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConstraintResult:
    constraint_type: str
    passed: bool
    expected: Any
    actual: Any
    details: str = ""


@dataclass
class AllConstraintResults:
    results: list[ConstraintResult] = field(default_factory=list)

    @property
    def cpr(self) -> float:
        if not self.results:
            return 1.0
        return sum(r.passed for r in self.results) / len(self.results)


class ConstraintChecker:
    """
    Evaluates a model response against the three ROBIN constraint types.
    Uses pre-computed verification_regex from the Phase 1 dataset where available.
    """

    def check_all(self, response: str, constraints: list[dict]) -> AllConstraintResults:
        return AllConstraintResults([self._check_one(response, c) for c in constraints])

    def _check_one(self, response: str, constraint: dict) -> ConstraintResult:
        ct = constraint["constraint_type"]
        if ct == "keyword":
            return self._keyword(response, constraint)
        elif ct == "length":
            return self._length(response, constraint)
        elif ct == "format":
            return self._format(response, constraint)
        return ConstraintResult(ct, False, None, None, f"Unknown type: {ct}")

    # ------------------------------------------------------------------
    # Keyword
    # ------------------------------------------------------------------

    def _keyword(self, response: str, constraint: dict) -> ConstraintResult:
        regex = constraint.get("verification_regex", "")
        words = constraint.get("target_value", [])

        # Use the pre-computed lookahead regex; add IGNORECASE since
        # models may capitalise keywords at sentence start.
        passed = bool(re.search(regex, response, re.IGNORECASE | re.DOTALL))

        missing = [w for w in words if not re.search(re.escape(w), response, re.IGNORECASE)]
        found = [w for w in words if w not in missing]

        return ConstraintResult(
            constraint_type="keyword",
            passed=passed,
            expected=words,
            actual=found,
            details="All keywords found" if passed else f"Missing: {missing}",
        )

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def _length(self, response: str, constraint: dict) -> ConstraintResult:
        min_w, max_w = constraint["target_value"]
        word_count = len(response.split())
        passed = min_w <= word_count <= max_w

        return ConstraintResult(
            constraint_type="length",
            passed=passed,
            expected=[min_w, max_w],
            actual=word_count,
            details=f"{word_count} words (range {min_w}-{max_w})",
        )

    # ------------------------------------------------------------------
    # Format
    # ------------------------------------------------------------------

    def _format(self, response: str, constraint: dict) -> ConstraintResult:
        fmt = constraint["target_value"]
        regex = constraint.get("verification_regex", "")

        if fmt == "json":
            passed = self._is_json(response) or bool(re.search(regex, response, re.DOTALL))
        elif fmt in ("numbered_list", "bullet_list"):
            # MULTILINE makes ^ match the start of every line, not just
            # the start of the whole string — required for list detection.
            passed = bool(re.search(regex, response, re.MULTILINE))
        elif fmt == "table":
            passed = bool(re.search(regex, response))
        else:
            passed = bool(re.search(regex, response)) if regex else False

        return ConstraintResult(
            constraint_type="format",
            passed=passed,
            expected=fmt,
            actual="detected" if passed else "not_detected",
            details=f"Format '{fmt}': {'pass' if passed else 'fail'}",
        )

    @staticmethod
    def _is_json(text: str) -> bool:
        text = text.strip()
        # Strip markdown code fences if present
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
        try:
            json.loads(text)
            return True
        except Exception:
            return False
