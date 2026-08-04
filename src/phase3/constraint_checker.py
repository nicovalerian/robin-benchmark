"""
ROBIN Phase 3 — Stream A: Rule-based constraint checker.

Each ROBIN sample carries three constraints stored as a list in the dataset:
  1. keyword  — every target_value term must match outside longer words
  2. length   — whitespace-delimited word count must fall within [min, max]
  3. format   — explicit JSON, list-presence, and table-presence heuristics

All checks produce a binary pass/fail ConstraintResult.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from constraint_rules import count_whitespace_delimited_words, match_required_keywords


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

    Known constraints use target_value and explicit operational rules. Serialized
    verification_regex values remain compatible metadata and a fallback only for
    unknown format identifiers.
    """

    _NUMBERED_LIST_MARKER = re.compile(r"^\s*\d+[.)](?:\s+|$)", re.MULTILINE)
    _BULLET_LIST_MARKER = re.compile(r"^\s*[-*•](?:\s+|$)", re.MULTILINE)
    _FULL_JSON_FENCE = re.compile(
        r"```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```",
        re.IGNORECASE,
    )

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
        words = constraint.get("target_value", [])
        found, missing = match_required_keywords(response, words)
        passed = not missing

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
        word_count = count_whitespace_delimited_words(response)
        passed = min_w <= word_count <= max_w

        return ConstraintResult(
            constraint_type="length",
            passed=passed,
            expected=[min_w, max_w],
            actual=word_count,
            details=(
                f"{word_count} whitespace-delimited words "
                f"(range {min_w}-{max_w})"
            ),
        )

    # ------------------------------------------------------------------
    # Format
    # ------------------------------------------------------------------

    def _format(self, response: str, constraint: dict) -> ConstraintResult:
        fmt = constraint["target_value"]
        regex = constraint.get("verification_regex", "")

        if fmt == "json":
            passed = self._is_json(response)
        elif fmt == "numbered_list":
            passed = bool(self._NUMBERED_LIST_MARKER.search(response))
        elif fmt == "bullet_list":
            passed = bool(self._BULLET_LIST_MARKER.search(response))
        elif fmt == "table":
            passed = self._has_markdown_table(response)
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
    def _is_json_container(text: str) -> bool:
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(value, (dict, list))

    @classmethod
    def _is_json(cls, text: str) -> bool:
        """Accept only a raw or full-response fenced JSON object/array."""
        stripped = text.strip()
        fence = cls._FULL_JSON_FENCE.fullmatch(stripped)
        if fence:
            stripped = fence.group("body").strip()
        elif "```" in stripped:
            return False
        return cls._is_json_container(stripped)

    @staticmethod
    def _table_column_count(line: str) -> int | None:
        """Return a pipe-delimited row's cell count, or None for non-rows."""
        row = line.strip()
        if "|" not in row:
            return None
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        cells = row.split("|")
        return len(cells) if len(cells) >= 2 else None

    @classmethod
    def _has_markdown_table(cls, text: str) -> bool:
        """Detect two or more consecutive rows with a stable column count."""
        block: list[int] = []
        for line in [*text.splitlines(), ""]:
            column_count = cls._table_column_count(line)
            if column_count is not None:
                block.append(column_count)
                continue
            if len(block) >= 2 and len(set(block)) == 1:
                return True
            block = []
        return False
