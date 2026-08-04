from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constraint_rules import count_whitespace_delimited_words
from phase1.constraint_injector import ConstraintInjector
from phase3.constraint_checker import ConstraintChecker


CHECKER = ConstraintChecker()


def check(response: str, constraint_type: str, target_value, regex: str = ""):
    constraint = {
        "constraint_type": constraint_type,
        "target_value": target_value,
        "verification_regex": regex,
    }
    return CHECKER.check_all(response, [constraint]).results[0]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("data", True),
        ("Data", True),
        ("data,", True),
        ("(data)", True),
        ("data.", True),
        ("database", False),
        ("metadata", False),
    ],
)
def test_keyword_exact_term_boundaries(response: str, expected: bool) -> None:
    assert check(response, "keyword", ["data"]).passed is expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [("Sudah di-review.", True), ("Sudah direview.", False)],
)
def test_keyword_hyphen_boundary(response: str, expected: bool) -> None:
    assert check(response, "keyword", ["review"]).passed is expected


def test_multiple_keyword_results_and_diagnostics_use_the_same_matches() -> None:
    passed = check("Data sudah di-review.", "keyword", ["data", "review"])
    assert passed.passed is True
    assert passed.actual == ["data", "review"]
    assert passed.details == "All keywords found"

    failed = check("database sudah di-review.", "keyword", ["data", "review"])
    assert failed.passed is False
    assert failed.actual == ["review"]
    assert failed.details == "Missing: ['data']"


def test_legacy_substring_regex_cannot_override_target_keyword_semantics() -> None:
    result = check("database", "keyword", ["data"], regex=r"(?=.*data)")
    assert result.passed is False
    assert result.actual == []
    assert result.details == "Missing: ['data']"


def test_phase1_keyword_metadata_uses_exact_term_semantics() -> None:
    constraint = ConstraintInjector()._generate_keyword_constraint(
        "coding", "data review"
    )
    assert re.search(constraint.verification_regex, "Data di-review.")
    assert not re.search(constraint.verification_regex, "database direview")


@pytest.mark.parametrize(
    ("response", "expected_count"),
    [
        ("di-review", 1),
        ("direview", 1),
        ("data,", 1),
        ("satu   dua", 2),
        ("satu\tdua\ntiga", 3),
    ],
)
def test_whitespace_delimited_word_count(response: str, expected_count: int) -> None:
    assert count_whitespace_delimited_words(response) == expected_count
    result = check(response, "length", [expected_count, expected_count])
    assert result.passed is True
    assert result.actual == expected_count


def test_length_range_boundaries_are_inclusive() -> None:
    assert check("one two", "length", [2, 3]).passed is True
    assert check("one two three", "length", [2, 3]).passed is True
    assert check("one", "length", [2, 3]).passed is False
    assert check("one two three four", "length", [2, 3]).passed is False


@pytest.mark.parametrize(
    "response",
    [
        '{"ok": true}',
        '[1, 2, 3]',
        '```json\n{"ok": true}\n```',
        '```\n[1, 2, 3]\n```',
        '  ```json\n{"ok": true}\n```  ',
    ],
)
def test_valid_json_objects_arrays_and_full_response_fences(response: str) -> None:
    assert check(response, "format", "json").passed is True


@pytest.mark.parametrize(
    "response",
    [
        "{ok: true}",
        '{"ok": true',
        'Here: {"ok": true}',
        '{"ok": true} done',
        'commentary\n```json\n{"ok": true}\n```',
        '```json\n{"ok": true}\n```\ncommentary',
        "42",
        '"text"',
    ],
)
def test_invalid_or_wrapped_json_is_rejected(response: str) -> None:
    assert check(response, "format", "json", regex=r"\{[\s\S]*\}").passed is False


@pytest.mark.parametrize(
    "response",
    ["1. Item", "1) Item", "1. First\n2. Second", "Intro\n1. Item"],
)
def test_numbered_list_markers_at_line_start_pass(response: str) -> None:
    assert check(response, "format", "numbered_list").passed is True


@pytest.mark.parametrize(
    "response",
    ["- Item", "* Item", "• Item", "- First\n- Second", "Intro\n* Item"],
)
def test_bullet_list_markers_at_line_start_pass(response: str) -> None:
    assert check(response, "format", "bullet_list").passed is True


def test_list_markers_in_prose_do_not_pass() -> None:
    assert check("Prose contains 1. an example.", "format", "numbered_list").passed is False
    assert check("A well-known fact.", "format", "bullet_list").passed is False


@pytest.mark.parametrize(
    "response",
    [
        "| Name | Value |\n|:---|---:|\n| A | 1 |",
        "Name | Value\nA | 1",
        "| Name | Value\n| A | 1",
        "Name | Value |\nA | 1 |",
    ],
)
def test_consistent_multiline_tables_pass(response: str) -> None:
    assert check(response, "format", "table").passed is True


@pytest.mark.parametrize(
    "response",
    [
        "| only one | row |",
        "| A | B |\n| 1 | 2 | 3 |",
        "Prose with an isolated | pipe.",
        "A | B\n\n1 | 2",
    ],
)
def test_malformed_or_nonconsecutive_tables_fail(response: str) -> None:
    assert check(response, "format", "table", regex=r"\|.+\|").passed is False
