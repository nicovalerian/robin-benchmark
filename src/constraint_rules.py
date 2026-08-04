"""Shared operational rules for ROBIN constraint generation and checking."""

from __future__ import annotations

import re
from collections.abc import Sequence


def exact_term_pattern(term: str) -> str:
    """Return a regex that matches *term* outside longer Unicode words."""
    return rf"(?<!\w){re.escape(term)}(?!\w)"


def build_keyword_verification_regex(keywords: Sequence[str]) -> str:
    """Build compatible metadata using the same exact-term keyword semantics."""
    lookaheads = "".join(
        rf"(?=.*{exact_term_pattern(keyword)})" for keyword in keywords
    )
    return rf"(?is){lookaheads}"


def match_required_keywords(
    text: str,
    keywords: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return found and missing keywords using case-insensitive exact terms."""
    found: list[str] = []
    missing: list[str] = []
    for keyword in keywords:
        destination = (
            found
            if re.search(exact_term_pattern(keyword), text, re.IGNORECASE)
            else missing
        )
        destination.append(keyword)
    return found, missing


def count_whitespace_delimited_words(text: str) -> int:
    """Count non-empty spans separated by whitespace, equivalent to split()."""
    return len(text.split())
