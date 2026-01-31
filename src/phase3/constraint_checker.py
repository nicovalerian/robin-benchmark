import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ConstraintResult:
    constraint_type: str
    passed: bool
    expected: Any
    actual: Any
    details: str = ""


class ConstraintChecker:
    def check_keyword_constraint(
        self,
        response: str,
        keywords: list[str],
        case_sensitive: bool = False,
    ) -> ConstraintResult:
        if not case_sensitive:
            response_lower = response.lower()
            keywords_lower = [k.lower() for k in keywords]
        else:
            response_lower = response
            keywords_lower = keywords
        
        found_keywords = []
        missing_keywords = []
        
        for keyword in keywords_lower:
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, response_lower, re.IGNORECASE if not case_sensitive else 0):
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        passed = len(missing_keywords) == 0
        
        return ConstraintResult(
            constraint_type="keyword",
            passed=passed,
            expected=keywords,
            actual=found_keywords,
            details=f"Missing: {missing_keywords}" if missing_keywords else "All keywords found",
        )
    
    def check_length_constraint(
        self,
        response: str,
        min_words: int | None = None,
        max_words: int | None = None,
        min_sentences: int | None = None,
        max_sentences: int | None = None,
    ) -> ConstraintResult:
        word_count = len(response.split())
        sentence_count = len(re.findall(r"[.!?]+", response))
        
        passed = True
        details = []
        
        if min_words is not None and word_count < min_words:
            passed = False
            details.append(f"Too few words: {word_count} < {min_words}")
        
        if max_words is not None and word_count > max_words:
            passed = False
            details.append(f"Too many words: {word_count} > {max_words}")
        
        if min_sentences is not None and sentence_count < min_sentences:
            passed = False
            details.append(f"Too few sentences: {sentence_count} < {min_sentences}")
        
        if max_sentences is not None and sentence_count > max_sentences:
            passed = False
            details.append(f"Too many sentences: {sentence_count} > {max_sentences}")
        
        return ConstraintResult(
            constraint_type="length",
            passed=passed,
            expected={"min_words": min_words, "max_words": max_words},
            actual={"word_count": word_count, "sentence_count": sentence_count},
            details="; ".join(details) if details else f"Word count: {word_count}",
        )
    
    def check_all_constraints(
        self,
        response: str,
        constraints: list[dict],
    ) -> list[ConstraintResult]:
        results = []
        
        for constraint in constraints:
            ctype = constraint.get("constraint_type")
            
            if ctype == "keyword":
                keywords = constraint.get("target_value", [])
                result = self.check_keyword_constraint(response, keywords)
                results.append(result)
            
            elif ctype == "length":
                target = constraint.get("target_value", [])
                if isinstance(target, list) and len(target) >= 2:
                    min_words, max_words = target[0], target[1]
                else:
                    min_words, max_words = None, None
                result = self.check_length_constraint(
                    response, min_words=min_words, max_words=max_words
                )
                results.append(result)
        
        return results
    
    def get_pass_rate(self, results: list[ConstraintResult]) -> float:
        if not results:
            return 1.0
        passed = sum(1 for r in results if r.passed)
        return passed / len(results)
