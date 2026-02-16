"""Answer extraction and normalization utilities."""

from __future__ import annotations

import re
from fractions import Fraction


# ---------------------------------------------------------------------------
# Multiple-choice answer extraction
# ---------------------------------------------------------------------------

_MCQ_PATTERNS: list[re.Pattern[str]] = [
    # "Answer: A" or "Answer: $A$"
    re.compile(r"(?i)Answer\s*:\s*\$?([A-D])\$?"),
    # "The correct answer is (A)" / "I choose B"
    re.compile(r"(?i)(?:correct answer|choose|select)\s+(?:is\s+)?(?:\()?([A-D])(?:\))?"),
    # "The answer is A"
    re.compile(r"(?i)the\s+answer\s+is\s+\(?([A-D])\)?"),
]


def extract_mcq_answer(text: str) -> str | None:
    """Extract a multiple-choice answer letter (A-D) from model response.

    Uses cascading regex patterns from most specific to most general.
    """
    if not text:
        return None

    # Try each pattern in priority order
    for pattern in _MCQ_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    # Fallback: check if the last non-empty line is a single letter
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        last_line = lines[-1].strip("().")
        if last_line.upper() in "ABCD" and len(last_line) == 1:
            return last_line.upper()

    # Last resort: find a standalone A-D letter in the last 3 lines only
    # (avoids false positives from letters in reasoning text)
    tail = "\n".join(lines[-3:]) if len(lines) >= 3 else text
    standalone = re.search(r"\b([A-D])\b", tail)
    if standalone:
        return standalone.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Math boxed answer extraction
# ---------------------------------------------------------------------------


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract answer from \boxed{...}, handling nested braces."""
    if not text:
        return None

    # Find the last \boxed{ occurrence
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1

    if depth != 0:
        return None

    return text[start : end - 1].strip()


def extract_answer_is(text: str) -> str | None:
    """Extract answer from 'the answer is ...' pattern."""
    if not text:
        return None

    match = re.search(
        r"(?i)(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)", text, re.DOTALL
    )
    if match:
        answer = match.group(1).strip().rstrip(".")
        if answer:
            return answer
    return None


def extract_last_expression(text: str) -> str | None:
    """Extract the last mathematical expression or number from text."""
    if not text:
        return None

    # Match numbers, fractions, expressions at end of text
    matches = re.findall(
        r"(?:[-+]?\d+(?:\.\d+)?(?:/\d+)?|\\frac\{[^}]+\}\{[^}]+\})", text
    )
    if matches:
        return matches[-1].strip()
    return None


def extract_math_answer(text: str) -> tuple[str | None, str]:
    """Multi-strategy math answer extraction.

    Returns (answer, match_method) where match_method is one of:
    'boxed', 'answer_is', 'last_expression', 'none'.
    """
    # Strategy 1: \boxed{} extraction (primary)
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer, "boxed"

    # Strategy 2: "the answer is" pattern
    answer = extract_answer_is(text)
    if answer is not None:
        return answer, "answer_is"

    # Strategy 3: Last expression fallback
    answer = extract_last_expression(text)
    if answer is not None:
        return answer, "last_expression"

    return None, "none"


# ---------------------------------------------------------------------------
# Math answer normalization
# ---------------------------------------------------------------------------


def normalize_math_answer(answer: str | None) -> str | None:
    """Normalize a math answer for comparison.

    Handles:
    - LaTeX cleanup (\\frac, \\dfrac, \\text, \\mbox, dollar signs, units)
    - Fraction normalization
    - Trailing zeros
    - Whitespace and cosmetic LaTeX
    """
    if answer is None:
        return None

    s = answer.strip()

    # Remove dollar signs (including escaped \$)
    s = s.replace("\\$", "")
    s = s.replace("$", "")

    # Remove \text{...}, \textbf{...}, \mathrm{...}, \mbox{...}
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)

    # Remove trailing unit words/symbols that follow the numeric answer
    # e.g. "864 inches^2", "5.4 cents", "90 degrees"
    s = re.sub(
        r"\s*(?:inches|feet|meters|centimeters|cents|degrees|cm|mm|m|kg|lbs?)\^?\d*\s*$",
        "", s, flags=re.IGNORECASE,
    )

    # Convert \dfrac{a}{b} and \frac{a}{b} → a/b
    s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    # Handle shorthand \frac12 or \frac{1}2 or \frac1{2} (no braces)
    s = re.sub(r"\\d?frac\{([^}]+)\}(\d)", r"\1/\2", s)
    s = re.sub(r"\\d?frac(\d)\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\d?frac(\d)(\d)", r"\1/\2", s)

    # Remove \left, \right delimiters
    s = s.replace("\\left", "").replace("\\right", "")

    # Remove LaTeX spacing commands
    s = s.replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\!", "").replace("\\:", "")

    # Remove ^\circ (degree symbol)
    s = re.sub(r"\^\\circ", "", s)
    s = re.sub(r"°", "", s)

    # Normalize \sqrt{x} — remove optional braces around single char:
    # \sqrt{2} and \sqrt2 should match
    s = re.sub(r"\\sqrt\{(\w)\}", r"\\sqrt\1", s)

    # Normalize subscripts: _{5} → _5
    s = re.sub(r"_\{(\w+)\}", r"_\1", s)

    # Remove LaTeX thousands separators: ,\! or just \!
    s = s.replace(",\\!", "")

    # Remove plain commas used as thousands separators in numbers
    # e.g. "10,080" → "10080", but don't remove commas in tuples like "(1,2)"
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)
    # Repeat for numbers with multiple groups like "1,000,000"
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)

    # \cdot and \times → *
    s = re.sub(r"\\(?:cdot|times)", "*", s)

    # Remove remaining cosmetic LaTeX commands we don't need
    s = re.sub(r"\\(?:displaystyle|phantom|hspace|vspace)\{[^}]*\}", "", s)

    # Collapse all internal whitespace to nothing for structural comparison
    # but keep content readable — strip spaces around operators/commas
    s = re.sub(r"\s+", "", s)

    # Remove leading/trailing whitespace and periods
    s = s.strip().rstrip(".")

    # Try to simplify fractions
    try:
        if "/" in s and not any(c.isalpha() for c in s):
            frac = Fraction(s).limit_denominator(10000)
            if frac.denominator == 1:
                s = str(frac.numerator)
            else:
                s = f"{frac.numerator}/{frac.denominator}"
    except (ValueError, ZeroDivisionError):
        pass

    # Remove trailing zeros from decimals
    if "." in s:
        try:
            val = float(s)
            if val == int(val):
                s = str(int(val))
            else:
                s = f"{val:g}"
        except ValueError:
            pass

    return s


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code_from_response(text: str) -> str:
    """Extract Python code from a model response, stripping markdown fences."""
    if not text:
        return ""

    # Try to extract from ```python ... ``` blocks
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        # Return the last code block (usually the final solution)
        return matches[-1].strip()

    # If no fences, return the full text (model might have returned raw code)
    return text.strip()
