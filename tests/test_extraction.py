"""Tests for answer extraction utilities."""

import pytest

from eval_harness.utils.extraction import (
    extract_boxed_answer,
    extract_code_from_response,
    extract_math_answer,
    extract_mcq_answer,
    normalize_math_answer,
)


class TestExtractMCQAnswer:
    """Test multiple-choice answer extraction."""

    def test_answer_colon_format(self):
        assert extract_mcq_answer("Answer: A") == "A"
        assert extract_mcq_answer("Answer: B") == "B"
        assert extract_mcq_answer("answer: c") == "C"

    def test_answer_with_dollar_signs(self):
        assert extract_mcq_answer("Answer: $D$") == "D"

    def test_the_answer_is_format(self):
        assert extract_mcq_answer("The answer is A.") == "A"
        assert extract_mcq_answer("the answer is (B)") == "B"

    def test_correct_answer_format(self):
        assert extract_mcq_answer("The correct answer is C") == "C"

    def test_last_line_single_letter(self):
        text = "Some reasoning\nMore reasoning\nB"
        assert extract_mcq_answer(text) == "B"

    def test_standalone_letter_fallback(self):
        text = "I believe D is correct based on the analysis."
        assert extract_mcq_answer(text) == "D"

    def test_no_answer(self):
        assert extract_mcq_answer("I don't know the answer") is None
        assert extract_mcq_answer("") is None
        assert extract_mcq_answer(None) is None

    def test_answer_in_long_response(self):
        text = (
            "Let me think step by step...\n"
            "First, we consider option A which is about...\n"
            "Then option B which states...\n"
            "After careful analysis, Answer: C"
        )
        assert extract_mcq_answer(text) == "C"


class TestExtractBoxedAnswer:
    """Test \\boxed{} extraction."""

    def test_simple_boxed(self):
        assert extract_boxed_answer("\\boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_boxed_in_context(self):
        text = "The answer is \\boxed{3x + 5} as shown."
        assert extract_boxed_answer(text) == "3x + 5"

    def test_last_boxed(self):
        text = "First \\boxed{wrong}, then \\boxed{correct}"
        assert extract_boxed_answer(text) == "correct"

    def test_no_boxed(self):
        assert extract_boxed_answer("No boxed answer here") is None
        assert extract_boxed_answer("") is None
        assert extract_boxed_answer(None) is None


class TestExtractMathAnswer:
    """Test multi-strategy math answer extraction."""

    def test_boxed_priority(self):
        text = "The answer is 5. \\boxed{42}"
        answer, method = extract_math_answer(text)
        assert answer == "42"
        assert method == "boxed"

    def test_answer_is_fallback(self):
        text = "After calculation, the answer is 17."
        answer, method = extract_math_answer(text)
        assert answer == "17"
        assert method == "answer_is"

    def test_last_expression_fallback(self):
        text = "Solving the equation gives us 3.14"
        answer, method = extract_math_answer(text)
        assert answer == "3.14"
        assert method == "last_expression"

    def test_no_answer(self):
        answer, method = extract_math_answer("No answer here at all")
        assert answer is None
        assert method == "none"


class TestNormalizeMathAnswer:
    """Test math answer normalization."""

    def test_fraction_simplification(self):
        assert normalize_math_answer("2/4") == "1/2"

    def test_trailing_zeros(self):
        assert normalize_math_answer("3.0") == "3"
        assert normalize_math_answer("3.00") == "3"

    def test_latex_frac(self):
        result = normalize_math_answer("\\frac{1}{2}")
        assert result == "1/2"

    def test_dollar_signs(self):
        assert normalize_math_answer("$42$") == "42"

    def test_text_removal(self):
        assert normalize_math_answer("\\text{hello}") == "hello"

    def test_none_input(self):
        assert normalize_math_answer(None) is None

    def test_integer_fraction(self):
        assert normalize_math_answer("4/2") == "2"

    def test_decimal_precision(self):
        assert normalize_math_answer("0.5") == "0.5"


class TestExtractCodeFromResponse:
    """Test code extraction from model responses."""

    def test_python_fenced(self):
        text = "Here's the solution:\n```python\nprint('hello')\n```\n"
        assert extract_code_from_response(text) == "print('hello')"

    def test_generic_fenced(self):
        text = "```\nprint('hello')\n```"
        assert extract_code_from_response(text) == "print('hello')"

    def test_no_fences(self):
        text = "print('hello')"
        assert extract_code_from_response(text) == "print('hello')"

    def test_empty(self):
        assert extract_code_from_response("") == ""

    def test_multiple_blocks(self):
        text = "```python\nfirst()\n```\nSome text\n```python\nsecond()\n```"
        assert extract_code_from_response(text) == "second()"
