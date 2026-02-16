"""Tests for MATH500 task."""

from eval_harness.config import EvalConfig
from eval_harness.tasks.math500 import Math500Task
from eval_harness.utils.extraction import normalize_math_answer


class TestMath500Scoring:
    """Test MATH500 scoring logic."""

    def test_exact_match(self):
        config = EvalConfig.default()
        task = Math500Task(config)
        assert task.score("42", "42") == 1.0

    def test_normalized_match(self):
        config = EvalConfig.default()
        task = Math500Task(config)
        # "2/4" normalizes to "1/2"
        assert task.score("2/4", "1/2") == 1.0

    def test_no_match(self):
        config = EvalConfig.default()
        task = Math500Task(config)
        assert task.score("42", "43") == 0.0

    def test_none_prediction(self):
        config = EvalConfig.default()
        task = Math500Task(config)
        assert task.score(None, "42") == 0.0

    def test_ground_truth_extraction(self):
        config = EvalConfig.default()
        task = Math500Task(config)
        sample = {
            "solution": "The answer is \\boxed{42}.",
            "answer": "42",
        }
        assert task.get_ground_truth(sample) == "42"

    def test_ground_truth_fallback(self):
        config = EvalConfig.default()
        task = Math500Task(config)
        sample = {
            "solution": "No boxed answer here.",
            "answer": "17",
        }
        assert task.get_ground_truth(sample) == "17"
