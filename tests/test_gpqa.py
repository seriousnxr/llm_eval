"""Tests for GPQA task — answer permutation and prompt formatting."""

from unittest.mock import MagicMock, patch

import pytest

from eval_harness.config import EvalConfig


class TestGPQAPermutation:
    """Test that GPQA answer permutation is correct and reproducible."""

    def test_permutation_reproducibility(self):
        """Same seed should produce same permutation."""
        from eval_harness.tasks.gpqa import GPQATask

        config = EvalConfig.default()
        task1 = GPQATask(config, seed=42)
        task2 = GPQATask(config, seed=42)

        sample = {
            "Question": "What is 2+2?",
            "Correct Answer": "4",
            "Incorrect Answer 1": "3",
            "Incorrect Answer 2": "5",
            "Incorrect Answer 3": "6",
        }

        choices1, letter1 = task1._permute_choices(sample)
        choices2, letter2 = task2._permute_choices(sample)

        assert choices1 == choices2
        assert letter1 == letter2

    def test_correct_answer_tracked(self):
        """The correct answer should always map to the returned letter."""
        from eval_harness.tasks.gpqa import GPQATask

        config = EvalConfig.default()
        task = GPQATask(config, seed=123)

        sample = {
            "Question": "Test?",
            "Correct Answer": "CORRECT",
            "Incorrect Answer 1": "WRONG1",
            "Incorrect Answer 2": "WRONG2",
            "Incorrect Answer 3": "WRONG3",
        }

        choices, correct_letter = task._permute_choices(sample)
        assert choices[correct_letter] == "CORRECT"

    def test_all_choices_present(self):
        """All four choices should appear in the permuted dict."""
        from eval_harness.tasks.gpqa import GPQATask

        config = EvalConfig.default()
        task = GPQATask(config, seed=0)

        sample = {
            "Question": "Test?",
            "Correct Answer": "A_ans",
            "Incorrect Answer 1": "B_ans",
            "Incorrect Answer 2": "C_ans",
            "Incorrect Answer 3": "D_ans",
        }

        choices, _ = task._permute_choices(sample)
        values = set(choices.values())
        assert values == {"A_ans", "B_ans", "C_ans", "D_ans"}
