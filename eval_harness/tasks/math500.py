"""MATH500 evaluation task — 500 math problems with multi-strategy extraction."""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from eval_harness.config import EvalConfig
from eval_harness.tasks.base import BaseTask, SampleResult
from eval_harness.utils.extraction import (
    extract_math_answer,
    normalize_math_answer,
)

logger = logging.getLogger(__name__)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n{problem}"
)


class Math500Task(BaseTask):
    """MATH500 evaluation (500 problems, \\boxed{} extraction)."""

    task_name = "math500"

    def __init__(self, config: EvalConfig) -> None:
        super().__init__(config)
        self._dataset: list[dict[str, Any]] | None = None

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load MATH-500 dataset (500 problems)."""
        if self._dataset is not None:
            return self._dataset

        logger.info("Loading MATH500 dataset...")
        ds = load_dataset("math-ai/math500", split="test")
        self._dataset = [dict(row) for row in ds]
        logger.info("Loaded %d MATH samples", len(self._dataset))
        return self._dataset

    def format_prompt(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Format a MATH problem as chat messages with \\boxed{} instruction."""
        problem = sample.get("problem", "")
        user_content = MATH_PROMPT_TEMPLATE.format(problem=problem)

        return [
            {
                "role": "system",
                "content": (
                    "You are a mathematical problem solver. Show your work step "
                    "by step and put your final answer inside \\boxed{}."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def extract_answer(self, response: str) -> str | None:
        """Multi-strategy math answer extraction."""
        answer, _ = extract_math_answer(response)
        return answer

    def _get_match_method(self, response: str) -> str:
        """Get which extraction method was used."""
        _, method = extract_math_answer(response)
        return method

    def normalize_answer(self, answer: str | None) -> str | None:
        """Normalize extracted answer for comparison."""
        return normalize_math_answer(answer)

    def get_ground_truth(self, sample: dict[str, Any]) -> str:
        """Extract ground truth from the solution field."""
        solution = sample.get("solution", "")
        # MATH dataset has the answer in \boxed{} within the solution
        from eval_harness.utils.extraction import extract_boxed_answer

        boxed = extract_boxed_answer(solution)
        if boxed is not None:
            return boxed

        # Fallback: use the 'answer' field if available
        return sample.get("answer", "")

    def score(self, predicted: str | None, ground_truth: str) -> float:
        """Score via normalized string comparison."""
        if predicted is None:
            return 0.0

        norm_pred = normalize_math_answer(predicted)
        norm_gt = normalize_math_answer(ground_truth)

        if norm_pred is None or norm_gt is None:
            return 0.0

        return 1.0 if norm_pred == norm_gt else 0.0

    def evaluate_sample(
        self, client: "EvalClient", sample: dict[str, Any], sample_id: str
    ) -> SampleResult:
        """Override to add match_method to extra fields."""
        result = super().evaluate_sample(client, sample, sample_id)

        # Add MATH-specific extra fields
        match_method = "none"
        if result.raw_response:
            match_method = self._get_match_method(result.raw_response)
        result.extra["match_method"] = match_method
        result.extra["raw_answer"] = result.parsed_answer

        return result
