"""GPQA Diamond evaluation task — 198 graduate-level multiple-choice questions."""

from __future__ import annotations

import logging
import random
from typing import Any

from datasets import load_dataset

from eval_harness.config import EvalConfig
from eval_harness.tasks.base import BaseTask
from eval_harness.utils.extraction import extract_mcq_answer

logger = logging.getLogger(__name__)

# Few-shot examples for GPQA (hand-crafted; domain-general to avoid leakage)
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the primary function of mitochondria in eukaryotic cells?",
        "choices": {
            "A": "Protein synthesis",
            "B": "ATP production through oxidative phosphorylation",
            "C": "DNA replication",
            "D": "Cell division regulation",
        },
        "answer": "B",
        "reasoning": (
            "Mitochondria are known as the powerhouses of the cell. "
            "They produce ATP through oxidative phosphorylation, which is the "
            "primary energy currency of the cell. While they contain their own DNA "
            "and ribosomes, their main function is energy production."
        ),
    },
    {
        "question": (
            "In quantum mechanics, what does the Heisenberg uncertainty principle state?"
        ),
        "choices": {
            "A": "Energy is always conserved in quantum systems",
            "B": "Particles can exist in multiple states simultaneously",
            "C": (
                "It is impossible to simultaneously know the exact position and "
                "momentum of a particle"
            ),
            "D": "Wave functions always collapse upon measurement",
        },
        "answer": "C",
        "reasoning": (
            "The Heisenberg uncertainty principle fundamentally states that there "
            "is a limit to the precision with which certain pairs of complementary "
            "variables (like position and momentum) can be known simultaneously. "
            "This is not due to measurement limitations but is an inherent property "
            "of quantum systems."
        ),
    },
]

SYSTEM_MESSAGE = (
    "You are an expert assistant. Answer multiple choice questions by reasoning "
    "step by step, then providing your final answer on the last line in the format: "
    "'Answer: $LETTER' where LETTER is one of A, B, C, D."
)


def _format_few_shot_example(ex: dict[str, Any]) -> str:
    """Format one few-shot example as a string."""
    lines = [ex["question"], ""]
    for letter in "ABCD":
        lines.append(f"{letter}) {ex['choices'][letter]}")
    lines.append("")
    lines.append(ex["reasoning"])
    lines.append(f"\nAnswer: {ex['answer']}")
    return "\n".join(lines)


class GPQATask(BaseTask):
    """GPQA Diamond evaluation (198 questions, few-shot MCQ)."""

    task_name = "gpqa"

    def __init__(self, config: EvalConfig, seed: int = 42) -> None:
        super().__init__(config)
        self.seed = seed
        self.rng = random.Random(seed)
        self._dataset: list[dict[str, Any]] | None = None

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load GPQA Diamond split from HuggingFace."""
        if self._dataset is not None:
            return self._dataset

        logger.info("Loading GPQA Diamond dataset...")
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        self._dataset = [dict(row) for row in ds]
        logger.info("Loaded %d GPQA samples", len(self._dataset))
        return self._dataset

    def _permute_choices(
        self, sample: dict[str, Any]
    ) -> tuple[dict[str, str], str]:
        """Randomly permute answer choices and return (choices_dict, correct_letter).

        Returns:
            choices_dict: {"A": text, "B": text, "C": text, "D": text}
            correct_letter: the letter (A-D) that maps to the correct answer
        """
        choices = [
            sample["Correct Answer"],
            sample["Incorrect Answer 1"],
            sample["Incorrect Answer 2"],
            sample["Incorrect Answer 3"],
        ]
        perm = list(range(4))
        self.rng.shuffle(perm)
        shuffled = [choices[i] for i in perm]

        correct_index = shuffled.index(sample["Correct Answer"])
        correct_letter = "ABCD"[correct_index]

        choices_dict = {
            "A": shuffled[0],
            "B": shuffled[1],
            "C": shuffled[2],
            "D": shuffled[3],
        }
        return choices_dict, correct_letter

    def format_prompt(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Format a GPQA sample as few-shot chat messages."""
        # Build few-shot prefix
        few_shot_parts = []
        for ex in FEW_SHOT_EXAMPLES:
            few_shot_parts.append(_format_few_shot_example(ex))
        few_shot_text = "\n\n---\n\n".join(few_shot_parts)

        # Build target question
        choices_dict, correct_letter = self._permute_choices(sample)
        # Store correct letter for later scoring
        sample["_correct_letter"] = correct_letter

        question_text = sample["Question"]
        target = f"{question_text}\n\n"
        for letter in "ABCD":
            target += f"{letter}) {choices_dict[letter]}\n"

        user_content = (
            f"Answer the following multiple choice question. The last line of your "
            f"response should be of the following format: 'Answer: $LETTER' "
            f"(without quotes) where LETTER is one of ABCD. Think step by step "
            f"before answering.\n\n"
            f"Here are some examples:\n\n"
            f"{few_shot_text}\n\n"
            f"---\n\n"
            f"Now answer this question:\n\n"
            f"{target}"
        )

        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]

    def extract_answer(self, response: str) -> str | None:
        """Extract answer letter from model response."""
        return extract_mcq_answer(response)

    def normalize_answer(self, answer: str | None) -> str | None:
        """MCQ answers don't need normalization beyond uppercasing."""
        if answer is None:
            return None
        return answer.upper()

    def get_ground_truth(self, sample: dict[str, Any]) -> str:
        """Return the correct answer letter."""
        return sample.get("_correct_letter", "A")

    def score(self, predicted: str | None, ground_truth: str) -> float:
        """Score: 1.0 if predicted matches ground truth letter, else 0.0."""
        if predicted is None:
            return 0.0
        return 1.0 if predicted == ground_truth else 0.0
