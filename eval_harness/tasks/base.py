"""Abstract base class for evaluation tasks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from eval_harness.client import EvalClient
from eval_harness.config import EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""

    task: str
    sample_id: str
    prompt: str
    raw_response: str
    parsed_answer: str | None
    normalized_answer: str | None
    ground_truth: str
    score: float
    error_category: str
    latency_ms: float
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    """Abstract base for evaluation tasks.

    Subclasses implement dataset loading, prompt formatting,
    answer extraction, and scoring.
    """

    task_name: str = "base"

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.partial_results: list[SampleResult] = []
        self._executor: ThreadPoolExecutor | None = None

    @abstractmethod
    def load_dataset(self) -> list[dict[str, Any]]:
        """Load and return the evaluation dataset."""
        ...

    @abstractmethod
    def format_prompt(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Format a sample into chat messages for the API."""
        ...

    @abstractmethod
    def extract_answer(self, response: str) -> str | None:
        """Extract the predicted answer from the model response."""
        ...

    @abstractmethod
    def normalize_answer(self, answer: str | None) -> str | None:
        """Normalize the extracted answer for comparison."""
        ...

    @abstractmethod
    def get_ground_truth(self, sample: dict[str, Any]) -> str:
        """Return the ground truth answer for a sample."""
        ...

    @abstractmethod
    def score(self, predicted: str | None, ground_truth: str) -> float:
        """Score a prediction against ground truth. Returns 0.0 or 1.0."""
        ...

    def evaluate_sample(
        self, client: EvalClient, sample: dict[str, Any], sample_id: str
    ) -> SampleResult:
        """Run evaluation on a single sample."""
        messages = self.format_prompt(sample)
        prompt_text = messages[-1]["content"] if messages else ""

        response = client.chat_completion(messages)

        if response.error_category != "none":
            return SampleResult(
                task=self.task_name,
                sample_id=sample_id,
                prompt=prompt_text,
                raw_response=response.content,
                parsed_answer=None,
                normalized_answer=None,
                ground_truth=self.get_ground_truth(sample),
                score=0.0,
                error_category=response.error_category,
                latency_ms=response.latency_ms,
            )

        parsed = self.extract_answer(response.content)
        normalized = self.normalize_answer(parsed)
        ground_truth = self.get_ground_truth(sample)
        result_score = self.score(normalized, ground_truth)

        error_cat = "none"
        if parsed is None:
            error_cat = "parse_failure"

        return SampleResult(
            task=self.task_name,
            sample_id=sample_id,
            prompt=prompt_text,
            raw_response=response.content,
            parsed_answer=parsed,
            normalized_answer=normalized,
            ground_truth=ground_truth,
            score=result_score,
            error_category=error_cat,
            latency_ms=response.latency_ms,
        )

    def run(self, client: EvalClient) -> list[SampleResult]:
        """Run the full evaluation with a single thread-pool.

        All samples are submitted to one ``ThreadPoolExecutor`` whose
        ``max_workers`` caps concurrency.  Workers are never idle between
        batches because there are no batch boundaries — as soon as one
        sample completes, the executor immediately starts the next queued
        sample.

        ``batch_size`` is used only for periodic checkpoint logging so the
        operator can see progress at a coarser grain than per-sample.

        Each sample results in exactly one ``/v1/chat/completions`` call
        (the standard OpenAI endpoint does not support multi-prompt
        requests).
        """
        dataset = self.load_dataset()
        total = len(dataset)
        # Reset for this run; accessible to the runner's interrupt handler.
        self.partial_results = []

        max_workers = self.config.concurrency.max_workers
        batch_size = self.config.concurrency.batch_size

        logger.info(
            "%s: %d samples, max_workers=%d, checkpoint_interval=%d",
            self.task_name,
            total,
            max_workers,
            batch_size,
        )

        def _eval(args: tuple[int, dict[str, Any]]) -> SampleResult:
            idx, sample = args
            return self.evaluate_sample(client, sample, str(idx))

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self._executor = executor
            futures = {
                executor.submit(_eval, (idx, sample)): idx
                for idx, sample in enumerate(dataset)
            }
            with tqdm(total=total, desc=self.task_name) as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception:
                        sample_idx = futures[future]
                        logger.exception(
                            "%s: sample %d raised an unexpected error",
                            self.task_name,
                            sample_idx,
                        )
                        result = SampleResult(
                            task=self.task_name,
                            sample_id=str(sample_idx),
                            prompt="",
                            raw_response="",
                            parsed_answer=None,
                            normalized_answer=None,
                            ground_truth="",
                            score=0.0,
                            error_category="unhandled_error",
                            latency_ms=0.0,
                        )
                    self.partial_results.append(result)
                    pbar.update(1)

                    completed += 1
                    if completed % batch_size == 0:
                        logger.debug(
                            "%s: checkpoint — %d/%d samples complete",
                            self.task_name,
                            completed,
                            total,
                        )

        self._executor = None

        # Sort by sample_id to maintain deterministic order
        self.partial_results.sort(key=lambda r: int(r.sample_id))
        return self.partial_results
