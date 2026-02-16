"""Abstract base class for evaluation tasks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

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


# Type alias for the optional per-batch checkpoint callback.
BatchCheckpointCallback = Callable[[str, list[SampleResult], int], None]


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

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_result(
        future: Future[SampleResult],
        future_to_idx: dict[Future[SampleResult], int],
        task_name: str,
    ) -> SampleResult:
        """Resolve a future into a SampleResult, handling exceptions."""
        try:
            return future.result()
        except Exception:
            sample_idx = future_to_idx[future]
            logger.exception(
                "%s: sample %d raised an unexpected error",
                task_name,
                sample_idx,
            )
            return SampleResult(
                task=task_name,
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

    def run(
        self,
        client: EvalClient,
        on_batch_complete: BatchCheckpointCallback | None = None,
    ) -> list[SampleResult]:
        """Run the full evaluation, processing samples in batches.

        Samples are dispatched in chunks of ``batch_size``.  Each batch
        submits its samples to a shared ``ThreadPoolExecutor`` (capped at
        ``max_workers``) and waits only for that batch to finish before
        moving to the next.  This gives us:

        * **Bounded memory** — at most ``batch_size`` futures + results
          are alive at once.
        * **Periodic checkpointing** — the caller can supply an
          ``on_batch_complete`` callback that is invoked after every batch
          with ``(task_name, partial_results, batch_end_idx)`` so it can
          flush results to disk.
        * **Backpressure** — if the server is slow we don't queue
          hundreds of pending HTTP requests.
        * **No straggler stall** — within each batch the executor keeps
          all ``max_workers`` threads busy via ``as_completed``.

        Each sample still results in exactly one
        ``/v1/chat/completions`` call.

        Args:
            client: The API client used to send requests.
            on_batch_complete: Optional callback invoked after each batch.
                Signature: ``(task_name, partial_results, batch_end_idx)``.
        """
        dataset = self.load_dataset()
        total = len(dataset)
        self.partial_results = []

        max_workers = self.config.concurrency.max_workers
        batch_size = self.config.concurrency.batch_size
        num_batches = (total + batch_size - 1) // batch_size

        logger.info(
            "%s: %d samples, batch_size=%d (%d batches), max_workers=%d",
            self.task_name,
            total,
            batch_size,
            num_batches,
            max_workers,
        )

        def _eval(args: tuple[int, dict[str, Any]]) -> SampleResult:
            idx, sample = args
            return self.evaluate_sample(client, sample, str(idx))

        with tqdm(total=total, desc=self.task_name) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                self._executor = executor

                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, total)

                    # Submit only this batch's samples
                    future_to_idx: dict[Future[SampleResult], int] = {
                        executor.submit(_eval, (idx, dataset[idx])): idx
                        for idx in range(start, end)
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_idx):
                        result = self._collect_result(
                            future, future_to_idx, self.task_name
                        )
                        self.partial_results.append(result)
                        pbar.update(1)

                    logger.debug(
                        "%s: batch %d/%d done (%d-%d)",
                        self.task_name,
                        batch_idx + 1,
                        num_batches,
                        start,
                        end - 1,
                    )

                    # Checkpoint callback — lets the runner save to disk
                    if on_batch_complete is not None:
                        on_batch_complete(
                            self.task_name,
                            self.partial_results,
                            end,
                        )

        self._executor = None

        # Sort by sample_id to maintain deterministic order
        self.partial_results.sort(key=lambda r: int(r.sample_id))
        return self.partial_results
