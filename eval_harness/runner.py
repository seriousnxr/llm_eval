"""Evaluation runner — orchestrates benchmark tasks and generates reports."""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Any

from eval_harness.client import EvalClient
from eval_harness.config import EvalConfig
from eval_harness.tasks.base import SampleResult
from eval_harness.tasks.gpqa import GPQATask
from eval_harness.tasks.livecodebench import LiveCodeBenchTask
from eval_harness.tasks.math500 import Math500Task
from eval_harness.utils.logging import setup_logging, write_jsonl
from eval_harness.utils.reporting import (
    build_summary_report,
    results_to_jsonl_records,
    write_summary_report,
)

logger = logging.getLogger(__name__)

# Which tasks to run by default
AVAILABLE_TASKS = ("gpqa", "math500", "livecodebench")


class EvalRunner:
    """Main orchestrator for running evaluation benchmarks.

    Features:
    - Runs selected tasks against the model server
    - Saves per-sample JSONL results with per-batch checkpointing
    - Generates summary report
    - Graceful interruption with partial result saving
    """

    def __init__(
        self,
        config: EvalConfig,
        tasks: list[str] | None = None,
    ) -> None:
        self.config = config
        self.requested_tasks = tasks or list(AVAILABLE_TASKS)
        self.client = EvalClient(config)
        self.results: dict[str, list[SampleResult]] = {}
        self._interrupted = False
        self._current_task: Any = None  # track in-flight task for interrupt
        self._start_time: float = 0.0
        self._task_wall_times: dict[str, float] = {}
        self._current_task_start: float = 0.0

        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle SIGINT (Ctrl+C) gracefully — save partial results and exit."""
        if self._interrupted:
            logger.warning("Force quit requested — exiting immediately")
            sys.exit(1)

        logger.warning(
            "Interrupt received — saving partial results and exiting..."
        )
        self._interrupted = True

        # Capture partial results from the in-flight task
        task = self._current_task
        if task is not None:
            # Cancel pending futures so the executor exits immediately
            # instead of waiting for all queued samples to complete.
            executor = getattr(task, "_executor", None)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

            partial = getattr(task, "partial_results", [])
            if partial:
                self.results[task.task_name] = list(partial)
                # Record elapsed time for the interrupted task
                if task.task_name not in self._task_wall_times:
                    self._task_wall_times[task.task_name] = (
                        time.perf_counter() - self._current_task_start
                    )

        self._save_results()

        # Generate summary report from whatever we have so far
        wall_clock = (
            time.perf_counter() - self._start_time
            if self._start_time > 0
            else 0.0
        )
        if self.results:
            results_dir = self.config.ensure_output_dir()
            report = build_summary_report(
                self.results, wall_clock, self._task_wall_times
            )
            write_summary_report(report, results_dir / "summary_report.json")
            logger.info("Partial summary report saved")

        sys.exit(0)

    def _build_task(self, task_name: str) -> GPQATask | Math500Task | LiveCodeBenchTask:
        """Instantiate a task by name."""
        if task_name == "gpqa":
            return GPQATask(self.config)
        elif task_name == "math500":
            return Math500Task(self.config)
        elif task_name == "livecodebench":
            return LiveCodeBenchTask(self.config)
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def _save_results(self) -> None:
        """Save all accumulated results to disk."""
        results_dir = self.config.ensure_output_dir()

        for task_name, results in self.results.items():
            if not results:
                continue
            records = results_to_jsonl_records(results)
            write_jsonl(records, results_dir / f"{task_name}_results.jsonl")

        # Write error log from client
        if self.client.error_log:
            error_records = [
                {
                    "timestamp": e.timestamp,
                    "error_type": e.error_type,
                    "status_code": e.status_code,
                    "message": e.message,
                    "retry_count": e.retry_count,
                    "prompt_preview": e.prompt_preview,
                }
                for e in self.client.error_log
            ]
            write_jsonl(error_records, results_dir / "error_log.jsonl")

    def _on_batch_complete(
        self,
        task_name: str,
        partial_results: list[SampleResult],
        batch_end_idx: int,
    ) -> None:
        """Checkpoint callback — flush partial results to disk after each batch."""
        self.results[task_name] = list(partial_results)
        results_dir = self.config.ensure_output_dir()
        records = results_to_jsonl_records(partial_results)
        write_jsonl(records, results_dir / f"{task_name}_results.jsonl")
        logger.info(
            "%s: checkpoint saved — %d results written to disk (through sample %d)",
            task_name,
            len(partial_results),
            batch_end_idx - 1,
        )

    def run(self) -> dict[str, Any]:
        """Run all requested tasks and return the summary report.

        Returns:
            Summary report dict.
        """
        setup_logging()
        results_dir = self.config.ensure_output_dir()

        logger.info("=" * 60)
        logger.info("LLM Evaluation Harness — Starting")
        logger.info("Tasks: %s", ", ".join(self.requested_tasks))
        logger.info("Server: %s", self.config.server.base_url)
        logger.info("Workers: %d", self.config.concurrency.max_workers)
        logger.info("Results dir: %s", results_dir)
        logger.info("=" * 60)

        self._start_time = time.perf_counter()
        self._task_wall_times = {}

        for task_name in self.requested_tasks:
            if self._interrupted:
                break

            logger.info("--- Running task: %s ---", task_name)
            self._current_task_start = time.perf_counter()

            try:
                task = self._build_task(task_name)
                self._current_task = task
                task_results = task.run(self.client, on_batch_complete=self._on_batch_complete)
                self._current_task = None
                self.results[task_name] = task_results

                task_elapsed = time.perf_counter() - self._current_task_start
                self._task_wall_times[task_name] = task_elapsed
                correct = sum(1 for r in task_results if r.score > 0)
                total = len(task_results)
                accuracy = correct / total if total > 0 else 0.0

                logger.info(
                    "Task %s complete: %d/%d correct (%.2f%%) in %.1fs",
                    task_name,
                    correct,
                    total,
                    accuracy * 100,
                    task_elapsed,
                )
            except Exception:
                logger.exception("Task %s failed", task_name)
                self.results[task_name] = []

        wall_clock = time.perf_counter() - self._start_time

        # Save results
        self._save_results()

        # Generate summary report
        report = build_summary_report(self.results, wall_clock, self._task_wall_times)
        write_summary_report(report, results_dir / "summary_report.json")

        logger.info("=" * 60)
        logger.info("Evaluation complete in %.1fs", wall_clock)
        for task_name, task_info in report.get("tasks", {}).items():
            logger.info(
                "  %s: %.2f%% accuracy (%d/%d)",
                task_name,
                task_info["accuracy"] * 100,
                task_info["correct"],
                task_info["total_samples"],
            )
        logger.info("=" * 60)

        return report


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Evaluation Harness",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(AVAILABLE_TASKS),
        default=None,
        help="Tasks to run (default: all)",
    )
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.config)
    runner = EvalRunner(config, tasks=args.tasks)
    runner.run()


if __name__ == "__main__":
    main()
