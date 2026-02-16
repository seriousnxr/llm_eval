"""Summary report generation for evaluation runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_harness.tasks.base import SampleResult

logger = logging.getLogger(__name__)


def build_summary_report(
    results_by_task: dict[str, list[SampleResult]],
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Build a summary report from evaluation results.

    Args:
        results_by_task: Mapping of task name → list of SampleResult.
        wall_clock_seconds: Total wall-clock time for the evaluation run.

    Returns:
        Summary report dict ready for JSON serialization.
    """
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "tasks": {},
    }

    total_samples = 0
    total_correct = 0

    for task_name, results in results_by_task.items():
        n = len(results)
        correct = sum(1 for r in results if r.score > 0)
        accuracy = correct / n if n > 0 else 0.0

        # Error breakdown
        error_counts: dict[str, int] = {}
        for r in results:
            cat = r.error_category
            error_counts[cat] = error_counts.get(cat, 0) + 1

        # Latency stats
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
        p95_idx = int(len(latencies) * 0.95) if latencies else 0
        p95 = sorted(latencies)[min(p95_idx, len(latencies) - 1)] if latencies else 0.0

        # Throughput
        total_latency_s = sum(latencies) / 1000.0
        throughput = n / wall_clock_seconds if wall_clock_seconds > 0 else 0.0

        task_report = {
            "total_samples": n,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "error_breakdown": error_counts,
            "latency_ms": {
                "mean": round(avg_latency, 1),
                "p50": round(p50, 1),
                "p95": round(p95, 1),
            },
            "throughput_rps": round(throughput, 2),
        }

        # Add LiveCodeBench-specific metrics
        if task_name == "livecodebench":
            pass_at_1_scores = [
                r.extra.get("pass_at_1", 0) for r in results
                if "pass_at_1" in r.extra
            ]
            if pass_at_1_scores:
                task_report["pass_at_1"] = round(
                    sum(pass_at_1_scores) / len(pass_at_1_scores), 4
                )

            # Breakdown by difficulty
            diff_counts: dict[str, dict[str, int]] = {}
            for r in results:
                diff = r.extra.get("difficulty", "unknown")
                if diff not in diff_counts:
                    diff_counts[diff] = {"total": 0, "passed": 0}
                diff_counts[diff]["total"] += 1
                if r.score > 0:
                    diff_counts[diff]["passed"] += 1
            task_report["by_difficulty"] = diff_counts

        report["tasks"][task_name] = task_report
        total_samples += n
        total_correct += correct

    report["overall"] = {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy": round(
            total_correct / total_samples if total_samples > 0 else 0.0, 4
        ),
        "throughput_rps": round(
            total_samples / wall_clock_seconds if wall_clock_seconds > 0 else 0.0, 2
        ),
    }

    return report


def write_summary_report(report: dict[str, Any], path: str | Path) -> None:
    """Write summary report to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Summary report written to %s", path)


def results_to_jsonl_records(results: list[SampleResult]) -> list[dict[str, Any]]:
    """Convert SampleResult objects to JSONL-compatible dicts."""
    records = []
    for r in results:
        record = {
            "task": r.task,
            "sample_id": r.sample_id,
            "prompt": r.prompt,
            "raw_response": r.raw_response,
            "parsed_answer": r.parsed_answer,
            "normalized_answer": r.normalized_answer,
            "ground_truth": r.ground_truth,
            "score": r.score,
            "error_category": r.error_category,
            "latency_ms": round(r.latency_ms, 1),
        }
        # Merge extra fields
        record.update(r.extra)
        records.append(record)
    return records
