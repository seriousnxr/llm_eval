"""Tests for summary report generation."""

import pytest

from eval_harness.tasks.base import SampleResult
from eval_harness.utils.reporting import (
    build_summary_report,
    results_to_jsonl_records,
)


def _make_result(
    task: str = "test",
    score: float = 1.0,
    error_category: str = "none",
    latency_ms: float = 100.0,
    **extra,
) -> SampleResult:
    return SampleResult(
        task=task,
        sample_id="0",
        prompt="test prompt",
        raw_response="test response",
        parsed_answer="A",
        normalized_answer="A",
        ground_truth="A",
        score=score,
        error_category=error_category,
        latency_ms=latency_ms,
        extra=extra,
    )


class TestBuildSummaryReport:
    """Test summary report generation."""

    def test_single_task(self):
        results = {"gpqa": [_make_result(score=1.0), _make_result(score=0.0)]}
        report = build_summary_report(results, wall_clock_seconds=10.0)

        assert "gpqa" in report["tasks"]
        assert report["tasks"]["gpqa"]["total_samples"] == 2
        assert report["tasks"]["gpqa"]["correct"] == 1
        assert report["tasks"]["gpqa"]["accuracy"] == 0.5

    def test_overall_stats(self):
        results = {
            "gpqa": [_make_result(score=1.0)],
            "math500": [_make_result(score=1.0), _make_result(score=0.0)],
        }
        report = build_summary_report(results, wall_clock_seconds=5.0)

        assert report["overall"]["total_samples"] == 3
        assert report["overall"]["total_correct"] == 2

    def test_error_breakdown(self):
        results = {
            "gpqa": [
                _make_result(error_category="none"),
                _make_result(error_category="parse_failure"),
                _make_result(error_category="api_error"),
                _make_result(error_category="parse_failure"),
            ]
        }
        report = build_summary_report(results, wall_clock_seconds=1.0)
        breakdown = report["tasks"]["gpqa"]["error_breakdown"]

        assert breakdown["none"] == 1
        assert breakdown["parse_failure"] == 2
        assert breakdown["api_error"] == 1

    def test_empty_results(self):
        report = build_summary_report({}, wall_clock_seconds=0.0)
        assert report["overall"]["total_samples"] == 0

    def test_wall_clock_recorded(self):
        report = build_summary_report({}, wall_clock_seconds=42.5)
        assert report["wall_clock_seconds"] == 42.5


class TestResultsToJsonl:
    """Test JSONL record conversion."""

    def test_required_fields(self):
        result = _make_result()
        records = results_to_jsonl_records([result])
        assert len(records) == 1
        record = records[0]

        required = [
            "task", "sample_id", "prompt", "raw_response",
            "parsed_answer", "normalized_answer", "ground_truth",
            "score", "error_category", "latency_ms",
        ]
        for field in required:
            assert field in record, f"Missing required field: {field}"

    def test_extra_fields_merged(self):
        result = _make_result(match_method="boxed", raw_answer="42")
        records = results_to_jsonl_records([result])
        assert records[0]["match_method"] == "boxed"
        assert records[0]["raw_answer"] == "42"
