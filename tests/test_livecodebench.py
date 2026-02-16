"""Tests for LiveCodeBench task logic — decoding, prompt formatting, scoring, and JSONL fields."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from eval_harness.tasks.livecodebench import (
    LiveCodeBenchTask,
    TestCaseResult,
    _decode_test_cases,
    _get_test_inputs_outputs,
)


# ---------------------------------------------------------------------------
# _decode_test_cases
# ---------------------------------------------------------------------------

class TestDecodeTestCases:
    """Test the various test-case decoding paths."""

    def test_plain_json_list(self):
        raw = json.dumps([{"input": "1", "output": "2"}])
        cases = _decode_test_cases(raw)
        assert len(cases) == 1
        assert cases[0]["input"] == "1"

    def test_already_list(self):
        data = [{"input": "a", "output": "b"}]
        assert _decode_test_cases(data) == data

    def test_invalid_string_returns_empty(self):
        assert _decode_test_cases("not-json-or-compressed") == []


# ---------------------------------------------------------------------------
# _get_test_inputs_outputs
# ---------------------------------------------------------------------------

class TestGetTestInputsOutputs:
    """Test extraction of inputs/outputs from sample dicts."""

    def test_input_output_field(self):
        sample = {
            "input_output": json.dumps({
                "inputs": ["1\n", "2\n"],
                "outputs": ["1\n", "4\n"],
                "fn_name": None,
            })
        }
        inputs, outputs, fn_name = _get_test_inputs_outputs(sample)
        assert inputs == ["1\n", "2\n"]
        assert outputs == ["1\n", "4\n"]
        assert fn_name is None

    def test_functional_fn_name(self):
        sample = {
            "input_output": json.dumps({
                "inputs": ["[1,2]"],
                "outputs": ["3"],
                "fn_name": "twoSum",
            })
        }
        _, _, fn_name = _get_test_inputs_outputs(sample)
        assert fn_name == "twoSum"

    def test_fallback_to_public_test_cases(self):
        cases = json.dumps([
            {"input": "hello", "output": "world", "testtype": "stdin"},
        ])
        sample = {"public_test_cases": cases}
        inputs, outputs, fn_name = _get_test_inputs_outputs(sample)
        assert inputs == ["hello"]
        assert outputs == ["world"]
        assert fn_name is None

    def test_empty_sample(self):
        inputs, outputs, fn_name = _get_test_inputs_outputs({})
        assert inputs == []
        assert outputs == []
        assert fn_name is None


# ---------------------------------------------------------------------------
# LiveCodeBenchTask unit tests (mock client & executor)
# ---------------------------------------------------------------------------

@pytest.fixture
def lcb_task():
    """Create a LiveCodeBenchTask with a dummy config."""
    config = MagicMock()
    config.base_url = "http://localhost:8000"
    config.model = "test-model"
    config.temperature = 0.0
    config.max_tokens = 1024
    config.max_retries = 1
    config.timeout = 5
    config.max_workers = 1
    config.batch_size = 10
    task = LiveCodeBenchTask(config)
    return task


class TestFormatPrompt:
    """Test prompt formatting for stdin vs functional problems."""

    def test_stdin_prompt(self, lcb_task):
        sample = {"question_content": "Print input reversed."}
        msgs = lcb_task.format_prompt(sample)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "Read the inputs from stdin" in msgs[1]["content"]

    def test_functional_prompt_with_starter(self, lcb_task):
        sample = {
            "question_content": "Return the sum.",
            "starter_code": "class Solution:\n    def solve(self, x):",
        }
        msgs = lcb_task.format_prompt(sample)
        assert "starter code" in msgs[1]["content"]
        assert "class Solution" in msgs[1]["content"]


class TestExtractAnswer:
    """Test code extraction from model responses."""

    def test_fenced_python(self, lcb_task):
        resp = "Here's the answer:\n```python\nprint(42)\n```"
        assert lcb_task.extract_answer(resp) == "print(42)"

    def test_no_code_returns_none(self, lcb_task):
        assert lcb_task.extract_answer("") is None


class TestRunTestCase:
    """Test the _run_test_case method with a real SafeExecutor."""

    def test_passing_stdin(self, lcb_task):
        code = "x = input()\nprint(x)"
        result = lcb_task._run_test_case(code, "hello", "hello", fn_name=None)
        assert result.passed
        assert result.error_type == "none"
        assert result.execution_time_ms > 0

    def test_wrong_answer(self, lcb_task):
        code = "print('wrong')"
        result = lcb_task._run_test_case(code, "", "correct", fn_name=None)
        assert not result.passed
        assert result.error_type == "wrong_answer"

    def test_timeout_in_test(self, lcb_task):
        code = "import time\nwhile True: time.sleep(0.1)"
        result = lcb_task._run_test_case(code, "", "anything", fn_name=None)
        assert not result.passed
        assert result.error_type == "timeout"

    def test_runtime_error_in_test(self, lcb_task):
        code = "raise RuntimeError('boom')"
        result = lcb_task._run_test_case(code, "", "anything", fn_name=None)
        assert not result.passed
        assert result.error_type == "runtime_error"


class TestEvaluateSampleJSONLFields:
    """Verify that evaluate_sample produces all required JSONL extra fields."""

    def _make_sample(self):
        return {
            "question_id": "lcb_42",
            "question_content": "Print '42'.",
            "input_output": json.dumps({
                "inputs": [""],
                "outputs": ["42"],
                "fn_name": None,
            }),
        }

    def test_all_extra_fields_on_success(self, lcb_task):
        """After a successful solve, all LCB JSONL fields must be present."""
        mock_response = MagicMock()
        mock_response.content = "```python\nprint('42')\n```"
        mock_response.error_category = "none"
        mock_response.latency_ms = 123.0

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        result = lcb_task.evaluate_sample(mock_client, self._make_sample(), "0")

        # Required per task.md §5 Reporting
        assert result.extra["problem_id"] == "lcb_42"
        assert result.extra["generated_code"] is not None
        assert isinstance(result.extra["test_results"], list)
        assert result.extra["pass_at_1"] in (0, 1)
        assert result.extra["execution_time_ms"] >= 0
        assert result.error_category in (
            "none", "wrong_answer", "timeout", "runtime_error",
        )

    def test_pass_at_1_logic(self, lcb_task):
        """pass@1 is 1 only when ALL test cases pass."""
        mock_response = MagicMock()
        mock_response.content = "```python\nprint('42')\n```"
        mock_response.error_category = "none"
        mock_response.latency_ms = 50.0

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        result = lcb_task.evaluate_sample(mock_client, self._make_sample(), "0")
        assert result.score == 1.0
        assert result.extra["pass_at_1"] == 1

    def test_api_error_still_has_fields(self, lcb_task):
        """Even on API error, extra dict must contain the required fields."""
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.error_category = "server_error"
        mock_response.latency_ms = 200.0

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        result = lcb_task.evaluate_sample(mock_client, self._make_sample(), "0")

        assert result.extra["problem_id"] == "lcb_42"
        assert result.extra["generated_code"] is None
        assert result.extra["test_results"] == []
        assert result.extra["pass_at_1"] == 0
        assert result.extra["execution_time_ms"] == 0.0
        assert result.error_category == "server_error"

    def test_parse_failure_has_fields(self, lcb_task):
        """When code extraction fails, extra dict still has required fields."""
        mock_response = MagicMock()
        mock_response.content = ""  # empty → extract_answer returns None
        mock_response.error_category = "none"
        mock_response.latency_ms = 80.0

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        result = lcb_task.evaluate_sample(mock_client, self._make_sample(), "0")

        assert result.error_category == "parse_failure"
        assert result.extra["problem_id"] == "lcb_42"
        assert result.extra["generated_code"] is None
        assert result.extra["pass_at_1"] == 0
