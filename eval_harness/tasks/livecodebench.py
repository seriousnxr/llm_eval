"""LiveCodeBench evaluation task — code generation with sandboxed execution."""

from __future__ import annotations

import base64
import json
import logging
import pickle
import time
import zlib
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from eval_harness.client import EvalClient
from eval_harness.config import EvalConfig
from eval_harness.sandbox.executor import ExecutionResult, SafeExecutor
from eval_harness.tasks.base import BaseTask, SampleResult
from eval_harness.utils.extraction import extract_code_from_response

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)


@dataclass
class TestCaseResult:
    """Result of a single test case execution."""

    passed: bool
    input_preview: str
    expected_preview: str
    actual_preview: str
    error_type: str  # "none", "timeout", "runtime_error", "wrong_answer", etc.
    execution_time_ms: float


def _decode_test_cases(raw: str | list) -> list[dict[str, str]]:
    """Decode test cases from dataset, handling both JSON and compressed formats.

    Returns list of dicts with 'input', 'output', and 'testtype' keys.
    """
    if isinstance(raw, list):
        return raw

    # Try plain JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try compressed format: base64 -> zlib -> pickle -> JSON
    try:
        decoded = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
        if isinstance(decoded, list):
            return decoded
        return [decoded]
    except Exception:
        pass

    return []


def _get_test_inputs_outputs(sample: dict[str, Any]) -> tuple[list[str], list[str], str | None]:
    """Extract test inputs, outputs, and fn_name from a sample.

    Returns:
        (inputs, outputs, fn_name) where fn_name is None for stdin-based problems.
    """
    # Try input_output field first (evaluation format)
    if "input_output" in sample and sample["input_output"]:
        try:
            io_data = json.loads(sample["input_output"]) if isinstance(sample["input_output"], str) else sample["input_output"]
            inputs = io_data.get("inputs", [])
            outputs = io_data.get("outputs", [])
            fn_name = io_data.get("fn_name", None)
            if isinstance(inputs, str):
                inputs = [inputs]
            if isinstance(outputs, str):
                outputs = [outputs]
            return inputs, outputs, fn_name
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to public + private test cases
    all_tests = []

    for field in ("public_test_cases", "private_test_cases"):
        raw = sample.get(field, "[]")
        cases = _decode_test_cases(raw)
        all_tests.extend(cases)

    if not all_tests:
        return [], [], None

    inputs = [t.get("input", "") for t in all_tests]
    outputs = [t.get("output", "") for t in all_tests]

    # Detect test type from first test case
    first_type = all_tests[0].get("testtype", "stdin")
    fn_name = None
    if first_type == "functional":
        # Need metadata for fn_name
        metadata = sample.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        fn_name = metadata.get("func_name", None)

    return inputs, outputs, fn_name


def _build_stdin_test_code(code: str, stdin_input: str) -> str:
    """Build a test script that runs the solution code with stdin input.

    For stdin-based problems, the generated code reads from stdin and
    writes to stdout. We just run it directly.
    """
    return code


def _build_functional_test_code(
    code: str, fn_name: str, test_input: str, expected_output: str
) -> str:
    """Build a test script for call-based (LeetCode) problems.

    Wraps the solution in a test harness that:
    1. Imports the solution
    2. Calls the function with inputs
    3. Prints the result for comparison
    """
    # Parse input arguments from JSON
    test_script = f"""\
import json
import sys

{code}

# Parse test inputs
raw_inputs = {json.dumps(test_input)}
inputs = [json.loads(line) for line in raw_inputs.split("\\n") if line.strip()]

# Create solution instance and call method
sol = Solution()
result = sol.{fn_name}(*inputs)

# Output result as JSON for comparison
print(json.dumps(result))
"""
    return test_script


class LiveCodeBenchTask(BaseTask):
    """LiveCodeBench evaluation (~400 coding problems with sandboxed execution)."""

    task_name = "livecodebench"

    def __init__(
        self,
        config: EvalConfig,
        timeout_per_test: int = 5,
        easy_only: bool = False,
    ) -> None:
        super().__init__(config)
        self.timeout_per_test = timeout_per_test
        self.easy_only = easy_only
        self._dataset: list[dict[str, Any]] | None = None
        self.executor = SafeExecutor(
            timeout_seconds=timeout_per_test,
            memory_limit_mb=256,
        )

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load LiveCodeBench code generation dataset."""
        if self._dataset is not None:
            return self._dataset

        logger.info("Loading LiveCodeBench dataset (release_v1)...")
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            version_tag="release_v1",
            trust_remote_code=True,
        )
        samples = [dict(row) for row in ds]

        if self.easy_only:
            samples = [
                s for s in samples
                if s.get("difficulty", "").lower() == "easy"
            ]
            logger.info("Filtered to %d easy problems", len(samples))

        self._dataset = samples
        logger.info("Loaded %d LiveCodeBench problems", len(self._dataset))
        return self._dataset

    def format_prompt(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Format a coding problem as chat messages."""
        question_content = sample.get("question_content", "")
        starter_code = sample.get("starter_code", "")

        # Build user prompt following LiveCodeBench's generic format
        prompt = f"### Question:\n{question_content}\n\n"

        if starter_code:
            prompt += (
                "### Format: You will use the following starter code to write "
                "the solution to the problem and enclose your code within delimiters.\n"
            )
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += (
                "### Format: Read the inputs from stdin solve the problem and "
                "write the answer to stdout (do not directly test on the sample "
                "inputs). Enclose your code within delimiters as follows. "
                "Ensure that when the python program runs, it reads the inputs, "
                "runs the algorithm and writes output to STDOUT.\n"
            )
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"

        prompt += "### Answer: (use the provided format with backticks)\n\n"

        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]

    def extract_answer(self, response: str) -> str | None:
        """Extract Python code from model response."""
        code = extract_code_from_response(response)
        return code if code else None

    def normalize_answer(self, answer: str | None) -> str | None:
        """Code answers don't need normalization."""
        return answer

    def get_ground_truth(self, sample: dict[str, Any]) -> str:
        """Ground truth for code tasks is the test cases (not a simple string)."""
        return sample.get("question_id", sample.get("question_title", "unknown"))

    def score(self, predicted: str | None, ground_truth: str) -> float:
        """Scoring is handled in evaluate_sample with test execution."""
        return 0.0  # Overridden in evaluate_sample

    def _run_test_case(
        self,
        code: str,
        test_input: str,
        expected_output: str,
        fn_name: str | None,
    ) -> TestCaseResult:
        """Execute code against a single test case in the sandbox."""
        start = time.perf_counter()

        if fn_name is not None:
            # Call-based (LeetCode style)
            test_code = _build_functional_test_code(
                code, fn_name, test_input, expected_output
            )
            result = self.executor.execute(test_code)
        else:
            # Standard input/output
            result = self.executor.execute(code, stdin_input=test_input)

        elapsed_ms = (time.perf_counter() - start) * 1000

        if result.timed_out:
            return TestCaseResult(
                passed=False,
                input_preview=test_input[:100],
                expected_preview=expected_output[:100],
                actual_preview="",
                error_type="timeout",
                execution_time_ms=elapsed_ms,
            )

        if result.exit_code != 0:
            return TestCaseResult(
                passed=False,
                input_preview=test_input[:100],
                expected_preview=expected_output[:100],
                actual_preview=result.stderr[:200],
                error_type=result.error_type,
                execution_time_ms=elapsed_ms,
            )

        # Compare output (strip trailing whitespace from each line)
        actual_lines = [
            line.strip() for line in result.stdout.strip().split("\n")
        ]
        expected_lines = [
            line.strip() for line in expected_output.strip().split("\n")
        ]

        passed = actual_lines == expected_lines

        # If exact match fails, try JSON comparison for call-based problems
        if not passed and fn_name is not None:
            try:
                actual_json = json.loads(result.stdout.strip())
                expected_json = json.loads(expected_output.strip())
                passed = actual_json == expected_json
            except (json.JSONDecodeError, TypeError):
                pass

        return TestCaseResult(
            passed=passed,
            input_preview=test_input[:100],
            expected_preview=expected_output[:100],
            actual_preview=result.stdout[:200],
            error_type="none" if passed else "wrong_answer",
            execution_time_ms=elapsed_ms,
        )

    def evaluate_sample(
        self,
        client: EvalClient,
        sample: dict[str, Any],
        sample_id: str,
    ) -> SampleResult:
        """Evaluate a single coding problem: prompt → generate → execute → score."""
        messages = self.format_prompt(sample)
        prompt_text = messages[-1]["content"] if messages else ""

        # Get model response
        response = client.chat_completion(messages)

        problem_id = sample.get("question_id", sample.get("question_title", sample_id))

        if response.error_category != "none":
            return SampleResult(
                task=self.task_name,
                sample_id=sample_id,
                prompt=prompt_text,
                raw_response=response.content,
                parsed_answer=None,
                normalized_answer=None,
                ground_truth=str(problem_id),
                score=0.0,
                error_category=response.error_category,
                latency_ms=response.latency_ms,
                extra={
                    "problem_id": problem_id,
                    "generated_code": None,
                    "test_results": [],
                    "pass_at_1": 0,
                    "execution_time_ms": 0.0,
                },
            )

        # Extract code from response
        generated_code = self.extract_answer(response.content)

        if generated_code is None:
            return SampleResult(
                task=self.task_name,
                sample_id=sample_id,
                prompt=prompt_text,
                raw_response=response.content,
                parsed_answer=None,
                normalized_answer=None,
                ground_truth=str(problem_id),
                score=0.0,
                error_category="parse_failure",
                latency_ms=response.latency_ms,
                extra={
                    "problem_id": problem_id,
                    "generated_code": None,
                    "test_results": [],
                    "pass_at_1": 0,
                    "execution_time_ms": 0.0,
                },
            )

        # Get test cases
        inputs, outputs, fn_name = _get_test_inputs_outputs(sample)

        if not inputs or not outputs:
            logger.warning("No test cases for problem %s", problem_id)
            return SampleResult(
                task=self.task_name,
                sample_id=sample_id,
                prompt=prompt_text,
                raw_response=response.content,
                parsed_answer=generated_code,
                normalized_answer=generated_code,
                ground_truth=str(problem_id),
                score=0.0,
                error_category="no_test_cases",
                latency_ms=response.latency_ms,
                extra={
                    "problem_id": problem_id,
                    "generated_code": generated_code,
                    "test_results": [],
                    "pass_at_1": 0,
                    "execution_time_ms": 0.0,
                },
            )

        # Run all test cases
        test_results: list[dict[str, Any]] = []
        total_exec_ms = 0.0
        all_passed = True
        error_category = "none"

        for i, (inp, expected) in enumerate(zip(inputs, outputs)):
            tc_result = self._run_test_case(generated_code, inp, expected, fn_name)
            total_exec_ms += tc_result.execution_time_ms

            test_results.append({
                "test_id": i,
                "passed": tc_result.passed,
                "error_type": tc_result.error_type,
                "execution_time_ms": tc_result.execution_time_ms,
            })

            if not tc_result.passed:
                all_passed = False
                if error_category == "none":
                    error_category = tc_result.error_type

        pass_at_1 = 1 if all_passed else 0

        return SampleResult(
            task=self.task_name,
            sample_id=sample_id,
            prompt=prompt_text,
            raw_response=response.content,
            parsed_answer=generated_code,
            normalized_answer=generated_code,
            ground_truth=str(problem_id),
            score=float(pass_at_1),
            error_category=error_category,
            latency_ms=response.latency_ms,
            extra={
                "problem_id": problem_id,
                "generated_code": generated_code,
                "test_results": test_results,
                "pass_at_1": pass_at_1,
                "execution_time_ms": total_exec_ms,
                "difficulty": sample.get("difficulty", "unknown"),
                "platform": sample.get("platform", "unknown"),
            },
        )
