"""Tests for sandboxed code execution."""

import pytest

from eval_harness.sandbox.executor import SafeExecutor


@pytest.fixture
def executor():
    return SafeExecutor(timeout_seconds=3, memory_limit_mb=128)


class TestSafeExecutor:
    """Test sandbox execution safety and correctness."""

    def test_simple_stdout(self, executor):
        result = executor.execute("print('hello world')")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello world"
        assert result.error_type == "none"
        assert not result.timed_out

    def test_stdin_input(self, executor):
        code = "x = input()\nprint(f'got: {x}')"
        result = executor.execute(code, stdin_input="test_input")
        assert result.exit_code == 0
        assert "got: test_input" in result.stdout

    def test_timeout_enforcement(self, executor):
        code = "import time\nwhile True: time.sleep(0.1)"
        result = executor.execute(code)
        assert result.timed_out
        assert result.error_type == "timeout"

    def test_runtime_error(self, executor):
        code = "raise ValueError('test error')"
        result = executor.execute(code)
        assert result.exit_code != 0
        assert result.error_type == "runtime_error"
        assert "ValueError" in result.stderr

    def test_syntax_error(self, executor):
        code = "def foo(\n  invalid syntax"
        result = executor.execute(code)
        assert result.exit_code != 0
        assert result.error_type == "compilation_error"

    def test_temp_directory_cleanup(self, executor):
        """Verify execution completes without leaking temp files."""
        import glob
        import tempfile

        before = set(glob.glob(f"{tempfile.gettempdir()}/eval_sandbox_*"))
        executor.execute("print('cleanup test')")
        after = set(glob.glob(f"{tempfile.gettempdir()}/eval_sandbox_*"))
        # No new sandbox dirs should remain
        new_dirs = after - before
        assert len(new_dirs) == 0

    def test_multiline_output(self, executor):
        code = "for i in range(3):\n    print(i)"
        result = executor.execute(code)
        assert result.stdout.strip() == "0\n1\n2"
