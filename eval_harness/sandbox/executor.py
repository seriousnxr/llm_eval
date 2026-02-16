"""Sandboxed code execution with timeout, memory limits, and process cleanup.

Works on both macOS and Linux without adjustments:
- Memory limits: resource.setrlimit (RLIMIT_AS on Linux, RLIMIT_DATA/RSS fallback on macOS)
- Network isolation: sandbox-exec (macOS) or unshare -rn (Linux), auto-validated at init
- Process cleanup: os.setsid + os.killpg (POSIX, both platforms)
"""

from __future__ import annotations

import logging
import os
import platform
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# macOS sandbox profile to deny all network access
_MACOS_SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"


def _probe_sandbox_exec() -> bool:
    """Validate that macOS sandbox-exec actually works (not just present)."""
    if platform.system() != "Darwin" or shutil.which("sandbox-exec") is None:
        return False
    try:
        result = subprocess.run(
            ["sandbox-exec", "-p", _MACOS_SANDBOX_PROFILE, "/usr/bin/true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _probe_unshare() -> bool:
    """Validate that Linux unshare -rn actually works.

    The binary may exist but unprivileged user namespaces can be disabled
    (kernel.unprivileged_userns_clone=0), so we probe with a real command.
    """
    if platform.system() != "Linux" or shutil.which("unshare") is None:
        return False
    try:
        result = subprocess.run(
            ["unshare", "-rn", "/bin/true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    error_type: str  # "none", "timeout", "runtime_error", "memory_limit", "compilation_error"
    execution_time_ms: float


def _set_limits(memory_limit_mb: int = 256) -> None:
    """Set resource limits for the child process (preexec_fn).

    Called in the forked child before exec.  Works on both Linux and macOS.
    """
    mem_bytes = memory_limit_mb * 1024 * 1024

    # Memory limit — try RLIMIT_AS (Linux), fall back to RLIMIT_DATA (macOS)
    for limit_type in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
        try:
            resource.setrlimit(limit_type, (mem_bytes, mem_bytes))
            break
        except (ValueError, resource.error, AttributeError):
            continue

    # CPU time limit (generous — timeout handles wall clock)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except (ValueError, resource.error):
        pass

    # Create a new process group so we can kill all children
    os.setsid()


class SafeExecutor:
    """Execute untrusted Python code in a sandboxed subprocess.

    Safety features:
    - Per-execution timeout with hard kill
    - Memory limits via resource.setrlimit
    - Network isolation (macOS sandbox-exec / Linux unshare -rn)
    - Process group kill to prevent process leaks
    - Temporary directory isolation per execution
    - Stdout/stderr capture
    """

    def __init__(
        self,
        timeout_seconds: int = 5,
        memory_limit_mb: int = 256,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        # Probe platform-specific network isolation at init time
        # (binary existence alone is not enough — e.g. unprivileged
        #  user namespaces may be disabled on Linux)
        self._use_sandbox_exec = _probe_sandbox_exec()
        self._use_unshare = not self._use_sandbox_exec and _probe_unshare()
        if self._use_sandbox_exec:
            logger.info("Network isolation: macOS sandbox-exec (validated)")
        elif self._use_unshare:
            logger.info("Network isolation: Linux unshare -rn (validated)")
        else:
            logger.warning(
                "Network isolation: not available on this platform — "
                "code will run without network restrictions"
            )

    def execute(
        self,
        code: str,
        stdin_input: str = "",
    ) -> ExecutionResult:
        """Execute Python code in a sandboxed subprocess.

        Args:
            code: Python source code to execute.
            stdin_input: String to pipe as stdin.

        Returns:
            ExecutionResult with stdout, stderr, timing, and error info.
        """
        tmpdir = tempfile.mkdtemp(prefix="eval_sandbox_")
        code_path = os.path.join(tmpdir, "solution.py")
        start_time = time.perf_counter()

        try:
            # Write code to temp file
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Encode stdin — handle both text and binary gracefully
            if isinstance(stdin_input, str):
                stdin_bytes = stdin_input.encode("utf-8", errors="replace")
            else:
                stdin_bytes = stdin_input

            # Build command with optional network isolation
            base_cmd = ["python3", code_path]
            if self._use_sandbox_exec:
                cmd = [
                    "sandbox-exec", "-p", _MACOS_SANDBOX_PROFILE,
                ] + base_cmd
            elif self._use_unshare:
                cmd = ["unshare", "-rn"] + base_cmd
            else:
                cmd = base_cmd

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                preexec_fn=lambda: _set_limits(self.memory_limit_mb),
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
                    "HOME": tmpdir,
                    "TMPDIR": tmpdir,
                },
            )

            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    input=stdin_bytes,
                    timeout=self.timeout_seconds,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")

                # Detect error types
                error_type = "none"
                if proc.returncode != 0:
                    if "MemoryError" in stderr or "Cannot allocate" in stderr:
                        error_type = "memory_limit"
                    elif "SyntaxError" in stderr or "IndentationError" in stderr:
                        error_type = "compilation_error"
                    else:
                        error_type = "runtime_error"

                return ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode,
                    timed_out=False,
                    error_type=error_type,
                    execution_time_ms=elapsed_ms,
                )

            except subprocess.TimeoutExpired:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                # Kill the entire process group
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                proc.wait()

                return ExecutionResult(
                    stdout="",
                    stderr="Execution timed out",
                    exit_code=-1,
                    timed_out=True,
                    error_type="timeout",
                    execution_time_ms=elapsed_ms,
                )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                timed_out=False,
                error_type="runtime_error",
                execution_time_ms=elapsed_ms,
            )

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
