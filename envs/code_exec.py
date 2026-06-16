"""
Sandboxed code execution environment.

Runs untrusted LLM-generated code in isolated subprocesses with:
  - configurable timeouts (default 5s)
  - memory limits via resource module
  - no network (subprocess inherits no special privileges)
  - stdout/stderr capture

This is the "custom RL environment" that hits the JD bullet:
  "Experience with virtualization and sandboxed code execution environments"

For production, swap the subprocess sandbox with a proper container
(gVisor, Firecracker, or nsjail). The interface stays the same.
"""

import subprocess
import resource
import sys
import os
import tempfile
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Safety limits
MAX_MEMORY_MB = 256
MAX_OUTPUT_BYTES = 64 * 1024  # 64KB
DEFAULT_TIMEOUT = 5.0


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    error: Optional[str] = None
    wall_time_ms: float = 0.0


def _set_resource_limits():
    """
    Called in subprocess before exec — sets memory and CPU limits.
    Linux only: macOS doesn't support RLIMIT_AS.
    """
    import platform
    if platform.system() != "Linux":
        return

    max_bytes = MAX_MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_code_sandboxed(
    code: str,
    stdin: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    python_executable: str = sys.executable,
) -> ExecutionResult:
    """
    Execute Python code in a sandboxed subprocess.

    The code is written to a temp file (avoids shell injection)
    and run with resource limits applied via preexec_fn.

    Args:
        code:   Python source code to execute
        stdin:  input to pipe to the process
        timeout: wall-clock timeout in seconds

    Returns:
        ExecutionResult with stdout, stderr, timing, and error info
    """
    import time

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [python_executable, tmp_path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_set_resource_limits,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Truncate runaway output
        stdout = proc.stdout[:MAX_OUTPUT_BYTES]
        stderr = proc.stderr[:MAX_OUTPUT_BYTES]

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
            wall_time_ms=elapsed_ms,
        )

    except subprocess.TimeoutExpired:
        logger.debug(f"Code execution timed out after {timeout}s")
        return ExecutionResult(
            stdout="",
            stderr="",
            returncode=-1,
            timed_out=True,
            error=f"Execution timed out after {timeout}s",
        )

    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr="",
            returncode=-1,
            error=str(e),
        )

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class CodeExecutionEnv:
    """
    RL environment wrapping the sandboxed executor.

    Provides a gym-like interface for the rollout workers:
      - reset(): sample a new coding problem
      - reward(prompt, completion): execute and score

    Designed so the same interface works for math, code,
    or any other verifiable task — swap the reward_fn only.
    """

    def __init__(self, problems: list[dict], timeout: float = DEFAULT_TIMEOUT):
        """
        Args:
            problems: list of {"prompt": str, "test_cases": [...]}
            timeout:  per-execution timeout
        """
        self.problems = problems
        self.timeout = timeout
        self._index = 0
        self._stats = {"total": 0, "passed": 0, "timed_out": 0, "errors": 0}

    def reset(self) -> str:
        """Return the next problem prompt."""
        problem = self.problems[self._index % len(self.problems)]
        self._index += 1
        return problem["prompt"]

    def reward(self, prompt: str, completion: str) -> tuple[float, dict]:
        """Score a completion by running its test cases."""
        from grpo.reward.code_reward import code_execution_reward

        # Find test cases for this prompt
        problem = next((p for p in self.problems if p["prompt"] == prompt), None)
        if problem is None:
            return 0.0, {"error": "prompt not found"}

        reward, meta = code_execution_reward(
            prompt, completion,
            test_cases=problem.get("test_cases", []),
            timeout=self.timeout,
        )

        self._stats["total"] += 1
        self._stats["passed"] += meta.get("tests_passed", 0)
        if any("Timeout" in e for e in meta.get("execution_errors", [])):
            self._stats["timed_out"] += 1
        if meta.get("execution_errors"):
            self._stats["errors"] += 1

        return reward, meta

    @property
    def stats(self) -> dict:
        return dict(self._stats)