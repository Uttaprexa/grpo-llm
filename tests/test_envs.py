"""Tests for sandboxed code execution environment."""
import pytest
from envs.code_exec import run_code_sandboxed, ExecutionResult


class TestRunCodeSandboxed:
    def test_simple_print(self):
        result = run_code_sandboxed("print('hello')")
        assert result.stdout.strip() == "hello"
        assert result.returncode == 0
        assert not result.timed_out

    def test_math_output(self):
        result = run_code_sandboxed("print(2 + 2)")
        assert result.stdout.strip() == "4"

    def test_stdin_input(self):
        code = "x = input()\nprint(int(x) * 2)"
        result = run_code_sandboxed(code, stdin="21")
        assert result.stdout.strip() == "42"

    def test_syntax_error_captured(self):
        result = run_code_sandboxed("def f(:\n    pass")
        assert result.returncode != 0
        assert "SyntaxError" in result.stderr

    def test_timeout_enforced(self):
        result = run_code_sandboxed("while True: pass", timeout=1.0)
        assert result.timed_out

    def test_runtime_error_captured(self):
        result = run_code_sandboxed("1/0")
        assert result.returncode != 0
        assert "ZeroDivisionError" in result.stderr

    def test_output_isolation(self):
        # Two runs should not share state
        r1 = run_code_sandboxed("x = 42\nprint(x)")
        r2 = run_code_sandboxed("print(x)")  # x not defined
        assert r1.stdout.strip() == "42"
        assert r2.returncode != 0