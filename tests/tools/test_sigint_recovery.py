# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BashSession timeout behavior.

Verifies that on timeout the session resets cleanly: cwd preserved,
commands work again immediately, and the whole cycle completes quickly.
"""

import os
import shlex
import sys
import time

import pytest

from aster.tools._bash_session import BashSession


def _sleep(seconds: int) -> str:
    return f"Start-Sleep -Seconds {seconds}" if os.name == "nt" else f"sleep {seconds}"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'" if os.name == "nt" else shlex.quote(value)


@pytest.fixture
async def session(tmp_path):
    """Create a bash session in a temp directory."""
    s = BashSession(cwd=tmp_path)
    await s.start()
    yield s
    await s.close()


class TestTimeoutRecovery:
    """Test that timeout triggers a fast reset and the session stays usable."""

    async def test_timeout_preserves_cwd(self, session, tmp_path):
        """After a timeout, the working directory should be preserved."""
        subdir = tmp_path / "mydir"
        subdir.mkdir()

        await session.run(f"cd {_quote(str(subdir))}")
        assert session.cwd == subdir

        # This will timeout and reset
        _, _, code = await session.run(_sleep(30), timeout=2)
        assert code == 124

        # cwd should still be the subdir (preserved across reset)
        out, _, _ = await session.run("(Get-Location).Path" if os.name == "nt" else "pwd")
        assert str(subdir) in out, f"cwd lost after timeout. Got: {out!r}"

    async def test_timeout_returns_partial_output(self, session):
        """Output produced before timeout should be captured."""
        out, err, code = await session.run(
            f"echo before_timeout; {_sleep(30)}",
            timeout=2,
        )
        assert "before_timeout" in out
        assert code == 124

    async def test_timeout_returns_quickly(self, session):
        """Timeout + reset should complete well within the timeout window."""
        start = time.monotonic()
        _, _, code = await session.run(_sleep(60), timeout=2)
        elapsed = time.monotonic() - start
        # Should complete within timeout + reset overhead (not 60s!)
        assert elapsed < 12, f"Timeout took too long: {elapsed:.1f}s"
        assert code == 124

    async def test_timeout_exit_code(self, session):
        """Timed-out commands should return exit code 124."""
        _, _, code = await session.run(_sleep(30), timeout=2)
        assert code == 124

    async def test_session_works_after_timeout(self, session):
        """After timeout+reset, session should still execute commands."""
        _, _, code = await session.run(_sleep(30), timeout=2)
        assert code == 124

        out, _, code = await session.run("echo still_alive")
        assert code == 0
        assert "still_alive" in out

    async def test_multiple_timeouts_session_survives(self, session):
        """Session should survive multiple consecutive timeouts."""
        for _ in range(3):
            _, _, code = await session.run(_sleep(30), timeout=2)
            assert code == 124

        out, _, code = await session.run("echo works")
        assert code == 0
        assert "works" in out

    async def test_timeout_kills_pipeline(self, session):
        """Timeout should kill all processes in a pipeline."""
        start = time.monotonic()
        command = f"{_sleep(30)} | Out-Null" if os.name == "nt" else "sleep 30 | cat"
        _, _, code = await session.run(command, timeout=2)
        elapsed = time.monotonic() - start
        assert elapsed < 12, f"Pipeline timeout took too long: {elapsed:.1f}s"
        assert code == 124

        out, _, code = await session.run("echo alive")
        assert code == 0
        assert "alive" in out

    async def test_timeout_kills_subshell(self, session):
        """Timeout should kill commands in a subshell."""
        command = "& { Start-Sleep -Seconds 30 }" if os.name == "nt" else "(sleep 30)"
        _, _, code = await session.run(command, timeout=2)
        assert code == 124

        out, _, code = await session.run("echo works")
        assert code == 0
        assert "works" in out

    async def test_env_lifecycle_after_timeout(self, session):
        """Windows restarts its Job; POSIX preserves env when child-only kill succeeds."""
        await session.run(
            "$env:TIMEOUT_TEST = 'hello'" if os.name == "nt" else "export TIMEOUT_TEST=hello"
        )
        read_env = "echo $env:TIMEOUT_TEST" if os.name == "nt" else "echo $TIMEOUT_TEST"
        out, _, _ = await session.run(read_env)
        assert "hello" in out

        before = session._start_count
        _, _, code = await session.run(_sleep(30), timeout=2)
        assert code == 124
        out, _, code = await session.run(read_env)
        assert code == 0  # Session works regardless
        if os.name == "nt":
            assert "hello" not in out
            assert session._start_count == before + 1
            return
        # Env preservation is best-effort (requires pgrep)
        if "hello" not in out:
            import warnings

            warnings.warn(
                "pgrep-based recovery not available; env vars lost on timeout", stacklevel=2
            )

    async def test_timeout_with_nested_processes(self, session):
        """Timeout also stops a command waiting for its own child process."""
        await session.run(
            "$env:NESTED_VAR = 'deep'" if os.name == "nt" else "export NESTED_VAR=deep"
        )
        if os.name == "nt":
            script = (
                "import subprocess,sys; "
                "subprocess.run([sys.executable, '-c', 'import time; time.sleep(30)'], check=True)"
            )
            command = f"& {_quote(sys.executable)} -c {_quote(script)}"
        else:
            command = "bash -c 'sleep 30'"
        _, _, code = await session.run(command, timeout=2)
        assert code == 124

        out, _, code = await session.run(
            "echo $env:NESTED_VAR" if os.name == "nt" else "echo $NESTED_VAR"
        )
        assert code == 0  # Session works regardless
        if os.name == "nt":
            assert "deep" not in out
            return
        # Env preservation is best-effort (requires pgrep)
        if "deep" not in out:
            import warnings

            warnings.warn(
                "pgrep-based recovery not available; env vars lost on timeout", stacklevel=2
            )

    async def test_fast_command_after_timeout(self, session):
        """Fast commands work immediately after timeout recovery."""
        _, _, code = await session.run(_sleep(30), timeout=2)
        assert code == 124
        # Should return instantly, not wait any residual time
        import time

        t0 = time.monotonic()
        out, _, code = await session.run("echo fast")
        elapsed = time.monotonic() - t0
        assert code == 0
        assert "fast" in out
        # Windows recreates its runspace and compiles the native stdin bridge.
        budget = 10 if os.name == "nt" else 2
        assert elapsed < budget, f"Post-timeout command took {elapsed:.1f}s"
