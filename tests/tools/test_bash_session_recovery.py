# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BashSession auto-recovery and bulletproofing (issue #178).

Tests verify:
1. Dead process recovery (auto-restart on killed/exited bash)
2. Stdin isolation (commands cannot steal control-pipe sentinels)
3. Large output handling (no pipe deadlock on >192KB output)
4. EOF on control fd reports non-zero exit code
5. Pipelines and redirects work correctly
"""

import asyncio
import os
import shlex
import sys

import pytest

from aster.tools._bash_session import BashSession


def _python_command(code: str) -> str:
    if os.name == "nt":
        executable = sys.executable.replace("'", "''")
        return f"& '{executable}' -c '{code.replace(chr(39), chr(39) * 2)}'"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


@pytest.fixture
async def session(tmp_path):
    """Fresh BashSession for each test."""
    s = BashSession(cwd=tmp_path)
    await s.start()
    yield s
    await s.close()


class TestAutoRecoveryFromDeadProcess:
    """BashSession should auto-recover when the bash process dies."""

    async def test_recovers_after_process_killed(self, session):
        """After killing the bash process, the next command should still work."""
        before = session._start_count
        session._process.kill()
        await asyncio.wait_for(session._process.wait(), timeout=5)

        stdout, stderr, code = await session.run("echo recovered")
        assert stdout.splitlines() == ["recovered"]
        assert code == 0, stderr
        assert session._start_count == before + 1

    async def test_recovers_after_process_exits(self, session):
        """After bash exits on its own, the next command should still work."""
        before = session._start_count
        request = b'{"op":"close"}\n' if os.name == "nt" else b"exit 0\n"
        session._process.stdin.write(request)
        await session._process.stdin.drain()
        await asyncio.wait_for(session._process.wait(), timeout=5)

        stdout, stderr, code = await session.run("echo recovered")
        assert stdout.splitlines() == ["recovered"]
        assert code == 0, stderr
        assert session._start_count == before + 1


class TestStdinIsolation:
    """Commands cannot steal from the control pipe."""

    async def test_cat_no_args_does_not_hang(self, session):
        """cat with no file args should get EOF quickly (fd 3 architecture)."""
        command = "[Console]::In.ReadToEnd()" if os.name == "nt" else "cat"
        stdout, stderr, code = await session.run(command, timeout=5.0)
        assert code == 0, stderr
        assert not stdout.strip()

        # Session should still be usable afterwards
        stdout2, _, _ = await session.run("echo alive")
        assert stdout2.splitlines() == ["alive"]

    async def test_read_builtin_does_not_hang(self, session):
        """read from stdin should get EOF, not block."""
        command = (
            '$line = [Console]::In.ReadLine(); Write-Output "got=$line"'
            if os.name == "nt"
            else "read line; echo got=$line"
        )
        stdout, stderr, code = await session.run(command, timeout=5.0)
        assert code == 0, stderr
        assert stdout.splitlines() == ["got="]

        stdout2, _, _ = await session.run("echo still_works")
        assert stdout2.splitlines() == ["still_works"]


class TestLargeOutput:
    """Commands producing large output should not deadlock."""

    async def test_200kb_stdout_does_not_deadlock(self, session):
        """A command producing >192KB of output should complete without timeout."""
        # Generate 200KB of output (well above the 128KB StreamReader + 64KB pipe limit)
        stdout, stderr, code = await session.run(
            _python_command("print('A' * 200000)"), timeout=10.0
        )
        assert code == 0
        assert len(stdout) >= 30000  # Truncated by MAX_OUTPUT_CHARS but proves no deadlock

    async def test_large_stderr_does_not_deadlock(self, session):
        """Large stderr should also complete without deadlock."""
        stdout, stderr, code = await session.run(
            _python_command("import sys; sys.stderr.write('B' * 200000)"), timeout=10.0
        )
        assert code == 0
        assert len(stderr) >= 30000

    async def test_session_usable_after_large_output(self, session):
        """After large output, session should still work for normal commands."""
        await session.run(_python_command("print('X' * 200000)"), timeout=10.0)
        stdout, _, code = await session.run("echo still_alive")
        assert stdout.splitlines() == ["still_alive"]
        assert code == 0


class TestEOFExitCode:
    """EOF on control fd (bash died) should report non-zero exit code."""

    async def test_killed_process_returns_nonzero(self, session):
        """If bash is killed during execution, exit code should be non-zero."""
        # Kill bash while it's running a sleep
        process = session._process

        async def kill_later():
            await asyncio.sleep(0.5)
            process.kill()

        killer = asyncio.create_task(kill_later())
        try:
            if os.name == "nt":
                with pytest.raises((EOFError, OSError)):
                    await session.run("Start-Sleep -Seconds 10", timeout=5.0)
            else:
                _, _, code = await session.run("sleep 10", timeout=5.0)
                assert code != 0
        finally:
            await killer
        stdout, stderr, code = await session.run("echo recovered")
        assert code == 0, stderr
        assert stdout.splitlines() == ["recovered"]


class TestPipelinesAndRedirects:
    """Pipelines and redirects must work correctly."""

    async def test_simple_pipe(self, session):
        """echo | grep should work."""
        command = (
            "Write-Output hello | Select-String hello"
            if os.name == "nt"
            else "echo hello | grep hello"
        )
        stdout, stderr, code = await session.run(command)
        assert "hello" in stdout
        assert code == 0

    async def test_multi_pipe(self, session):
        """Multi-stage pipeline."""
        command = (
            "'b', 'a', 'c' | Sort-Object | Select-Object -First 1"
            if os.name == "nt"
            else "echo -e 'b\\na\\nc' | sort | head -1"
        )
        stdout, stderr, code = await session.run(command)
        assert code == 0, stderr
        assert stdout.splitlines() == ["a"]

    async def test_stdin_redirect_from_file(self, session, tmp_path):
        """Explicit stdin redirect from a file should work."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("file_content\n")
        if os.name == "nt":
            command = "Get-Content -Raw -LiteralPath 'input.txt'"
        else:
            command = f"cat < {shlex.quote(str(test_file))}"
        stdout, stderr, code = await session.run(command)
        assert code == 0, stderr
        assert "file_content" in stdout

    async def test_heredoc(self, session):
        """Heredoc should work correctly."""
        command = "@'\nheredoc_line\n'@" if os.name == "nt" else "cat <<EOF\nheredoc_line\nEOF"
        stdout, stderr, code = await session.run(command)
        assert code == 0, stderr
        assert "heredoc_line" in stdout

    async def test_command_substitution(self, session):
        """Command substitution should work."""
        stdout, stderr, code = await session.run("echo $(echo nested)")
        assert code == 0, stderr
        assert stdout.splitlines() == ["nested"]
