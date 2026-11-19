# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for persistent bash session."""

import shlex
import sys

import pytest

from aster.tools._bash_session import BashSession


def _python_command(script: str) -> str:
    arguments = [sys.executable, "-c", script]
    if sys.platform == "win32":
        return "& " + " ".join("'" + argument.replace("'", "''") + "'" for argument in arguments)
    return shlex.join(arguments)


@pytest.fixture
async def session(tmp_path):
    """Create a bash session in a temp directory."""
    s = BashSession(cwd=tmp_path)
    await s.start()
    yield s
    await s.close()


class TestBashSession:
    async def test_simple_command(self, session):
        out, err, code = await session.run("echo hello")
        assert code == 0
        assert "hello" in out

    async def test_exit_code(self, session):
        out, err, code = await session.run(_python_command("import sys; sys.exit(7)"))
        assert code == 7

    async def test_stderr(self, session):
        out, err, code = await session.run(
            _python_command("import sys; print('oops', file=sys.stderr)")
        )
        assert code == 0
        assert "oops" in err

    async def test_cd_persists(self, session, tmp_path):
        """cd in one command should persist to the next."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        await session.run(f"cd {subdir}")
        out, _, _ = await session.run("pwd")
        assert str(subdir) in out

    async def test_cwd_tracking(self, session, tmp_path):
        """Session.cwd should update after cd."""
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)

        await session.run(f"cd {subdir}")
        assert session.cwd == subdir

    async def test_env_persists(self, session):
        """Environment variables should persist."""
        assignment, read = "export MY_VAR=hello123", "echo $MY_VAR"
        if sys.platform == "win32":
            assignment, read = "$env:MY_VAR='hello123'", "Write-Output $env:MY_VAR"
        _, _, code = await session.run(assignment)
        assert code == 0
        out, _, _ = await session.run(read)
        assert "hello123" in out

    async def test_multiline_output(self, session):
        out, _, code = await session.run("echo line1; echo line2; echo line3")
        assert code == 0
        assert "line1" in out
        assert "line2" in out
        assert "line3" in out

    async def test_start_idempotent(self, session):
        """Calling start() twice should be safe."""
        await session.start()  # already started by fixture
        out, _, _ = await session.run("echo still_works")
        assert "still_works" in out

    async def test_output_truncation(self, session):
        """Very long output should be truncated."""
        out, _, code = await session.run(_python_command("print('x' * 50000)"))
        assert code == 0
        assert "<truncated-output>" in out
        assert len(out) <= 31000  # MAX_OUTPUT_CHARS + truncation message

    async def test_truncation_keeps_the_head_and_the_tail(self, session):
        """A long output keeps its end too: a failing command usually ends with its error."""
        script = "import sys; print('H' * 20000 + 'M' * 20000 + 'T' * 20000); "
        script += "sys.stderr.write('h' * 20000 + 'm' * 20000 + 't' * 20000)"
        out, err, code = await session.run(_python_command(script))
        assert code == 0
        for text, head, middle, tail in ((out, "H", "M", "T"), (err, "h", "m", "t")):
            # The standard notice of TruncatingStringIO, then the head, then the tail.
            assert text.startswith("<truncated-output>\nOutput too large (")
            assert head * 14000 in text
            assert tail * 14000 in text
            assert text.index(head * 14000) < text.index(tail * 14000)
            assert middle * 1000 not in text
            assert len(text) <= 31000

    def test_bounded_matches_one_write_and_keeps_both_ends(self):
        """Chunked feeding gives the same text as one write, for every size around the limit."""
        from aster.agentdoc import TruncatingStringIO
        from aster.tools._bash_session import MAX_OUTPUT_CHARS, _bounded

        for size in (MAX_OUTPUT_CHARS, MAX_OUTPUT_CHARS + 1, 65_536, 65_537, 300_000):
            text = "".join(chr(ord("a") + i % 26) for i in range(size))
            reference = TruncatingStringIO(limit=MAX_OUTPUT_CHARS)
            reference.write(text)
            assert _bounded(text) == reference.getvalue()
            if size > MAX_OUTPUT_CHARS:
                assert text[: MAX_OUTPUT_CHARS // 2] in _bounded(text)
                assert text[-(MAX_OUTPUT_CHARS // 2) :] in _bounded(text)

    async def test_close_and_restart(self, tmp_path):
        """Session should be closeable and re-startable."""
        s = BashSession(cwd=tmp_path)
        await s.start()
        out1, _, _ = await s.run("echo first")
        assert "first" in out1
        await s.close()

        # Start fresh
        await s.start()
        out2, _, _ = await s.run("echo second")
        assert "second" in out2
        await s.close()

    async def test_pipe_commands(self, session):
        command = "printf 'a\\nb\\nc\\n' | sort -r"
        if sys.platform == "win32":
            command = "'a', 'b', 'c' | Sort-Object -Descending"
        out, _, code = await session.run(command)
        assert code == 0
        assert out.splitlines() == ["c", "b", "a"]

    async def test_command_with_quotes(self, session):
        out, _, code = await session.run("echo 'hello world'")
        assert code == 0
        assert "hello world" in out

    async def test_file_operations(self, session, tmp_path):
        """Write and read a file through the session."""
        await session.run(f"echo 'test content' > {tmp_path}/test.txt")
        out, _, _ = await session.run(f"cat {tmp_path}/test.txt")
        assert "test content" in out
