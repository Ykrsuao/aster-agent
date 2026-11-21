# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core behavior of the DEFAULT ShellTools (run / read / write_file / replace).

This keeps the *default* ShellTools — the one agents actually use — covered for
its primary file/run surface. Search-anchor behavior is covered separately in
test_shell_tools_modern.py.
"""

import shlex
import sys

import pytest

from aster.tools.shell_tools import Match, ShellResult, ShellTools

_IS_WINDOWS = sys.platform == "win32"
_PWD_COMMAND = "(Get-Location).Path" if _IS_WINDOWS else "pwd"


def _python_command(code: str) -> str:
    arguments = [sys.executable, "-c", code]
    if _IS_WINDOWS:
        return "& " + " ".join("'" + argument.replace("'", "''") + "'" for argument in arguments)
    return shlex.join(arguments)


def test_shell_tools_directs_agents_to_its_file_and_command_methods():
    doc = ShellTools.__doc__
    assert doc is not None
    assert "Always use these four methods rather than Python builtins" in doc
    assert "For shell commands and file operations" in doc


def test_run_stream_documents_standalone_usage():
    doc = ShellTools.run_stream.__doc__
    assert doc is not None
    assert "pyp" not in doc
    assert "async for event in self.shell.run_stream(" in doc
    assert "event.returncode" in doc
    assert "event.timed_out" in doc


def test_run_and_run_stream_have_matching_arguments():
    import inspect

    def arguments(method):
        return [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()]

    assert arguments(ShellTools.run_stream) == arguments(ShellTools.run)


@pytest.mark.parametrize("payload", ["", "hello", "one\ntwo\n", "'\" $HOME $(echo nope) `pwd` λ\n"])
async def test_run_stream_accepts_stdin_verbatim_and_keeps_exit_status(tmp_path, payload):
    shell = ShellTools(cwd=str(tmp_path))
    try:
        command = (
            _python_command(
                "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); "
                "sys.stderr.buffer.write(b'problem\\n'); sys.exit(7)"
            )
            if _IS_WINDOWS
            else "cat; printf 'problem\\n' >&2; (exit 7)"
        )
        buffered = await shell.run(command, stdin=payload, timeout=5.0)
        events = [event async for event in shell.run_stream(command, stdin=payload, timeout=5.0)]
        assert "".join(event.text for event in events if event.kind == "stdout") == payload
        assert "".join(event.text for event in events if event.kind == "stderr") == "problem\n"
        assert events[-1].kind == "done"
        assert events[-1].returncode == 7
        # PowerShell preserves trailing newlines; Bash buffered results strip them.
        assert buffered.stdout == (payload if _IS_WINDOWS else payload.rstrip("\n"))
        assert buffered.stderr == ("problem\n" if _IS_WINDOWS else "problem")
        assert buffered.returncode == events[-1].returncode
        assert not events[-1].timed_out
        assert sum(event.kind == "done" for event in events) == 1
        ready = "[Console]::Write('ready')" if _IS_WINDOWS else "printf ready"
        assert (await shell.run(ready)).stdout == "ready"
    finally:
        await shell.close()


@pytest.fixture
async def sh(tmp_path):
    shell = ShellTools(cwd=str(tmp_path))
    try:
        yield shell
    finally:
        await shell.close()


@pytest.mark.asyncio
async def test_run_persists_state(sh, tmp_path):
    r = await sh.run("echo hello")
    assert r.success
    assert "hello" in r.stdout
    # cd persists across calls in the same session.
    (tmp_path / "sub").mkdir()
    await sh.run("cd sub")
    r2 = await sh.run(_PWD_COMMAND)
    assert r2.stdout.strip().endswith("sub")


@pytest.mark.asyncio
async def test_run_reports_failure(sh):
    r = await sh.run(_python_command("import sys; sys.exit(1)"))
    assert not r.success
    assert r.returncode == 1
    assert r.timed_out is False


def test_match_requires_resolved_path():
    with pytest.raises(TypeError, match="resolved_path"):
        Match("example.py", 1, 1, "value\n")  # type: ignore[call-arg]


def test_shell_result_timeout_flag_preserves_positional_matches_argument(tmp_path):
    match = Match("example.py", 1, 1, "value\n", resolved_path=tmp_path / "example.py")
    result = ShellResult("value", "", 0, [match], timed_out=True)

    assert result.matches == [match]
    assert result.timed_out is True


def test_shell_result_prints_its_class_and_attribute_names():
    # Models copy names from what they see: the printout must name the real
    # class and attributes, and always show the return code.
    assert repr(ShellResult("ok\n")) == "ShellResult(returncode=0, stdout='ok\\n')"
    failed = ShellResult("", "boom\n", 3)
    assert str(failed) == "ShellResult(returncode=3, stdout='', stderr='boom\\n')"
    assert failed.returncode == 3 and failed.stderr == "boom\n"
    timed_out = ShellResult("", "", 124, timed_out=True, timeout_seconds=30.0)
    assert repr(timed_out) == (
        "ShellResult(returncode=124, timed_out=True, timeout_seconds=30.0, stdout='')"
    )


@pytest.mark.asyncio
async def test_write_file_then_read(sh, tmp_path):
    await sh.write_file("f.txt", "line1\nline2\nline3\n")
    assert (tmp_path / "f.txt").read_text() == "line1\nline2\nline3\n"
    # read with a numbered gutter (default) -> Match; inspect via .numbered/.text.
    view = await sh.read("f.txt")
    assert "line2" in view.numbered
    # read a line window -> Match for just that line.
    window = await sh.read("f.txt", (2, 2))
    assert "line2" in window.text
    assert "line1" not in window.text


@pytest.mark.asyncio
async def test_replace_path_unique(sh, tmp_path):
    await sh.write_file("f.py", "x = 1\ny = 2\nz = 3\n")
    await sh.replace("f.py", "y = 2", "y = 22")
    assert (tmp_path / "f.py").read_text() == "x = 1\ny = 22\nz = 3\n"


@pytest.mark.asyncio
async def test_replace_path_ambiguous_errors(sh, tmp_path):
    await sh.write_file("f.py", "a = 1\na = 1\n")
    with pytest.raises(ValueError, match="matched 2 times"):
        # Two matches -> must error rather than guess.
        await sh.replace("f.py", "a = 1", "a = 2")


@pytest.mark.asyncio
@pytest.mark.parametrize("lines", [None, (2, 2)])
@pytest.mark.parametrize("replacement", ["return a + b", ""])
@pytest.mark.parametrize("keyword", [False, True])
async def test_replace_match_rejects_old_new_without_modifying_file(
    sh, tmp_path, lines, replacement, keyword
):
    """Both whole-file and sliced matches reject the path-form argument pattern."""
    original = "def calc(a, b):\n    return a * b\n"
    path = tmp_path / "calc.py"
    path.write_bytes(original.encode("utf-8"))
    match = await sh.read("calc.py", lines)
    with pytest.raises(ValueError) as error:
        if keyword:
            await sh.replace(match, "return a * b", new=replacement)
        else:
            await sh.replace(match, "return a * b", replacement)
    assert path.read_bytes() == original.encode("utf-8")
    assert "replace(match, new_text)" in str(error.value)
    assert "replace(path, old, new)" in str(error.value)
    assert "no file was changed" in str(error.value)


@pytest.mark.asyncio
async def test_replace_match_argument_guard_runs_before_file_access(sh, tmp_path):
    """An invalid call is rejected even when its old Match points to a missing file."""
    match = Match("missing.py", 1, 1, "old", resolved_path=tmp_path / "missing.py")
    with pytest.raises(ValueError, match="ambiguous"):
        await sh.replace(match, "old", "new")
    assert not (tmp_path / "missing.py").exists()


@pytest.mark.asyncio
async def test_write_file_is_overwrite(sh, tmp_path):
    await sh.write_file("f.txt", "old")
    await sh.write_file("f.txt", "new")
    assert (tmp_path / "f.txt").read_text() == "new"


@pytest.mark.asyncio
async def test_file_operations_allow_paths_outside_cwd(sh, tmp_path):
    sibling = tmp_path.parent / f"{tmp_path.name}-sibling"
    sibling.mkdir()
    relative = f"../{sibling.name}/relative.txt"
    absolute = sibling / "absolute.txt"

    await sh.write_file(relative, "one\ntwo\n")
    assert (await sh.read(relative)).text == "one\ntwo\n"

    await sh.replace(relative, "one", "changed")
    await sh.replace(
        Match(relative, 2, 2, "two\n", resolved_path=sibling / "relative.txt"), "replaced"
    )
    # The region reaches EOF in a file that ended with a newline, so the
    # replacement is re-terminated rather than stripping the final byte.
    assert (sibling / "relative.txt").read_text() == "changed\nreplaced\n"

    await sh.write_file(str(absolute), "absolute")
    assert (await sh.read(str(absolute))).text == "absolute"


@pytest.mark.asyncio
async def test_match_from_read_stays_bound_after_cwd_change(sh, tmp_path):
    original = tmp_path / "original.txt"
    original.write_text("before\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "original.txt").write_text("wrong file\n")

    match = await sh.read("original.txt")
    sliced = match[1:1]
    await sh.run("cd other")
    await sh.replace(sliced, "after")

    # Whole-file region at EOF in a newline-terminated file keeps its final
    # newline instead of silently stripping it.
    assert original.read_text() == "after\n"
    assert (other / "original.txt").read_text() == "wrong file\n"


@pytest.mark.asyncio
async def test_close_terminates_underlying_shell_session(sh):
    """Verify close() terminates the session and the shell lazily restarts."""
    r = await sh.run("echo started")
    assert r.success
    assert sh._session._process is not None

    await sh.close()

    assert sh._session._process is None
    assert not sh._session._started

    # The shell remains reusable after close(); a fresh session starts lazily.
    r2 = await sh.run("echo restarted")
    assert r2.success
    assert "restarted" in r2.stdout
    await sh.close()


def test_match_rejects_a_relative_resolved_path(tmp_path, monkeypatch):
    """The anchor must be absolute, or it silently binds to the process cwd.

    Path.resolve() on a relative path resolves against os.getcwd(), which is
    not the shell cwd — so a caller passing a relative path would produce a
    Match pointing at a different file, with no error. Every caller passes an
    absolute path today; this keeps that a rule rather than a convention.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        Match("f.txt", 1, 1, "hello\n", resolved_path="f.txt")


def test_match_keeps_an_absolute_resolved_path(tmp_path):
    """The supported form is unaffected."""
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    match = Match("f.txt", 1, 1, "hello\n", resolved_path=target)
    assert match.resolved_path == str(target.resolve())


@pytest.mark.asyncio
async def test_replace_match_at_eof_keeps_trailing_newline(sh, tmp_path):
    """A replacement reaching end-of-file must not strip the file's final newline."""
    await sh.write_file("f.py", "a = 1\nb = 2\n")
    match = await sh.read("f.py", (2, 2))
    await sh.replace(match, "b = 20")  # no trailing newline in the replacement
    assert (tmp_path / "f.py").read_text() == "a = 1\nb = 20\n"


@pytest.mark.asyncio
async def test_replace_match_at_eof_preserves_missing_newline(sh, tmp_path):
    """A file that genuinely lacks a final newline must not gain one."""
    await sh.write_file("f.py", "a = 1\nb = 2")  # no trailing newline
    match = await sh.read("f.py", (2, 2))
    await sh.replace(match, "b = 20")
    assert (tmp_path / "f.py").read_text() == "a = 1\nb = 20"


@pytest.mark.parametrize("absolute", [False, True])
async def test_run_with_cwd_runs_there_and_leaves_the_shell_directory(sh, tmp_path, absolute):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    home = sh.cwd
    target = str((tmp_path / "sub").resolve()) if absolute else "sub"
    try:
        r = await sh.run(_PWD_COMMAND, cwd=target)
        assert r.success
        assert r.stdout.strip() == str((tmp_path / "sub").resolve())
        # A cd inside the scoped command does not move the shell either.
        command = "cd deeper; (Get-Location).Path" if _IS_WINDOWS else "cd deeper && pwd"
        r = await sh.run(command, cwd=target)
        assert r.success
        assert r.stdout.strip().endswith("deeper")
        assert sh.cwd == home
        assert (await sh.run(_PWD_COMMAND)).stdout.strip() == str(home)
    finally:
        await sh.close()


async def test_run_with_a_missing_cwd_fails_without_running_the_command(sh, tmp_path):
    try:
        command = (
            "New-Item -ItemType File marker; Write-Output 'ran'"
            if _IS_WINDOWS
            else "touch marker; echo ran"
        )
        r = await sh.run(command, cwd="missing")
        assert not r.success
        assert "missing" in r.stderr
        assert "ran" not in r.stdout
        assert not (tmp_path / "marker").exists()
        assert (await sh.run(_PWD_COMMAND)).stdout.strip() == str(sh.cwd)
    finally:
        await sh.close()


async def test_run_with_cwd_keeps_environment_changes(sh, tmp_path):
    """``cwd=`` runs in the session's own shell, not a subshell: exports persist."""
    (tmp_path / "sub").mkdir()
    try:
        command = "$env:ASTER_CWD_PROBE = 'kept'" if _IS_WINDOWS else "export ASTER_CWD_PROBE=kept"
        assert (await sh.run(command, cwd="sub")).success
        command = "Write-Output $env:ASTER_CWD_PROBE" if _IS_WINDOWS else "echo $ASTER_CWD_PROBE"
        assert (await sh.run(command)).stdout.strip() == "kept"
    finally:
        await sh.close()


async def test_run_with_cwd_returns_the_command_status(sh, tmp_path):
    (tmp_path / "sub").mkdir()
    try:
        r = await sh.run(_python_command("import sys; sys.exit(1)"), cwd="sub")
        assert r.returncode == 1
        command = (
            _python_command("import sys; sys.exit(7)")
            if _IS_WINDOWS
            else "exit_code() { return 7; }; exit_code"
        )
        r = await sh.run(command, cwd="sub")
        assert r.returncode == 7
        assert (await sh.run(_PWD_COMMAND)).stdout.strip() == str(sh.cwd)
    finally:
        await sh.close()


@pytest.mark.skipif(_IS_WINDOWS, reason="CDPATH is a POSIX shell feature")
async def test_run_with_a_relative_cwd_ignores_cdpath(sh, tmp_path):
    """A relative ``cwd`` is the shell directory's child, whatever CDPATH says."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "elsewhere" / "sub").mkdir(parents=True)
    try:
        await sh.run(f"export CDPATH={shlex.quote(str(tmp_path / 'elsewhere'))}")
        r = await sh.run("pwd", cwd="sub")
        assert r.stdout == str((tmp_path / "sub").resolve())
    finally:
        await sh.close()


async def test_run_with_stdin_keeps_shell_changes(sh, tmp_path):
    """``stdin=`` runs in the session's own shell, not a subshell: cd and exports persist."""
    (tmp_path / "sub").mkdir()
    try:
        command = (
            "$env:ASTER_STDIN_PROBE = [Console]::In.ReadLine()"
            if _IS_WINDOWS
            else "read value; export ASTER_STDIN_PROBE=$value"
        )
        r = await sh.run(command, stdin="kept\n")
        assert r.success
        command = (
            "Write-Output $env:ASTER_STDIN_PROBE" if _IS_WINDOWS else "echo $ASTER_STDIN_PROBE"
        )
        assert (await sh.run(command)).stdout.strip() == "kept"
        await sh.run("cd sub", stdin="ignored\n")
        assert sh.cwd == (tmp_path / "sub").resolve()
    finally:
        await sh.close()


async def test_run_with_stdin_and_cwd_returns_to_the_shell_directory(sh, tmp_path):
    (tmp_path / "sub").mkdir()
    try:
        home = sh.cwd
        command = (
            "[Console]::Write([Console]::In.ReadToEnd()); (Get-Location).Path"
            if _IS_WINDOWS
            else "cat; pwd"
        )
        r = await sh.run(command, stdin="line\n", cwd="sub")
        assert r.success
        assert r.stdout.splitlines() == ["line", str((tmp_path / "sub").resolve())]
        assert sh.cwd == home
        assert (
            await sh.run(_python_command("import sys; sys.exit(1)"), stdin="x\n")
        ).returncode == 1
        # The session keeps reading its own commands after the redirection ends.
        command = "Write-Output 'still here'" if _IS_WINDOWS else "echo still here"
        assert (await sh.run(command)).stdout.strip() == "still here"
    finally:
        await sh.close()


async def test_run_stream_with_cwd_runs_there(sh, tmp_path):
    (tmp_path / "sub").mkdir()
    try:
        events = [event async for event in sh.run_stream(_PWD_COMMAND, cwd="sub")]
        out = "".join(event.text for event in events if event.kind == "stdout")
        assert out.strip() == str((tmp_path / "sub").resolve())
        assert events[-1].returncode == 0
    finally:
        await sh.close()


def test_run_documents_cwd_for_the_model():
    from aster.agentdoc import doc

    rendered = doc(ShellTools.run)
    assert "cwd: str | Path | None = None" in rendered
    assert "cwd: Directory for this command only" in rendered
    assert "run(command, stdin=, timeout=, cwd=)" in doc(ShellTools)
