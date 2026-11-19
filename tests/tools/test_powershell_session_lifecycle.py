# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PowerShell command serialization across restarts and lifecycle operations."""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace

import pytest

from aster.tools._powershell_session import PowerShellSession


@pytest.mark.parametrize("entrypoint", ["run", "run_with_timeout_flag", "run_stream"])
async def test_loop_change_keeps_commands_serialized(tmp_path, monkeypatch, entrypoint):
    """A second caller must wait while the first command restarts its shell."""
    session = PowerShellSession(cwd=tmp_path)
    session._started = True
    session._started_on_loop = object()
    command_entered = asyncio.Event()
    allow_command = asyncio.Event()
    active = peak = 0

    async def execute(command, timeout, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        command_entered.set()
        await allow_command.wait()
        session._started_on_loop = asyncio.get_running_loop()
        active -= 1
        return command, "", 0, False

    async def invoke(command):
        if entrypoint == "run_stream":
            return [chunk async for chunk in session.run_stream(command)]
        return await getattr(session, entrypoint)(command)

    monkeypatch.setattr(session, "_run", execute)
    first = asyncio.create_task(invoke("first"))
    await asyncio.wait_for(command_entered.wait(), timeout=1)
    second = asyncio.create_task(invoke("second"))
    try:
        # run_stream schedules its command task one turn after taking the lock.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        entered_concurrently = peak
    finally:
        allow_command.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True), timeout=1
        )

    assert entered_concurrently == 1
    if entrypoint == "run_stream":
        assert results == [[("__done__", "0,0")], [("__done__", "0,0")]]
    elif entrypoint == "run_with_timeout_flag":
        assert results == [("first", "", 0, False), ("second", "", 0, False)]
    else:
        assert results == [("first", "", 0), ("second", "", 0)]


@pytest.mark.parametrize("channel", ["stdout", "stderr"])
async def test_replaced_shell_cannot_write_into_current_output(tmp_path, channel):
    session = PowerShellSession(cwd=tmp_path)
    previous_reader, current_reader = asyncio.StreamReader(), asyncio.StreamReader()
    session._process = SimpleNamespace(**{channel: previous_reader})
    session._buffers = {channel: StringIO()}
    session._events = asyncio.Queue()
    previous_drain = asyncio.create_task(session._drain(channel, previous_reader))
    try:
        await asyncio.sleep(0)
        session._process = SimpleNamespace(**{channel: current_reader})
        previous_reader.feed_data(b"previous shell output")
        previous_reader.feed_eof()
        await asyncio.wait_for(previous_drain, timeout=1)
        assert previous_reader.at_eof()  # Preserved pipes still need draining.
        assert session._buffers[channel].getvalue() == ""
        assert session._events.empty()

        current_reader.feed_data(b"current shell output")
        current_reader.feed_eof()
        await session._drain(channel, current_reader)
        assert session._buffers[channel].getvalue() == "current shell output"
        assert session._events.get_nowait() == (channel, "current shell output")
    finally:
        previous_drain.cancel()
        await asyncio.gather(previous_drain, return_exceptions=True)
        session._process = None
        session._events = None


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
@pytest.mark.parametrize("operation", ["start", "reset", "close"])
@pytest.mark.parametrize("streaming", [False, True], ids=["run", "run_stream"])
async def test_lifecycle_waits_for_running_command(tmp_path, monkeypatch, operation, streaming):
    """Public lifecycle operations cannot replace or stop an active command."""
    session = PowerShellSession(cwd=tmp_path)
    await session.start()
    process = session._process
    reader = session._control_reader
    read_response = reader.readline
    command_entered = asyncio.Event()
    allow_response = asyncio.Event()

    async def readline():
        command_entered.set()
        await allow_response.wait()
        return await read_response()

    async def execute():
        if streaming:
            return [chunk async for chunk in session.run_stream("Write-Output 'first'")]
        return await session.run("Write-Output 'first'")

    monkeypatch.setattr(reader, "readline", readline)
    first = asyncio.create_task(execute())
    await asyncio.wait_for(command_entered.wait(), timeout=3)
    lifecycle = asyncio.create_task(getattr(session, operation)())
    try:
        await asyncio.sleep(0)
        preserved_process = session._process is process
        lifecycle_waited = not lifecycle.done()
    finally:
        allow_response.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, lifecycle, return_exceptions=True), timeout=10
        )
        await session.close()

    assert preserved_process
    assert lifecycle_waited
    assert results[1] is None
    if streaming:
        assert results[0][-1] == ("__done__", "0,0")
        assert "".join(chunk for name, chunk in results[0] if name == "stdout") == "first\r\n"
        assert not any(name == "stderr" for name, _ in results[0])
    else:
        assert results[0] == ("first\r\n", "", 0)


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
async def test_startup_command_and_reset_do_not_reenter_lock(tmp_path):
    session = PowerShellSession(cwd=tmp_path, init_command="$answer = 42")
    try:
        for _ in range(2):
            assert await asyncio.wait_for(session.run("Write-Output $answer"), timeout=10) == (
                "42\r\n",
                "",
                0,
            )
            await asyncio.wait_for(session.reset(), timeout=10)
    finally:
        await session.close()


@contextmanager
def _retained_process(pid):
    import _winapi

    handle = _winapi.OpenProcess(_winapi.SYNCHRONIZE | 0x0001, False, pid)
    try:
        yield _winapi, handle
    finally:
        try:
            if _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_TIMEOUT:
                _winapi.TerminateProcess(handle, 1)
                _winapi.WaitForSingleObject(handle, 5000)
        finally:
            _winapi.CloseHandle(handle)


async def _start_native_background(session, directory):
    worker = directory / "background worker.py"
    worker.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "release = Path(__file__).with_suffix('.go')\n"
        "while not release.exists(): time.sleep(0.01)\n"
        "print('previous shell stdout', flush=True)\n"
        "print('previous shell stderr', file=sys.stderr, flush=True)\n"
        "Path(__file__).with_suffix('.done').write_text('survived', encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    def quoted(value):
        return "'" + str(value).replace("'", "''") + "'"

    arguments = f'"{worker}"'
    command = (
        f"$background = Start-Process -FilePath {quoted(sys.executable)} "
        f"-ArgumentList {quoted(arguments)} "
        "-PassThru -WindowStyle Hidden "
        f"-RedirectStandardOutput {quoted(directory / 'background.out')} "
        f"-RedirectStandardError {quoted(directory / 'background.err')}; "
        "Write-Output $background.Id"
    )
    try:
        stdout, stderr, code = await session.run(command)
    except Exception as error:
        captured = {name: buffer.getvalue() for name, buffer in session._buffers.items()}
        raise AssertionError(f"Background startup failed: {captured}") from error
    assert code == 0, (stdout, stderr, code)
    return int(stdout.strip())


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows Job lifecycle")
@pytest.mark.parametrize("keep", [False, True], ids=["stop", "keep"])
async def test_close_handles_native_background_processes(tmp_path, keep):
    session = PowerShellSession(cwd=tmp_path, keep_background_on_close=keep)
    try:
        pid = await _start_native_background(session, tmp_path)
        with _retained_process(pid) as (kernel, handle):
            assert kernel.WaitForSingleObject(handle, 0) == kernel.WAIT_TIMEOUT
            await session.close()
            if keep:
                assert kernel.WaitForSingleObject(handle, 0) == kernel.WAIT_TIMEOUT
                (tmp_path / "background worker.go").touch()
                marker = tmp_path / "background worker.done"
                async with asyncio.timeout(5):
                    while not marker.exists():
                        await asyncio.sleep(0.01)
                assert marker.read_text(encoding="utf-8") == "survived"
            else:
                assert kernel.WaitForSingleObject(handle, 5000) == kernel.WAIT_OBJECT_0
    finally:
        await session.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows Job lifecycle")
@pytest.mark.parametrize("streaming", [False, True], ids=["run", "run_stream"])
async def test_reopened_session_keeps_preserved_background_running(tmp_path, streaming):
    session = PowerShellSession(cwd=tmp_path, keep_background_on_close=True)
    try:
        pid = await _start_native_background(session, tmp_path)
        with _retained_process(pid) as (kernel, handle):
            await session.close()
            assert kernel.WaitForSingleObject(handle, 0) == kernel.WAIT_TIMEOUT
            command = (
                "New-Item -ItemType File 'background worker.go' | Out-Null; "
                "while (-not (Test-Path 'background worker.done')) { Start-Sleep -Milliseconds 10 }; "
                "Start-Sleep -Milliseconds 200; Write-Output 'new session'"
            )
            if streaming:
                events = [event async for event in session.run_stream(command, timeout=10)]
                assert events[-1] == ("__done__", "0,0")
                stdout = "".join(value for name, value in events if name == "stdout")
                stderr = "".join(value for name, value in events if name == "stderr")
            else:
                stdout, stderr, code = await session.run(command, timeout=10)
                assert code == 0
            assert stdout.splitlines() == ["new session"]
            assert stderr == ""
            assert (tmp_path / "background.out").read_text() == "previous shell stdout\n"
            assert (tmp_path / "background.err").read_text() == "previous shell stderr\n"
            assert kernel.WaitForSingleObject(handle, 0) == kernel.WAIT_TIMEOUT
    finally:
        await session.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows Job lifecycle")
@pytest.mark.parametrize("kill_fails", [False, True], ids=["force-shell", "failure-cleans-tree"])
async def test_keep_close_owns_tree_until_shutdown_finishes(tmp_path, monkeypatch, kill_fails):
    session = PowerShellSession(cwd=tmp_path, keep_background_on_close=True)
    process = None
    try:
        pid = await _start_native_background(session, tmp_path)
        process = session._process
        with _retained_process(pid) as (kernel, handle):
            real_timeout = asyncio.timeout
            first_wait = True

            def short_grace_timeout(delay):
                nonlocal first_wait
                if first_wait:
                    first_wait = False
                    return real_timeout(0.1)
                return real_timeout(delay)

            with monkeypatch.context() as patch:
                # Fault injection: the shell never receives close or stdin EOF.
                patch.setattr(process.stdin, "write", lambda _: None)
                patch.setattr(process.stdin, "close", lambda: None)
                patch.setattr(asyncio, "timeout", short_grace_timeout)
                if kill_fails:

                    def failed_kill():
                        raise OSError("forced shutdown failed")

                    patch.setattr(process, "kill", failed_kill)
                    with pytest.raises(OSError, match="forced shutdown failed"):
                        await session.close()
                else:
                    await session.close()
            if kill_fails:
                assert kernel.WaitForSingleObject(handle, 5000) == kernel.WAIT_OBJECT_0
                await asyncio.wait_for(process.wait(), timeout=5)
            else:
                assert kernel.WaitForSingleObject(handle, 0) == kernel.WAIT_TIMEOUT
            assert process.returncode is not None
            assert session._process is None
            assert session._job is None
            assert session._control_writer is None
    finally:
        if process is not None:
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.wait(), timeout=5)
        await session.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows Job lifecycle")
async def test_timeout_stops_native_background_tree_and_recovers(tmp_path):
    session = PowerShellSession(cwd=tmp_path, keep_background_on_close=True)
    try:
        pid = await _start_native_background(session, tmp_path)
        with _retained_process(pid) as (kernel, handle):
            _, _, code, timed_out = await session.run_with_timeout_flag(
                "Start-Sleep -Seconds 60", timeout=0.2
            )
            assert code == 124 and timed_out
            assert kernel.WaitForSingleObject(handle, 5000) == kernel.WAIT_OBJECT_0
            out, err, code = await session.run("Write-Output 'recovered'")
            assert out.splitlines() == ["recovered"]
            assert (err, code) == ("", 0)
    finally:
        await session.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
async def test_failed_startup_command_cleans_up_without_reentering_lock(tmp_path):
    session = PowerShellSession(cwd=tmp_path, init_command="Write-Error 'setup failed'")
    try:
        with pytest.raises(RuntimeError, match="Shell init command failed"):
            await asyncio.wait_for(session.run("Write-Output 'unreachable'"), timeout=10)
        assert session._process is None
        assert session._job is None
        session._init_command = "$answer = 42"
        assert await asyncio.wait_for(session.run("Write-Output $answer"), timeout=10) == (
            "42\r\n",
            "",
            0,
        )
    finally:
        await session.close()
