# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live Python backend changes prepare, commit, and cancel through the ACP boundary."""

from __future__ import annotations

import asyncio
import json
import platform
import shlex
import sys
from contextlib import asynccontextmanager
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp import RequestError, text_block
from aster_acp.server import CodingACPAdapter

from aster.events import ExecutionResult, PythonOutput
from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall, registry


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("ASTER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("ASTER_SETTINGS", "ASTER_MODEL", "ASTER_LLM_CONFIG", "ASTER_RICH_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})


@pytest.fixture
def executors(monkeypatch):
    created = []

    def create(agent, config, **kwargs):
        executor = SimpleNamespace(
            config=config,
            options=kwargs,
            degraded_guards=(),
            run_cell=AsyncMock(return_value=ExecutionResult()),
            aclose=AsyncMock(),
        )
        created.append(executor)
        return executor

    monkeypatch.setattr("aster.runtime.sandbox.executor.SandboxedExecutor", create)
    return created


def _reply(code=None):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="execute" if code else "done",
                name="execute_python" if code else "return_result",
                arguments=json.dumps(
                    {"code": code}
                    if code
                    else {"result": {"message": "Finished.", "explanation": "complete"}}
                ),
            )
        ],
    )


@asynccontextmanager
async def _session(tmp_path, *responses, enabled=True, timeout=15):
    llm = FakeLLMClient(responses, strict_exhaustion=True)
    adapter = CodingACPAdapter(lambda: llm)
    client = SimpleNamespace(session_update=AsyncMock())
    adapter.on_connect(client)
    await adapter.initialize(1, aster={"runtimeControls": enabled})
    new = await adapter.new_session(cwd=str(tmp_path), mcp_servers=[])
    session = (await adapter._sessions.get(new.session_id)).value
    session.dispatcher._strategy.config = session.dispatcher._strategy.config.model_copy(
        update={"prefill": None}
    )
    try:
        async with asyncio.timeout(timeout):
            yield adapter, session, client
    finally:
        await adapter.cancel(new.session_id)
        await adapter.close()


async def _configure(adapter, session, **changes):
    return await adapter.ext_method(
        "aster/session/configure", {"sessionId": session.handle.id, **changes}
    )


async def test_switch_routes_next_prompt_and_disable_restores_native_execution(tmp_path, executors):
    target = tmp_path / "host-only.txt"
    code = f"open({str(target)!r}, 'w').write('native')"
    async with _session(tmp_path, _reply(code), _reply(), _reply(code), _reply()) as (
        adapter,
        session,
        client,
    ):
        state = session.controls.status()
        assert state["executionBackend"] == "native"
        assert state["executionBackends"] == ["native", "sandbox"]
        await _configure(adapter, session, permissionMode="auto")
        changed = await _configure(adapter, session, executionBackend="sandbox")
        assert changed["executionBackend"] == "sandbox"
        assert changed["permissionMode"] == "auto"
        probe = executors[0]
        probe.run_cell.assert_awaited_once_with("pass")
        probe.aclose.assert_awaited_once()
        assert probe.config.workspace == str(tmp_path.resolve())
        assert probe.config.require is True

        result = await adapter.prompt(session.handle.id, [text_block("Run in the sandbox")])
        assert result.stop_reason == "end_turn"
        assert not target.exists()
        assert len(executors) == 2
        execution = executors[1]
        assert execution.config == probe.config
        assert any(call.args[0] == code for call in execution.run_cell.await_args_list)
        execution.aclose.assert_awaited_once()

        changed = await _configure(adapter, session, executionBackend="native")
        assert changed["executionBackend"] == "native"
        result = await adapter.prompt(session.handle.id, [text_block("Run on the host")])
        assert result.stop_reason == "end_turn"
        assert target.read_text() == "native"
        assert len(executors) == 2
        assert session.handle.info.turn_count == 2


async def test_selecting_current_backend_does_not_probe(tmp_path, executors):
    async with _session(tmp_path) as (adapter, session, client):
        previous = session.dispatcher._strategy
        await _configure(adapter, session, executionBackend="native")
        assert session.dispatcher._strategy is previous
        assert not executors
        await _configure(adapter, session, executionBackend="sandbox")
        previous = session.dispatcher._strategy
        await _configure(adapter, session, executionBackend="sandbox")
        assert session.dispatcher._strategy is previous
        assert len(executors) == 1


@pytest.mark.parametrize("failure", ["construct", "result", "raise", "close"])
async def test_failed_preparation_keeps_previous_backend(tmp_path, monkeypatch, failure):
    executor = SimpleNamespace(
        run_cell=AsyncMock(return_value=ExecutionResult()), aclose=AsyncMock()
    )
    error = RuntimeError("microVM unavailable")
    if failure == "result":
        executor.run_cell.return_value = ExecutionResult(error=error)
    elif failure == "raise":
        executor.run_cell.side_effect = error
    else:
        executor.aclose.side_effect = error

    def create(*args, **kwargs):
        if failure == "construct":
            raise OSError("microVM unavailable")
        return executor

    monkeypatch.setattr("aster.runtime.sandbox.executor.SandboxedExecutor", create)
    async with _session(tmp_path, _reply()) as (adapter, session, client):
        previous = session.dispatcher._strategy
        with pytest.raises(RequestError) as caught:
            await _configure(adapter, session, executionBackend="sandbox")
        assert "microVM unavailable" in caught.value.data["reason"]
        assert session.dispatcher._strategy is previous
        assert session.controls.status()["executionBackend"] == "native"
        assert executor.aclose.await_count == (0 if failure == "construct" else 1)
        assert (
            await adapter.prompt(session.handle.id, [text_block("Still usable")])
        ).stop_reason == ("end_turn")


@pytest.mark.parametrize("cancel_via", ["session", "request"])
async def test_cancelled_preparation_keeps_previous_backend(tmp_path, monkeypatch, cancel_via):
    entered = asyncio.Event()

    async def prepare(code):
        entered.set()
        await asyncio.Event().wait()

    executor = SimpleNamespace(run_cell=prepare, aclose=AsyncMock())
    monkeypatch.setattr(
        "aster.runtime.sandbox.executor.SandboxedExecutor", lambda *a, **k: executor
    )
    async with _session(tmp_path, _reply()) as (adapter, session, client):
        previous = session.dispatcher._strategy
        change = asyncio.create_task(_configure(adapter, session, executionBackend="sandbox"))
        await entered.wait()
        assert session.controls.status()["executionBackend"] == "native"
        with pytest.raises(RequestError):
            await adapter.prompt(session.handle.id, [text_block("Must wait")])
        with pytest.raises(RequestError):
            await _configure(adapter, session, executionBackend="native")
        if cancel_via == "session":
            await adapter.cancel(session.handle.id)
            with pytest.raises(RequestError) as caught:
                await change
            assert "cancelled" in caught.value.data["reason"]
        else:
            change.cancel()
            with pytest.raises(asyncio.CancelledError):
                await change
        assert session.dispatcher._strategy is previous
        assert session.controls.status()["executionBackend"] == "native"
        assert not session.dispatcher.active
        executor.aclose.assert_awaited_once()
        assert (
            await adapter.prompt(session.handle.id, [text_block("Still usable")])
        ).stop_reason == ("end_turn")


async def test_background_job_prevents_switching(tmp_path, executors):
    async with _session(tmp_path) as (adapter, session, client):
        session.agent.queue_manager.queue("work")
        job = session.agent.queue_manager.spawn(asyncio.Event().wait(), channel="work")
        with pytest.raises(RequestError) as caught:
            await _configure(adapter, session, executionBackend="sandbox")
        assert "background jobs" in caught.value.data["reason"]
        assert session.controls.status()["executionBackend"] == "native"
        assert not executors
        await session.agent.queue_manager.shutdown()
        assert job.state == "cancelled"


async def test_cancelled_switch_retains_turn_until_cleanup_finishes(tmp_path, monkeypatch):
    probe_entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def prepare(code):
        probe_entered.set()
        await asyncio.Event().wait()

    executor = SimpleNamespace(run_cell=prepare, aclose=AsyncMock())
    monkeypatch.setattr(
        "aster.runtime.sandbox.executor.SandboxedExecutor", lambda *a, **k: executor
    )
    async with _session(tmp_path, _reply()) as (adapter, session, client):
        shutdown = session.agent.queue_manager.shutdown

        async def delayed_shutdown():
            cleanup_entered.set()
            await release_cleanup.wait()
            await shutdown()

        monkeypatch.setattr(session.agent.queue_manager, "shutdown", delayed_shutdown)
        change = asyncio.create_task(_configure(adapter, session, executionBackend="sandbox"))
        await probe_entered.wait()
        cancel = asyncio.create_task(adapter.cancel(session.handle.id))
        try:
            await cleanup_entered.wait()
            done, _ = await asyncio.wait({change}, timeout=0.02)
            assert not done
            with pytest.raises(RequestError):
                await adapter.prompt(session.handle.id, [text_block("Still cleaning up")])
        finally:
            release_cleanup.set()
            await cancel
        with pytest.raises(RequestError) as caught:
            await change
        assert "cancelled" in caught.value.data["reason"]
        assert not session.dispatcher.active
        assert (await adapter.prompt(session.handle.id, [text_block("Now ready")])).stop_reason == (
            "end_turn"
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"executionBackend": "host"},
        {"executionBackend": ""},
        {"executionBackend": None},
        {"executionBackend": True},
        {"executionBackend": []},
        {"executionBackend": "sandbox", "permissionMode": "auto"},
        {"executionBackend": "sandbox", "reasoningLevel": None},
    ],
)
async def test_invalid_backend_requests_do_not_mutate_settings(tmp_path, executors, changes):
    async with _session(tmp_path) as (adapter, session, client):
        previous = session.dispatcher._strategy
        with pytest.raises(RequestError):
            await _configure(adapter, session, **changes)
        assert session.dispatcher._strategy is previous
        assert session.controls.status()["executionBackend"] == "native"
        assert session.controls.status()["permissionMode"] == "manual"
        assert not executors


async def test_session_has_execution_controls_without_negotiation(tmp_path, executors):
    async with _session(tmp_path, enabled=False) as (adapter, session, client):
        assert session.agent._execution_control is session.controls
        assert session.controls.approval == "ask"
        assert session.controls.execution_backend == "native"
        assert not executors


@pytest.mark.sandbox
async def test_real_microvm_switch_roundtrip(tmp_path):
    guest = "import platform\nassert platform.system() == 'Linux'\nprint('sandbox verified')"
    host = (
        f"import platform\nassert platform.system() == {platform.system()!r}\n"
        "print('native verified')"
    )
    async with _session(tmp_path, _reply(guest), _reply(), _reply(host), _reply(), timeout=240) as (
        adapter,
        session,
        client,
    ):
        await _configure(adapter, session, permissionMode="auto")
        state = await _configure(adapter, session, executionBackend="sandbox")
        assert state["executionBackend"] == "sandbox"
        assert (
            await adapter.prompt(session.handle.id, [text_block("Verify Linux")])
        ).stop_reason == ("end_turn")
        state = await _configure(adapter, session, executionBackend="native")
        assert state["executionBackend"] == "native"
        assert (
            await adapter.prompt(session.handle.id, [text_block("Verify host")])
        ).stop_reason == ("end_turn")
        outputs = [
            event
            for event in session.agent.event_manager.all_events()
            if isinstance(event, PythonOutput) and event.tool_call_id == "execute"
        ]
        assert len(outputs) == 2
        assert all(not output.error for output in outputs)
        assert [output.stdout.strip() for output in outputs] == [
            "sandbox verified",
            "native verified",
        ]


@pytest.mark.sandbox
async def test_real_microvm_isolates_cells_but_host_tools_keep_host_access(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "host-only.txt"
    outside.write_text("test canary", encoding="utf-8")
    broker_output = tmp_path / "host-tool-output.txt"
    monkeypatch.setenv("ASTER_SANDBOX_TEST_CANARY", "host environment canary")

    async def respond(reader, writer):
        try:
            await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\nConnection: close\r\n\r\nsandbox-probe"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(respond, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        network_probe = (
            "import urllib.request; "
            f"print(urllib.request.urlopen('http://127.0.0.1:{port}/', timeout=5).read().decode())"
        )
        if sys.platform == "win32":
            command = "& " + " ".join(
                "'" + argument.replace("'", "''") + "'"
                for argument in (sys.executable, "-c", network_probe)
            )
        else:
            command = shlex.join((sys.executable, "-c", network_probe))
        child_probe = dedent("""\
            import socket
            for family in (socket.AF_INET, socket.AF_INET6):
                for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
                    try:
                        with socket.socket(family, kind):
                            pass
                    except PermissionError as error:
                        assert error.errno == 13
                    else:
                        raise AssertionError('child process opened an internet socket')
            print('blocked')
            """)
        code = dedent(f"""\
            import asyncio, errno, json, os, platform, sys
            from pathlib import Path
            report = {{'platform': platform.system(), 'uid': os.getuid()}}
            report['host_environment_hidden'] = 'ASTER_SANDBOX_TEST_CANARY' not in os.environ
            Path('guest-output.txt').write_text('guest write', encoding='utf-8')
            report['workspace_write'] = Path('guest-output.txt').read_text() == 'guest write'
            for name, target in (
                ('host_file_hidden', Path({str(outside)!r})),
                ('parent_traversal_blocked', Path.cwd().parent / 'host-only.txt'),
            ):
                try:
                    target.read_text()
                except (PermissionError, FileNotFoundError):
                    report[name] = True
                else:
                    report[name] = False
            try:
                with open('/etc/hosts', 'r+b'):
                    pass
            except OSError as error:
                report['system_write_blocked'] = error.errno in (errno.EACCES, errno.EROFS)
            else:
                report['system_write_blocked'] = False
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection('127.0.0.1', {port}), timeout=5)
            except PermissionError as error:
                report['direct_network_blocked'] = error.errno == errno.EACCES
            else:
                writer.close()
                await writer.wait_closed()
                report['direct_network_blocked'] = False
            child = await asyncio.create_subprocess_exec(sys.executable, '-c', {child_probe!r}, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            child_stdout, child_stderr = await asyncio.wait_for(child.communicate(), timeout=5)
            report['child_network_blocked'] = child.returncode == 0 and child_stdout.strip() == b'blocked'
            host_file = await self.shell.read({str(outside)!r})
            report['host_tool_reads_outside'] = host_file.text.strip() == 'test canary'
            await self.shell.write_file({str(broker_output)!r}, 'host tool write')
            response = await self.shell.run({command!r}, timeout=15)
            report['host_tool_network_works'] = response.returncode == 0 and 'sandbox-probe' in response.stdout
            print('SANDBOX_REPORT ' + json.dumps(report, sort_keys=True))
            """)
        async with _session(workspace, _reply(code), _reply(), timeout=240) as (
            adapter,
            session,
            client,
        ):
            assert session.dispatcher._strategy.config.sandbox.network is False
            await _configure(adapter, session, permissionMode="auto")
            await _configure(adapter, session, executionBackend="sandbox")
            result = await adapter.prompt(
                session.handle.id, [text_block("Probe sandbox boundaries")]
            )
            assert result.stop_reason == "end_turn"
            outputs = [
                event
                for event in session.agent.event_manager.all_events()
                if isinstance(event, PythonOutput) and event.tool_call_id == "execute"
            ]
            assert len(outputs) == 1
            assert not outputs[0].error, outputs[0].error
            report = json.loads(outputs[0].stdout.split("SANDBOX_REPORT ", 1)[1])
            assert report.pop("platform") == "Linux"
            assert report.pop("uid") == 1000
            assert all(value is True for value in report.values()), report
            assert (workspace / "guest-output.txt").read_text() == "guest write"
            assert outside.read_text() == "test canary"
            assert broker_output.read_text() == "host tool write"
            print(json.dumps(report, sort_keys=True))
