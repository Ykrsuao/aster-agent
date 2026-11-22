# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ACP JSON-RPC subprocess test."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.connection import StreamDirection
from acp.schema import (
    AgentMessageChunk,
    AvailableCommandsUpdate,
    ContentToolCallContent,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

# Bounds a hang, not the expected duration. Spawning an interpreter and running
# a turn takes well under a second here, but a loaded CI runner is a different
# machine — and a flaky test gets deleted, which is worse than a slow one. A
# real deadlock still fails, just later.
_HANG_TIMEOUT = 30


@pytest.mark.parametrize("entrypoint", ["serve", "cli"])
@pytest.mark.parametrize("prepare", [False, True])
async def test_provider_diagnostics_do_not_corrupt_protocol_stdout(
    tmp_path, monkeypatch, entrypoint, prepare
):
    """Inspect raw wire bytes: the SDK otherwise logs and skips corrupt lines."""
    from acp.transports import default_environment

    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    arguments = [sys.executable, str(fixture), "--noisy"]
    if entrypoint == "cli":
        arguments.append("--cli")
    process = await asyncio.create_subprocess_exec(
        *arguments,
        cwd=tmp_path,
        env=default_environment(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    diagnostics = asyncio.create_task(process.stderr.read())

    async def request(request_id, method, params):
        process.stdin.write(
            (
                json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
                + "\n"
            ).encode()
        )
        await process.stdin.drain()
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=_HANG_TIMEOUT)
            assert line, (await diagnostics).decode("utf-8", errors="replace")
            message = json.loads(line)
            if message.get("method") == "session/request_permission":
                answer = {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
                reply = {"jsonrpc": "2.0", "id": message["id"], "result": answer}
                process.stdin.write((json.dumps(reply) + "\n").encode())
                await process.stdin.drain()
                continue
            if message.get("id") == request_id:
                assert "error" not in message, message
                return message["result"]

    try:
        await request(1, "initialize", {"protocolVersion": PROTOCOL_VERSION})
        session = await request(
            2,
            "session/new",
            {"cwd": str(tmp_path), "mcpServers": [], "_meta": {"aster": {"prepare": prepare}}},
        )
        before_prompt = await request(20, "session/list", {"cwd": str(tmp_path)})
        assert [item["sessionId"] for item in before_prompt["sessions"]] == (
            [] if prepare else [session["sessionId"]]
        )
        response = await request(
            3,
            "session/prompt",
            {
                "sessionId": session["sessionId"],
                "prompt": [{"type": "text", "text": "run smoke test"}],
            },
        )
        assert response["stopReason"] == "end_turn"
        await request(4, "session/close", {"sessionId": session["sessionId"]})
        history = await request(
            5,
            "session/list",
            {"cwd": str(tmp_path), "_meta": {"aster": {"includeInputHistory": True}}},
        )
        assert history["_meta"]["aster"]["inputHistory"] == ["run smoke test"]
        assert [item["sessionId"] for item in history["sessions"]] == [session["sessionId"]]
        process.stdin.close()
        await asyncio.wait_for(process.wait(), timeout=_HANG_TIMEOUT)
        # Every remaining notification must also be a complete JSON-RPC line.
        for line in (await process.stdout.read()).splitlines():
            json.loads(line)
        stderr = (await diagnostics).decode("utf-8", errors="replace")
        assert "PROVIDER_INITIALIZATION_DIAGNOSTIC" in stderr
        assert "PROVIDER_REQUEST_DIAGNOSTIC" in stderr
        if entrypoint == "cli":
            assert "CONFIGURATION_DIAGNOSTIC" in stderr
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
        await diagnostics


def _is_prompt_activity(update: object) -> bool:
    return isinstance(update, ToolCallStart | ToolCallProgress) and update.tool_call_id.startswith(
        "aster-progress-"
    )


def test_protocol_subprocess_imports_this_checkout(tmp_path):
    import json
    import subprocess

    from acp.transports import default_environment

    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, aster, aster_acp; print(json.dumps([aster.__file__, aster_acp.__file__]))",
        ],
        cwd=tmp_path,
        env=default_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert [Path(path).resolve() for path in json.loads(result.stdout)] == [
        root / "src/aster/__init__.py",
        root / "packages/aster-acp/src/aster_acp/__init__.py",
    ]


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.tool_started = asyncio.Event()
        self.commands_updated = asyncio.Event()
        self.sandbox_ready = asyncio.Event()
        self.preparing = asyncio.Event()
        self.waiting_for_model = asyncio.Event()
        self.smoke_received = asyncio.Event()
        self.response_ready = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        )

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))
        if isinstance(update, ToolCallStart) and not _is_prompt_activity(update):
            self.tool_started.set()
        if isinstance(update, ToolCallStart | ToolCallProgress) and _is_prompt_activity(update):
            if update.title == "Preparing execution environment":
                self.preparing.set()
            if update.title == "Waiting for model":
                self.waiting_for_model.set()
            if update.title == "Response ready" and update.status == "completed":
                self.response_ready.set()
            if update.title == "Cancelled" and update.status == "failed":
                self.cancelled.set()
        if isinstance(update, AvailableCommandsUpdate):
            self.commands_updated.set()
        if isinstance(update, AgentMessageChunk) and update.content.text == "Sandbox ready":
            self.sandbox_ready.set()
        if (
            isinstance(update, AgentMessageChunk)
            and update.content.text == "ASTER ACP smoke test passed."
        ):
            self.smoke_received.set()


def _mcp_stdio_config(workspace: Path, *, label: str = "project"):
    fixture = Path(__file__).parent / "fixtures" / "controlled_mcp.py"
    control = workspace / f"{label}-control.json"
    events = workspace / f"{label}-events.jsonl"
    control.write_text(json.dumps({"label": label}), encoding="utf-8")
    return (
        {
            "command": sys.executable,
            "args": [str(fixture), str(control), str(events)],
            "env": {"ASTER_MCP_TEST_LITERAL": "${UNCHANGED_MCP_TEST_VALUE}"},
        },
        control,
        events,
    )


def _mcp_events(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if line.endswith("\n")
    ]


def _mcp_process_alive(pid: int) -> bool:
    import os

    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE, never terminate.
        if not handle:
            if ctypes.get_last_error() == 87:  # No such process.
                return False
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            state = kernel.WaitForSingleObject(handle, 0)
            if state == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            return state == 258  # WAIT_TIMEOUT: still running.
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _mcp_status(connection, session_id: str, *, reconnect: bool = False):
    params = {"sessionId": session_id}
    if reconnect:
        params["name"] = "probe"
    return await asyncio.wait_for(
        connection.ext_method(
            "aster/session/mcp/reconnect" if reconnect else "aster/session/mcp/status", params
        ),
        timeout=_HANG_TIMEOUT,
    )


async def test_acp_subprocess_mcp_stdio_reload_resume_and_cleanup(tmp_path, monkeypatch):
    """A real peer follows the session cwd, reconnects, and is closed after every operation."""
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    config, control, events = _mcp_stdio_config(tmp_path)
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"probe": config}}), encoding="utf-8"
    )
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    # Deliberately start the backend outside the session workspace.
    backend_cwd = tmp_path / "backend"
    backend_cwd.mkdir()
    async with spawn_agent_process(
        client, sys.executable, str(fixture), "--mcp", cwd=backend_cwd
    ) as (connection, _process):
        initialized = await connection.initialize(PROTOCOL_VERSION)
        assert initialized.field_meta["aster"]["mcp"] is True
        session = await connection.new_session(str(tmp_path))
        status = await _mcp_status(connection, session.session_id)
        assert [(row["name"], row["source"], row["state"]) for row in status["servers"]] == [
            ("probe", "project", "ready")
        ]
        assert status["servers"][0]["toolCount"] == 1
        await connection.prompt(session.session_id, [text_block("call the MCP probe")])
        assert any(
            isinstance(update, AgentMessageChunk)
            and update.content.text == "MCP probe: project: wire"
            for _, update in client.updates
        )

        control.write_text(json.dumps({"label": "project", "mode": "offline"}), encoding="utf-8")
        failed = await _mcp_status(connection, session.session_id, reconnect=True)
        assert failed["servers"][0]["state"] == "failed"
        assert failed["servers"][0]["error"]
        control.write_text(json.dumps({"label": "recovered"}), encoding="utf-8")
        recovered = await _mcp_status(connection, session.session_id, reconnect=True)
        assert recovered["servers"][0]["state"] == "ready"
        await connection.close_session(session.session_id)

        await connection.load_session(str(tmp_path), session.session_id)
        assert (await _mcp_status(connection, session.session_id))["servers"][0]["state"] == "ready"
        await connection.prompt(session.session_id, [text_block("call again after resume")])
        assert any(
            isinstance(update, AgentMessageChunk)
            and update.content.text == "MCP probe: recovered: wire"
            for _, update in client.updates
        )
        assert any(
            isinstance(update, AgentMessageChunk)
            and update.content.text == "MCP durable state restored"
            for _, update in client.updates
        )
        await connection.close_session(session.session_id)

    recorded = _mcp_events(events)
    started = [event for event in recorded if event["event"] == "started"]
    assert len(started) >= 6
    assert {event["pid"] for event in started} == {
        event["pid"] for event in recorded if event["event"] == "stopped"
    }
    assert all(Path(event["cwd"]).resolve() == tmp_path.resolve() for event in started)
    assert all(event["literal"] == "${UNCHANGED_MCP_TEST_VALUE}" for event in started)


async def test_acp_subprocess_mcp_trust_and_client_precedence(tmp_path, monkeypatch):
    from acp.schema import EnvVariable, McpServerStdio

    from aster import workspace_trust

    user_dir = tmp_path / "user-config"
    user_dir.mkdir()
    monkeypatch.setenv("ASTER_USER_DIR", str(user_dir))
    monkeypatch.delenv("ASTER_TRUST_ALL_WORKSPACES", raising=False)
    user, _, user_events = _mcp_stdio_config(tmp_path, label="user")
    project, _, project_events = _mcp_stdio_config(tmp_path, label="project")
    forwarded, _, client_events = _mcp_stdio_config(tmp_path, label="client")
    (user_dir / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"probe": user}}), encoding="utf-8"
    )
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"probe": project}}), encoding="utf-8"
    )
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(
        _RecordingClient(), sys.executable, str(fixture), "--enforce-trust", cwd=tmp_path
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        status = await _mcp_status(connection, session.session_id)
        assert status["servers"][0]["source"] == "user"
        assert status["servers"][0]["state"] == "ready"
        assert status["warnings"]
        assert not project_events.exists()
        await connection.close_session(session.session_id)

        client_server = McpServerStdio(
            name="probe",
            command=forwarded["command"],
            args=forwarded["args"],
            env=[EnvVariable(name=key, value=value) for key, value in forwarded["env"].items()],
        )
        duplicate = client_server.model_copy(update={"args": project["args"]})
        session = await connection.new_session(
            str(tmp_path), mcp_servers=[client_server, duplicate]
        )
        status = await _mcp_status(connection, session.session_id)
        assert status["servers"][0]["source"] == "client"
        assert status["servers"][0]["state"] == "ready"
        assert status["warnings"]
        assert not project_events.exists()
        await connection.close_session(session.session_id)

        workspace_trust.trust(tmp_path)
        session = await connection.new_session(str(tmp_path))
        status = await _mcp_status(connection, session.session_id)
        assert status["servers"][0]["source"] == "project"
        assert status["servers"][0]["state"] == "ready"
        await connection.close_session(session.session_id)

    for path in (user_events, project_events, client_events):
        assert any(event["event"] == "started" for event in _mcp_events(path))


@pytest.mark.parametrize("operation", ["call", "reconnect"])
async def test_acp_subprocess_mcp_close_cancels_inflight_work(tmp_path, monkeypatch, operation):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    config, control, events = _mcp_stdio_config(tmp_path)
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"probe": config}}), encoding="utf-8"
    )
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(
        _RecordingClient(), sys.executable, str(fixture), "--mcp", cwd=tmp_path
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        before = len(_mcp_events(events))
        control.write_text(
            json.dumps(
                {"label": "project", "mode": "block_call" if operation == "call" else "block_start"}
            ),
            encoding="utf-8",
        )
        pending = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("call a blocked MCP peer")])
            if operation == "call"
            else _mcp_status(connection, session.session_id, reconnect=True)
        )
        try:
            async with asyncio.timeout(_HANG_TIMEOUT):
                while not any(
                    event["event"] == ("call" if operation == "call" else "started")
                    for event in _mcp_events(events)[before:]
                ):
                    await asyncio.sleep(0.02)
                await connection.close_session(session.session_id)
                result = (await asyncio.gather(pending, return_exceptions=True))[0]
                while any(
                    _mcp_process_alive(event["pid"])
                    for event in _mcp_events(events)
                    if event["event"] == "started"
                ):
                    await asyncio.sleep(0.02)
            if operation == "call":
                assert result.stop_reason == "cancelled"
            else:
                assert isinstance(result, Exception)
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_acp_subprocess_mcp_http_credentials_errors_and_recovery(tmp_path, monkeypatch):
    """Use real HTTP with explicit headers; a rejected credential never reaches status or turns."""
    import socket

    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from starlette.responses import PlainTextResponse

    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    mcp = FastMCP("http-probe", stateless_http=True, json_response=True, log_level="ERROR")
    calls = []

    @mcp.tool()
    async def echo(value: str) -> str:
        calls.append(value)
        return f"http: {value}"

    app = mcp.streamable_http_app()
    accepted = True
    credential = "Bearer fixture-token-never-real"
    rejected_body = "fixture-response-body-secret"

    async def authenticated(scope, receive, send):
        if scope["type"] == "http":
            authorization = dict(scope["headers"]).get(b"authorization", b"").decode()
            if not accepted or authorization != credential:
                await PlainTextResponse(f"{rejected_body}: {credential}", status_code=401)(
                    scope, receive, send
                )
                return
        await app(scope, receive, send)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(authenticated, log_level="error", access_log=False, lifespan="on")
        )
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(_HANG_TIMEOUT):
                while not server.started:
                    if serving.done():
                        await serving
                    await asyncio.sleep(0.02)

            config_path = tmp_path / ".mcp.json"
            config = {
                "type": "http",
                "url": f"http://127.0.0.1:{port}/mcp?credential=fixture-url-secret",
                "headers": {"Authorization": "Bearer rejected-test-token"},
            }

            def write_config():
                config_path.write_text(
                    json.dumps({"mcpServers": {"probe": config}}), encoding="utf-8"
                )

            write_config()
            client = _RecordingClient()
            fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
            async with spawn_agent_process(
                client, sys.executable, str(fixture), "--mcp", cwd=tmp_path
            ) as (connection, _process):
                await connection.initialize(PROTOCOL_VERSION)
                session = await connection.new_session(str(tmp_path))
                failed = await _mcp_status(connection, session.session_id)
                assert failed["servers"][0]["state"] == "failed"
                assert "401" in failed["servers"][0]["error"]

                config["headers"]["Authorization"] = credential
                write_config()
                restored = await _mcp_status(connection, session.session_id, reconnect=True)
                assert restored["servers"][0]["state"] == "ready"
                await connection.prompt(session.session_id, [text_block("call HTTP MCP")])
                assert calls == ["wire"]

                accepted = False
                await connection.prompt(session.session_id, [text_block("observe an HTTP failure")])
                call_failed = await _mcp_status(connection, session.session_id)
                assert call_failed["servers"][0]["state"] == "failed"
                assert "401" in call_failed["servers"][0]["error"]

                accepted = True
                await _mcp_status(connection, session.session_id, reconnect=True)
                await connection.close_session(session.session_id)
                await connection.load_session(str(tmp_path), session.session_id)
                assert (await _mcp_status(connection, session.session_id))["servers"][0][
                    "state"
                ] == "ready"
                await connection.prompt(
                    session.session_id, [text_block("call HTTP MCP after resume")]
                )
                assert calls == ["wire", "wire"]
                await connection.close_session(session.session_id)

            wire = json.dumps(
                {
                    "status": [failed, call_failed, restored],
                    "updates": [update.model_dump(mode="json") for _, update in client.updates],
                }
            )
            for secret in (credential, rejected_body, "rejected-test-token", "fixture-url-secret"):
                assert secret not in wire
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=_HANG_TIMEOUT)


def _write_protocol_skill(workspace: Path) -> None:
    skills_root = workspace / "external-skills"
    package = skills_root / "protocol_skill"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "protocol-skill"\n\n'
        '[project.entry-points."aster.skills"]\n'
        '"test.protocol" = "protocol_skill:ProtocolSkill"\n'
    )
    (package / "__init__.py").write_text(
        "from aster.skill import Skill, slash_command\n\n"
        "class ProtocolSkill(Skill):\n"
        "    @slash_command(\n"
        "        'protocol-check', argument_hint='<value>', output_to_agent=False\n"
        "    )\n"
        "    def check(self, args: str) -> str:\n"
        '        """Check ACP command dispatch."""\n'
        "        return f'Check {args}.'\n"
    )
    config_dir = workspace / ".aster"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs:\n    - {skills_root}\n"
    )
    text_skill = workspace / ".agents" / "skills" / "protocol-text"
    text_skill.mkdir(parents=True)
    (text_skill / "SKILL.md").write_text(
        "---\nname: protocol-text\ndescription: Test the text command protocol\n---\n"
        "Inspect the supplied task.",
        encoding="utf-8",
    )


async def test_acp_subprocess_transcript(tmp_path, monkeypatch):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "isolated-home"))
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    _write_protocol_skill(tmp_path)
    incoming: list[dict[str, object]] = []

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        cwd=tmp_path,
    ) as (connection, _process):
        connection._conn.add_observer(  # type: ignore[attr-defined]
            lambda event: (
                incoming.append(event.message)
                if event.direction is StreamDirection.INCOMING
                else None
            )
        )
        initialized = await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await asyncio.wait_for(client.commands_updated.wait(), timeout=5)
        response = await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("run smoke test")]),
            timeout=_HANG_TIMEOUT,
        )

    assert initialized.agent_info is not None
    assert initialized.agent_info.name == "aster-acp"
    assert response.stop_reason == "end_turn"
    assert {update_session for update_session, _ in client.updates} == {session.session_id}
    new_session_response = next(
        index
        for index, message in enumerate(incoming)
        if message.get("id") == 1 and "result" in message
    )
    commands_notification = next(
        index
        for index, message in enumerate(incoming)
        if message.get("method") == "session/update"
        and message.get("params", {}).get("update", {}).get("sessionUpdate")
        == "available_commands_update"
    )
    assert new_session_response < commands_notification
    commands = next(
        update for _, update in client.updates if isinstance(update, AvailableCommandsUpdate)
    )
    assert [command.name for command in commands.available_commands] == [
        "protocol-check",
        "protocol-text",
    ]
    assert commands.available_commands[0].input is not None
    assert commands.available_commands[0].input.root.hint == "<value>"
    raw_commands = incoming[commands_notification]["params"]["update"]["availableCommands"]
    text_command = next(item for item in raw_commands if item["name"] == "protocol-text")
    assert text_command["input"] == {"hint": "[task]"}
    assert text_command["_meta"]["aster"]["kind"] == "skill"
    assert text_command["_meta"]["aster"]["aliases"] == ["cmd.protocol-text"]
    started = next(
        update
        for _, update in client.updates
        if isinstance(update, ToolCallStart) and not _is_prompt_activity(update)
    )
    assert started.kind == "other"
    assert started.status == "in_progress"
    assert started.raw_input is None
    assert started.content is not None
    source_content = started.content[0]
    assert isinstance(source_content, ContentToolCallContent)
    assert isinstance(source_content.content, TextContentBlock)
    assert source_content.content.text.startswith("```python\n")
    assert "print(sum([20, 22]))" in source_content.content.text
    assert (
        started.model_dump(mode="json", by_alias=True, exclude_none=True)["content"][0]["content"][
            "type"
        ]
        == "text"
    )

    completed = next(
        update
        for _, update in client.updates
        if isinstance(update, ToolCallProgress) and not _is_prompt_activity(update)
    )
    assert completed.content is not None
    assert len(completed.content) == 2
    assert completed.title == "Ran Python"
    assert any(
        isinstance(update, AgentMessageChunk)
        and update.content.text == "ASTER ACP smoke test passed."
        for _, update in client.updates
    )


async def test_acp_subprocess_dispatches_advertised_slash_command(tmp_path, monkeypatch):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "isolated-home"))
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    _write_protocol_skill(tmp_path)

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        response = await asyncio.wait_for(
            connection.prompt(
                session.session_id,
                [text_block("/protocol-check ready")],
            ),
            timeout=_HANG_TIMEOUT,
        )

    assert response.stop_reason == "end_turn"
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "Check ready."
        for _, update in client.updates
    )
    assert not any(
        isinstance(update, ToolCallStart) and not _is_prompt_activity(update)
        for _, update in client.updates
    )


async def test_acp_subprocess_cancellation_finishes_open_tools(tmp_path):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        "--blocking",
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt_task = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("wait forever")])
        )
        await asyncio.wait_for(client.tool_started.wait(), timeout=_HANG_TIMEOUT)
        await connection.cancel(session.session_id)
        response = await asyncio.wait_for(prompt_task, timeout=_HANG_TIMEOUT)

    assert response.stop_reason == "cancelled"
    started = next(
        update
        for _, update in client.updates
        if isinstance(update, ToolCallStart) and not _is_prompt_activity(update)
    )
    failed = next(
        update
        for _, update in client.updates
        if isinstance(update, ToolCallProgress)
        and update.status == "failed"
        and not _is_prompt_activity(update)
    )
    assert failed.tool_call_id == started.tool_call_id


async def test_acp_subprocess_closes_a_session_over_the_wire(tmp_path):
    """session/close must work through the router, not just on the adapter.

    initialize advertises the close capability, and the library registers that
    method as unstable — so an adapter-level test passes while a real client
    gets "method not found" and can never release a session.
    """
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        cwd=tmp_path,
    ) as (connection, _process):
        initialized = await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await connection.close_session(session.session_id)

        # Routable is only half of it: a no-op handler leaks the runtime for the
        # process lifetime. Prompting a closed session must now be rejected.
        with pytest.raises(Exception, match="(?i)not found|no such|unknown"):
            await asyncio.wait_for(
                connection.prompt(session.session_id, [text_block("still there?")]),
                timeout=_HANG_TIMEOUT,
            )

    capabilities = initialized.agent_capabilities.session_capabilities
    assert capabilities is not None and capabilities.close is not None


async def test_cancelling_a_turn_says_so_in_the_conversation(tmp_path):
    """A cancelled turn must leave a visible trace, not just stop.

    stop_reason=cancelled and the tool card carry the outcome, but a collapsed
    card shows the user nothing at all — the turn simply goes quiet.
    """
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        "--blocking",
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt_task = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("wait forever")])
        )
        await asyncio.wait_for(client.tool_started.wait(), timeout=_HANG_TIMEOUT)
        await connection.cancel(session.session_id)
        response = await asyncio.wait_for(prompt_task, timeout=_HANG_TIMEOUT)

    assert response.stop_reason == "cancelled"
    messages = [
        update.content.text for _, update in client.updates if isinstance(update, AgentMessageChunk)
    ]
    assert any("stopped" in text.lower() for text in messages), messages


async def test_cancelling_a_shell_command_reports_it_as_cancellation(tmp_path):
    """Cancellation during a real shell command, which --blocking never reaches."""
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        "--shell",
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt_task = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("run it")])
        )
        await asyncio.wait_for(client.tool_started.wait(), timeout=_HANG_TIMEOUT)
        await connection.cancel(session.session_id)
        response = await asyncio.wait_for(prompt_task, timeout=_HANG_TIMEOUT)

    assert response.stop_reason == "cancelled"
    rendered = "".join(str(update) for _, update in client.updates)
    assert "Cancelled by user." in rendered
    assert "CancelledError" not in rendered


@pytest.mark.sandbox
async def test_acp_sandbox_subprocess_runs_python_in_linux_guest(tmp_path):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(
        client, sys.executable, str(fixture), "--sandbox", cwd=tmp_path
    ) as (connection, process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        result = await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("run smoke test")]), timeout=180
        )
        assert result.stop_reason == "end_turn"
        await asyncio.wait_for(client.smoke_received.wait(), timeout=_HANG_TIMEOUT)
        assert any(
            isinstance(update, AgentMessageChunk)
            and update.content.text == "Interpreter platform: linux"
            for _, update in client.updates
        )
        assert any(
            isinstance(update, AgentMessageChunk)
            and update.content.text == "ASTER ACP smoke test passed."
            for _, update in client.updates
        )
        await connection.close_session(session.session_id)


@pytest.mark.sandbox
async def test_acp_sandbox_subprocess_cancels_running_guest(tmp_path):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(
        client, sys.executable, str(fixture), "--sandbox", "--blocking", cwd=tmp_path
    ) as (connection, process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        pending = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("wait forever")])
        )
        await asyncio.wait_for(client.sandbox_ready.wait(), timeout=180)
        await connection.cancel(session.session_id)
        result = await asyncio.wait_for(pending, timeout=30)
        assert result.stop_reason == "cancelled"
        await connection.close_session(session.session_id)


@pytest.mark.sandbox
async def test_acp_sandbox_cancels_before_setup_then_reuses_session(tmp_path):
    """Cancel a deterministically held prefill, then run a real guest in the same session."""
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(
        client,
        sys.executable,
        str(fixture),
        "--sandbox",
        "--pause-prefill",
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        pending = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("cancel during setup")])
        )
        try:
            await asyncio.wait_for(client.preparing.wait(), timeout=_HANG_TIMEOUT)
            assert not any(isinstance(update, AgentMessageChunk) for _, update in client.updates)
            await connection.cancel(session.session_id)
            result = await asyncio.wait_for(pending, timeout=_HANG_TIMEOUT)
            assert result.stop_reason == "cancelled"
            await asyncio.wait_for(client.cancelled.wait(), timeout=_HANG_TIMEOUT)
            activity = [update for _, update in client.updates if _is_prompt_activity(update)]
            assert activity[-1].title == "Cancelled"
            assert activity[-1].status == "failed"
            assert len({update.tool_call_id for update in activity}) == 1
            assert sum(update.status in {"failed", "completed"} for update in activity) == 1

            client.updates.clear()
            result = await asyncio.wait_for(
                connection.prompt(session.session_id, [text_block("run the smoke test now")]),
                timeout=180,
            )
            assert result.stop_reason == "end_turn"
            # ACP resolves responses before its queued notification callbacks finish.
            await asyncio.wait_for(
                asyncio.gather(client.smoke_received.wait(), client.response_ready.wait()),
                timeout=_HANG_TIMEOUT,
            )
            answers = [
                update.content.text
                for _, update in client.updates
                if isinstance(update, AgentMessageChunk)
            ]
            assert "Interpreter platform: linux" in answers
            assert "ASTER ACP smoke test passed." in answers
            activity = [update for _, update in client.updates if _is_prompt_activity(update)]
            assert activity[-1].status == "completed"
            assert activity[-1].title == "Response ready"
            assert sum(update.status in {"failed", "completed"} for update in activity) == 1
            await connection.close_session(session.session_id)
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
