# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-facing MCP status cannot change calls or expose credentials."""

import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("mcp")

from aster import Agent  # noqa: E402
from aster.mcp import client as client_module  # noqa: E402
from aster.mcp import tool as tool_module  # noqa: E402
from aster.mcp.tool import MCPCallEvent, MCPManager, MCPTool, safe_mcp_error  # noqa: E402
from aster.skill_registry import SkillRegistry  # noqa: E402
from aster.storage.snapshot import AgentSnapshot  # noqa: E402
from aster.unifiedllm import FakeLLMClient  # noqa: E402


def _http_error(status=401):
    request = httpx.Request("POST", "https://secret.invalid/?token=secret-leak-marker")
    response = httpx.Response(
        status,
        request=request,
        text="secret-leak-marker",
        extensions={"reason_phrase": b"secret-leak-marker"},
    )
    return httpx.HTTPStatusError("secret-leak-marker", request=request, response=response)


class _Client:
    def __init__(self):
        self.session = SimpleNamespace(
            initialize=AsyncMock(),
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="echo",
                            description="Echo",
                            inputSchema={
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                            },
                        )
                    ]
                )
            ),
            call_tool=AsyncMock(
                return_value=SimpleNamespace(content=[SimpleNamespace(text="done")])
            ),
        )
        self.closed = 0

    @asynccontextmanager
    async def connect_to_server(self):
        owner = asyncio.current_task()
        try:
            yield self.session
        finally:
            assert asyncio.current_task() is owner
            self.closed += 1


@pytest.mark.parametrize("transport", ["stdio", "streamable-http", "sse"])
@pytest.mark.parametrize("generated", [False, True], ids=["base", "generated"])
async def test_runtime_tools_do_not_break_strict_snapshots_or_persist_credentials(
    monkeypatch, transport, generated
):
    agent = Agent(llm=FakeLLMClient())
    restored = Agent(llm=FakeLLMClient())
    agent.user_state = {"notes": ["durable"], "count": 3}
    registry = SkillRegistry(agent)
    credential = "MCP_CLIENT_CREDENTIAL_CANARY"
    callback_state = {"secret": "MCP_CALLBACK_CANARY", "agent": agent}

    def observe(event):
        callback_state["event"] = event

    if transport == "stdio":
        config = {"command": "mock-server", "env": {"TOKEN": credential}}
    else:
        config = {
            "url": "https://example.invalid/mcp",
            "headers": {"Authorization": credential},
        }
    client = client_module.create_mcp_client(transport=transport, **config)
    monkeypatch.setattr(client, "connect_to_server", _Client().connect_to_server)
    monkeypatch.setattr(tool_module, "create_mcp_client", lambda **kwargs: client)
    try:
        if not generated:
            tool = MCPTool(client, "probe", call_observer=observe)
        elif transport == "stdio":
            tool = await MCPManager.create_stdio_server("probe", **config, call_observer=observe)
        else:
            tool = await MCPManager.create_url_server(
                "probe", **config, transport=transport, call_observer=observe
            )
        assert credential in str(client.server_config)
        assert tool._call_observer is observe
        registry.register("mcp.probe", tool)
        registry.activate(["mcp.probe"])

        saved = AgentSnapshot.from_agent(agent, strict=True).model_dump_json()
        assert credential not in saved
        assert callback_state["secret"] not in saved
        AgentSnapshot.model_validate_json(saved).restore(restored, strict=True)
        assert restored.user_state == {"notes": ["durable"], "count": 3}
        assert not hasattr(restored, "probe")
        assert agent.probe is tool
    finally:
        await agent.aclose()
        await restored.aclose()


@pytest.mark.parametrize("transport", ["stdio", "streamable-http", "sse"])
async def test_factories_observe_final_success_and_release_each_connection(
    monkeypatch, tmp_path, transport
):
    client = _Client()
    captured = []
    events = []

    def create(**kwargs):
        captured.append(kwargs)
        return client

    monkeypatch.setattr(tool_module, "create_mcp_client", create)
    if transport == "stdio":
        tool = await MCPManager.create_stdio_server(
            "example",
            "server",
            cwd=tmp_path,
            call_observer=events.append,
        )
        assert captured[0]["cwd"] == tmp_path
    else:
        tool = await MCPManager.create_url_server(
            "example",
            "https://example.invalid/mcp",
            transport=transport,
            call_observer=events.append,
            allow_oauth_refresh=False,
        )
    assert client.closed == 1
    assert events == []
    assert await tool.echo("hello") == "done"
    assert client.closed == 2
    assert events == [MCPCallEvent("echo", "succeeded")]


async def test_stdio_working_directory_reaches_sdk_without_changing_host_cwd(monkeypatch, tmp_path):
    client = _Client()
    captured = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    before = Path.cwd()

    @asynccontextmanager
    async def stdio(params):
        captured.append(params)
        yield object(), object()

    @asynccontextmanager
    async def session(*args, **kwargs):
        yield client.session

    monkeypatch.setattr(client_module, "stdio_client", stdio)
    monkeypatch.setattr(client_module, "ClientSession", session)
    args = ["-c", "if True:\n\tprint('${UNEXPANDED}')\n"]
    env = {"CERTIFICATE": "first\r\nsecond\t${UNEXPANDED}"}
    tool = await MCPManager.create_stdio_server(
        "example", "server", args=args, env=env, cwd=workspace
    )
    assert await tool.echo("hello") == "done"
    assert len(captured) == 2
    assert all(params.args == args and params.env == env for params in captured)
    assert all(Path(params.cwd) == workspace for params in captured)
    assert Path.cwd() == before
    assert tool._client.cwd == workspace
    assert tool._client.server_config["cwd"] == str(workspace)


@pytest.mark.parametrize("safe_errors", [False, True])
async def test_header_only_unauthorized_call_never_enters_oauth(monkeypatch, safe_errors):
    client = _Client()
    client.session.call_tool.side_effect = _http_error()
    events = []
    oauth = AsyncMock(side_effect=AssertionError("OAuth must not be used"))
    monkeypatch.setattr(tool_module, "create_mcp_client", lambda **kwargs: client)
    monkeypatch.setattr("aster.mcp.oauth.handle_mcp_oauth", oauth)
    tool = await MCPManager.create_url_server(
        "example",
        "https://example.invalid",
        headers={"Authorization": "Bearer explicit"},
        allow_oauth_refresh=False,
        call_observer=events.append,
        safe_errors=safe_errors,
    )
    with pytest.raises(RuntimeError) as caught:
        await tool.echo("hello")
    oauth.assert_not_called()
    assert events == [MCPCallEvent("echo", "failed", "MCP authentication failed (HTTP 401).")]
    assert client.closed == 2
    if safe_errors:
        assert str(caught.value) == "MCP authentication failed (HTTP 401)."
        assert "secret-leak-marker" not in "".join(traceback.format_exception(caught.value))
    else:
        assert "secret-leak-marker" in str(caught.value)


async def test_oauth_retry_reports_only_the_final_result(monkeypatch):
    client = _Client()
    result = client.session.call_tool.return_value
    client.session.call_tool.side_effect = [_http_error(), result]
    events = []
    tool = MCPTool(
        client,
        "example",
        refresh_ctx={"server_url": "https://example.invalid"},
        call_observer=events.append,
    )
    refresh = AsyncMock(return_value=True)
    monkeypatch.setattr(tool, "_refresh_access_token", refresh)
    assert await tool._call_tool("echo", {"value": "hello"}) == "done"
    refresh.assert_awaited_once()
    assert client.session.call_tool.await_count == 2
    assert events == [MCPCallEvent("echo", "succeeded")]


async def test_tool_level_failure_has_safe_observation_and_safe_traceback():
    client = _Client()
    client.session.call_tool.return_value = SimpleNamespace(
        isError=True,
        content=[SimpleNamespace(text="secret-leak-marker")],
    )
    events = []
    tool = MCPTool(client, "example", call_observer=events.append, safe_errors=True)
    with pytest.raises(RuntimeError) as caught:
        await tool._call_tool("echo", {})
    assert events == [MCPCallEvent("echo", "failed", "MCP operation failed.")]
    assert str(caught.value) == "MCP operation failed."
    assert "secret-leak-marker" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("safe_errors", [False, True])
async def test_cancellation_is_preserved_and_reported_after_transport_cleanup(safe_errors):
    client = _Client()
    started = asyncio.Event()
    events = []

    async def call(*args):
        started.set()
        await asyncio.Event().wait()

    client.session.call_tool.side_effect = call
    tool = MCPTool(client, "example", call_observer=events.append, safe_errors=safe_errors)
    task = asyncio.create_task(tool._call_tool("echo", {}))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed == 1
    assert events == [MCPCallEvent("echo", "cancelled")]


@pytest.mark.parametrize("observer_error", [RuntimeError, asyncio.CancelledError])
async def test_observer_errors_cannot_change_success_or_failure(monkeypatch, observer_error):
    def observe(event):
        raise observer_error("observer failure")

    tool = MCPTool(_Client(), "example", call_observer=observe)
    assert await tool._call_tool("echo", {}) == "done"
    failure = RuntimeError("original failure")
    monkeypatch.setattr(tool, "_call_tool_with_refresh", AsyncMock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        await tool._call_tool("echo", {})
    assert caught.value is failure


@pytest.mark.parametrize(
    "error, expected",
    [
        (_http_error(401), "MCP authentication failed (HTTP 401)."),
        (_http_error(403), "MCP authentication failed (HTTP 403)."),
        (_http_error(503), "MCP server returned HTTP 503."),
        (TimeoutError("secret-leak-marker"), "MCP operation timed out."),
        (httpx.ReadTimeout("secret-leak-marker"), "MCP operation timed out."),
        (FileNotFoundError("secret-leak-marker"), "MCP server executable was not found."),
        (PermissionError("secret-leak-marker"), "MCP connection was denied."),
        (ConnectionError("secret-leak-marker"), "MCP connection failed."),
        (httpx.ConnectError("secret-leak-marker"), "MCP connection failed."),
        (ValueError("secret-leak-marker"), "MCP operation failed."),
        (asyncio.CancelledError("secret-leak-marker"), "MCP operation cancelled."),
    ],
)
def test_safe_errors_never_echo_exception_values(error, expected):
    assert safe_mcp_error(error) == expected


def test_safe_error_handles_nested_groups_chains_and_cycles():
    nested = ExceptionGroup("secret-leak-marker", [ExceptionGroup("hidden", [_http_error(403)])])
    wrapper = RuntimeError("secret-leak-marker")
    wrapper.__cause__ = nested
    nested.__context__ = wrapper
    assert safe_mcp_error(wrapper) == "MCP authentication failed (HTTP 403)."


def test_timeout_is_not_misclassified_as_its_internal_cancellation():
    timeout = TimeoutError("secret-leak-marker")
    timeout.__cause__ = asyncio.CancelledError()
    assert safe_mcp_error(timeout) == "MCP operation timed out."
