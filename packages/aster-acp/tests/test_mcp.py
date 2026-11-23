# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP availability and configuration at the ACP session boundary."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from acp import RequestError
from acp.schema import AcpMcpServer, McpServerStdio
from aster_acp.server import CodingACPAdapter

from aster import workspace_trust
from aster.mcp.tool import MCPCallEvent
from aster.paths import get_user_dir
from aster.unifiedllm import FakeLLMClient


class LookupTools:
    _tool_method_names = frozenset({"lookup"})

    async def lookup(self, query: str) -> str:
        """Look up a test value."""
        return query


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    user = home / "config"
    user.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("ASTER_USER_DIR", str(user))
    monkeypatch.delenv("ASTER_PROJECT_DIR", raising=False)
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    return user


def configure(path: Path, servers: dict) -> None:
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


@asynccontextmanager
async def session(workspace: Path, *, servers=None):
    llm = FakeLLMClient([], strict_exhaustion=True)
    adapter = CodingACPAdapter(lambda: llm)
    adapter.on_connect(AsyncMock())
    try:
        initialized = await adapter.initialize(1)
        created = await adapter.new_session(str(workspace), mcp_servers=servers)
        runtime = await adapter._get_runtime(created.session_id)
        yield adapter, created.session_id, runtime, llm, initialized
    finally:
        await adapter.close()


async def status(adapter, session_id):
    return await adapter.ext_method("aster/session/mcp/status", {"sessionId": session_id})


async def reconnect(adapter, session_id, name="lookup"):
    return await adapter.ext_method(
        "aster/session/mcp/reconnect", {"sessionId": session_id, "name": name}
    )


async def test_configuration_sources_and_duplicate_clients_are_session_scoped(tmp_path):
    configure(get_user_dir(".mcp.json"), {"lookup": {"command": "user-server"}})
    configure(tmp_path / ".mcp.json", {"lookup": {"command": "project-server"}})
    forwarded = [
        McpServerStdio(name="lookup", command="client-server", args=[], env=[]),
        McpServerStdio(name="lookup", command="ignored-server", args=[], env=[]),
    ]
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=LookupTools()),
    ) as create:
        async with session(tmp_path, servers=forwarded) as (adapter, sid, runtime, llm, init):
            assert init.field_meta["aster"]["mcp"] is True
            snapshot = await status(adapter, sid)
            assert snapshot["servers"] == [
                {
                    "name": "lookup",
                    "source": "client",
                    "transport": "stdio",
                    "state": "ready",
                    "toolCount": 1,
                    "error": None,
                }
            ]
            assert any("same name" in warning for warning in snapshot["warnings"])
            assert create.await_count == 1
            assert create.await_args.kwargs["command"] == "client-server"
            assert create.await_args.kwargs["cwd"] == str(tmp_path.resolve())
            assert create.await_args.kwargs["safe_errors"] is True
            assert "mcp.lookup" in runtime.value.agent.skills.activated()
            assert llm.call_count == 0


@pytest.mark.parametrize("deny_metadata", [False, True])
async def test_untrusted_project_is_not_read_but_user_configuration_remains_available(
    tmp_path, monkeypatch, deny_metadata
):
    configure(get_user_dir(".mcp.json"), {"lookup": {"command": "user-server"}})
    project_file = tmp_path / ".mcp.json"
    configure(project_file, {"private-project-name": {"command": "must-not-run"}})
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        assert path != project_file, "Untrusted MCP configuration was read"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    if deny_metadata:
        original_is_file = Path.is_file

        def guarded_is_file(path):
            if path == project_file:
                raise PermissionError("private-metadata-error")
            return original_is_file(path)

        monkeypatch.setattr(Path, "is_file", guarded_is_file)
    monkeypatch.delenv("ASTER_TRUST_ALL_WORKSPACES", raising=False)
    with (
        workspace_trust.enforced(),
        patch(
            "aster_acp._mcp.MCPManager.create_stdio_server",
            new=AsyncMock(return_value=LookupTools()),
        ) as create,
    ):
        async with session(tmp_path) as (adapter, sid, _, _, _):
            snapshot = await status(adapter, sid)
            assert [item["source"] for item in snapshot["servers"]] == ["user"]
            assert "private-project-name" not in json.dumps(snapshot)
            assert "private-metadata-error" not in json.dumps(snapshot)
            assert snapshot["warnings"]
            assert create.await_args.kwargs["command"] == "user-server"


async def test_failed_discovery_is_safe_and_reconnect_reads_updated_configuration(tmp_path):
    project_file = tmp_path / ".mcp.json"
    configure(project_file, {"lookup": {"command": "old-server"}})
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(side_effect=RuntimeError("secret-token http://user:password@host")),
    ) as create:
        async with session(tmp_path) as (adapter, sid, runtime, llm, _):
            snapshot = await status(adapter, sid)
            assert snapshot["servers"][0]["state"] == "failed"
            assert "secret-token" not in json.dumps(snapshot)
            assert "password" not in json.dumps(snapshot)
            assert "secret-token" not in " ".join(runtime.value.startup_warnings)
            configure(project_file, {"lookup": {"command": "repaired-server"}})
            create.side_effect = None
            create.return_value = LookupTools()
            snapshot = await reconnect(adapter, sid)
            assert snapshot["servers"][0]["state"] == "ready"
            assert create.await_args.kwargs["command"] == "repaired-server"
            assert "mcp.lookup" in runtime.value.agent.skills.activated()
            assert llm.call_count == 0


@pytest.mark.parametrize(
    ("updated", "expected"),
    [
        ({"lookup": {"disabled": True}}, "disabled"),
        ({"lookup": {"command": 42}}, "failed"),
        ({}, None),
    ],
)
async def test_reconnect_withdraws_tools_when_configuration_is_disabled_invalid_or_removed(
    tmp_path, updated, expected
):
    project_file = tmp_path / ".mcp.json"
    configure(project_file, {"lookup": {"command": "old-server"}})
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=LookupTools()),
    ) as create:
        async with session(tmp_path) as (adapter, sid, runtime, _, _):
            assert hasattr(runtime.value.agent, "lookup")
            configure(project_file, updated)
            snapshot = await reconnect(adapter, sid)
            assert "mcp.lookup" not in runtime.value.agent.skills.loaded()
            assert not hasattr(runtime.value.agent, "lookup")
            assert create.await_count == 1
            if expected is None:
                assert snapshot["servers"] == []
                assert any("no longer configured" in item for item in snapshot["warnings"])
            else:
                assert snapshot["servers"][0]["state"] == expected
                assert snapshot["servers"][0]["toolCount"] == 0


async def test_reconnect_preserves_a_skill_that_took_over_the_registration(tmp_path):
    configure(tmp_path / ".mcp.json", {"lookup": {"command": "server"}})
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=LookupTools()),
    ) as create:
        async with session(tmp_path) as (adapter, sid, runtime, _, _):
            replacement = LookupTools()
            runtime.value.agent.skills.register("mcp.lookup", replacement)
            snapshot = await reconnect(adapter, sid)
            assert runtime.value.agent.lookup is replacement
            assert snapshot["servers"][0]["state"] == "failed"
            assert "owns" in snapshot["servers"][0]["error"]
            assert create.await_count == 1


async def test_status_is_available_while_busy_but_reconnect_is_rejected(tmp_path, monkeypatch):
    configure(tmp_path / ".mcp.json", {"lookup": {"command": "server"}})
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=LookupTools()),
    ) as create:
        async with session(tmp_path) as (adapter, sid, runtime, llm, _):
            async with runtime.turn():
                snapshot = await asyncio.wait_for(status(adapter, sid), 1)
                assert snapshot["servers"][0]["state"] == "ready"
                with pytest.raises(RequestError):
                    await reconnect(adapter, sid)
            monkeypatch.setattr(runtime.value.agent.queue_manager, "running_handles", lambda: [1])
            with pytest.raises(RequestError):
                await reconnect(adapter, sid)
            assert create.await_count == 1
            assert llm.call_count == 0


@pytest.mark.parametrize("name", [None, "", " ", 7, "x" * 257])
async def test_reconnect_rejects_invalid_names_before_discovery(tmp_path, name):
    async with session(tmp_path) as (adapter, sid, _, _, _):
        with pytest.raises(RequestError):
            await reconnect(adapter, sid, name)


async def test_call_outcomes_update_availability_without_stale_notifications(tmp_path):
    configure(tmp_path / ".mcp.json", {"lookup": {"command": "server"}})
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(side_effect=[LookupTools(), LookupTools()]),
    ) as create:
        async with session(tmp_path) as (adapter, sid, _, _, _):
            observe = create.await_args.kwargs["call_observer"]
            observe(MCPCallEvent("lookup", "failed", "MCP server is unavailable."))
            assert (await status(adapter, sid))["servers"][0]["state"] == "failed"
            observe(MCPCallEvent("lookup", "succeeded", None))
            assert (await status(adapter, sid))["servers"][0]["state"] == "ready"
            await reconnect(adapter, sid)
            observe(MCPCallEvent("lookup", "failed", "MCP server is unavailable."))
            assert (await status(adapter, sid))["servers"][0]["state"] == "ready"


async def test_closing_a_session_cancels_discovery_before_waiting_for_the_turn_lock(tmp_path):
    configure(tmp_path / ".mcp.json", {"lookup": {"command": "server"}})
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=LookupTools()),
    ) as create:
        async with session(tmp_path) as (adapter, sid, _, _, _):
            create.side_effect = blocked
            operation = asyncio.create_task(reconnect(adapter, sid))
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.wait_for(adapter.close_session(sid), 1)
            with pytest.raises(RequestError):
                await operation
            assert cleaned.is_set()
            with pytest.raises(RequestError):
                await status(adapter, sid)


async def test_different_workspaces_load_their_own_configuration(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    configure(first / ".mcp.json", {"first": {"command": "first-server"}})
    configure(second / ".mcp.json", {"second": {"command": "second-server"}})
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(side_effect=lambda *args, **kwargs: LookupTools()),
    ) as create:
        async with session(first) as one, session(second) as two:
            assert (await status(one[0], one[1]))["servers"][0]["name"] == "first"
            assert (await status(two[0], two[1]))["servers"][0]["name"] == "second"
            assert [call.kwargs["cwd"] for call in create.await_args_list] == [
                str(first),
                str(second),
            ]


async def test_mcp_names_with_the_same_python_attribute_do_not_replace_each_other(tmp_path):
    configure(
        tmp_path / ".mcp.json",
        {"foo-bar": {"command": "first-server"}, "foo_bar": {"command": "second-server"}},
    )
    first = LookupTools()
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server", new=AsyncMock(return_value=first)
    ) as create:
        async with session(tmp_path) as (adapter, sid, runtime, _, _):
            snapshot = await status(adapter, sid)
            assert [server["state"] for server in snapshot["servers"]] == ["ready", "failed"]
            assert runtime.value.agent.foo_bar is first
            assert create.await_count == 1


async def test_mcp_does_not_replace_a_different_skill_with_the_same_leaf(tmp_path):
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server", new=AsyncMock(return_value=LookupTools())
    ) as create:
        async with session(tmp_path) as (adapter, sid, runtime, _, _):
            existing = LookupTools()
            runtime.value.agent.skills.register("custom.lookup", existing)
            configure(tmp_path / ".mcp.json", {"lookup": {"command": "server"}})
            snapshot = await reconnect(adapter, sid)
            assert snapshot["servers"][0]["state"] == "failed"
            assert runtime.value.agent.lookup is existing
            create.assert_not_awaited()


async def test_request_cancellation_still_propagates_and_leaves_the_session_usable(tmp_path):
    configure(tmp_path / ".mcp.json", {"lookup": {"command": "server"}})
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=LookupTools()),
    ) as create:
        async with session(tmp_path) as (adapter, sid, _, _, _):
            create.side_effect = blocked
            operation = asyncio.create_task(reconnect(adapter, sid))
            await asyncio.wait_for(started.wait(), 1)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert cleaned.is_set()
            assert (await status(adapter, sid))["servers"][0]["state"] == "failed"
            create.side_effect = None
            assert (await reconnect(adapter, sid))["servers"][0]["state"] == "ready"


async def test_startup_discovers_healthy_and_hung_servers_concurrently(tmp_path, monkeypatch):
    configure(
        tmp_path / ".mcp.json",
        {name: {"command": name} for name in ("slow-one", "slow-two", "healthy")},
    )
    monkeypatch.setattr("aster_acp._mcp._DISCOVERY_TIMEOUT", 0.5)
    entered: set[str] = set()
    cleaned: set[str] = set()
    all_entered = asyncio.Event()

    async def discover(name, **kwargs):
        entered.add(name)
        if len(entered) == 3:
            all_entered.set()
        try:
            await all_entered.wait()
            if name == "healthy":
                assert not cleaned, "Healthy discovery waited for another server's timeout"
                return LookupTools()
            await asyncio.Event().wait()
        finally:
            if name != "healthy":
                cleaned.add(name)

    with patch("aster_acp._mcp.MCPManager.create_stdio_server", side_effect=discover):
        async with session(tmp_path) as (adapter, sid, _, llm, _):
            snapshot = await status(adapter, sid)
            assert [server["state"] for server in snapshot["servers"]] == [
                "failed",
                "failed",
                "ready",
            ]
            assert entered == {"slow-one", "slow-two", "healthy"}
            assert cleaned == {"slow-one", "slow-two"}
            assert llm.call_count == 0


async def test_cancelled_startup_waits_for_every_discovery_cleanup(tmp_path):
    names = {"one", "two", "three"}
    configure(tmp_path / ".mcp.json", {name: {"command": name} for name in sorted(names)})
    entered: set[str] = set()
    cleaned: set[str] = set()
    all_entered = asyncio.Event()

    async def discover(name, **kwargs):
        entered.add(name)
        if entered == names:
            all_entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.add(name)

    adapter = CodingACPAdapter(lambda: FakeLLMClient([], strict_exhaustion=True))
    adapter.on_connect(AsyncMock())
    try:
        with patch("aster_acp._mcp.MCPManager.create_stdio_server", side_effect=discover):
            operation = asyncio.create_task(adapter.new_session(str(tmp_path)))
            await asyncio.wait_for(all_entered.wait(), 2)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(operation, 2)
        assert cleaned == names
        assert await adapter._sessions.ids() == ()
        configure(tmp_path / ".mcp.json", {})
        created = await adapter.new_session(str(tmp_path))
        assert (await status(adapter, created.session_id))["servers"] == []
    finally:
        await adapter.close()


@pytest.mark.parametrize("first", [{"disabled": True}, {"command": 42}])
async def test_disabled_or_invalid_servers_do_not_reserve_a_python_attribute(tmp_path, first):
    configure(tmp_path / ".mcp.json", {"foo-bar": first, "foo_bar": {"command": "server"}})
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server", new=AsyncMock(return_value=LookupTools())
    ) as create:
        async with session(tmp_path) as (adapter, sid, _, _, _):
            snapshot = await status(adapter, sid)
            assert snapshot["servers"][1]["state"] == "ready"
            assert create.await_count == 1
            assert create.await_args.args[0] == "foo_bar"


@pytest.mark.parametrize("name", ["close", "handle", "context_manager"])
async def test_mcp_cannot_replace_agent_methods_or_fields(tmp_path, name):
    configure(tmp_path / ".mcp.json", {name: {"command": "server"}})
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server", new=AsyncMock(return_value=LookupTools())
    ) as create:
        async with session(tmp_path) as (adapter, sid, runtime, _, _):
            snapshot = await status(adapter, sid)
            assert snapshot["servers"][0]["state"] == "failed"
            assert callable(runtime.value.agent.close)
            assert callable(runtime.value.agent.handle)
            assert runtime.value.agent.context_manager is not None
            create.assert_not_awaited()


@pytest.mark.parametrize("later_client", [False, True])
async def test_unsupported_first_client_blocks_same_name_fallback(tmp_path, later_client):
    configure(get_user_dir(".mcp.json"), {"lookup": {"command": "user-server"}})
    configure(tmp_path / ".mcp.json", {"lookup": {"command": "project-server"}})
    servers = [AcpMcpServer(name="lookup", id="proxy-lookup", type="acp")]
    if later_client:
        servers.append(McpServerStdio(name="lookup", command="later-client", args=[], env=[]))
    with patch(
        "aster_acp._mcp.MCPManager.create_stdio_server", new=AsyncMock(return_value=LookupTools())
    ) as create:
        async with session(tmp_path, servers=servers) as (adapter, sid, runtime, _, _):
            snapshot = await status(adapter, sid)
            assert snapshot["servers"][0]["source"] == "client"
            assert snapshot["servers"][0]["state"] == "failed"
            assert "mcp.lookup" not in runtime.value.agent.skills.loaded()
            assert any("unsupported ACP server type" in item for item in snapshot["warnings"])
            if later_client:
                assert any("same name" in item for item in snapshot["warnings"])
            create.assert_not_awaited()
