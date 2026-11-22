# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-owned MCP configuration, tool registration and availability."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer

from aster import workspace_trust
from aster.mcp import MCPManager, MCPTool
from aster.mcp.config import MCPConfigResult, MCPServerConfig, load_mcp_config
from aster.mcp.tool import MCPCallEvent, safe_mcp_error

if TYPE_CHECKING:
    from aster_cli.coding.agent import CodingAgent

_DISCOVERY_TIMEOUT = 30.0


@dataclass(slots=True)
class _Server:
    definition: MCPServerConfig
    state: str = "connecting"
    error: str | None = None
    tool: MCPTool | None = None
    tool_count: int = 0


class SessionMCP:
    """Keep each workspace's MCP state separate without holding network connections."""

    def __init__(
        self,
        agent: CodingAgent,
        workspace: Path,
        client_servers: list[Any] | None,
    ) -> None:
        self.agent = agent
        self.registry = agent.skills
        self.workspace = workspace
        self._client_servers, self._client_warnings = _forwarded_servers(client_servers)
        self._servers: dict[str, _Server] = {}
        self._warnings: list[str] = []
        self._connecting: set[asyncio.Task[MCPTool]] = set()
        self._closed = False
        self.on_change: Callable[[], None] | None = None

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _configuration(self) -> MCPConfigResult:
        trusted = workspace_trust.is_trusted(self.workspace)
        result = load_mcp_config(
            self.workspace,
            client_servers=self._client_servers,
            project_trusted=trusted,
        )
        self._warnings = [*result.warnings, *self._client_warnings]
        if not trusted:
            try:
                show_notice = (self.workspace / ".mcp.json").is_file()
            except OSError:
                show_notice = True
            if show_notice:
                self._warnings.append(
                    "Project .mcp.json was skipped because this workspace is not trusted. "
                    "Run `aster trust` in the workspace to use its MCP servers."
                )
        return result

    async def start(self) -> None:
        self._servers = {
            definition.name: _Server(definition)
            for definition in self._configuration().servers.values()
        }
        reserved: set[str] = set()
        tasks: list[asyncio.Task[None]] = []
        for server in self._servers.values():
            definition = server.definition
            if (
                not definition.disabled
                and definition.error is None
                and definition.config is not None
            ):
                leaf = _attribute_name(definition.name)
                if leaf in reserved:
                    server.state = "failed"
                    server.error = "Another MCP server already uses this agent attribute."
                    continue
                reserved.add(leaf)
            tasks.append(asyncio.create_task(self._connect(server), name="aster-acp-mcp-start"))
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def snapshot(self) -> dict[str, Any]:
        return {
            "servers": [
                {
                    "name": server.definition.name,
                    "source": server.definition.source,
                    "transport": (server.definition.config or {}).get("transport", ""),
                    "state": server.state,
                    "toolCount": server.tool_count,
                    "error": server.error,
                }
                for server in self._servers.values()
            ],
            "warnings": list(self._warnings),
        }

    def startup_warnings(self) -> tuple[str, ...]:
        return (
            *self._warnings,
            *(
                f"MCP server {server.definition.name!r} was not loaded: {server.error}"
                for server in self._servers.values()
                if server.state == "failed"
            ),
        )

    async def reconnect(self, name: str) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("The session's MCP tools are closed.")
        definition = self._configuration().servers.get(name)
        previous = self._servers.get(name)
        if definition is None and previous is None:
            raise KeyError(name)
        if previous is not None and previous.tool is not None:
            if not self.registry.unregister(f"mcp.{name}", instance=previous.tool):
                previous.state = "failed"
                previous.error = "Another skill now owns this MCP name."
                previous.tool = None
                previous.tool_count = 0
                self._changed()
                return self.snapshot()
        if definition is None:
            self._servers.pop(name, None)
            self._warnings.append(f"MCP server {name!r} is no longer configured.")
            self._changed()
            return self.snapshot()
        server = _Server(definition)
        self._servers[name] = server
        self._changed()
        await self._connect(server)
        return self.snapshot()

    async def _connect(self, server: _Server) -> None:
        definition = server.definition
        if definition.disabled:
            server.state = "disabled"
            self._changed()
            return
        if definition.error is not None or definition.config is None:
            server.state = "failed"
            server.error = definition.error or "MCP configuration is invalid."
            self._changed()
            return
        registry_name = f"mcp.{definition.name}"
        if self._registration_conflict(registry_name):
            server.state = "failed"
            server.error = "Another skill already uses this MCP name."
            self._changed()
            return
        task = asyncio.create_task(self._discover(server), name="aster-acp-mcp-discovery")
        self._connecting.add(task)
        try:
            async with asyncio.timeout(_DISCOVERY_TIMEOUT):
                tool = await task
            if self._closed:
                raise asyncio.CancelledError
            try:
                if self._registration_conflict(registry_name):
                    raise ValueError("Another skill uses this MCP attribute.")
                self.registry.register(registry_name, tool)
                self.registry.activate([registry_name])
            except Exception:
                self.registry.unregister(registry_name, instance=tool)
                server.state = "failed"
                server.error = "A skill or reserved agent attribute already uses this name."
            else:
                server.tool = tool
                server.tool_count = len(getattr(tool, "_tool_method_names", ()))
                server.state = "ready"
                server.error = None
        except asyncio.CancelledError:
            server.state = "closed" if self._closed else "failed"
            server.error = None if self._closed else "MCP discovery was cancelled."
            raise
        except Exception as exc:
            server.state = "failed"
            server.error = safe_mcp_error(exc)
        finally:
            self._connecting.discard(task)
            self._changed()

    def _registration_conflict(self, registry_name: str) -> bool:
        # register() binds the normalized final component as an agent attribute.
        leaf = _attribute_name(registry_name)
        try:
            inspect.getattr_static(self.agent, leaf)
        except AttributeError:
            pass
        else:
            return True
        return any(_attribute_name(name) == leaf for name in self.registry.discovered())

    async def _discover(self, server: _Server) -> MCPTool:
        config = server.definition.config
        assert config is not None

        def observe(event: MCPCallEvent) -> None:
            self._observe_call(server, event)

        if config["transport"] == "stdio":
            return await MCPManager.create_stdio_server(
                server.definition.name,
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env", {}),
                cwd=str(self.workspace),
                call_observer=observe,
                safe_errors=True,
            )
        return await MCPManager.create_url_server(
            server.definition.name,
            config["url"],
            headers=config.get("headers", {}),
            transport=config["transport"],
            call_observer=observe,
            allow_oauth_refresh=False,
            safe_errors=True,
        )

    def _observe_call(self, server: _Server, event: MCPCallEvent) -> None:
        if self._closed or self._servers.get(server.definition.name) is not server:
            return
        if event.outcome == "cancelled":
            return
        server.state = "ready" if event.outcome == "succeeded" else "failed"
        server.error = event.error
        self._changed()

    def _changed(self) -> None:
        if self.on_change is not None and not self._closed:
            self.on_change()

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(self._connecting)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for server in self._servers.values():
            server.state = "closed"


def _attribute_name(name: str) -> str:
    return name.rsplit(".", 1)[-1].replace("-", "_")


def _forwarded_servers(servers: list[Any] | None) -> tuple[dict[str, dict], list[str]]:
    configurations: dict[str, dict] = {}
    warnings: list[str] = []
    for server in servers or []:
        name = getattr(server, "name", "<unnamed>")
        if name in configurations:
            warnings.append(
                f"MCP server {name!r} was not loaded: another server has the same name."
            )
            continue
        if not isinstance(server, (McpServerStdio, HttpMcpServer, SseMcpServer)):
            configurations[name] = {"transport": "unsupported-acp"}
            warnings.append(
                f"MCP server {name!r} was not loaded: unsupported ACP server type "
                f"{type(server).__name__}."
            )
            continue
        if isinstance(server, McpServerStdio):
            configurations[name] = {
                "transport": "stdio",
                "command": server.command,
                "args": server.args,
                "env": {item.name: item.value for item in server.env},
            }
        else:
            configurations[name] = {
                "transport": "streamable-http" if isinstance(server, HttpMcpServer) else "sse",
                "url": server.url,
                "headers": {item.name: item.value for item in server.headers},
            }
    return configurations, warnings
