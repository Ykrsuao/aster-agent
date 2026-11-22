# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve terminal MCP configuration without starting servers or expanding host values."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from aster.paths import get_user_dir

MCPConfigSource = Literal["user", "project", "client"]


@dataclass(frozen=True)
class MCPServerConfig:
    """One selected server, including invalid and explicitly disabled entries."""

    name: str
    source: MCPConfigSource
    config: dict[str, Any] | None = field(default=None, repr=False)
    error: str | None = None
    disabled: bool = False


@dataclass(frozen=True)
class MCPConfigResult:
    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class _JSONObject(dict[str, Any]):
    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__()
        self.duplicates: set[str] = set()
        for key, value in pairs:
            if key in self:
                self.duplicates.add(key)
            self[key] = value


def _has_duplicates(value: Mapping[str, Any]) -> bool:
    return isinstance(value, _JSONObject) and bool(value.duplicates)


def _text(value: Any, *, empty: bool = True) -> bool:
    return (
        isinstance(value, str)
        and (empty or bool(value.strip()))
        and (not value or value.isprintable())
    )


def _stdio_value(value: Any) -> bool:
    # argv and environment values reach the process directly; whitespace is data.
    return isinstance(value, str) and "\x00" not in value


def _string_mapping(value: Any, *, headers: bool = False) -> dict[str, str]:
    label = "headers" if headers else "env"
    error = f"MCP {label} must contain valid string keys and values."
    if not isinstance(value, Mapping) or _has_duplicates(value):
        raise ValueError(error)
    result: dict[str, str] = {}
    seen: set[str] = set()
    for key, item in value.items():
        if not _text(key, empty=False) or not (_text(item) if headers else _stdio_value(item)):
            raise ValueError(error)
        if headers:
            if (
                re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) is None
                or not item.isascii()
                or key.lower() in seen
            ):
                raise ValueError(error)
            seen.add(key.lower())
        elif "=" in key:
            raise ValueError(error)
        result[key] = item
    return result


def _server_config(value: Any, workspace: Path) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(value, Mapping):
        raise ValueError("MCP server configuration must be an object.")
    if _has_duplicates(value):
        raise ValueError("MCP server configuration contains duplicate fields.")
    disabled = value.get("disabled", False)
    if not isinstance(disabled, bool):
        raise ValueError("MCP disabled must be a boolean.")
    if disabled:
        return None, True

    transport = value.get("transport", value.get("type"))
    aliases = {
        "stdio": "stdio",
        "http": "streamable-http",
        "streamable-http": "streamable-http",
        "sse": "sse",
    }
    if "transport" in value or "type" in value:
        if not isinstance(transport, str) or transport not in aliases:
            raise ValueError("MCP transport must be stdio, http, streamable-http or sse.")
        transport = aliases[transport]
        if "type" in value:
            kind = value["type"]
            if not isinstance(kind, str) or aliases.get(kind) != transport:
                raise ValueError("MCP type and transport must agree.")
    else:
        transport = "streamable-http" if "url" in value else "stdio"

    if transport == "stdio":
        if "url" in value or "headers" in value:
            raise ValueError("MCP stdio configuration cannot contain URL transport fields.")
        command = value.get("command")
        if not _text(command, empty=False):
            raise ValueError(
                "MCP stdio command must be a nonempty string without control characters."
            )
        args = value.get("args", [])
        if not isinstance(args, list) or any(not _stdio_value(arg) for arg in args):
            raise ValueError("MCP stdio args must be a list of strings without NUL characters.")
        return {
            "transport": "stdio",
            "command": command,
            "args": list(args),
            "env": _string_mapping(value.get("env", {})),
            # All terminal servers run in the session workspace, regardless of file origin.
            "cwd": str(workspace),
        }, False

    if any(key in value for key in ("command", "args", "env")):
        raise ValueError("MCP URL configuration cannot contain stdio transport fields.")
    url = value.get("url")
    if not _text(url, empty=False) or any(char.isspace() for char in url):
        raise ValueError("MCP URL must be an HTTP or HTTPS URL with a hostname.")
    try:
        parsed = urlsplit(url)
        valid_url = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
        _ = parsed.port
    except ValueError:
        valid_url = False
    if not valid_url:
        raise ValueError("MCP URL must be an HTTP or HTTPS URL with a hostname.")
    return {
        "transport": transport,
        "url": url,
        "headers": _string_mapping(value.get("headers", {}), headers=True),
    }, False


def _read_servers(path: Path, source: MCPConfigSource, warnings: list[str]) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, ValueError):
        warnings.append(f"Could not read {source} MCP configuration.")
        return {}
    if not text.strip():
        return {}
    try:
        data = json.loads(text, object_pairs_hook=_JSONObject)
    except (ValueError, RecursionError):
        warnings.append(f"Could not parse {source} MCP configuration.")
        return {}
    if not isinstance(data, Mapping) or _has_duplicates(data):
        warnings.append(f"Invalid {source} MCP configuration object.")
        return {}
    servers = data.get("mcpServers", {})
    if not isinstance(servers, Mapping):
        warnings.append(f"Invalid {source} MCP server collection.")
        return {}
    return servers


def load_mcp_config(
    workspace: Path,
    *,
    client_servers: Mapping[str, dict] | None = None,
    project_trusted: bool,
    user_file: Path | None = None,
) -> MCPConfigResult:
    """Resolve user → trusted project → client entries, replacing whole servers.

    Invalid higher-priority entries remain failures instead of reviving a lower
    layer's server or combining its credentials with a different endpoint.
    Values, including ``${VAR}``, stay literal; OAuth is never performed here.
    """
    workspace = Path(workspace).resolve()
    user_path = Path(user_file if user_file is not None else get_user_dir(".mcp.json")).resolve()
    project_path = workspace / ".mcp.json"
    result = MCPConfigResult()
    layers: list[tuple[MCPConfigSource, Mapping[str, Any]]] = [
        ("user", _read_servers(user_path, "user", result.warnings))
    ]
    if project_trusted:
        layers.append(("project", _read_servers(project_path, "project", result.warnings)))
    if client_servers is not None:
        if isinstance(client_servers, Mapping):
            layers.append(("client", client_servers))
        else:
            result.warnings.append("Invalid client MCP server collection.")

    for source, servers in layers:
        for name, value in servers.items():
            if not _text(name, empty=False) or len(name) > 256 or name != name.strip():
                result.warnings.append(f"Invalid server name in {source} MCP configuration.")
                continue
            try:
                if isinstance(servers, _JSONObject) and name in servers.duplicates:
                    raise ValueError("Duplicate MCP server name in this configuration.")
                config, disabled = _server_config(value, workspace)
                entry = MCPServerConfig(name, source, config=config, disabled=disabled)
            except ValueError as exc:
                entry = MCPServerConfig(name, source, error=str(exc))
            result.servers[name] = entry
    return result
