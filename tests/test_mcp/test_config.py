# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration resolution stays local, isolated and safe to diagnose."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from aster.mcp import config as mcp_config  # noqa: E402
from aster.mcp.config import load_mcp_config  # noqa: E402


def _write(path: Path, servers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def test_layers_replace_whole_servers_and_keep_invalid_overrides(tmp_path):
    workspace = tmp_path / "workspace"
    user = tmp_path / "user" / ".mcp.json"
    _write(
        user,
        {
            "project-wins": {"url": "https://user.invalid", "headers": {"Authorization": "secret"}},
            "client-wins": {"command": "user-command"},
            "invalid": {"command": "must-not-start"},
            "disabled": {"command": "must-not-start"},
            "user-only": {"command": "user-command"},
        },
    )
    _write(
        workspace / ".mcp.json",
        {
            "project-wins": {"url": "https://project.invalid"},
            "client-wins": {"command": "project-command", "env": {"KEY": "project-secret"}},
            "invalid": None,
            "disabled": {"disabled": True},
        },
    )
    result = load_mcp_config(
        workspace,
        user_file=user,
        project_trusted=True,
        client_servers={"client-wins": {"command": "client-command"}},
    )

    assert not result.warnings
    project = result.servers["project-wins"]
    assert project.source == "project"
    assert project.config == {
        "transport": "streamable-http",
        "url": "https://project.invalid",
        "headers": {},
    }
    client = result.servers["client-wins"]
    assert client.source == "client"
    assert client.config["command"] == "client-command"
    assert client.config["env"] == {}
    assert client.config["cwd"] == str(workspace.resolve())
    assert result.servers["invalid"].source == "project"
    assert result.servers["invalid"].config is None
    assert result.servers["invalid"].error == "MCP server configuration must be an object."
    assert result.servers["disabled"].disabled is True
    assert result.servers["disabled"].config is None
    assert result.servers["disabled"].error is None
    assert result.servers["user-only"].source == "user"


def test_untrusted_project_is_not_read_and_default_user_path_is_used(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    user = tmp_path / "user" / ".mcp.json"
    _write(user, {"user": {"command": "user-server"}})
    project = workspace / ".mcp.json"
    _write(project, {"project": {"command": "must-not-start"}})
    requested = []
    read_text = Path.read_text

    def checked_read(path, *args, **kwargs):
        assert path != project
        requested.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked_read)
    monkeypatch.setattr(
        mcp_config, "get_user_dir", lambda name: user if name == ".mcp.json" else None
    )
    result = load_mcp_config(workspace, project_trusted=False)
    assert list(result.servers) == ["user"]
    assert requested == [user]


def test_each_load_uses_its_workspace_and_current_config_without_changing_cwd(
    tmp_path, monkeypatch
):
    first, second = tmp_path / "first", tmp_path / "second"
    user = tmp_path / "missing-user.json"
    _write(first / ".mcp.json", {"server": {"command": "first-server", "cwd": "ignored"}})
    _write(second / ".mcp.json", {"server": {"command": "second-server"}})
    monkeypatch.chdir(tmp_path)
    before = Path.cwd()
    first_result = load_mcp_config(first, user_file=user, project_trusted=True)
    second_result = load_mcp_config(second, user_file=user, project_trusted=True)
    _write(first / ".mcp.json", {"server": {"command": "changed-server"}})
    changed = load_mcp_config(first, user_file=user, project_trusted=True)
    assert Path.cwd() == before
    assert first_result.servers["server"].config["command"] == "first-server"
    assert first_result.servers["server"].config["cwd"] == str(first)
    assert second_result.servers["server"].config["cwd"] == str(second)
    assert changed.servers["server"].config["command"] == "changed-server"


@pytest.mark.parametrize("contents", ["", " \r\n\t", "\ufeff", "{}", '{"mcpServers":{}}'])
def test_empty_configuration_never_produces_servers(tmp_path, contents):
    user = tmp_path / ".mcp.json"
    user.write_text(contents, encoding="utf-8")
    result = load_mcp_config(tmp_path / "absent", user_file=user, project_trusted=False)
    assert result.servers == {}
    assert result.warnings == []


@pytest.mark.parametrize(
    "contents",
    [
        '{"secret-leak-marker":',
        "null",
        "[]",
        '{"mcpServers":null}',
        '{"mcpServers":[]}',
        '{"mcpServers":{},"mcpServers":{}}',
    ],
)
def test_bad_layer_has_safe_warning_and_other_layers_still_load(tmp_path, contents):
    user = tmp_path / "secret-leak-marker.json"
    user.write_text(contents, encoding="utf-8")
    result = load_mcp_config(
        tmp_path / "workspace",
        user_file=user,
        project_trusted=True,
        client_servers={"valid": {"command": "server"}},
    )
    assert list(result.servers) == ["valid"]
    assert len(result.warnings) == 1
    assert "secret-leak-marker" not in result.warnings[0]


def test_duplicate_names_fail_only_that_server_and_do_not_revive_user_entry(tmp_path):
    user = tmp_path / "user.json"
    _write(user, {"same": {"command": "old"}})
    (tmp_path / ".mcp.json").write_text(
        '{"mcpServers":{"same":{"command":"first"},'
        '"valid":{"command":"other"},"same":{"command":"second"}}}',
        encoding="utf-8",
    )
    result = load_mcp_config(tmp_path, user_file=user, project_trusted=True)
    assert result.servers["same"].source == "project"
    assert result.servers["same"].config is None
    assert result.servers["same"].error == "Duplicate MCP server name in this configuration."
    assert result.servers["valid"].config["command"] == "other"


@pytest.mark.parametrize(
    "definition",
    [
        '{"command":"a","command":"b"}',
        '{"command":"a","env":{"KEY":"first","KEY":"second"}}',
        '{"url":"https://x.invalid","headers":{"X-Key":"first","X-Key":"second"}}',
    ],
)
def test_duplicate_config_fields_are_not_silently_selected(tmp_path, definition):
    user = tmp_path / "user.json"
    user.write_text('{"mcpServers":{"server":' + definition + "}}", encoding="utf-8")
    result = load_mcp_config(tmp_path, user_file=user, project_trusted=False)
    assert result.servers["server"].config is None
    assert result.servers["server"].error


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        "secret-leak-marker",
        {},
        {"command": None},
        {"command": 12},
        {"command": " \t"},
        {"command": "secret-leak-marker\x1b[31m"},
        {"command": "server", "args": None},
        {"command": "server", "args": "secret-leak-marker"},
        {"command": "server", "args": [None]},
        {"command": "server", "args": ["bad\x00arg"]},
        {"command": "server", "env": None},
        {"command": "server", "env": {"KEY": 1}},
        {"command": "server", "env": {"BAD=KEY": "secret-leak-marker"}},
        {"command": "server", "env": {"BAD\x00KEY": "secret-leak-marker"}},
        {"command": "server", "env": {"KEY": "secret-leak-marker\x00"}},
        {"command": "server", "disabled": "false"},
        {"disabled": None},
        {"disabled": 0},
        {"command": "server", "transport": None},
        {"command": "server", "type": []},
        {"command": "server", "transport": "secret-leak-marker"},
        {"type": "http", "transport": "stdio", "command": "server"},
        {"command": "server", "url": "https://x.invalid"},
        {"url": None},
        {"url": "file:///secret-leak-marker"},
        {"url": "https:///secret-leak-marker"},
        {"url": "https://x.invalid:99999"},
        {"url": "https://[malformed/secret-leak-marker"},
        {"url": "https://x.invalid/\nsecret"},
        {"url": "https://x.invalid", "headers": None},
        {"url": "https://x.invalid", "headers": {"Authorization": 5}},
        {"url": "https://x.invalid", "headers": {"Invalid Header": "secret-leak-marker"}},
        {"url": "https://x.invalid", "headers": {"Authorization": "Bearer secret\r\n"}},
        {
            "url": "https://x.invalid",
            "headers": {"Authorization": "first", "authorization": "second"},
        },
    ],
)
def test_invalid_server_is_retained_as_failure_without_echoing_values(tmp_path, invalid):
    result = load_mcp_config(
        tmp_path,
        user_file=tmp_path / "missing.json",
        project_trusted=False,
        client_servers={"bad": invalid, "valid": {"command": "server"}},
    )
    assert result.servers["bad"].config is None
    assert result.servers["bad"].source == "client"
    assert result.servers["bad"].error
    assert "secret-leak-marker" not in result.servers["bad"].error
    assert result.servers["valid"].config


def test_placeholders_stay_literal_and_input_mutations_do_not_change_results(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_HOST_SECRET", "host-secret-canary")
    servers = {
        "local": {
            "command": "${MCP_HOST_SECRET}",
            "args": ["${MCP_HOST_SECRET}"],
            "env": {"KEY": "${MCP_HOST_SECRET}"},
            "expand_env_vars": True,
        },
        "remote": {
            "type": "http",
            "url": "https://x.invalid/${MCP_HOST_SECRET}",
            "headers": {"Authorization": "Bearer ${MCP_HOST_SECRET}"},
        },
    }
    result = load_mcp_config(
        tmp_path,
        user_file=tmp_path / "missing.json",
        project_trusted=False,
        client_servers=servers,
    )
    servers["local"]["args"].append("mutated")
    servers["local"]["env"]["KEY"] = "mutated"
    servers["remote"]["headers"]["Authorization"] = "mutated"
    local = result.servers["local"].config
    remote = result.servers["remote"].config
    assert local["command"] == "${MCP_HOST_SECRET}"
    assert local["args"] == ["${MCP_HOST_SECRET}"]
    assert local["env"] == {"KEY": "${MCP_HOST_SECRET}"}
    assert remote["headers"]["Authorization"] == "Bearer ${MCP_HOST_SECRET}"
    assert remote["url"] == "https://x.invalid/${MCP_HOST_SECRET}"
    assert "host-secret-canary" not in repr((local, remote))


@pytest.mark.parametrize("source", ["user", "project", "client"])
def test_stdio_multiline_args_and_environment_stay_literal(tmp_path, source):
    script = "if True:\n\tprint('${UNEXPANDED}')\n"
    certificate = (
        "-----BEGIN CERTIFICATE-----\r\ncanary\t${UNEXPANDED}\r\n-----END CERTIFICATE-----\r\n"
    )
    servers = {
        "local": {
            "command": "python",
            "args": ["-c", script, ""],
            "env": {"CERTIFICATE": certificate, "EMPTY": ""},
        }
    }
    user_file = tmp_path / "user" / ".mcp.json"
    if source != "client":
        _write(user_file if source == "user" else tmp_path / ".mcp.json", servers)
    result = load_mcp_config(
        tmp_path,
        user_file=user_file,
        project_trusted=source == "project",
        client_servers=servers if source == "client" else None,
    )
    server = result.servers["local"]
    assert server.source == source
    assert server.error is None
    assert server.config["args"] == ["-c", script, ""]
    assert server.config["env"] == {"CERTIFICATE": certificate, "EMPTY": ""}


@pytest.mark.parametrize("transport", ["http", "streamable-http", "sse"])
def test_url_transport_aliases_and_disabled_entries(tmp_path, transport):
    result = load_mcp_config(
        tmp_path,
        user_file=tmp_path / "missing.json",
        project_trusted=False,
        client_servers={
            "remote": {"type": transport, "url": "https://x.invalid", "oauth_manual": True},
            "disabled": {"disabled": True, "url": None},
        },
    )
    assert result.servers["remote"].config["transport"] == (
        "streamable-http" if transport == "http" else transport
    )
    assert "oauth_manual" not in result.servers["remote"].config
    assert result.servers["disabled"].disabled
    assert result.servers["disabled"].error is None


def test_invalid_names_and_invalid_client_collection_do_not_echo_data(tmp_path):
    invalid_names = ["", " secret-leak-marker ", "secret-leak-marker\x1b[31m", "bad\nname", None]
    result = load_mcp_config(
        tmp_path,
        user_file=tmp_path / "missing.json",
        project_trusted=False,
        client_servers={name: {"command": "server"} for name in invalid_names},
    )
    assert not result.servers
    assert len(result.warnings) == len(invalid_names)
    assert "secret-leak-marker" not in repr(result.warnings)
    result = load_mcp_config(
        tmp_path,
        user_file=tmp_path / "missing.json",
        project_trusted=False,
        client_servers=["secret-leak-marker"],
    )
    assert not result.servers
    assert result.warnings == ["Invalid client MCP server collection."]


@pytest.mark.parametrize("source", ["user", "project", "client"])
def test_server_name_limit_matches_the_reconnect_contract(tmp_path, source):
    accepted = "a" * 256
    rejected = "secret-name-canary".ljust(257, "x")
    servers = {
        accepted: {"command": "server"},
        rejected: {"command": "server"},
    }
    user_file = tmp_path / "user" / ".mcp.json"
    if source != "client":
        _write(user_file if source == "user" else tmp_path / ".mcp.json", servers)
    result = load_mcp_config(
        tmp_path,
        user_file=user_file,
        project_trusted=source == "project",
        client_servers=servers if source == "client" else None,
    )
    assert list(result.servers) == [accepted]
    assert result.servers[accepted].error is None
    assert result.warnings == [f"Invalid server name in {source} MCP configuration."]
    assert "secret-name-canary" not in repr(result.warnings)


def test_read_errors_do_not_echo_filename_and_config_repr_hides_credentials(tmp_path):
    unreadable = tmp_path / "secret-leak-marker.json"
    unreadable.mkdir()
    result = load_mcp_config(
        tmp_path,
        user_file=unreadable,
        project_trusted=False,
        client_servers={
            "remote": {
                "url": "https://x.invalid",
                "headers": {"Authorization": "secret-leak-marker"},
            }
        },
    )
    assert result.warnings == ["Could not read user MCP configuration."]
    assert "secret-leak-marker" not in repr(result)
